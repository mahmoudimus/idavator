import itertools
import logging
import struct
import typing
from contextlib import suppress

import llvmlite.binding as llvm
import llvmlite.ir as ir
import numpy as np

import ida_auto
import ida_bytes
import ida_funcs
import ida_hexrays
import ida_ida
import ida_idaapi
import ida_nalt
import ida_name
import ida_segment
import ida_typeinf
import idaapi
import idautils

from idavator.events import EventEmitter
from idavator.persistence import (
    FIDELITY_EVENT,
    SEVERITY_CORRUPTION,
    SEVERITY_HARD_FAIL,
    SEVERITY_IMPRECISION,
    FidelityEvent,
)
from idavator.type_providers import TypeProvider, resolve_lvar_type

# ============================================================================
# Global Configuration & Cache Constants
# ============================================================================
i8ptr = ir.IntType(8).as_pointer()
ptrsize = 64 if ida_ida.inf_is_64bit() else 32
ptext = {}  # Cache decompiled functions: {address: decompiled_result}
refreshed_funcs = set()  # Prevent redundant triggers of decompilation without cache

# Standard FS segment size standard on Windows
FS_SEGMENT_SIZE = 0x10000  # 64KB

# Default fallback value for float extraction failures
DEFAULT_FLOAT_VALUE = 1.0

# Floating-point compare opcode -> (ordered?, cmpop) for LLVM ``fcmp``.
#
# The decompiler REUSES the integer set/jump opcodes for FPU compares, flagged by
# ``minsn_t.is_fpinsn()`` (SDK ``*F`` marker on m_setp/m_setnz/m_setz/m_setae/m_setb/
# m_seta/m_setbe and the matching m_j* jumps). On x86 a NaN operand of ``ucomiss``
# sets CF=ZF=PF=1, so the carry-based codes are ASYMMETRIC w.r.t. ordering:
#   ja / jae  (CF=0[/&ZF=0])  -> taken ONLY when ordered      -> ogt / oge
#   jb / jbe  (CF=1[|ZF=1])   -> taken when less-or UNORDERED  -> ult / ule
#   jz        (ZF=1)          -> equal or UNORDERED            -> ueq
#   jnz       (ZF=0)          -> ordered and not-equal         -> one
# Verified against LLVM's own x86 backend (``llc -mtriple=x86_64`` of each ``fcmp``
# predicate): ogt->seta, oge->setae, ult->setb, ule->setbe, ueq->sete, one->setne.
# This is the polarity the int-cast+icmp path silently destroyed (it truncated both
# float operands to int and compared the bit patterns), inverting compares such as
# ``growth_threshold > 0.1`` into ``(unsigned)0.1`` == 0 nonsense. The OPCODE (not the
# generic cmpop string) decides ordered-vs-unordered, so this map is keyed by opcode.
def _fp_cmp_predicate_table():
    hx = ida_hexrays
    return {
        hx.m_seta: (True, ">"),    # ogt
        hx.m_ja: (True, ">"),
        hx.m_setae: (True, ">="),  # oge
        hx.m_jae: (True, ">="),
        hx.m_setb: (False, "<"),   # ult
        hx.m_jb: (False, "<"),
        hx.m_setbe: (False, "<="),  # ule
        hx.m_jbe: (False, "<="),
        hx.m_setz: (False, "=="),  # ueq
        hx.m_jz: (False, "=="),
        hx.m_setnz: (True, "!="),  # one
        hx.m_jnz: (True, "!="),
    }


_FP_CMP_PREDICATE = _fp_cmp_predicate_table()


def _fp_compare(builder, l, r, ida_insn):
    """Emit an LLVM ``fcmp`` for a floating-point microcode compare, or ``None`` if
    the operands are not float / the opcode has no FP form. ``l`` and ``r`` are the
    lifted operand values; the ordered-vs-unordered predicate is chosen from the
    microcode OPCODE (see ``_FP_CMP_PREDICATE``)."""
    if l is None or r is None:
        return None
    l_fp = isinstance(getattr(l, "type", None), (ir.FloatType, ir.DoubleType))
    r_fp = isinstance(getattr(r, "type", None), (ir.FloatType, ir.DoubleType))
    if not (l_fp or r_fp):
        return None
    entry = _FP_CMP_PREDICATE.get(ida_insn.opcode)
    if entry is None:
        return None
    # Promote operands to a common float type so an asymmetric pair (a double field
    # compared against a float literal) is WIDENED, never truncated -- the bug being
    # fixed truncated BOTH sides to int. The wider float type wins (double > float);
    # an int operand (e.g. a constant 0 lifted as i32) is converted to that float
    # type. Width is taken from the actual lifted operand IR types when float (so no
    # precision is lost vs the load), falling back to the microcode operand size.
    def _fp_bits(val, mc_size):
        t = getattr(val, "type", None)
        if isinstance(t, ir.DoubleType):
            return 64
        if isinstance(t, ir.FloatType):
            return 32
        return mc_size * 8

    typ = float_type(max(_fp_bits(l, ida_insn.l.size), _fp_bits(r, ida_insn.r.size)) // 8)
    l = typecast(l, typ, builder)
    r = typecast(r, typ, builder)
    ordered, cmpop = entry
    if ordered:
        return builder.fcmp_ordered(cmpop, l, r)
    return builder.fcmp_unordered(cmpop, l, r)

# Fidelity ledger: each lossy lift decision is emitted here and persisted
# asynchronously to sqlite3 (see idavator.persistence). The store subscribes to
# this emitter in lift_binary_to_llvm and drains on stop().
fidelity_emitter = EventEmitter()

# Lift-time type providers (e.g. CiRCLE struct recovery). Populated by
# lift_binary_to_llvm from outside; empty by default (no behavior change). Consulted
# when allocating function locals to type recovered struct pointers.
type_providers: list[TypeProvider] = []


def _emit_fidelity(kind, severity, *, function=None, ea=None, detail=None):
    """Emit a single fidelity-loss event onto the module-global emitter."""
    fidelity_emitter.emit(
        FIDELITY_EVENT,
        FidelityEvent(
            kind=kind, severity=severity, function=function, ea=ea, detail=detail
        ),
    )


def lift_tif(tif: ida_typeinf.tinfo_t, width: int = -1) -> ir.Type:
    """
    Translates an IDA type into its corresponding LLVM type.
    If the IDA type is an array, struct, or compound type, the translation
    is performed recursively.

    :param tif: The IDA type to convert
    :type tif: ida_typeinf.tinfo_t
    :raises NotImplementedError: Variadic or unsupported structure types
    :return: The converted LLVM type
    :rtype: ir.Type
    """
    if tif.is_func():
        ida_rettype = tif.get_rettype()
        ida_args = (tif.get_nth_arg(i) for i in range(tif.get_nargs()))
        is_vararg = tif.is_vararg_cc()
        llvm_rettype = lift_tif(ida_rettype)
        llvm_args = (lift_tif(arg) for arg in ida_args)
        return ir.FunctionType(
            i8ptr if isinstance(llvm_rettype, ir.VoidType) else llvm_rettype,
            llvm_args,
            var_arg=is_vararg,
        )

    elif tif.is_ptr():
        child_tif = tif.get_ptrarr_object()
        if child_tif.is_void():
            return ir.IntType(8).as_pointer()
        return lift_tif(child_tif).as_pointer()

    elif tif.is_array():
        child_tif = tif.get_ptrarr_object()
        element = lift_tif(child_tif)
        count = tif.get_array_nelems()
        if count == 0:
            # An array with an indeterminate number of elements defaults to a pointer type
            tif.convert_array_to_ptr()
            return lift_tif(tif)
        return ir.ArrayType(element, count)

    elif tif.is_void():
        return ir.VoidType()

    elif tif.is_udt():
        udt_data = ida_typeinf.udt_type_data_t()
        tif.get_udt_details(udt_data)
        type_name = tif.get_type_name() or "struct"
        context = ir.context.global_context

        if type_name not in context.identified_types:
            struct_t = context.get_identified_type(type_name)
            element_types = []
            for idx in range(udt_data.size()):
                udt_member = udt_data.at(idx)
                if tif.is_varstruct() and idx == udt_data.size() - 1:
                    continue
                member_name = udt_member.type.get_type_name()
                if member_name in context.identified_types:
                    element_types.append(context.identified_types[member_name])
                else:
                    element_types.append(lift_tif(udt_member.type))
            struct_t.set_body(*element_types)
            if tif.is_varstruct():
                logging.debug(
                    "variadic struct %s: lifted fixed prefix only (%d members)",
                    type_name,
                    len(element_types),
                )
                _emit_fidelity(
                    "varstruct_truncated", SEVERITY_IMPRECISION, detail=type_name
                )
        return context.get_identified_type(type_name)

    elif tif.is_bool():
        return ir.IntType(1)

    elif tif.is_char():
        return ir.IntType(8)

    elif tif.is_float():
        return ir.FloatType()

    elif tif.is_double():
        return ir.DoubleType()

    elif tif.is_decl_int() or tif.is_decl_uint() or tif.is_uint() or tif.is_int():
        return ir.IntType(tif.get_size() * 8)

    elif (
        tif.is_decl_int16() or tif.is_decl_uint16() or tif.is_uint16() or tif.is_int16()
    ):
        return ir.IntType(tif.get_size() * 8)

    elif (
        tif.is_decl_int32() or tif.is_decl_uint32() or tif.is_uint32() or tif.is_int32()
    ):
        return ir.IntType(tif.get_size() * 8)

    elif (
        tif.is_decl_int64() or tif.is_decl_uint64() or tif.is_uint64() or tif.is_int64()
    ):
        return ir.IntType(tif.get_size() * 8)

    elif (
        tif.is_decl_int128()
        or tif.is_decl_uint128()
        or tif.is_uint128()
        or tif.is_int128()
    ):
        return ir.IntType(tif.get_size() * 8)

    elif tif.is_ext_arithmetic() or tif.is_arithmetic():
        size_bits = tif.get_size() * 8
        # Prevent creating excessively large integer types (limited to 128 bits)
        if size_bits > 128 or size_bits <= 0:
            _emit_fidelity(
                "type_fallback_ptrsize",
                SEVERITY_IMPRECISION,
                detail=f"arithmetic size_bits={size_bits}",
            )
            return ir.IntType(ptrsize)
        return ir.IntType(size_bits)

    else:
        if width != -1:
            # Prevent creating excessively large arrays (limited to 1MB)
            if width > 1024 * 1024 or width <= 0:
                _emit_fidelity(
                    "type_fallback_ptrsize",
                    SEVERITY_IMPRECISION,
                    detail=f"oversized width={width}",
                )
                return ir.IntType(ptrsize)
            _emit_fidelity(
                "type_fallback_bytearray", SEVERITY_IMPRECISION, detail=f"width={width}"
            )
            return ir.ArrayType(ir.IntType(8), width)
        else:
            _emit_fidelity(
                "type_fallback_ptrsize", SEVERITY_IMPRECISION, detail="unknown type"
            )
            return ir.IntType(ptrsize)


def typecast(
    src: ir.Value, dst_type: ir.Type, builder: ir.IRBuilder, signed: bool = False
) -> ir.Value:
    """
    Casts a source value to a destination type.
    The generated casting instructions are inserted into the builder.

    :param src: Value to convert
    :type src: ir.Value
    :param dst_type: Target type
    :type dst_type: ir.Type
    :param builder: Instruction builder
    :type builder: ir.IRBuilder
    :param signed: Whether to preserve signedness, defaults to False
    :type signed: bool, optional
    :raises NotImplementedError: Unsupported type casting operations
    :return: Casted value
    :rtype: ir.Value
    """
    if not hasattr(src, "type"):
        logging.warning("cannot cast non-LLVM value %r to %s", src, dst_type)
        _emit_fidelity(
            "value_zero_substituted",
            SEVERITY_CORRUPTION,
            detail=f"non-LLVM src -> {dst_type}",
        )
        return _zero_initializer_for_type(dst_type)
    if isinstance(src.type, ir.VoidType):
        _emit_fidelity(
            "value_zero_substituted",
            SEVERITY_CORRUPTION,
            detail=f"void src -> {dst_type}",
        )
        return _zero_initializer_for_type(dst_type)
    if src.type != dst_type:
        if isinstance(src.type, ir.PointerType) and isinstance(
            dst_type, ir.PointerType
        ):
            return builder.bitcast(src, dst_type)
        elif isinstance(src.type, ir.PointerType) and isinstance(dst_type, ir.IntType):
            return builder.ptrtoint(src, dst_type)
        elif isinstance(src.type, ir.IntType) and isinstance(dst_type, ir.PointerType):
            return builder.inttoptr(src, dst_type)
        elif isinstance(src.type, ir.IntType) and isinstance(dst_type, ir.FloatType):
            return builder.uitofp(src, dst_type)
        elif (
            isinstance(src.type, ir.FloatType) or isinstance(src.type, ir.DoubleType)
        ) and isinstance(dst_type, ir.IntType):
            return (
                builder.fptosi(src, dst_type)
                if signed
                else builder.fptoui(src, dst_type)
            )
        elif isinstance(src.type, ir.FloatType) and isinstance(dst_type, ir.FloatType):
            return src
        elif (
            isinstance(src.type, ir.IntType)
            and isinstance(dst_type, ir.IntType)
            and src.type.width < dst_type.width
        ):
            return (
                builder.sext(src, dst_type) if signed else builder.zext(src, dst_type)
            )
        elif (
            isinstance(src.type, ir.IntType)
            and isinstance(dst_type, ir.IntType)
            and src.type.width > dst_type.width
        ):
            return builder.trunc(src, dst_type)
        elif isinstance(src.type, ir.IntType) and isinstance(dst_type, ir.DoubleType):
            return builder.uitofp(src, dst_type)
        elif isinstance(src.type, ir.FloatType) and isinstance(dst_type, ir.DoubleType):
            return builder.fpext(src, dst_type)
        elif isinstance(src.type, ir.DoubleType) and isinstance(dst_type, ir.FloatType):
            return builder.fptrunc(src, dst_type)
        elif isinstance(src.type, (ir.DoubleType, ir.FloatType)) and isinstance(
            dst_type, ir.PointerType
        ):
            tmp = (
                builder.fptosi(src, ir.IntType(ptrsize))
                if signed
                else builder.fptoui(src, ir.IntType(ptrsize))
            )
            return builder.inttoptr(tmp, dst_type)
        elif isinstance(dst_type, (ir.DoubleType, ir.FloatType)) and isinstance(
            src.type, ir.PointerType
        ):
            tmp = builder.ptrtoint(src, ir.IntType(ptrsize))
            return builder.uitofp(tmp, dst_type)
        elif isinstance(
            dst_type, (ir.IdentifiedStructType, ir.ArrayType)
        ) or isinstance(src.type, (ir.IdentifiedStructType, ir.ArrayType)):
            with builder.goto_entry_block():
                tmp = builder.alloca(src.type)
            builder.store(src, tmp)
            src = builder.load(builder.bitcast(tmp, dst_type.as_pointer()))
        else:
            return builder.bitcast(src, dst_type)
    return src


def _zero_initializer_for_type(typ: ir.Type) -> ir.Constant:
    if isinstance(typ, (ir.FloatType, ir.DoubleType)):
        return ir.Constant(typ, 0.0)
    if isinstance(typ, ir.PointerType):
        return ir.Constant(typ, None)
    if isinstance(typ, ir.IntType):
        return ir.Constant(typ, 0)
    return ir.Constant(typ, None)


def storecast(src, dst, builder):
    """
    Casts the type of dst into a pointer of the src type.
    """
    if src is None or isinstance(src.type, ir.VoidType):
        return dst
    if dst is not None and dst.type != src.type.as_pointer():
        dst = typecast(dst, src.type.as_pointer(), builder)
    return dst


def get_offset_to(builder: ir.IRBuilder, arg: ir.Value, off: int = 0) -> ir.Value:
    """
    Indexes a value relative to a given byte offset.

    :param arg: Base value to index from
    :type arg: ir.Value
    :param off: Index offset in bytes, defaults to 0
    :type off: int, optional
    :return: Indexed value after applying the offset
    :rtype: ir.Value
    """
    assert hasattr(arg, "type"), "arg must be a Value"
    if not isinstance(arg.type, ir.PointerType) or (
        (pointee := getattr(arg.type, "pointee", None)) is None
    ):
        if pointee is None:
            logging.warning(f"pointee is None for {arg.type}")
        return arg

    rval: ir.Value = arg
    pointee = typing.cast(ir.Type, pointee)
    if isinstance(pointee, ir.ArrayType):
        arr = pointee
        td = llvm.create_target_data("e")
        size = arr.element.get_abi_size(td)
        rval = builder.gep(
            arg,
            (
                ir.Constant(ir.IntType(32), 0),
                ir.Constant(ir.IntType(32), off // size),
            ),
        )
    elif isinstance(pointee, ir.IdentifiedStructType):
        # A struct-typed local addressed at a member byte offset (e.g. the IDA
        # microcode `ldx ..., new_ent@8` writing `new_ent.st_ino`). The base is
        # decayed to `i8*`, and a non-zero `off` must be carried as a byte GEP --
        # otherwise every member store collapses onto offset 0 and the later
        # writes become dead overwrites of the first (the missing-param-store
        # drop: `seen_file` lost `st_ino`/`st_dev`). The microcode preserved the
        # offset; only this decay erased it.
        rval = typecast(arg, ir.IntType(8).as_pointer(), builder)
        if off > 0:
            rval = builder.gep(rval, (ir.Constant(ir.IntType(32), off),))
    elif off > 0:
        td = llvm.create_target_data("e")
        size = pointee.get_abi_size(td)
        rval = builder.gep(arg, (ir.Constant(ir.IntType(32), off // size),))
    return rval


def dedereference(arg: ir.Value) -> ir.Value:
    """
    Performs a "de-dereference" operation: extracts the raw memory address from a loaded value.

    In LLVM, a LoadInstruction loads a value from a memory address (dereferencing).
    When we require the original memory address, we "de-dereference".

    Why this is required:
    - IDA microcode treats all local variables (LVARS) as register-like values.
    - During lifting, we treat all LVARS as stack-allocated variables (conforming to LLVM SSA).

    :param arg: Value to de-dereference
    :type arg: ir.Value
    :raises NotImplementedError: If the argument is not a LoadInstr or PointerType
    :return: The underlying raw memory address
    :rtype: ir.Value
    """
    if isinstance(arg, ir.LoadInstr):
        return arg.operands[0]
    arg_type = getattr(arg, "type", None)
    if arg_type is None:
        logging.warning(f"type is None for {arg}")
        return arg
    if isinstance(arg_type, ir.PointerType):
        return arg
    if isinstance(arg, ir.Constant) and isinstance(arg_type, ir.IntType):
        return arg
    raise NotImplementedError(
        f"not implemented: get reference for object {arg} of type {arg_type}"
    )


def lift_type_from_address(ea: int, pfunc=None):
    """Retrieves type information from a given effective address."""
    if (
        ida_funcs.get_func(ea) is not None
        and ida_segment.segtype(ea) & ida_segment.SEG_XTRN
    ):
        # Prefer the real prototype IDA already has for this import (libc imports
        # are typed via IDA's type libraries, e.g. strlen -> size_t(const char *)).
        imported = ida_typeinf.tinfo_t()
        if ida_nalt.get_tinfo(imported, ea) and (
            imported.is_func() or imported.is_funcptr()
        ):
            return imported

        # No real type available: synthesize a variadic void(...) as a last resort.
        ida_func_details = ida_typeinf.func_type_data_t()
        void = ida_typeinf.tinfo_t()
        void.create_simple_type(ida_typeinf.BTF_VOID)
        ida_func_details.rettype = void
        ida_func_details.set_cc(ida_typeinf.CM_CC_ELLIPSIS | ida_typeinf.CC_CDECL_OK)

        function_tinfo = ida_typeinf.tinfo_t()
        function_tinfo.create_func(ida_func_details)
        _emit_fidelity(
            "import_sig_synthesized",
            SEVERITY_IMPRECISION,
            ea=ea,
            function=ida_name.get_name(ea) or None,
        )
        return function_tinfo

    if ea in ptext:
        pfunc_cached = ptext.get(ea)
        if pfunc_cached is not None and getattr(pfunc_cached, "type", None) is not None:
            return pfunc_cached.type
        # Stale or invalid cache entry, drop and fall back to IDA type info.
        with suppress(KeyError):
            del ptext[ea]

    tif = ida_typeinf.tinfo_t()
    has_tinfo = ida_nalt.get_tinfo(tif, ea)
    if not has_tinfo:
        ida_typeinf.guess_tinfo(tif, ea)
        _emit_fidelity(
            "type_guessed",
            SEVERITY_IMPRECISION,
            ea=ea,
            function=ida_name.get_name(ea) or None,
        )
    return tif


def analyze_insn(module, ida_insn, ea):
    """
    Analyzes function call instructions for parameter count mismatches.

    Problem: Sometimes IDA's type propagation is incomplete, causing a mismatch
    between the number of arguments in a call and the function's actual signature.
    For example, function A calls function B with 3 arguments, but B's signature
    expects 4 arguments.

    Solution: Force re-decompilation of both caller and callee to refresh type
    information and resolve the mismatch.

    :param module: LLVM module being constructed
    :param ida_insn: Microcode instruction to analyze
    :param ea: Address of the instruction
    """
    if ida_insn.opcode == ida_hexrays.m_call:
        callnum = len(ida_insn.d.f.args)
        if ida_insn.l.t == ida_hexrays.mop_v:
            temp_ea = ida_insn.l.g
            func_name = ida_name.get_name(temp_ea)
            temp_func = ida_funcs.get_func(temp_ea)
            if temp_func is not None and (temp_func.flags & ida_funcs.FUNC_THUNK):
                tfunc_ea, _ptr = ida_funcs.calc_thunk_func_target(temp_func)
                if tfunc_ea != ida_idaapi.BADADDR:
                    temp_ea = tfunc_ea
                    func_name = ida_name.get_name(temp_ea)

            tif = lift_type_from_address(temp_ea)
            if tif.is_func() or tif.is_funcptr():
                argnum = tif.get_nargs()
                with suppress(KeyError):
                    if func_name == "":
                        func_name = f"data_{hex(ea)[2:]}"
                    if hasattr(module.get_global(func_name), "args"):
                        argnum = len(module.get_global(func_name).args)

                if callnum != argnum:
                    ida_hf = ida_hexrays.hexrays_failure_t()
                    if temp_ea not in refreshed_funcs:
                        refreshed_funcs.add(temp_ea)
                        try:
                            pfunc = ida_hexrays.decompile(
                                temp_ea, ida_hf, ida_hexrays.DECOMP_NO_CACHE
                            )
                            if pfunc is not None:
                                ptext[temp_ea] = pfunc
                        except Exception:
                            pass

                    if ea not in refreshed_funcs:
                        refreshed_funcs.add(ea)
                        try:
                            pfunc = ida_hexrays.decompile(
                                ea, ida_hf, ida_hexrays.DECOMP_NO_CACHE
                            )
                            if pfunc is not None:
                                ptext[ea] = pfunc
                        except Exception:
                            return

    if ida_insn.l.t == ida_hexrays.mop_d:
        analyze_insn(module, ida_insn.l.d, ea)
    if ida_insn.r.t == ida_hexrays.mop_d:
        analyze_insn(module, ida_insn.r.d, ea)
    if ida_insn.d.t == ida_hexrays.mop_d:
        analyze_insn(module, ida_insn.d.d, ea)


def lift_from_address(
    module: ir.Module, ea: int, typ: typing.Optional[ir.Type] = None
) -> ir.Value:
    if typ is None:
        tif = lift_type_from_address(ea)
        typ = lift_tif(tif)
    return _lift_from_address(module, ea, typ)


def _lift_from_address(module: ir.Module, ea: int, typ: ir.Type):
    if isinstance(typ, ir.FunctionType):
        func_name = ida_name.get_name(ea) or f"data_{hex(ea)[2:]}"
        res = module.get_global(func_name)
        logging.debug(
            "res: %s, res.type: %s, type(res): %s",
            res,
            getattr(res, "type", None),
            type(res),
        )
        res.lvars = {}
        if ea in ptext:
            pfunc = ptext[ea]
        else:
            return res

        # Refresh function call configuration.
        # Sometimes there is a mismatch between caller and callee signatures.
        # We trigger a re-decompilation to fix this.
        mba = pfunc.mba
        for index in range(mba.qty):
            ida_blk = mba.get_mblock(index)
            ida_insn = ida_blk.head
            while ida_insn is not None:
                analyze_insn(module, ida_insn, ea)
                ida_insn = ida_insn.next

        if ea in ptext:
            pfunc = ptext[ea]
        else:
            return res

        mba = pfunc.mba
        for index in range(mba.qty):
            res.append_basic_block(name=f"@{index}")

        ida_func_details = ida_typeinf.func_type_data_t()
        tif = lift_type_from_address(ea, pfunc)
        tif.get_func_details(ida_func_details)
        names = []

        builder = ir.IRBuilder(res.entry_basic_block)

        with builder.goto_entry_block():
            # Declare function results as stack variables
            if not isinstance(typ.return_type, ir.VoidType):
                res.lvars["funcresult"] = builder.alloca(
                    typ.return_type, name="funcresult"
                )

            for lvar in list(pfunc.lvars):
                if lvar.is_result_var:
                    continue
                # Prefer a provider-supplied type for this local (e.g. CiRCLE
                # struct recovery); fall back to the IDA-derived type otherwise.
                arg_t = resolve_lvar_type(type_providers, ea, lvar.name)
                if arg_t is None:
                    arg_t = lift_tif(lvar.tif)
                res.lvars[lvar.name] = builder.alloca(arg_t, name=lvar.name)
                if lvar.is_arg_var:
                    names.append(lvar.name)

            # If function is variadic, declare va_start intrinsic
            if tif.is_vararg_cc() and typ.var_arg:
                ptr = builder.alloca(ir.IntType(8).as_pointer(), name="ArgList")
                res.lvars["ArgList"] = ptr
                va_start = module.declare_intrinsic(
                    "llvm.va_start",
                    fnty=ir.FunctionType(ir.VoidType(), [ir.IntType(8).as_pointer()]),
                )
                ptr = builder.load(ptr)
                builder.call(va_start, (ptr,))

            # Store stack variables
            for arg, arg_n in zip(res.args, names):
                arg = typecast(arg, res.lvars[arg_n].type.pointee, builder)
                builder.store(arg, res.lvars[arg_n])

        with builder.goto_block(res.blocks[-1]):
            if isinstance(typ.return_type, ir.VoidType):
                builder.ret_void()
            else:
                builder.ret(builder.load(res.lvars["funcresult"]))

        # Lift each basic block in CFG
        for index, blk in enumerate(res.blocks):
            ida_blk = mba.get_mblock(index)
            ida_insn = ida_blk.head
            while ida_insn is not None:
                lift_insn(ida_insn, blk, builder)
                ida_insn = ida_insn.next

            if not blk.is_terminated and index + 1 < len(res.blocks):
                with builder.goto_block(blk):
                    builder.branch(res.blocks[index + 1])

        # If function is variadic, declare va_end intrinsic
        if tif.is_vararg_cc() and typ.var_arg:
            ptr = res.lvars["ArgList"]
            va_end = module.declare_intrinsic(
                "llvm.va_end",
                fnty=ir.FunctionType(ir.VoidType(), [ir.IntType(8).as_pointer()]),
            )
            with builder.goto_block(res.blocks[-1]):
                ptr = builder.load(ptr)
                builder.call(va_end, (ptr,))
        return res
    elif isinstance(typ, ir.IntType):
        # should probably check endianness, BOOL type is IntType(1)
        r = ida_bytes.get_bytes(ea, 1 if typ.width // 8 < 1 else typ.width // 8)
        return typ(0) if r is None else typ(int.from_bytes(r, "little"))
    elif isinstance(typ, ir.FloatType):
        r = ida_bytes.get_bytes(ea, 4)
        value = struct.unpack("f", r)
        return typ(np.float32(0)) if r is None else typ(np.float32(value[0]))
    elif isinstance(typ, ir.DoubleType):
        r = ida_bytes.get_bytes(ea, 8)
        value = struct.unpack("d", r)
        return typ(np.float64(0)) if r is None else typ(np.float64(value[0]))
    elif isinstance(typ, ir.PointerType):
        return ir.Constant(typ, None)
    elif isinstance(typ, ir.ArrayType):
        td = llvm.create_target_data("e")
        sub_size = typ.element.get_abi_size(td)
        array = [
            lift_from_address(module, sub_ea, typ.element)
            for sub_ea in range(ea, ea + sub_size * typ.count, sub_size)
        ]
        return ir.Constant.literal_array(array)
    elif isinstance(typ, (ir.LiteralStructType, ir.IdentifiedStructType)):
        td = llvm.create_target_data("e")
        struct_eles = []
        for el in typ.elements:
            struct_ele = lift_from_address(module, ea, el)
            struct_eles.append(struct_ele)
            sub_size = el.get_abi_size(td)
            ea += sub_size
        return ir.Constant(typ, struct_eles)
    else:
        raise NotImplementedError(f"object at {hex(ea)} is of unsupported type {typ}")


def str2size(str_size: str) -> int:
    """
    Converts a string representing memory size into its size in bits.

    :param str_size: String describing size (e.g., 'byte', 'word')
    :type str_size: str
    :return: Size of the description, in bits
    :rtype: int
    """
    size_map = {"byte": 8, "word": 16, "dword": 32, "qword": 64}
    if str_size not in size_map:
        raise AssertionError(
            f"String size must be one of {list(size_map.keys())}, but got '{str_size}'"
        )
    return size_map[str_size]


def lift_intrinsic_function(module: ir.Module, func_name: str):
    """
    Lifts IDA macros to corresponding LLVM intrinsics.

    Hexray's decompiler recognises higher-level functions at the Microcode level.
        Such ida_hexrays:mop_t objects are typed as ida_hexrays.mop_h (auxillary function member)

        This improves decompiler output, representing operations that cannot be mapped to nice C code
        (https://hex-rays.com/blog/igors-tip-of-the-week-67-decompiler-helpers/).

        For relevant #define macros, refer to IDA SDK: `defs.h` and `pro.h`.

    LLVM intrinsics have well known names and semantics and are required to follow certain restrictions.

    :param module: _description_
    :type module: ir.Module
    :param func_name: _description_
    :type func_name: str
    :raises NotImplementedError: _description_
    :return: _description_
    :rtype: _type_
    """
    with suppress(KeyError):
        return module.get_global(func_name)

    if func_name == "sadd_overflow":
        typ = ir.LiteralStructType((ir.IntType(64), ir.IntType(1)))
        return module.declare_intrinsic(
            "sadd_overflow",
            fnty=ir.FunctionType(typ.as_pointer(), [ir.IntType(64), ir.IntType(64)]),
        )

    elif func_name == "__OFSUB__":
        return module.declare_intrinsic(
            "__OFSUB__",
            fnty=ir.FunctionType(ir.IntType(1), [ir.IntType(64), ir.IntType(64)]),
        )

    elif func_name == "_mm_cvtsi128_si32":
        return module.declare_intrinsic(
            "_mm_cvtsi128_si32", fnty=ir.FunctionType(ir.IntType(32), [ir.IntType(128)])
        )

    elif func_name == "_BitScanReverse":
        return module.declare_intrinsic(
            "_BitScanReverse",
            fnty=ir.FunctionType(i8ptr, [ir.IntType(32), ir.IntType(32)]),
        )

    elif func_name == "__FYL2X__":
        return module.declare_intrinsic(
            "__FYL2X__",
            fnty=ir.FunctionType(ir.DoubleType(), [ir.DoubleType(), ir.DoubleType()]),
        )

    elif func_name == "__FYL2P__":
        return module.declare_intrinsic(
            "__FYL2P__",
            fnty=ir.FunctionType(ir.DoubleType(), [ir.DoubleType(), ir.DoubleType()]),
        )

    elif func_name == "fabs":
        return module.declare_intrinsic(
            "fabs", fnty=ir.FunctionType(ir.DoubleType(), [ir.DoubleType()])
        )

    elif func_name == "fabsf":
        return module.declare_intrinsic(
            "fabsf", fnty=ir.FunctionType(ir.FloatType(), [ir.FloatType()])
        )

    elif func_name == "fabsl":
        return module.declare_intrinsic(
            "fabs", fnty=ir.FunctionType(ir.DoubleType(), [ir.DoubleType()])
        )

    elif func_name == "memcpy":
        return module.declare_intrinsic(
            "memcpy", fnty=ir.FunctionType(i8ptr, [i8ptr, i8ptr, ir.IntType(64)])
        )

    elif func_name == "_byteswap_ulong":
        return module.declare_intrinsic(
            "_byteswap_ulong", fnty=ir.FunctionType(ir.IntType(32), [ir.IntType(32)])
        )

    elif func_name == "_byteswap_uint64":
        return module.declare_intrinsic(
            "_byteswap_uint64", fnty=ir.FunctionType(ir.IntType(64), [ir.IntType(64)])
        )

    elif func_name == "memset":
        return module.declare_intrinsic(
            "memset",
            fnty=ir.FunctionType(i8ptr, [i8ptr, ir.IntType(32), ir.IntType(32)]),
        )

    elif func_name == "abs64":
        return module.declare_intrinsic(
            "abs64", fnty=ir.FunctionType(ir.IntType(64), [ir.IntType(64)])
        )

    elif func_name == "__PAIR64__":
        return module.declare_intrinsic(
            "__PAIR64__",
            fnty=ir.FunctionType(ir.IntType(64), [ir.IntType(32), ir.IntType(32)]),
        )

    elif func_name == "__PAIR128__":
        return module.declare_intrinsic(
            "__PAIR128__",
            fnty=ir.FunctionType(ir.IntType(128), [ir.IntType(64), ir.IntType(64)]),
        )

    elif func_name == "__PAIR32__":
        return module.declare_intrinsic(
            "__PAIR32__",
            fnty=ir.FunctionType(ir.IntType(32), [ir.IntType(16), ir.IntType(16)]),
        )

    elif func_name == "__PAIR16__":
        return module.declare_intrinsic(
            "__PAIR16__",
            fnty=ir.FunctionType(ir.IntType(16), [ir.IntType(8), ir.IntType(8)]),
        )

    elif func_name == "_BitScanReverse64":
        return module.declare_intrinsic(
            "_BitScanReverse64",
            fnty=ir.FunctionType(i8ptr, [ir.IntType(64).as_pointer(), ir.IntType(64)]),
        )

    elif func_name == "_BitScanForward64":
        return module.declare_intrinsic(
            "_BitScanForward64",
            fnty=ir.FunctionType(i8ptr, [ir.IntType(64).as_pointer(), ir.IntType(64)]),
        )

    elif func_name == "__halt":
        fty = ir.FunctionType(ir.VoidType(), [])
        f = ir.Function(module, fty, "__halt")
        f.append_basic_block()
        builder = ir.IRBuilder(f.entry_basic_block)
        builder.asm(fty, "hlt", "", (), True)
        builder.ret_void()
        return f

    elif func_name == "is_mul_ok":
        return module.declare_intrinsic(
            "is_mul_ok",
            fnty=ir.FunctionType(ir.IntType(8), [ir.IntType(64), ir.IntType(64)]),
        )

    elif func_name == "va_start":
        return module.declare_intrinsic(
            "va_start",
            fnty=ir.FunctionType(ir.VoidType(), [ir.IntType(8).as_pointer()]),
        )

    elif func_name == "va_arg":
        return module.declare_intrinsic(
            "va_arg", fnty=ir.FunctionType(ir.IntType(64), [ir.IntType(8).as_pointer()])
        )

    elif func_name == "va_end":
        return module.declare_intrinsic(
            "va_end", fnty=ir.FunctionType(ir.VoidType(), [ir.IntType(8).as_pointer()])
        )

    elif func_name == "_QWORD":
        return module.declare_intrinsic(
            "IDA_QWORD", fnty=ir.FunctionType(ir.IntType(8).as_pointer(), [])
        )

    elif func_name == "__ROL8__":
        return module.declare_intrinsic(
            "__ROL8__",
            fnty=ir.FunctionType(ir.IntType(64), [ir.IntType(64), ir.IntType(8)]),
        )

    elif func_name == "__ROL4__":
        return module.declare_intrinsic(
            "__ROL4__",
            fnty=ir.FunctionType(ir.IntType(64), [ir.IntType(64), ir.IntType(8)]),
        )

    elif func_name == "__ROR4__":
        return module.declare_intrinsic(
            "__ROR4__",
            fnty=ir.FunctionType(ir.IntType(64), [ir.IntType(64), ir.IntType(8)]),
        )

    elif func_name == "__ROR8__":
        return module.declare_intrinsic(
            "__ROR8__",
            fnty=ir.FunctionType(ir.IntType(64), [ir.IntType(64), ir.IntType(8)]),
        )

    elif func_name.startswith("__readfs"):
        _, size_str = func_name.split("__readfs")
        size = str2size(size_str)

        try:
            fs_reg = module.get_global("virtual_fs")
        except KeyError:
            fs_reg_typ = ir.ArrayType(ir.IntType(8), FS_SEGMENT_SIZE)
            fs_reg = ir.GlobalVariable(module, fs_reg_typ, "virtual_fs")
            fs_reg.initializer = fs_reg_typ(None)

        fty = ir.FunctionType(
            ir.IntType(size),
            [
                ir.IntType(32),
            ],
        )

        f = ir.Function(module, fty, func_name)
        (offset,) = f.args
        f.append_basic_block()
        builder = ir.IRBuilder(f.entry_basic_block)
        pointer = builder.gep(
            fs_reg,
            (ir.Constant(ir.IntType(32), 0), offset),
            inbounds=False,
        )
        pointer = typecast(pointer, ir.IntType(size).as_pointer(), builder)
        res = builder.load(pointer, align=1)
        builder.ret(res)
        return f

    elif func_name.startswith("__writefs"):
        _, size_str = func_name.split("__writefs")
        size = str2size(size_str)

        try:
            fs_reg = module.get_global("virtual_fs")
        except KeyError:
            fs_reg_typ = ir.ArrayType(ir.IntType(8), FS_SEGMENT_SIZE)
            fs_reg = ir.GlobalVariable(module, fs_reg_typ, "virtual_fs")
            fs_reg.initializer = fs_reg_typ(None)

        fty = ir.FunctionType(ir.VoidType(), [ir.IntType(32), ir.IntType(size)])

        f = ir.Function(module, fty, func_name)
        offset, value = f.args
        f.append_basic_block()
        builder = ir.IRBuilder(f.entry_basic_block)
        pointer = builder.gep(
            fs_reg,
            (ir.Constant(ir.IntType(32), 0), offset),
            inbounds=False,
        )
        pointer = typecast(pointer, ir.IntType(size).as_pointer(), builder)
        builder.store(value, pointer, align=1)
        builder.ret_void()
        return f

    elif func_name.startswith("sys_") or func_name.startswith(
        ("_InterlockedCompareExchange", "_InterlockedExchange")
    ):
        return ir.Function(
            module, ir.FunctionType(ir.IntType(64), [], var_arg=True), func_name
        )
    else:
        raise NotImplementedError(f"unsupported intrinsic helper: {func_name}")


def demangle_name(mangled_name: str) -> tuple[str, str]:
    """Return (demangled_name, original_mangled_name) for UI display."""
    demangled = ida_name.demangle_name(mangled_name, 0)
    if not demangled:
        demangled = mangled_name
    return demangled, mangled_name


def create_name_comment(demangled_name: str, mangled_name: str) -> str:
    """LLVM IR comment line preserving the original mangled symbol name."""
    if demangled_name.strip('"') != mangled_name:
        return f"; Mangled: {mangled_name}"
    return ""


def format_llvm_module(module: ir.Module) -> str:
    """Serialize a module to LLVM IR text, with mangled-name comments where useful."""
    comments: dict[str, str] = {}
    for func in module.functions:
        if not func.name.startswith("_Z"):
            continue
        demangled, mangled = demangle_name(func.name)
        comment = create_name_comment(demangled, mangled)
        if comment:
            comments[func.name] = comment

    if not comments:
        return str(module)

    out: list[str] = []
    for line in str(module).splitlines():
        stripped = line.lstrip()
        for mangled, comment in comments.items():
            if f"@{mangled}" in line and (stripped.startswith(("define ", "declare "))):
                out.append(comment)
                break
        out.append(line)
    text = "\n".join(out)
    return f"{text}\n" if text else ""


def lift_function(
    module: ir.Module,
    func_name: str,
    is_declare: bool,
    ea: typing.Optional[int] = None,
    tif: typing.Optional[ida_typeinf.tinfo_t] = None,
):
    """
    Declares a function given its name.
    If `is_declare` is False, the function is recursively defined by lifting
    its instructions from the IDA decompiler output.
    If `tif` is provided, the function type is enforced accordingly.
    The primary lifting logic resides in `lift_from_address`.

    :param module: Parent module of the function
    :type module: ir.Module
    :param func_name: Name of the function to lift
    :type func_name: str
    :param is_declare: Whether the function is declaration-only
    :type is_declare: bool
    :param tif: Function type metadata, defaults to None
    :type tif: ida_typeinf.tinfo_t, optional
    :return: The lifted LLVM function
    :rtype: ir.Function
    """
    if func_name == "" and ea is not None:
        func_name = f"data_{hex(ea)[2:]}"
    with suppress(NotImplementedError):
        return lift_intrinsic_function(module, func_name)

    with suppress(KeyError):
        return module.get_global(func_name)

    func_ea = ida_name.get_name_ea(ida_idaapi.BADADDR, func_name)
    if ida_segment.segtype(func_ea) & ida_segment.SEG_XTRN:
        is_declare = True

    if func_ea == ida_idaapi.BADADDR:
        func_ea = ea

    assert func_ea is not None
    assert func_ea != ida_idaapi.BADADDR
    if tif is None:
        tif = lift_type_from_address(func_ea)

    typ = lift_tif(tif)
    res = ir.Function(module, typ, func_name)
    if is_declare:
        return res
    return lift_from_address(module, func_ea, typ)


_SEGMENT_MREGS = frozenset({240, 248, 256, 264, 272})  # cs, ss, ds, fs, gs


def _mreg_is_segment(mop: ida_hexrays.mop_t) -> bool:
    if mop.r in _SEGMENT_MREGS:
        return True
    if hasattr(ida_hexrays, "get_mreg_name"):
        with suppress(Exception):
            return ida_hexrays.get_mreg_name(mop.r, mop.size) in (
                "cs",
                "ds",
                "es",
                "fs",
                "gs",
                "ss",
            )
    return False


def lift_microregister(
    mop: ida_hexrays.mop_t,
    blk: ir.Block,
    builder: ir.IRBuilder,
    *,
    dest: bool = False,
) -> ir.Value:
    """
    Lift a micro-register (mop_r).

    Segment selectors are ignored (flat address space); other registers use
    per-function stack slots.
    """
    size_bytes = mop.size if mop.size > 0 else 1
    typ = ir.IntType(size_bytes * 8)

    if _mreg_is_segment(mop):
        if dest:
            func = blk.parent
            name = f"seg_{mop.r}"
            with suppress(Exception):
                name = f"seg_{ida_hexrays.get_mreg_name(mop.r, mop.size)}"
            if name not in func.lvars:
                with builder.goto_entry_block():
                    func.lvars[name] = builder.alloca(typ, name=name)
            return func.lvars[name]
        return ir.Constant(typ, 0)

    func = blk.parent
    name = f"__mreg_{mop.r}_{size_bytes}"
    if name not in func.lvars:
        with builder.goto_entry_block():
            func.lvars[name] = builder.alloca(typ, name=name)
    slot = func.lvars[name]
    return slot if dest else builder.load(slot)


def calc_instsize(typ):
    """
    Calculates instruction width in bits.
    """
    if isinstance(typ, ir.PointerType):
        return ptrsize
    elif isinstance(typ, ir.ArrayType):
        return -1
    elif isinstance(typ, ir.IdentifiedStructType):
        return -1
    elif isinstance(typ, ir.FloatType):
        return 32
    elif isinstance(typ, ir.DoubleType):
        return 64
    else:
        return typ.width


def lift_mop(
    mop: ida_hexrays.mop_t,
    blk: ir.Block,
    builder: ir.IRBuilder,
    dest=False,
    knowntyp=None,
) -> typing.Optional[ir.Value]:
    """
    Lifts a microcode operand (mop) to an LLVM value.
    """
    builder.position_at_end(blk)
    if mop.t == ida_hexrays.mop_r:  # register value
        return lift_microregister(mop, blk, builder, dest=dest)
    elif mop.t == ida_hexrays.mop_n:  # immediate value
        res = ir.Constant(ir.IntType(mop.size * 8), mop.nnn.value)
        setattr(res, "parent", blk)
        return res
    elif mop.t == ida_hexrays.mop_d:  # another instruction
        d = lift_insn(mop.d, blk, builder)
        if d is None:
            logging.warning("nested instruction lift failed for %s", mop.dstr())
            return None
        if isinstance(d.type, ir.VoidType):
            pass
        elif mop.size == -1:
            pass
        elif isinstance(mop, ida_hexrays.mcallarg_t):
            lltype = lift_tif(mop.type)
            d = typecast(d, lltype, builder, signed=mop.type.is_signed())
        elif knowntyp is not None:
            d = typecast(d, knowntyp, builder)
        elif calc_instsize(d.type) != mop.size * 8:
            d = typecast(d, ir.IntType(mop.size * 8), builder)
        return d
    elif mop.t == ida_hexrays.mop_l:  # local variables
        lvar = mop.l.var()
        name = "funcresult" if lvar.is_result_var else lvar.name
        off = mop.l.off
        func = blk.parent
        llvm_arg = func.lvars[name]
        llvm_arg = get_offset_to(builder, llvm_arg, off)
        if mop.size == -1:
            pass
        elif knowntyp is not None:
            llvm_arg = typecast(llvm_arg, knowntyp, builder)
        elif calc_instsize(llvm_arg.type.pointee) != mop.size * 8:
            llvm_arg = typecast(
                llvm_arg, ir.IntType(mop.size * 8).as_pointer(), builder
            )
        return llvm_arg if dest else builder.load(llvm_arg)
    elif mop.t == ida_hexrays.mop_S:  # stack variables
        name = "stack"
        func = blk.parent
        if name not in func.lvars:
            with builder.goto_entry_block():
                func.lvars[name] = builder.alloca(ir.IntType(ptrsize), name=name)
        llvm_arg = func.lvars[name]
        llvm_arg = get_offset_to(builder, llvm_arg, mop.s.off)
        if mop.size == -1:
            pass
        elif knowntyp is not None:
            llvm_arg = typecast(llvm_arg, knowntyp, builder)
        elif calc_instsize(llvm_arg.type.pointee) != mop.size * 8:
            llvm_arg = typecast(
                llvm_arg, ir.IntType(mop.size * 8).as_pointer(), builder
            )
        #        if (hasattr(llvm_arg.type.pointee, "width") and llvm_arg.type.pointee.width != mop.size * 8) and mop.size != -1:
        #            llvm_arg = typecast(llvm_arg, ir.IntType(mop.size * 8).as_pointer(), builder)
        return llvm_arg if dest else builder.load(llvm_arg)
    elif mop.t == ida_hexrays.mop_b:  # block number (used in jmp/call instruction)
        return blk.parent.blocks[mop.b]
    elif mop.t == ida_hexrays.mop_v:  # global variable
        ea = mop.g
        name = ida_name.get_name(ea) or f"data_{hex(ea)[2:]}"
        tif = lift_type_from_address(ea)
        if tif.is_func() or tif.is_funcptr():
            with suppress(KeyError):
                return blk.parent.parent.get_global(name)
            if tif.is_funcptr():
                tif = tif.get_ptrarr_object()
            # if function is a thunk function, define the actual function instead
            func = ida_funcs.get_func(ea)
            if func is not None and (func.flags & ida_funcs.FUNC_THUNK):
                tfunc_ea, ptr = ida_funcs.calc_thunk_func_target(func)
                if tfunc_ea != ida_idaapi.BADADDR:
                    ea = tfunc_ea
                    name = ida_name.get_name(ea) or f"data_{hex(ea)[2:]}"
                    tif = lift_type_from_address(ea)

            # If there is no function definition, or it is a library/external func, declare only
            if (
                ida_funcs.get_func(ea) is None
                or (ida_funcs.get_func(ea).flags & ida_funcs.FUNC_LIB)
                or ida_segment.segtype(ea) & ida_segment.SEG_XTRN
            ):
                g = lift_function(blk.parent.parent, name, True, ea, tif)
            else:
                g = lift_function(blk.parent.parent, name, False, ea, tif)
            return g

        else:
            if name in blk.parent.parent.globals:
                g = blk.parent.parent.get_global(name)
            else:
                tif = lift_type_from_address(ea)
                typ = lift_tif(tif)
                g_cmt = lift_from_address(blk.parent.parent, ea, typ)
                g_cmt_type = getattr(g_cmt, "type", None)
                if g_cmt_type is None:
                    raise ValueError(f"g_cmt.type is None for {name}")
                g = ir.GlobalVariable(blk.parent.parent, g_cmt_type, name=name)
                g.initializer = g_cmt

            if isinstance(g.type.pointee, (ir.IdentifiedStructType, ir.ArrayType)):
                g = builder.gep(
                    g, (ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), 0))
                )
            if mop.size == -1:
                pass
            elif knowntyp is not None:
                g = typecast(g, knowntyp, builder)
            elif calc_instsize(g.type.pointee) != mop.size * 8:
                g = typecast(g, ir.IntType(mop.size * 8).as_pointer(), builder)
            return g if dest else builder.load(g)
    elif mop.t == ida_hexrays.mop_f:  # function call information
        mcallinfo = mop.f
        f_args = []
        f_ret = []
        for i in range(mcallinfo.retregs.size()):
            mopt = mcallinfo.retregs.at(i)
            f_ret.append(lift_mop(mopt, blk, builder, dest))
        for arg in mcallinfo.args:
            typ = lift_tif(arg.type)
            f_arg = lift_mop(arg, blk, builder, dest, typ.as_pointer())

            if arg.t == ida_hexrays.mop_h and f_arg is None:
                f_arg = blk.parent.parent.declare_intrinsic(
                    arg.helper, fnty=ir.FunctionType(typ, [])
                )

            if arg.t == ida_hexrays.mop_r and f_arg is None:
                name = "fs"
                func = blk.parent
                if name not in func.lvars:
                    with builder.goto_entry_block():
                        func.lvars[name] = builder.alloca(ir.IntType(16), name=name)
                llvm_arg = func.lvars[name]
                f_arg = llvm_arg if mop.size == -1 else builder.load(llvm_arg)

            if f_arg is None:
                logging.warning("call argument lift failed for %s", arg.dstr())
                nested = arg.d.dstr() if arg.t == ida_hexrays.mop_d else ""
                _emit_fidelity(
                    "value_zero_substituted",
                    SEVERITY_CORRUPTION,
                    function=getattr(blk.parent, "name", None),
                    detail=f"call arg mopt={arg.t} nested[{nested}] -> {typ}",
                )
                f_arg = _zero_initializer_for_type(typ)
            f_arg = typecast(f_arg, typ, builder)
            f_args.append(f_arg)
        return f_ret, f_args
    elif mop.t == ida_hexrays.mop_a:  # operand address
        mop_addr = mop.a
        val = lift_mop(mop_addr, blk, builder, True)
        if val is None:
            logging.warning("address operand lift failed for %s", mop.dstr())
            return None
        if isinstance(mop, ida_hexrays.mcallarg_t):
            lltype = lift_tif(mop.type)
            val = typecast(val, lltype, builder)
        elif knowntyp is not None:
            val = typecast(val, knowntyp, builder)
        return val
    elif mop.t == ida_hexrays.mop_h:  # auxiliary function number
        with suppress(NotImplementedError):
            return lift_intrinsic_function(blk.parent.parent, mop.helper)
        return None
    elif mop.t == ida_hexrays.mop_str:  # string constant
        str_csnt = mop.cstr
        str_type = ir.ArrayType(ir.IntType(8), len(str_csnt))
        g = ir.GlobalVariable(
            blk.parent.parent, str_type, name=f"cstr_{len(blk.parent.parent.globals)}"
        )
        g.initializer = ir.Constant(str_type, bytearray(str_csnt.encode("utf-8")))
        g.linkage = "private"
        g.global_constant = True
        return typecast(g, ir.IntType(8).as_pointer(), builder)
    elif mop.t == ida_hexrays.mop_c:  # switch cases and targets
        mcases = {}
        for i in range(mop.c.size()):
            dst = mop.c.targets[i]
            if mop.c.values[i].size() == 0:
                mcases["default"] = dst
            for j in range(mop.c.values[i].size()):
                src = mop.c.values[i][j]
                mcases[src] = dst
        return mcases
    elif mop.t == ida_hexrays.mop_fn:
        # IDA float value extraction can crash under certain edge cases
        try:
            fp = mop.fpc.fnum.float
        except (AttributeError, ValueError, SystemError) as e:
            logging.debug(
                "Failed to extract float value: %s, using default %s",
                e,
                DEFAULT_FLOAT_VALUE,
            )
            _emit_fidelity(
                "float_default_substituted", SEVERITY_CORRUPTION, detail=str(e)
            )
            fp = DEFAULT_FLOAT_VALUE
        typ = float_type(mop.size)
        return ir.Constant(typ, fp)
    elif mop.t == ida_hexrays.mop_p:
        f = lift_intrinsic_function(blk.parent.parent, f"__PAIR{mop.size * 8}__")
        l = lift_mop(mop.pair.hop, blk, builder, dest)
        r = lift_mop(mop.pair.lop, blk, builder, dest)
        l = typecast(l, ir.IntType(mop.size * 4), builder)
        r = typecast(r, ir.IntType(mop.size * 4), builder)
        return builder.call(f, (l, r))
    elif mop.t == ida_hexrays.mop_sc:
        pass
    elif mop.t == ida_hexrays.mop_z:
        return None
    mop_descs = {
        ida_hexrays.mop_r: "register value",
        ida_hexrays.mop_n: "immediate value",
        ida_hexrays.mop_d: "another instruction",
        ida_hexrays.mop_l: "local variables",
        ida_hexrays.mop_S: "stack variables",
        ida_hexrays.mop_b: "block number (used in jmp/call instruction)",
        ida_hexrays.mop_v: "global variable",
        ida_hexrays.mop_f: "function call information",
        ida_hexrays.mop_a: "operand address (mop_l\\mop_v\\mop_S\\mop_r)",
        ida_hexrays.mop_h: "auxiliary function member",
        ida_hexrays.mop_str: "string constant",
        ida_hexrays.mop_c: "switch cases and targets",
        ida_hexrays.mop_fn: "floating point constant",
        ida_hexrays.mop_p: "pair operations",
        ida_hexrays.mop_sc: "scattered operation information",
    }
    raise NotImplementedError(
        f"not implemented: {mop.dstr()} of type {mop_descs[mop.t]}"
    )


def _store_as(
    l: typing.Optional[ir.Value],
    d: typing.Optional[ir.Value],
    blk: ir.Block,
    builder: ir.IRBuilder,
    d_typ: typing.Optional[ir.Type] = None,
    signed: bool = True,
) -> typing.Optional[ir.Value]:
    """
    Private helper function to store a value to its destination address.
    """
    if d is None:  # destination does not exist
        return l

    if l is None:
        return

    d = dedereference(d)
    if d_typ:
        d = typecast(d, d_typ, builder, signed)

    d_type: typing.Optional[ir.Type] = getattr(d, "type", None)
    assert d_type and isinstance(d_type, ir.PointerType)
    d_pointee = getattr(d_type, "pointee", None)

    if d_pointee and isinstance(d_pointee, ir.ArrayType):
        arrtoptr = d_pointee.element.as_pointer()
        d = typecast(d, arrtoptr.as_pointer(), builder, signed)

    l_type: typing.Optional[ir.Type] = getattr(l, "type", None)
    if l_type is None or isinstance(l_type, ir.VoidType):
        return

    with suppress(AttributeError):
        l_pointee = getattr(l_type, "pointee", None)
        # A struct/array-pointer VALUE (`l`) reaches this store in two very different
        # IDA constructs that must NOT be conflated:
        #   (a) pointer copy   `p = q`  -- the destination is a POINTER SLOT (`T**`):
        #       store the 8-byte pointer through it, exactly like the native
        #       `v3 = o` alias. The 128/129 lifter memcpys in examples/cp.ll are this
        #       case (e.g. set_char_quoting's `memcpy(v3_slot, *o, 56)` clobbering the
        #       pointer slot and severing the write-through to the caller's struct).
        #   (b) in-place copy  `*p = *q` -- the destination ADDRESSES THE AGGREGATE
        #       (`d_pointee` is the struct/array itself): a genuine byte-copy.
        # Only (b) is a real memcpy; gate on the destination pointee, not `l`.
        if (
            l_pointee
            and isinstance(l_pointee, (ir.IdentifiedStructType, ir.ArrayType))
            and not isinstance(d_pointee, ir.PointerType)
        ):
            dest, src = d, l
            td = llvm.create_target_data("e")
            length = ir.Constant(ir.IntType(64), l_pointee.get_abi_size(td))
            memcpy = lift_intrinsic_function(blk.parent.parent, "memcpy")
            src = typecast(src, ir.IntType(8).as_pointer(), builder)
            dest = typecast(dest, ir.IntType(8).as_pointer(), builder)
            return builder.call(memcpy, (dest, src, length))

    if d_pointee:
        if isinstance(d_pointee, ir.IdentifiedStructType):
            d = typecast(d, l_type, builder)
        else:
            l = typecast(l, d_pointee, builder, signed)

    return builder.store(l, d)


def create_intrinsic_function(module: ir.Module, func_name: str, ftif):
    """
    Creates an intrinsic function declaration for an IDA helper function.
    """
    argtypes = []
    for arg in ftif.args:
        argtypes.append(lift_tif(arg.type))

    rettype = lift_tif(ftif.return_type)
    if isinstance(rettype, ir.VoidType):
        rettype = i8ptr
    return module.declare_intrinsic(func_name, fnty=ir.FunctionType(rettype, argtypes))


def float_type(size: int) -> ir.Type:
    """
    Returns appropriate LLVM floating point representation given its byte length.
    """
    return ir.FloatType() if size == 4 else ir.DoubleType()


# ============================================================================
# Lift Operation Handlers
# ============================================================================


def _handle_binary_arithmetic(
    l, r, d, op_func, blk, builder, ida_insn, allow_ptr=False
) -> typing.Optional[ir.Value]:
    """
    Internal helper to process binary arithmetic logic.
    """
    if l is None or r is None:
        logging.warning("binary operand lift failed for instruction %s", ida_insn)
        return None
    if isinstance(l.type, (ir.FloatType, ir.DoubleType)):
        l = builder.fptoui(l, ir.IntType(ida_insn.l.size * 8))
    if isinstance(r.type, (ir.FloatType, ir.DoubleType)):
        r = builder.fptoui(r, ir.IntType(ida_insn.r.size * 8))

    if allow_ptr:
        if isinstance(l.type, ir.PointerType) and isinstance(r.type, ir.IntType):
            cast_ptr = typecast(l, ir.IntType(8).as_pointer(), builder)
            math = builder.gep(cast_ptr, (r,))
            math = typecast(math, l.type, builder)
        elif isinstance(r.type, ir.PointerType) and isinstance(l.type, ir.IntType):
            cast_ptr = typecast(r, ir.IntType(8).as_pointer(), builder)
            math = builder.gep(cast_ptr, (l,))
            math = typecast(math, r.type, builder)
        elif isinstance(l.type, ir.IntType) and isinstance(r.type, ir.IntType):
            math = op_func(l, r)
        elif isinstance(l.type, ir.PointerType) and isinstance(r.type, ir.PointerType):
            ptr_type = ir.IntType(64)
            const1 = builder.ptrtoint(l, ptr_type)
            const2 = builder.ptrtoint(r, ptr_type)
            math = op_func(const1, const2)
        else:
            raise NotImplementedError(
                f"unsupported pointer/arithmetic types: {l.type} and {r.type}"
            )
    else:
        l = typecast(l, ir.IntType(ida_insn.d.size * 8), builder)
        r = typecast(r, ir.IntType(ida_insn.d.size * 8), builder)
        math = op_func(l, r)

    d = storecast(l, d, builder)
    return _store_as(math, d, blk, builder)


def _handle_comparison(l, r, d, cmp_op, blk, builder, ida_insn, signed=False):
    """
    Internal comparison builder helper.
    """
    if l is None or r is None:
        logging.warning("comparison operand lift failed for instruction %s", ida_insn)
        return None
    # A floating-point compare (``is_fpinsn`` set, FP operand) must emit ``fcmp`` with
    # the correct ordered/unordered predicate -- NOT cast both floats to int and
    # ``icmp`` them, which truncates the values and inverts the result.
    if ida_insn.is_fpinsn():
        fcond = _fp_compare(builder, l, r, ida_insn)
        if fcond is not None:
            result = builder.select(
                fcond,
                ir.IntType(ida_insn.d.size * 8)(1),
                ir.IntType(ida_insn.d.size * 8)(0),
            )
            return _store_as(result, d, blk, builder)
    l = typecast(l, ir.IntType(ida_insn.l.size * 8), builder)
    r = typecast(r, ir.IntType(ida_insn.r.size * 8), builder)

    cond = (
        builder.icmp_signed(cmp_op, l, r)
        if signed
        else builder.icmp_unsigned(cmp_op, l, r)
    )
    result = builder.select(
        cond, ir.IntType(ida_insn.d.size * 8)(1), ir.IntType(ida_insn.d.size * 8)(0)
    )
    return _store_as(result, d, blk, builder)


def _handle_conditional_jump(
    l, r, d, next_blk, cmp_op, builder, ida_insn, signed=False
):
    """
    Internal jump condition branch builder helper.
    """
    if l is None or r is None:
        logging.warning("jump operand lift failed for instruction %s", ida_insn)
        return None
    # FP conditional jump: branch on an ``fcmp`` with the right ordered/unordered
    # predicate instead of truncating both float operands to int and ``icmp``-ing.
    if ida_insn.is_fpinsn():
        fcond = _fp_compare(builder, l, r, ida_insn)
        if fcond is not None:
            return builder.cbranch(fcond, d, next_blk)
    l = typecast(l, ir.IntType(ida_insn.l.size * 8), builder)
    r = typecast(r, ir.IntType(ida_insn.r.size * 8), builder)

    cond = (
        builder.icmp_signed(cmp_op, l, r)
        if signed
        else builder.icmp_unsigned(cmp_op, l, r)
    )
    return builder.cbranch(cond, d, next_blk)


def _handle_float_binary_op(l, r, d, op_func, blk, builder, ida_insn):
    """
    Internal floating point math calculation builder helper.
    """
    if l is None or r is None:
        logging.warning("float operand lift failed for instruction %s", ida_insn)
        return None
    typ = float_type(ida_insn.l.size)
    l = typecast(l, typ, builder)
    r = typecast(r, typ, builder)
    math = op_func(l, r)
    d = storecast(l, d, builder)
    return _store_as(math, d, blk, builder)


def lift_insn(
    ida_insn: ida_hexrays.minsn_t, blk: ir.Block, builder: ir.IRBuilder
) -> typing.Optional[ir.Instruction]:
    """
    Lifts a single IDA microcode instruction to LLVM IR.

    This function lift microcode insn into llvm in following steps:
    1. Lift left, right and destination mop for each instruction.
    2. Lift instruction.

    ida_insn: microcode insn
    blk: current llvm block
    builder: llvm builder
    """
    builder.position_at_end(blk)
    l = lift_mop(ida_insn.l, blk, builder)

    # Load source operand is always an address VALUE (the pointer to dereference),
    # never the address-of-slot: `ldx ds, r, d` loads from the value held by `r`.
    # Lifting `r` with dest=True returned the SLOT ADDRESS for an lvalue operand
    # (mop_l/mop_S/mop_r), so m_ldx's single `load` recovered only the slot value
    # (one indirection short -- `x` instead of `*x`); the loss was invisible at
    # sub-pointer width (the drop's _ptr_deref_alias width rule patched it) but
    # corrupted pointer-width derefs (`free(*(void**)x)` -> `free(x)` in
    # triple_free/triple_hash/triple_compare/randint_all_free). Lift `r` as a VALUE
    # so the load dereferences it (`load(load(slot))`), distinct in the IR from a
    # plain slot read (`bitcast slot; load`) -- which is the disambiguation the
    # decompiler microcode (m_ldx vs mop_l) carried but the old lifter erased.
    r = lift_mop(ida_insn.r, blk, builder)

    # Target destination is always handled as reference pointer, except call argument setups
    d = lift_mop(
        ida_insn.d,
        blk,
        builder,
        dest=(
            ida_insn.opcode != ida_hexrays.m_call
            and ida_insn.opcode != ida_hexrays.m_icall
        ),
    )

    # m_stx (`stx l, {r=sel, d=off}`) stores THROUGH a pointer: `d` is the memory
    # ADDRESS to write, a pointer VALUE, NOT a destination slot. When that address is
    # a bare LOCAL/STACK pointer slot at offset 0 (`%new_bucket`, a mop_l/mop_S),
    # lift_mop(dest=True) returned the SLOT ADDRESS (&slot), so the store wrote the
    # slot itself (`new_bucket = data`, a pointer-slot DEFINE) instead of
    # dereferencing (`new_bucket->data = data`, a store THROUGH the pointer). The two
    # are byte-identical in the IR (`bitcast %slot; store`) -- neither the lifter nor
    # the drop renderer can tell them apart -- but the microcode does (an m_stx
    # *address* operand vs an m_ldx/m_mov *destination*). This is the exact STORE-side
    # mirror of the m_ldx `r`-as-VALUE fix (57e7c90).
    #
    # The fix LOADS the slot to recover the pointer value and stores through THAT
    # (`store l, load(slot)`), making the deref distinct from a slot define. It is
    # done here (not via lift_mop dest=False) because the generic `_store_as` path
    # `dedereference()`s its destination -- which would strip a bare `load(slot)` and
    # collapse back to a slot define (e.g. `*total_n_read = 0` -> `total_n_read =
    # NULL`). Scope is deliberately NARROW: only a bare mop_l/mop_S address operand at
    # offset 0 is re-pointed at its loaded value; offset>0 operands are a mop_d
    # (already a GEP value, deref-correct) and registers keep their lowering.
    stx_slot_addr = None
    if ida_insn.opcode == ida_hexrays.m_stx and ida_insn.d.t in (
        ida_hexrays.mop_l,
        ida_hexrays.mop_S,
    ):
        stx_slot_addr = d

    # Declare helper functions dynamically when referenced
    if ida_insn.l.t == ida_hexrays.mop_h and l is None:
        l = create_intrinsic_function(
            blk.parent.parent, ida_insn.l.helper, ida_insn.d.f
        )

    def need(value, operand_name: str) -> bool:
        if value is not None:
            return True
        logging.warning("%s is None for instruction %s", operand_name, ida_insn)
        return False

    blk_itr = iter(blk.parent.blocks)
    list(itertools.takewhile(lambda x: x.name != blk.name, blk_itr))
    next_blk = next(blk_itr, None)

    # Dispatch instructions based on their operation code (opcode)
    match ida_insn.opcode:
        case ida_hexrays.m_nop:  # 0x00, nop no operation
            return
        case ida_hexrays.m_stx:  # 0x01, stx l, {r=sel, d=off} store value to memory
            if not need(l, "l") or not need(d, "d"):
                return None
            if stx_slot_addr is not None:
                # Deref-store through a bare local/stack pointer slot: load the slot
                # to recover the pointer VALUE, then store `l` THROUGH it. This keeps
                # `new_bucket->data = data` (a store through the pointer) distinct in
                # the IR from a `new_bucket = data` slot define -- and, unlike routing
                # a loaded value back through `_store_as`, it is not undone by
                # `dedereference()` (which would strip the load and collapse to a slot
                # define, e.g. `*total_n_read = 0` -> `total_n_read = NULL`). See the
                # operand-lift note above.
                ptr_val = builder.load(stx_slot_addr)
                ptr_val = typecast(ptr_val, l.type.as_pointer(), builder)
                return typing.cast(
                    ir.Instruction, builder.store(l, ptr_val)
                )
            d = storecast(l, d, builder)
            return typing.cast(ir.Instruction, _store_as(l, d, blk, builder))
        case ida_hexrays.m_ldx:  # 0x02, ldx {l=sel, r=off}, d load value from memory
            # A value-context ldx (e.g. a memory-load passed as a call argument) has
            # no destination operand; only the address `r` is required. `d.size` is
            # still valid, and storecast/_store_as both handle a None destination by
            # returning the loaded value, so a missing `d` must not abort the lift.
            if not need(r, "r"):
                return None
            # A pointer-slot DEFINE -- `bucket = table->bucket` (a through-pointer
            # member load of a POINTER-typed field, at ANY byte offset: `->bucket`
            # at 0, `->next` at +8), where the destination `d` is a bare pointer
            # slot (`hash_entry**`, pointee is itself a pointer). The default lowers
            # the load as `IntType(d.size*8)` (i64), then storecast/_store_as bitcast
            # the pointer SLOT to `i64*` and store an i64 through it -- which is
            # byte-identical in the IR to a deref-write `*X = v`. The drop renderer's
            # _ptr_deref_alias rule keys off the stored value's TYPE (`_is_ptr_type`):
            # an i64 reads as a deref, so the slot-define `bucket = table->bucket`
            # rendered as `bucket->data = *(void**)a0` (one spurious deref). The
            # microcode carried no width distinction (both 8 bytes) -- the pointer-
            # ness was destroyed by typing the load `i64`. Preserve it: when `d` is a
            # pointer-to-pointer slot and the load is pointer-width, load `r` as that
            # pointee POINTER type and store the pointer VALUE into the slot, so the
            # IR store is pointer-typed (`store hash_entry* v, hash_entry** %bucket`)
            # -- distinct from a deref-write, matching native's `bucket = table->bucket`.
            # Scope is narrow: only a pointer-WIDTH load into a pointer-to-pointer
            # destination slot changes; sub-pointer-width loads (`*name` i8) keep the
            # IntType lowering and the drop's width-based deref rule, and a value-
            # context ldx (d is None) or a non-pointer destination is untouched.
            if (
                not ida_insn.is_fpinsn()
                and d is not None
                and ida_insn.d.size * 8 == ptrsize
                and isinstance(getattr(d, "type", None), ir.PointerType)
                and isinstance(d.type.pointee, ir.PointerType)
            ):
                slot_ptr_ty = d.type.pointee
                r = typecast(r, slot_ptr_ty.as_pointer(), builder)
                r = builder.load(r)
                return typing.cast(
                    ir.Instruction, builder.store(r, d)
                )
            typ = (
                float_type(ida_insn.d.size)
                if ida_insn.is_fpinsn()
                else ir.IntType(ida_insn.d.size * 8)
            )
            r = typecast(r, typ.as_pointer(), builder)
            r = builder.load(r)
            d = storecast(r, d, builder)
            return typing.cast(ir.Instruction, _store_as(r, d, blk, builder))
        case ida_hexrays.m_ldc:  # 0x03, ldc l=const, d load constant
            if not need(d, "d"):
                return None
            r_val = ir.Constant(ir.IntType(32), ida_insn.l.nnn)
            return typing.cast(ir.Instruction, _store_as(r_val, d, blk, builder))
        case ida_hexrays.m_mov:  # 0x04, mov l, d move
            if not need(l, "l"):
                return None
            d = storecast(l, d, builder)
            return typing.cast(ir.Instruction, _store_as(l, d, blk, builder))
        case ida_hexrays.m_neg:  # 0x05, neg l, d negate
            if not need(l, "l"):
                return None
            l = typecast(l, ir.IntType(ida_insn.d.size * 8), builder)
            l = builder.neg(l)
            d = storecast(l, d, builder)
            return typing.cast(ir.Instruction, _store_as(l, d, blk, builder))
        case ida_hexrays.m_lnot:  # 0x06, lnot l, d logical not
            if not need(l, "l"):
                return None
            zero_const = ir.IntType(ida_insn.l.size * 8)(0)
            assert l is not None and getattr(l, "type", None) is not None
            zero_const = typecast(zero_const, l.type, builder)
            cmp = builder.icmp_unsigned("==", l, zero_const)
            d = storecast(cmp, d, builder)
            return typing.cast(ir.Instruction, _store_as(cmp, d, blk, builder))
        case ida_hexrays.m_bnot:  # 0x07, bnot l, d bitwise not
            if not need(l, "l"):
                return None
            l = typecast(l, ir.IntType(ida_insn.d.size * 8), builder)
            l = builder.not_(l)
            return typing.cast(ir.Instruction, _store_as(l, d, blk, builder))
        case ida_hexrays.m_xds:  # 0x08, xds l, d extend (signed)
            if not need(l, "l"):
                return None
            return typing.cast(ir.Instruction, _store_as(l, d, blk, builder))
        case ida_hexrays.m_xdu:  # 0x09, xdu l, d extend (unsigned)
            if not need(l, "l"):
                return None
            return typing.cast(
                ir.Instruction, _store_as(l, d, blk, builder, signed=False)
            )
        case ida_hexrays.m_low:  # 0x0A, low l, d take low part
            if not need(l, "l"):
                return None
            return typing.cast(ir.Instruction, _store_as(l, d, blk, builder))
        case ida_hexrays.m_high:  # 0x0B, high l, d take high part
            if not need(l, "l"):
                return None
            return typing.cast(ir.Instruction, _store_as(l, d, blk, builder))
        case ida_hexrays.m_add:  # 0x0C, add l, r, d l + r -> dst
            return _handle_binary_arithmetic(
                l, r, d, builder.add, blk, builder, ida_insn, allow_ptr=True
            )
        case ida_hexrays.m_sub:  # 0x0D, sub l, r, d l - r -> dst
            if not need(l, "l") or not need(r, "r"):
                return None
            l_type: typing.Optional[ir.Type] = getattr(l, "type", None)
            if l_type and isinstance(l_type, (ir.FloatType, ir.DoubleType)):
                l = builder.fptoui(l, ir.IntType(ida_insn.l.size * 8))
            r_type: typing.Optional[ir.Type] = getattr(r, "type", None)
            if r_type and isinstance(r_type, (ir.FloatType, ir.DoubleType)):
                r = builder.fptoui(r, ir.IntType(ida_insn.r.size * 8))

            if (
                l_type
                and isinstance(l_type, ir.PointerType)
                and r_type
                and isinstance(r_type, ir.IntType)
            ):
                neg_r = builder.neg(r)
                cast_ptr = typecast(l, ir.IntType(8).as_pointer(), builder)
                math = builder.gep(cast_ptr, (neg_r,))
                math = typecast(math, l_type, builder)
            elif (
                r_type
                and isinstance(r_type, ir.PointerType)
                and l_type
                and isinstance(l_type, ir.IntType)
            ):
                neg_l = builder.neg(l)
                cast_ptr = typecast(r, ir.IntType(8).as_pointer(), builder)
                math = builder.gep(cast_ptr, (neg_l,))
                math = typecast(math, r_type, builder)
            elif isinstance(l_type, ir.IntType) and isinstance(r_type, ir.IntType):
                math = builder.sub(l, r)
            elif (
                l_type
                and isinstance(l_type, ir.PointerType)
                and r_type
                and isinstance(r_type, ir.PointerType)
            ):
                ptr_type = ir.IntType(64)
                const1 = builder.ptrtoint(l, ptr_type)
                const2 = builder.ptrtoint(r, ptr_type)
                math = builder.sub(const1, const2)
            else:
                logging.warning(
                    "unsupported subtraction between %s and %s for instruction %s",
                    l_type,
                    r_type,
                    ida_insn,
                )
                return None
            d = storecast(l, d, builder)
            return typing.cast(ir.Instruction, _store_as(math, d, blk, builder))
        case ida_hexrays.m_mul:  # 0x0E, mul l, r, d l * r -> dst
            return _handle_binary_arithmetic(
                l, r, d, builder.mul, blk, builder, ida_insn
            )
        case ida_hexrays.m_udiv:  # 0x0F, udiv l, r, d l / r -> dst
            return _handle_binary_arithmetic(
                l, r, d, builder.udiv, blk, builder, ida_insn
            )
        case ida_hexrays.m_sdiv:  # 0x10, sdiv l, r, d l / r -> dst
            return _handle_binary_arithmetic(
                l, r, d, builder.sdiv, blk, builder, ida_insn
            )
        case ida_hexrays.m_umod:  # 0x11, umod l, r, d l % r -> dst
            return _handle_binary_arithmetic(
                l, r, d, builder.urem, blk, builder, ida_insn
            )
        case ida_hexrays.m_smod:  # 0x12, smod l, r, d l % r -> dst
            return _handle_binary_arithmetic(
                l, r, d, builder.srem, blk, builder, ida_insn
            )
        case ida_hexrays.m_or:  # 0x13, or l, r, d bitwise or
            return _handle_binary_arithmetic(
                l, r, d, builder.or_, blk, builder, ida_insn
            )
        case ida_hexrays.m_and:  # 0x14, and l, r, d bitwise and
            return _handle_binary_arithmetic(
                l, r, d, builder.and_, blk, builder, ida_insn
            )
        case ida_hexrays.m_xor:  # 0x15, xor l, r, d bitwise xor
            return _handle_binary_arithmetic(
                l, r, d, builder.xor, blk, builder, ida_insn
            )
        case ida_hexrays.m_shl:  # 0x16, shl l, r, d shift logical left
            return _handle_binary_arithmetic(
                l, r, d, builder.shl, blk, builder, ida_insn
            )
        case ida_hexrays.m_shr:  # 0x17, shr l, r, d shift logical right
            return _handle_binary_arithmetic(
                l, r, d, builder.ashr, blk, builder, ida_insn
            )
        case ida_hexrays.m_sar:  # 0x18, sar l, r, d shift arithmetic right
            return _handle_binary_arithmetic(
                l, r, d, builder.ashr, blk, builder, ida_insn
            )
        case (
            ida_hexrays.m_cfadd
        ):  # 0x19, cfadd l, r, d=carry (calculate carry bit of l+r)
            l = typecast(l, ir.IntType(64), builder)
            r = typecast(r, ir.IntType(64), builder)
            math = builder.call(
                lift_intrinsic_function(blk.parent.parent, "sadd_overflow"), [l, r]
            )
            math = builder.gep(
                math,
                (
                    ir.IntType(32)(0),
                    ir.IntType(32)(0),
                ),
            )
            math = builder.load(math)
            d = storecast(l, d, builder)
            return typing.cast(ir.Instruction, _store_as(math, d, blk, builder))
        case (
            ida_hexrays.m_ofadd
        ):  # 0x1A, ofadd l, r, d=overf (calculate overflow bit of l+r)
            l = typecast(l, ir.IntType(64), builder)
            r = typecast(r, ir.IntType(64), builder)
            math = builder.call(
                lift_intrinsic_function(blk.parent.parent, "sadd_overflow"), [l, r]
            )
            math = builder.gep(
                math,
                (
                    ir.IntType(32)(0),
                    ir.IntType(32)(1),
                ),
            )
            math = builder.load(math)
            d = storecast(l, d, builder)
            return typing.cast(ir.Instruction, _store_as(math, d, blk, builder))
        case (
            ida_hexrays.m_cfshl
        ):  # 0x1B, cfshl l, r, d=carry (calculate carry bit of l<<r)
            l = typecast(l, ir.IntType(64), builder)
            r = typecast(r, ir.IntType(64), builder)
            func_name = f"m_cfshr_{ida_insn.d.size}"
            if func_name in blk.parent.parent.globals:
                f_func = blk.parent.parent.get_global(func_name)
            else:
                f_func = blk.parent.parent.declare_intrinsic(
                    func_name,
                    fnty=ir.FunctionType(
                        ir.DoubleType(), [ir.IntType(64), ir.IntType(64)]
                    ),
                )
            l = builder.call(f_func, [l, r])
            d = storecast(l, d, builder)
            return typing.cast(ir.Instruction, _store_as(l, d, blk, builder))
        case (
            ida_hexrays.m_cfshr
        ):  # 0x1C, cfshr l, r, d=carry (calculate carry bit of l>>r)
            l = typecast(l, ir.IntType(64), builder)
            r = typecast(r, ir.IntType(64), builder)
            func_name = f"m_cfshr_{ida_insn.d.size}"
            if func_name in blk.parent.parent.globals:
                f_func = blk.parent.parent.get_global(func_name)
            else:
                f_func = blk.parent.parent.declare_intrinsic(
                    func_name,
                    fnty=ir.FunctionType(
                        ir.DoubleType(), [ir.IntType(64), ir.IntType(64)]
                    ),
                )
            l = builder.call(f_func, [l, r])
            d = storecast(l, d, builder)
            return typing.cast(ir.Instruction, _store_as(l, d, blk, builder))
        case ida_hexrays.m_sets:  # 0x1D, sets l, d=byte SF=1Sign
            l = typecast(l, ir.IntType(ida_insn.l.size * 8), builder)
            r_const = ir.Constant(ir.IntType(ida_insn.l.size * 8), 0)
            cond = builder.icmp_unsigned("<", l, r_const)
            result = builder.select(
                cond,
                ir.IntType(ida_insn.d.size * 8)(1),
                ir.IntType(ida_insn.d.size * 8)(0),
            )
            return typing.cast(ir.Instruction, _store_as(result, d, blk, builder))
        case ida_hexrays.m_seto:  # 0x1E, seto l, r, d=byte OF=1Overflow of (l-r)
            l = typecast(l, ir.IntType(64), builder)
            r = typecast(r, ir.IntType(64), builder)
            math = builder.call(
                lift_intrinsic_function(blk.parent.parent, "__OFSUB__"), [l, r]
            )
            return typing.cast(ir.Instruction, _store_as(math, d, blk, builder))
        case ida_hexrays.m_setp:  # 0x1F, setp l, r, d=byte PF=1Unordered/Parity
            func_name = f"setp_{ida_insn.l.size}_{ida_insn.d.size}"
            if func_name in blk.parent.parent.globals:
                f_setp = blk.parent.parent.get_global(func_name)
            else:
                f_setp = blk.parent.parent.declare_intrinsic(
                    func_name,
                    fnty=ir.FunctionType(
                        ir.IntType(ida_insn.d.size * 8),
                        [ir.IntType(ida_insn.l.size * 8), ir.IntType(32)],
                    ),
                )
            l = typecast(l, ir.IntType(ida_insn.l.size * 8), builder)
            l = builder.call(f_setp, [l, ir.Constant(ir.IntType(32), ida_insn.d.size)])
            return typing.cast(ir.Instruction, _store_as(l, d, blk, builder))
        case ida_hexrays.m_setnz:  # 0x20, setnz l, r, d=byte ZF=0Not Equal
            return _handle_comparison(l, r, d, "!=", blk, builder, ida_insn)
        case ida_hexrays.m_setz:  # 0x21, setz l, r, d=byte ZF=1Equal
            return _handle_comparison(l, r, d, "==", blk, builder, ida_insn)
        case ida_hexrays.m_setae:  # 0x22, setae l, r, d=byte CF=0Above or Equal
            return _handle_comparison(l, r, d, ">=", blk, builder, ida_insn)
        case ida_hexrays.m_setb:  # 0x23, setb l, r, d=byte CF=1Below
            return _handle_comparison(l, r, d, "<", blk, builder, ida_insn)
        case ida_hexrays.m_seta:  # 0x24, seta l, r, d=byte CF=0 & ZF=0 Above
            return _handle_comparison(l, r, d, ">", blk, builder, ida_insn)
        case ida_hexrays.m_setbe:  # 0x25, setbe l, r, d=byte CF=1 | ZF=1 Below or Equal
            return _handle_comparison(l, r, d, "<=", blk, builder, ida_insn)
        case ida_hexrays.m_setg:  # 0x26, setg l, r, d=byte SF=OF & ZF=0 Greater
            return _handle_comparison(l, r, d, ">", blk, builder, ida_insn, signed=True)
        case ida_hexrays.m_setge:  # 0x27, setge l, r, d=byte SF=OF Greater or Equal
            return _handle_comparison(
                l, r, d, ">=", blk, builder, ida_insn, signed=True
            )
        case ida_hexrays.m_setl:  # 0x28, setl l, r, d=byte SF!=OF Less
            return _handle_comparison(l, r, d, "<", blk, builder, ida_insn, signed=True)
        case (
            ida_hexrays.m_setle
        ):  # 0x29, setle l, r, d=byte SF!=OF | ZF=1 Less or Equal
            return _handle_comparison(
                l, r, d, "<=", blk, builder, ida_insn, signed=True
            )
        case ida_hexrays.m_jcnd:  # 0x2A, jcnd l, d (d is mop_v or mop_b)
            l = typecast(l, ir.IntType(1), builder)
            return builder.cbranch(l, d, next_blk)
        case ida_hexrays.m_jnz:  # 0x2B, jnz l, r, d ZF=0Not Equal
            return _handle_conditional_jump(l, r, d, next_blk, "!=", builder, ida_insn)
        case ida_hexrays.m_jz:  # 0x2C, jz l, r, d ZF=1Equal
            return _handle_conditional_jump(l, r, d, next_blk, "==", builder, ida_insn)
        case ida_hexrays.m_jae:  # 0x2D, jae l, r, d CF=0Above or Equal
            return _handle_conditional_jump(l, r, d, next_blk, ">=", builder, ida_insn)
        case ida_hexrays.m_jb:  # 0x2E, jb l, r, d CF=1Below
            return _handle_conditional_jump(l, r, d, next_blk, "<", builder, ida_insn)
        case ida_hexrays.m_ja:  # 0x2F, ja l, r, d CF=0 & ZF=0 Above
            return _handle_conditional_jump(l, r, d, next_blk, ">", builder, ida_insn)
        case ida_hexrays.m_jbe:  # 0x30, jbe l, r, d CF=1 | ZF=1 Below or Equal
            return _handle_conditional_jump(l, r, d, next_blk, "<=", builder, ida_insn)
        case ida_hexrays.m_jg:  # 0x31, jg l, r, d SF=OF & ZF=0 Greater
            return _handle_conditional_jump(
                l, r, d, next_blk, ">", builder, ida_insn, signed=True
            )
        case ida_hexrays.m_jge:  # 0x32, jge l, r, d SF=OF Greater or Equal
            return _handle_conditional_jump(
                l, r, d, next_blk, ">=", builder, ida_insn, signed=True
            )
        case ida_hexrays.m_jl:  # 0x33, jl l, r, d SF!=OF Less
            return _handle_conditional_jump(
                l, r, d, next_blk, "<", builder, ida_insn, signed=True
            )
        case ida_hexrays.m_jle:  # 0x34, jle l, r, d SF!=OF | ZF=1 Less or Equal
            return _handle_conditional_jump(
                l, r, d, next_blk, "<=", builder, ida_insn, signed=True
            )
        case ida_hexrays.m_jtbl:  # 0x35, jtbl l, r=mcases Table jump
            l = typecast(l, ir.IntType(ida_insn.l.size * 8), builder)
            if "default" in r:
                switch = builder.switch(l, blk.parent.basic_blocks[r["default"]])
            else:
                switch = builder.switch(
                    l, blk.parent.basic_blocks[r[list(r.keys())[0]]]
                )
            for value in r.keys():
                if isinstance(value, int):
                    switch.add_case(value, blk.parent.basic_blocks[r[value]])
            return switch
        case (
            ida_hexrays.m_ijmp
        ):  # 0x36, ijmp {r=sel, d=off} indirect unconditional jump
            return
        case ida_hexrays.m_goto:  # 0x37, goto l (l is mop_v or mop_b)
            return builder.branch(l)
        case ida_hexrays.m_call:  # 0x38, call ld (l is mop_v or mop_b or mop_h)
            rets, args = d
            l_type: typing.Optional[ir.Type] = getattr(l, "type", None)
            if l_type is None:
                return None
            l_pointee: typing.Optional[ir.Type] = getattr(l_type, "pointee", None)
            if l_pointee is None:
                return None
            if not isinstance(l_type, ir.PointerType) or not isinstance(
                l_pointee, ir.FunctionType
            ):
                arg_type = [arg.type for arg in args]
                new_func_type = ir.FunctionType(
                    i8ptr, arg_type, var_arg=False
                ).as_pointer()
                l = typecast(l, new_func_type, builder)
                ret = builder.call(l, args)
                for dst in rets:
                    _store_as(ret, dst, blk, builder)
                return ret

            for i, llvm_type in enumerate(l_pointee.args):
                if i >= len(args):
                    args.append(ir.Constant(ir.IntType(32), 1))
                if args[i].type != llvm_type:
                    args[i] = typecast(args[i], llvm_type, builder)

            if l_pointee.var_arg:  # Function signature is variadic
                # Keep the surplus trailing arguments (the varargs) UNCHANGED and
                # widen the function type so the emitted `call` carries them; only
                # the fixed prefix above is typecast. Truncating here would drop
                # every vararg (the bug fixed by ida-23as).
                ltype = l_pointee
                new_args = list(ltype.args)
                for i in range(len(new_args), len(args)):
                    new_args.append(args[i].type)
                new_func_type = ir.FunctionType(
                    ltype.return_type, new_args, var_arg=True
                ).as_pointer()
                l = typecast(l, new_func_type, builder)
            else:
                # Non-variadic callee: drop any surplus operands beyond the fixed
                # signature (preserves prior behavior for fixed-arity calls).
                args = args[: len(l_pointee.args)]
            ret = builder.call(l, args)
            for dst in rets:
                _store_as(ret, dst, blk, builder)
            return ret
        case ida_hexrays.m_icall:  # 0x39, icall {l=sel, r=off} d (indirect call)
            rets, args = d
            if not need(r, "r"):
                return None
            f_type = ir.FunctionType(
                ir.IntType(8).as_pointer(), (arg.type for arg in args)
            )
            f = typecast(r, f_type.as_pointer(), builder)
            if not hasattr(f, "function_type"):
                logging.warning("indirect call target is not callable for %s", ida_insn)
                return None
            return builder.call(f, args)
        case ida_hexrays.m_ret:  # 0x3A, ret
            return
        case ida_hexrays.m_push:  # 0x3B, push l
            return
        case ida_hexrays.m_pop:  # 0x3C, popd
            return
        case ida_hexrays.m_und:  # 0x3D, undd undefine
            return
        case ida_hexrays.m_ext:  # 0x3E, ext (external instruction, not microcode)
            return
        case ida_hexrays.m_f2i:  # 0x3F, f2i (convert float to signed integer)
            # fp->signed int. Like the integer casts (m_xds/m_low) and m_f2f, the
            # conversion RESULT must be stored to the destination `d` when this is a
            # top-level insn (line ~636 discards lift_insn's return -> an unstored
            # result is lost, e.g. an `f2i v, STKVAR` to a merge slot vanishes). The
            # nested-operand case (mop_d, line ~1158) passes `d=None`; _store_as then
            # returns the value unchanged, preserving the chained-value behaviour.
            res = typecast(l, ir.IntType(ida_insn.d.size * 8), builder, signed=True)
            return typing.cast(ir.Instruction, _store_as(res, d, blk, builder))
        case ida_hexrays.m_f2u:  # 0x40, f2u (convert float to unsigned integer)
            res = typecast(l, ir.IntType(ida_insn.d.size * 8), builder, signed=False)
            return typing.cast(
                ir.Instruction, _store_as(res, d, blk, builder, signed=False)
            )
        case ida_hexrays.m_i2f:  # 0x41, i2f (convert integer to float)
            l = typecast(l, ir.IntType(ida_insn.l.size * 8), builder)
            typ = float_type(ida_insn.d.size)
            res = builder.sitofp(l, typ)
            return typing.cast(ir.Instruction, _store_as(res, d, blk, builder))
        case ida_hexrays.m_u2f:  # 0x42, u2f (convert unsigned integer to float)
            l = typecast(l, ir.IntType(ida_insn.l.size * 8), builder)
            typ = float_type(ida_insn.d.size)
            res = builder.uitofp(l, typ)
            return typing.cast(
                ir.Instruction, _store_as(res, d, blk, builder, signed=False)
            )
        case ida_hexrays.m_f2f:  # 0x43, f2f (change float precision)
            target_type = float_type(ida_insn.d.size)
            l = typecast(l, target_type, builder)
            if d is not None and getattr(d, "type", None) != target_type.as_pointer():
                d = typecast(d, target_type.as_pointer(), builder)
            return typing.cast(ir.Instruction, _store_as(l, d, blk, builder))
        case ida_hexrays.m_fneg:  # 0x44, fneg (negate float sign)
            return typing.cast(ir.Instruction, _store_as(l, d, blk, builder))
        case ida_hexrays.m_fadd:  # 0x45, fadd (float addition)
            return _handle_float_binary_op(
                l, r, d, builder.fadd, blk, builder, ida_insn
            )
        case ida_hexrays.m_fsub:  # 0x46, fsub (float subtraction)
            return _handle_float_binary_op(
                l, r, d, builder.fsub, blk, builder, ida_insn
            )
        case ida_hexrays.m_fmul:  # 0x47, fmul (float multiplication)
            return _handle_float_binary_op(
                l, r, d, builder.fmul, blk, builder, ida_insn
            )
        case ida_hexrays.m_fdiv:  # 0x48, fdiv (float division)
            return _handle_float_binary_op(
                l, r, d, builder.fdiv, blk, builder, ida_insn
            )
        case _:
            _emit_fidelity(
                "not_implemented",
                SEVERITY_HARD_FAIL,
                ea=ida_insn.ea,
                detail=ida_insn.dstr(),
            )
            raise NotImplementedError(f"not implemented opcode: {ida_insn.dstr()}")


class BIN2LLVMController:
    """
    Controller that coordinates binary lifting to LLVM IR.
    """

    def __init__(self, target_mode: str = "host"):
        """Initialize the controller with an empty LLVM module."""
        self.m = ir.Module()
        self._name_cache = {}
        self._func_cache = {}
        self.target_mode = target_mode
        if target_mode == "host":
            self._set_module_target_from_host()
        else:
            self._set_module_target_from_ida()

    def _get_name(self, ea):
        name = self._name_cache.get(ea)
        if name is None:
            name = ida_name.get_name(ea)
            self._name_cache[ea] = name
        return name

    def _get_func(self, ea):
        func = self._func_cache.get(ea, "__missing__")
        if func == "__missing__":
            func = ida_funcs.get_func(ea)
            self._func_cache[ea] = func
        return func

    def _set_module_target_from_host(self):
        """Sets module target triple and data layout based on host LLVM."""
        try:
            try:
                if hasattr(llvm, "initialize_all_targets"):
                    llvm.initialize_all_targets()
                if hasattr(llvm, "initialize_all_asmprinters"):
                    llvm.initialize_all_asmprinters()
            except Exception:
                pass

            triple = llvm.get_default_triple()
            target = llvm.Target.from_triple(triple)
            target_machine = target.create_target_machine()
            self.m.triple = triple
            self.m.data_layout = str(target_machine.target_data)
        except Exception as exc:
            logging.debug(
                "Host mode target setup failed: %s, falling back to IDA mode", exc
            )
            self._set_module_target_from_ida()

    def _set_module_target_from_ida(self):
        """Sets module target triple based on IDA's analysis information."""
        try:
            proc = ida_ida.inf_get_procname()
            is_64 = ida_ida.inf_is_64bit()
            arch = "unknown"

            if proc == "metapc":
                arch = "x86_64" if is_64 else "i386"
            elif proc in ("arm", "ARM"):
                arch = "aarch64" if is_64 else "arm"
            elif proc in ("aarch64", "arm64"):
                arch = "aarch64"
            elif proc == "mips":
                arch = "mips64" if is_64 else "mips"
            elif proc in ("ppc", "ppc64"):
                arch = "powerpc64" if is_64 else "powerpc"
            elif proc == "riscv":
                arch = "riscv64" if is_64 else "riscv32"

            os_name = "unknown"
            with suppress(Exception):
                ostype = ida_ida.inf_get_ostype()
                if hasattr(ida_ida, "OSTYPE_WIN") and ostype == ida_ida.OSTYPE_WIN:
                    os_name = "windows"
                elif (
                    hasattr(ida_ida, "OSTYPE_LINUX") and ostype == ida_ida.OSTYPE_LINUX
                ):
                    os_name = "linux"
                elif (
                    hasattr(ida_ida, "OSTYPE_MACOS") and ostype == ida_ida.OSTYPE_MACOS
                ):
                    os_name = "darwin"

            self.m.triple = f"{arch}-unknown-{os_name}"
            self.m.data_layout = ""
        except Exception as exc:
            logging.warning("Failed to set module target from IDA: %s", exc)

    def insertAllFunctions(self):
        """
        Lifts all executable functions (typically in the .text segment) to LLVM IR.
        """
        for f_ea in idautils.Functions():
            self.insertFunctionAtEa(f_ea)

    def insertFunctionAtEa(self, ea):
        """
        Lifts the specific function at the given effective address (ea) to LLVM IR.
        """
        if ea in ptext:
            pfunc = ptext.get(ea)
            if pfunc is None or getattr(pfunc, "type", None) is None:
                return
            typ = pfunc.type
            func_name = self._get_name(ea)
            lift_function(self.m, func_name, False, ea, typ)

    def create_global(self, ea, width, str_dict):
        """
        Creates LLVM global variables for IDA data items.
        1. get data item name and type.
        2. create global variables.

        ea: data address
        width: data width in ea
        str_dict: all known strings
        """
        name = self._get_name(ea) or f"data_{hex(ea)[2:]}"

        # If the data item is recognized as a string, create a global constant string
        if ea in str_dict:
            str_csnt = str_dict[ea][0]
            str_type = ir.ArrayType(ir.IntType(8), str_dict[ea][1])
            g = ir.GlobalVariable(self.m, str_type, name=name)
            g.initializer = ir.Constant(str_type, bytearray(str_csnt))
            g.linkage = "private"
            g.global_constant = True
            return

        tif = ida_typeinf.tinfo_t()
        if not ida_nalt.get_tinfo(tif, ea):
            ida_typeinf.guess_tinfo(tif, ea)

        # Handle external/library function setups
        elif tif.is_func() or tif.is_funcptr():
            if tif.is_funcptr():
                tif = tif.get_ptrarr_object()

            func = self._get_func(ea)
            # if function is a thunk function, define the actual function instead
            if func is not None and (func.flags & ida_funcs.FUNC_THUNK):
                tfunc_ea, ptr = ida_funcs.calc_thunk_func_target(func)
                if tfunc_ea != ida_idaapi.BADADDR:
                    ea = tfunc_ea
                    name = self._get_name(ea)
                    func = self._get_func(ea)

            if (
                func is None
                or (func.flags & ida_funcs.FUNC_LIB)
                or ida_segment.segtype(ea) & ida_segment.SEG_XTRN
            ):
                lift_function(self.m, name, True, ea, tif)

        # Fallback to other regular types (int, float, array, struct, etc.)
        else:
            typ = lift_tif(tif, width)
            g_cmt = lift_from_address(self.m, ea, typ)
            g = ir.GlobalVariable(self.m, typ, name=name)
            g.initializer = g_cmt

    def initialize(self):
        """
        Initializes the LLVM module by preparing all necessary metadata.

        1. Decompile all functions.
        2. Collect all strings.
        3. Create GlobalVariabel for all IDA data item.

        ptext: dict to save decompile results {ea:decompile}
        str_dict: dict to save all strings
        """
        # Step 1: Decompile all functions and cache the decompiled results
        for func in idautils.Functions():
            try:
                pfunc = ida_hexrays.decompile(func)
                if pfunc is not None:
                    ptext[func] = pfunc
            except Exception as e:
                logging.debug("Failed to decompile function at %s: %s", hex(func), e)

        # Step 2: Collect all string constants identified by IDA
        str_dict = {}
        for s in idautils.Strings():
            str_dict[s.ea] = [ida_bytes.get_bytes(s.ea, s.length), s.length]

        # Step 3: Iterate through data items in non-executable segments and construct globals
        for i in range(idaapi.get_segm_qty()):
            segm = idaapi.getnseg(i)
            if segm is None or (segm.perm & idaapi.SEGPERM_EXEC):
                continue
            for head in idautils.Heads(segm.start_ea, segm.end_ea):
                end = ida_bytes.get_item_end(head)
                if end <= head:
                    continue
                self.create_global(head, end - head, str_dict)

    def save_to_file(
        self,
        filename,
        *,
        annotate_concurrency: bool = False,
        ir_passes: tuple[str, ...] = (),
        source_binary: str | None = None,
    ):
        """
        Saves the LLVM IR module in text format to a file.
        """
        llvm_ir = format_llvm_module(self.m)
        if annotate_concurrency and "concurrency" not in ir_passes:
            ir_passes = (*ir_passes, "concurrency")
        if ir_passes:
            from .ir_passes import IRPassContext, run_ir_passes

            llvm_ir = run_ir_passes(
                llvm_ir,
                ir_passes,
                IRPassContext(
                    source_binary=source_binary,
                    target_mode=self.target_mode,
                ),
            )
        with open(filename, "w") as f:
            f.write(llvm_ir)
