"""LLVM IR -> Hex-Rays microcode DROP (Model 2).

The drop hooks the decompile at ``hxe_preoptimized``, rebuilds the target
function's microcode from the LLVM IR (full-body replace), and lets the normal
``decompile()`` pipeline wire the CFG, allocate lvars, optimize, and build the
ctree -> clean pseudocode. We build DIRECTLY on LLVM (LLVM IR is already the IR --
no  portable IR); ``cfg_verify`` decodes any INTERR.

Proven slices folded here: full-body replace, the op map, SSA results via
``alloc_kreg`` scratch registers, ABI arg mapping, casts. Multi-block control flow
(the 2-way if/else mechanic) is proven in tests/test_drop_controlflow.py and is the
next layered addition. See memory ``idavator_drop_microcode_hook_architecture``.
"""
from __future__ import annotations

import logging
import re
from contextlib import suppress

import ida_bytes
import ida_frame
import ida_funcs
import ida_hexrays as hx
import ida_ida
import ida_idaapi
import ida_idp
import ida_name
import ida_nalt
import ida_strlist
import ida_typeinf
import llvmlite.binding as llvm

from idavator.cfg_verify import try_verify

logger = logging.getLogger("idavator.llvm_drop")

# Instruction-tail opcodes (terminators / control flow) -- never cleared.
_TERM = frozenset({
    hx.m_goto, hx.m_ret, hx.m_jcnd, hx.m_jz, hx.m_jnz, hx.m_jtbl,
    hx.m_jae, hx.m_jb, hx.m_ja, hx.m_jbe, hx.m_jg, hx.m_jge, hx.m_jl, hx.m_jle,
})

_BINOP = {
    "add": hx.m_add, "sub": hx.m_sub, "mul": hx.m_mul, "and": hx.m_and,
    "or": hx.m_or, "xor": hx.m_xor, "shl": hx.m_shl, "lshr": hx.m_shr,
    "ashr": hx.m_sar,
    "udiv": hx.m_udiv, "sdiv": hx.m_sdiv, "urem": hx.m_umod, "srem": hx.m_smod,
}
# zext/sext/trunc -> microcode widening/narrowing.
_CAST = {"zext": hx.m_xdu, "sext": hx.m_xds, "trunc": hx.m_low}
# bit-identical reinterpretations -> no microcode, just alias the operand.
_NOOP_CAST = frozenset({"bitcast", "ptrtoint", "inttoptr"})

# icmp predicate -> a 2-way conditional jump that branches to ``d`` when the
# predicate holds (the fall-through, serial+1, takes the FALSE arm).
_ICMP_JMP = {
    "eq": hx.m_jz, "ne": hx.m_jnz,
    "ugt": hx.m_ja, "uge": hx.m_jae, "ult": hx.m_jb, "ule": hx.m_jbe,
    "sgt": hx.m_jg, "sge": hx.m_jge, "slt": hx.m_jl, "sle": hx.m_jle,
}

# icmp predicate -> a setcc that materialises the i1 result as a 1-byte value
# (``d = (l <pred> r) ? 1 : 0``). Used when an icmp result is CONSUMED as a value
# (a ``select`` condition / short-circuit arm) rather than folded into a branch.
_ICMP_SET = {
    "eq": hx.m_setz, "ne": hx.m_setnz,
    "ugt": hx.m_seta, "uge": hx.m_setae, "ult": hx.m_setb, "ule": hx.m_setbe,
    "sgt": hx.m_setg, "sge": hx.m_setge, "slt": hx.m_setl, "sle": hx.m_setle,
}

# IDA rotate intrinsics that survive into FAITHFUL pseudocode (e.g. rotr_sz ->
# `return __ROR8__(a0, a1)`) -> emit as a Hex-Rays helper call. Deliberately NOT
# the stack canary (`__readfsqword`/`__stack_chk_fail`): the decompiler ELIDES
# that boilerplate from final output, so reconstructing it would DIVERGE from the
# round-trip reference (and `make_helper` + reg-arg movs crashes the decompiler;
# the args must ride in the mcallinfo via create_helper_call).
_HELPER_INTRINSICS = frozenset({
    "__ROL1__", "__ROL2__", "__ROL4__", "__ROL8__",
    "__ROR1__", "__ROR2__", "__ROR4__", "__ROR8__",
})

# A private string-constant global used as a VALUE (e.g. an error message handed
# to a call/printf, decayed via getelementptr) stringifies in llvmlite as its
# full definition: ``@name = private [unnamed_addr ]constant [N x i8] c"..."``.
# LLVM truncates the symbol (``aInvalidKindInG``) and IDA auto-names the literal
# from its (longer) content (``aInvalidKindInGenTempname``), so get_name_ea on
# the LLVM name misses. We instead decode the c"..." body and match it against
# the IDB string table by exact content -> the literal's address.
_STRCONST_RE = re.compile(
    r'private\s+(?:unnamed_addr\s+)?constant\s+\[\d+\s+x\s+i8\]\s+c"(.*)"\s*$',
    re.S,
)


def _decode_llvm_cstr(body: str) -> bytes:
    """Decode an LLVM ``c"..."`` string body to raw bytes. LLVM escapes a byte as
    ``\\XX`` (two hex digits, e.g. ``\\22`` = ``"``, ``\\0a`` = ``\\n``, ``\\00`` =
    NUL); every other character is a literal byte. The trailing NUL is preserved
    here and stripped by the caller for matching against IDB strlit contents."""
    out = bytearray()
    i, n = 0, len(body)
    while i < n:
        ch = body[i]
        if ch == "\\" and i + 3 <= n:
            try:
                out.append(int(body[i + 1:i + 3], 16))
                i += 3
                continue
            except ValueError:
                pass
        out.append(ord(ch) & 0xFF)
        i += 1
    return bytes(out)


def _type_size(type_str) -> int:
    # llvmlite uses OPAQUE pointers (LLVM 14+): a pointer stringifies to "ptr",
    # not "i32*", and the pointee type is lost -- so a string "*" probe misses it.
    # Detect the pointer first via the structured ``is_pointer`` flag.
    if getattr(type_str, "is_pointer", False):
        return 8
    s = str(type_str)
    if s == "ptr" or "*" in s:
        return 8
    for tok, sz in (("i64", 8), ("i32", 4), ("i16", 2), ("i8", 1), ("i1", 1),
                    ("double", 8), ("float", 4)):
        if tok in s:
            return sz
    return 4


def _is_ptr_type(type_str) -> bool:
    """True if an LLVM type is a pointer (opaque ``ptr`` or a legacy ``T*``).
    Used to tell a full-pointer slot access from a sub-pointer deref on a
    pointer-typed alloca."""
    if getattr(type_str, "is_pointer", False):
        return True
    s = str(type_str).strip()
    return s == "ptr" or s.endswith("*")


def _round_up(x: int, a: int) -> int:
    return (x + a - 1) // a * a if a else x


def _split_fields(s: str) -> list[str]:
    """Top-level comma split of a struct body, respecting [] / {} nesting."""
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def _type_sa(tok: str, structs: dict) -> tuple[int, int]:
    """(size, align) of an LLVM type token; nested ``%name`` via ``structs``."""
    tok = tok.strip()
    if tok == "ptr" or tok.endswith("*"):
        return (8, 8)
    m = re.match(r"i(\d+)$", tok)
    if m:
        sz = max(1, (int(m.group(1)) + 7) // 8)
        return (sz, sz if sz in (1, 2, 4, 8) else 1)
    if tok in ("double", "float"):
        return (8, 8) if tok == "double" else (4, 4)
    m = re.match(r"\[\s*(\d+)\s+x\s+(.+)\]$", tok)
    if m:
        esz, eal = _type_sa(m.group(2), structs)
        return (int(m.group(1)) * _round_up(esz, eal), eal)
    # struct name -- llvmlite renders `%timespec`, the IR text declares
    # `%"timespec"`; canonicalize by stripping quotes.
    if tok.replace('"', "") in structs:
        return structs[tok.replace('"', "")]
    raise ValueError(f"unsized type {tok!r}")


def _parse_struct_layouts(ir_text: str) -> dict:
    """name -> (size, align) for every ``%name = type {..}`` whose layout is
    computable (natural C alignment). Un-computable structs are absent (the drop
    raises, the round-trip records a build error)."""
    raw = {m.group(1).strip().replace('"', ""): _split_fields(m.group(2))
           for m in re.finditer(r'(%[\w".:$]+)\s*=\s*type\s*\{(.*)\}', ir_text)}
    structs: dict = {}

    def layout(name, seen):
        name = name.replace('"', "")
        if name in structs:
            return structs[name]
        if name in seen:
            raise ValueError("recursive struct")
        off, maxal = 0, 1
        for f in raw[name]:
            if f.replace('"', "") in raw:
                layout(f, seen | {name})
            sz, al = _type_sa(f, structs)
            off = _round_up(off, al) + sz
            maxal = max(maxal, al)
        structs[name] = (_round_up(off, maxal), maxal)
        return structs[name]

    for nm in raw:
        with suppress(Exception):
            layout(nm, set())
    return structs


class LLVMDropConverter:
    """Drop a (straight-line) LLVM function into a host's decompiled output."""

    def __init__(self, ir_text: str):
        self._ir_text = ir_text
        self.module = llvm.parse_assembly(ir_text)
        self._sroa_module = None     # lazily-built SROA-optimized copy (fallback)
        self._struct_size = _parse_struct_layouts(ir_text)  # name -> (size, align)
        self._allocas: dict = {}  # set per-drop in _build (scalar-slot kregs)
        self._addr_taken: dict = {}  # name -> (stkoff, size) for &local allocas
        self._ptr_allocas: dict = {}  # ptr-typed addr_taken alloca name -> stkoff
        self._ptr_deref_alias: set = set()  # bitcast aliases rooted at a ptr alloca
        self._cur_mba = None         # current mba (make_stkvar needs it)
        self._call_spd_ea = None     # host resting-frame ea for stack-passing calls
        self._canary_kreg = None     # shared kreg for __readfsqword (canary fold)
        self._ret_off = None         # frame off of a promoted return slot, or None
        self._ret_kreg = None        # kreg of a promoted scalar return slot, or None
        self._ret_phi = None         # name of a phi whose result feeds `ret`, or None
        self._icmp_defs = {}         # icmp SSA name -> (pred, [operands]) for select
        self._str_index = None       # IDB string-literal content -> ea (lazy)

    # -- public API ------------------------------------------------------
    def drop(self, host_ea: int, llvm_fn_name: str):
        """Rebuild ``host_ea``'s microcode from the named LLVM function and return
        the resulting ``cfunc_t`` (or None on failure).

        SROA FALLBACK (scoped, zero-regression): the plain drop runs first. ONLY
        if it returns a LATE failure -- ``cf is None`` with NO build error and NO
        early INTERR caught by ``try_verify`` (the 50342-style value-numbering
        failure that surfaces only after ``preoptimized``) -- do we retry from an
        SROA-optimized copy of the module. SROA collapses the lifter's return-slot
        alloca into a return phi, which the return-phi promotion writes straight to
        the return reg (clearing 50342). Because the retry only runs when the plain
        drop already FAILED, every currently-passing function is untouched."""
        cf, box = self._drop_from_module(self.module, host_ea, llvm_fn_name)
        is_late_failure = (cf is None and box["err"] is None
                           and box["interr"] is None)
        if is_late_failure:
            opt = self._get_sroa_module()
            if opt is not None and any(
                    g.name == llvm_fn_name and not g.is_declaration
                    for g in opt.functions):
                logger.info("drop @%s: late failure -> SROA fallback retry",
                            llvm_fn_name)
                cf2, box2 = self._drop_from_module(opt, host_ea, llvm_fn_name)
                if cf2 is not None:
                    cf, box = cf2, box2
        if box["err"]:
            logger.error("drop build failed:\n%s", box["err"])
        self.last_interr = box["interr"]
        self.last_error = box["err"]
        return cf

    def _drop_from_module(self, module, host_ea: int, llvm_fn_name: str):
        """Rebuild ``host_ea`` from ``llvm_fn_name`` in ``module``; return
        ``(cfunc_t|None, box)`` where ``box`` carries ``interr``/``err``."""
        fn = next((g for g in module.functions
                   if g.name == llvm_fn_name and not g.is_declaration), None)
        if fn is None:
            raise ValueError(f"no definition for @{llvm_fn_name}")
        self._force_prototype(host_ea, fn)
        # A FUNC_NORET host would keep its __noreturn attribute and discard the
        # rebuilt return path; clear it so the dropped body (which returns) shows.
        hf = ida_funcs.get_func(host_ea)
        if hf is not None and (hf.flags & ida_funcs.FUNC_NORET):
            hf.flags &= ~ida_funcs.FUNC_NORET
            ida_funcs.update_func(hf)

        box = {"interr": None, "err": None}
        conv = self

        class _Hook(hx.Hexrays_Hooks):
            def preoptimized(self, mba):  # hxe_preoptimized: m_ret exists
                try:
                    conv._build(mba, fn)
                    ok, code = try_verify(mba, "after llvm drop")
                    box["interr"] = code
                except Exception:  # noqa: BLE001
                    import traceback
                    box["err"] = traceback.format_exc()
                return 0

        hook = _Hook()
        hook.hook()
        try:
            hx.mark_cfunc_dirty(host_ea)
            cf = hx.decompile(host_ea)
            # Type the struct-pointer cursor slots and re-drop so the type
            # propagation re-runs -- the only thing that beats the decompiler's own
            # param-propagated pointer inference is a persistent user type at the
            # cursor lvar's ACTUAL (post-decompile) location, applied between two
            # full decompiles. Build failures leave cf untyped (unchanged).
            if (cf is not None and box["err"] is None
                    and self._save_struct_ptr_lvar_types(host_ea, fn, cf)):
                hx.mark_cfunc_dirty(host_ea)
                cf = hx.decompile(host_ea)
        finally:
            hook.unhook()
        return cf, box

    def _get_sroa_module(self):
        """An SROA(+simplifycfg+instnamer)-optimized copy of the module, built
        lazily and cached. SROA promotes the lifter's return-slot alloca to a
        return phi (then promoted to the return reg); instnamer is required because
        SROA emits anonymous ``%0``/``%1`` that the converter keys by NAME. Returns
        None if the new-PM pipeline is unavailable (the plain drop result stands)."""
        if self._sroa_module is not None:
            return self._sroa_module
        try:
            opt = llvm.parse_assembly(self._ir_text)
            llvm.initialize_all_targets()
            llvm.initialize_native_target()
            llvm.initialize_native_asmprinter()
            tm = llvm.Target.from_default_triple().create_target_machine()
            pb = llvm.create_pass_builder(
                tm, llvm.create_pipeline_tuning_options())
            mpm = llvm.create_new_module_pass_manager()
            mpm.add_sroa_pass()
            mpm.add_simplify_cfg_pass()
            mpm.add_instruction_namer_pass()
            mpm.run(opt, pb)
            self._sroa_module = opt
        except Exception:  # noqa: BLE001
            import traceback
            logger.warning("SROA fallback unavailable:\n%s",
                           traceback.format_exc())
            self._sroa_module = None
        return self._sroa_module

    # -- internals -------------------------------------------------------
    @staticmethod
    def _pointee_size(fn, arg) -> int:
        """Opaque pointers drop the pointee type; recover the access width from
        the first load/store that consumes ``arg`` (default 4)."""
        for bb in fn.blocks:
            for ins in bb.instructions:
                ops = list(ins.operands)
                if ins.opcode == "load" and ops and ops[0].name == arg.name:
                    return _type_size(ins.type)
                if (ins.opcode == "store" and len(ops) >= 2
                        and ops[1].name == arg.name):
                    return _type_size(ops[0].type)
        return 4

    @classmethod
    def _arg_ctype(cls, arg, fn) -> str:
        if getattr(arg.type, "is_pointer", False) or str(arg.type) == "ptr":
            # Match the pointee C type to the access width so an N-byte ldx/stx
            # renders as a clean ``*a`` (no reinterpret cast).
            return {1: "char *", 2: "__int16 *", 4: "_DWORD *",
                    8: "__int64 *"}.get(cls._pointee_size(fn, arg), "_DWORD *")
        s = str(arg.type)
        if "i64" in s:
            return "__int64"
        if "i16" in s:
            return "__int16"
        if "i8" in s:
            return "char"
        return "int"

    @staticmethod
    def _ret_ctype(fn) -> str:
        for bb in fn.blocks:
            for ins in bb.instructions:
                if ins.opcode == "ret":
                    ops = list(ins.operands)
                    if not ops:
                        return "void"
                    t = ops[0].type
                    if getattr(t, "is_pointer", False) or str(t) == "ptr":
                        return "void *"
                    s = str(t)
                    if "i64" in s:
                        return "__int64"
                    if "i16" in s:
                        return "__int16"
                    if "i8" in s:
                        return "char"
                    return "int"
        return "void"

    @classmethod
    def _force_prototype(cls, host_ea: int, fn) -> None:
        args = list(fn.arguments)
        params = ", ".join(f"{cls._arg_ctype(a, fn)} a{i}"
                           for i, a in enumerate(args)) or "void"
        ret = cls._ret_ctype(fn)
        tif = ida_typeinf.tinfo_t()
        ida_typeinf.parse_decl(tif, None, f"{ret} __fastcall f({params});", 0)
        ida_typeinf.apply_tinfo(host_ea, tif, ida_typeinf.TINFO_DEFINITE)

    @staticmethod
    def _arg_reg_names():
        # Integer arg registers depend on the target ABI: a PE (Windows) target
        # uses the Microsoft x64 convention (rcx/rdx/r8/r9); everything else
        # (ELF/Mach-O) uses System V (rdi/rsi/rdx/rcx/r8/r9).
        if ida_ida.inf_get_filetype() == ida_ida.f_PE:
            return ("rcx", "rdx", "r8", "r9")
        return ("rdi", "rsi", "rdx", "rcx", "r8", "r9")

    @staticmethod
    def _abi():
        # Return reg = rax (rdi/... for args -- see _arg_reg_names).
        names = LLVMDropConverter._arg_reg_names()
        argregs = [hx.reg2mreg(ida_idp.str2reg(r)) for r in names]
        return (argregs, hx.reg2mreg(ida_idp.str2reg("rax")),
                hx.reg2mreg(ida_idp.str2reg("ds")))

    # -- microcode emit helpers -----------------------------------------
    def _fill(self, mop, d) -> None:
        kind, val, size = d
        if kind == "reg":
            mop.make_reg(val, size)
        elif kind == "gvar":
            mop.make_gvar(val)
            mop.size = size
        elif kind == "gvaraddr":
            # &global: mop_a wrapping a gvar (cf. &local = mop_a wrapping a
            # stkvar). A global used as a POINTER VALUE is its ADDRESS, not the
            # lvalue read at that address -- e.g. a decayed string constant
            # handed to a call is `gettext("msg")`, not
            # `gettext(*(const char **)"msg")`.
            inner = hx.mop_addr_t()
            inner.make_gvar(val)
            mop.t = hx.mop_a
            mop.a = inner
            mop.size = 8
        elif kind == "stkaddr":
            # &local: mop_a wrapping a stkvar (cf. &global = mop_a wrapping mop_v).
            inner = hx.mop_addr_t()
            inner.make_stkvar(self._cur_mba, val)
            mop.t = hx.mop_a
            mop.a = inner
            mop.size = 8
        elif kind == "stkvar":
            # The VALUE held in a host frame slot (an incoming >6th param spilled
            # to the caller stack -- cf. _fill's stkaddr is the slot's ADDRESS).
            mop.make_stkvar(self._cur_mba, val)
            mop.size = size
        else:
            mop.make_number(val & ((1 << (8 * size)) - 1), size)

    def _strconst_ea(self, operand_str: str):
        """Resolve a private string-constant operand (``@x = private constant
        [N x i8] c"..."``) to the address of the matching IDB string literal, or
        None. The LLVM symbol is truncated and IDA names the literal from its
        (longer) content, so name lookup fails; we match by exact content against
        the IDB string table instead (built once, cached on the instance)."""
        m = _STRCONST_RE.search(operand_str)
        if m is None:
            return None
        content = _decode_llvm_cstr(m.group(1)).rstrip(b"\x00")
        if self._str_index is None:
            self._str_index = {}
            ida_strlist.build_strlist()
            for i in range(ida_strlist.get_strlist_qty()):
                si = ida_strlist.string_info_t()
                if not ida_strlist.get_strlist_item(si, i):
                    continue
                raw = ida_bytes.get_strlit_contents(si.ea, si.length, si.type)
                if raw is None:
                    continue
                # First literal wins (lowest index == lowest ea for a content).
                self._str_index.setdefault(raw.rstrip(b"\x00"), si.ea)
        return self._str_index.get(content)

    def _desc(self, operand, vmap, default_size):
        """Resolve an LLVM operand to a value descriptor (reg kreg / numeric /
        gvar / gvaraddr / stkaddr). A global used as a POINTER VALUE is its
        ADDRESS (``gvaraddr`` = &global) -- an array/string global decays to a
        ``ptr`` whose value IS its address, so a decayed string constant handed
        to a call must render ``gettext("msg")``, not the lvalue read
        ``gettext(*(const char **)"msg")``. An address-taken alloca is the
        address of its frame slot (&local). A private string-constant value is
        its IDB string-literal address (matched by content)."""
        nm = operand.name
        if nm and nm in vmap:
            return vmap[nm]
        if nm and nm in self._addr_taken:
            return ("stkaddr", self._addr_taken[nm][0], 8)
        s = str(operand).strip()
        if nm and "@" in s:
            # A global operand of pointer type used as a VALUE is its address:
            # in opaque-pointer IR a global symbol's operand type is ``ptr`` and
            # its value IS its address (the array decayed). Materialise &global
            # (gvaraddr) rather than the lvalue read at that address (gvar).
            ptr_value = str(operand.type) == "ptr"
            ea = ida_name.get_name_ea(ida_idaapi.BADADDR, nm)
            if ea != ida_idaapi.BADADDR:
                return ("gvaraddr", ea, 8) if ptr_value \
                    else ("gvar", ea, default_size)
            sea = self._strconst_ea(s)
            if sea is not None:
                return ("gvaraddr", sea, 8) if ptr_value \
                    else ("gvar", sea, default_size)
        if s.split()[-1:] == ["undef"]:
            # an ``undef`` value is a don't-care (SROA leaves it on dead phi
            # incomings / masked-insert leftovers / a `ret <ty> undef` whose path
            # is pruned). Resolve it to zero of the operand's width -- any concrete
            # value is correct, and 0 keeps the value-numbering trivial.
            return ("num", 0, _type_size(s) or default_size)
        num = re.search(r"(-?\d+)\s*$", s)
        if num:
            return ("num", int(num.group(1)), _type_size(s) or default_size)
        if s.split()[-1:] == ["null"]:
            # a `null` pointer constant (e.g. a phi incoming after SROA folds the
            # return slot: ``[ null, %B ]``) -> the zero address.
            return ("num", 0, _type_size(s) or default_size)
        raise ValueError(f"unhandled operand {s!r}")

    def _global_ea(self, operand):
        """Resolve an LLVM global operand (``@name``) to its IDB address, or
        None if it is not a (resolvable) global. A private string-constant global
        resolves to its IDB string literal (matched by content)."""
        s = str(operand)
        if operand.name and "@" in s:
            ea = ida_name.get_name_ea(ida_idaapi.BADADDR, operand.name)
            if ea != ida_idaapi.BADADDR:
                return ea
            return self._strconst_ea(s)
        return None

    @staticmethod
    def _wire(blk, succs) -> None:
        """Replace blk's successors with ``succs``, fixing peer predsets."""
        mba = blk.mba
        for old in [int(s) for s in blk.succset]:
            blk.succset._del(old)
            ob = mba.get_mblock(old)
            if ob is not None:
                with suppress(Exception):
                    ob.predset._del(blk.serial)
                ob.mark_lists_dirty()
        for ns in succs:
            blk.succset.push_back(ns)
            nb = mba.get_mblock(ns)
            if nb is not None and blk.serial not in [int(p) for p in nb.predset]:
                nb.predset.push_back(blk.serial)
                nb.mark_lists_dirty()
        blk.mark_lists_dirty()

    @staticmethod
    def _clear(blk) -> None:
        ins = blk.head
        while ins is not None:
            nxt = ins.next
            blk.remove_from_block(ins)
            ins = nxt
        # A reused/minted host block may carry the host's stale MBL_NORET ("dead
        # end: doesn't return execution control") -- set on blocks that, in the
        # ORIGINAL function, sat after a noreturn call (xalloc_die / the
        # __stack_chk_fail path). The drop repurposes such a block as a normal
        # call-continuation or branch-arm that DOES fall through to the return.
        # If the stale flag survives, Hex-Rays (from MMAT_CALLS on) severs the
        # block's successor edge and DCEs its body -- silently dropping the
        # COMPUTED return value it produces (e.g. `return hash_lookup(...) != 0`),
        # leaving only the constant arm (`return 0`). The block's noreturn-ness
        # must be re-derived from its NEW content, so clear the flag on wipe; a
        # genuinely-noreturn tail relies on BLT_0WAY (set explicitly in PASS A),
        # not on this inherited bit. See memory idavator_drop_retslot_mbl_noret.
        blk.flags &= ~hx.MBL_NORET
        blk.mark_lists_dirty()

    def _emit_i1(self, mba, blk, anchor, ea, operand, vmap):
        """Materialise an i1 ``operand`` as a 1-byte value descriptor, returning
        ``(desc, anchor)``. An icmp result (otherwise folded into a branch, never
        in ``vmap``) is emitted here as a ``setcc`` (``d = (l <pred> r) ? 1 : 0``);
        a constant ``true``/``false`` -> 1/0; anything already resolvable (a prior
        select result, a value arg) -> ``_desc``. Needed because ``select`` (and a
        short-circuit ``select`` arm) consumes an i1 as a VALUE."""
        nm = operand.name
        if nm and nm in self._icmp_defs and nm not in vmap:
            pred, iops = self._icmp_defs[nm]
            isz = _type_size(iops[0].type)
            mi = hx.minsn_t(ea)
            mi.opcode = _ICMP_SET.get(pred, hx.m_setnz)
            self._fill(mi.l, self._desc(iops[0], vmap, isz))
            self._fill(mi.r, self._desc(iops[1], vmap, isz))
            kreg = mba.alloc_kreg(1)
            mi.d.make_reg(kreg, 1)
            blk.insert_into_block(mi, anchor)
            vmap[nm] = ("reg", kreg, 1)
            return ("reg", kreg, 1), mi
        s = str(operand).strip()
        if s.split()[-1:] == ["true"]:
            return ("num", 1, 1), anchor
        if s.split()[-1:] == ["false"]:
            return ("num", 0, 1), anchor
        d = self._desc(operand, vmap, 1)
        return (d[0], d[1], 1), anchor

    def _emit_value(self, mba, blk, anchor, ea, ins, vmap, ds):
        """Emit one non-terminator LLVM instruction into ``blk`` before
        ``anchor``; record its SSA result in ``vmap``. Returns the new anchor."""
        op = ins.opcode
        ops = list(ins.operands)
        if op in _BINOP:
            size = _type_size(ins.type)
            mi = hx.minsn_t(ea)
            mi.opcode = _BINOP[op]
            self._fill(mi.l, self._desc(ops[0], vmap, size))
            r_desc = self._desc(ops[1], vmap, size)
            if op in ("shl", "lshr", "ashr"):
                # shift-amount operand must be size 1 (INTERR 50835).
                r_desc = (r_desc[0], r_desc[1], 1)
            self._fill(mi.r, r_desc)
            kreg = mba.alloc_kreg(size)
            mi.d.make_reg(kreg, size)
            blk.insert_into_block(mi, anchor)
            vmap[ins.name] = ("reg", kreg, size)
            return mi
        if op in _NOOP_CAST:
            # ptrtoint/inttoptr/bitcast are usually bit-identical reinterpretations
            # (ptr<->int, ptr<->ptr) of the SAME width -- alias the operand, no insn.
            # But ptrtoint/inttoptr can also CHANGE width (e.g. `ptrtoint i8* %p to
            # i8` truncates 8->1, `inttoptr i32 %x to ptr` widens 4->8); a width
            # change is a REAL narrow/widen, not an alias, and must lower like
            # trunc/zext (m_low / m_xdu into a fresh kreg). Otherwise the consumer
            # -- typically an icmp folded to a conditional jump -- sees an operand
            # whose size mismatches its comparand (`m_jz l=kr.8 r=#0.1`) and the
            # verifier rejects it (INTERR 50831, verify.cpp conditional-branch
            # operand-size check requires l.size == r.size).
            in_sz = _type_size(ops[0].type)
            out_sz = _type_size(ins.type)
            if in_sz == out_sz or out_sz == 0 or in_sz == 0:
                vmap[ins.name] = self._desc(ops[0], vmap, 8)
                return anchor
            mi = hx.minsn_t(ea)
            mi.opcode = hx.m_low if out_sz < in_sz else hx.m_xdu
            self._fill(mi.l, self._desc(ops[0], vmap, in_sz))
            kreg = mba.alloc_kreg(out_sz)
            mi.d.make_reg(kreg, out_sz)
            blk.insert_into_block(mi, anchor)
            vmap[ins.name] = ("reg", kreg, out_sz)
            return mi
        if op in _CAST:
            in_sz = _type_size(ops[0].type)
            out_sz = _type_size(ins.type)
            if in_sz == out_sz:
                # Same BYTE width (e.g. `zext i1 to i8` -- both 1 byte): a no-op
                # reinterpretation, not a real widen/narrow. m_xdu/m_xds/m_low all
                # INTERR on equal l/d sizes (50837/50838), so alias the operand.
                vmap[ins.name] = self._desc(ops[0], vmap, out_sz)
                return anchor
            mi = hx.minsn_t(ea)
            mi.opcode = _CAST[op]
            self._fill(mi.l, self._desc(ops[0], vmap, in_sz))
            kreg = mba.alloc_kreg(out_sz)
            mi.d.make_reg(kreg, out_sz)
            blk.insert_into_block(mi, anchor)
            vmap[ins.name] = ("reg", kreg, out_sz)
            return mi
        if op == "alloca":
            # Storage modeled as a kreg (pre-allocated in _scan_allocas); the
            # alloca itself emits nothing -- load/store route to that kreg.
            return anchor
        if op == "load":
            out_sz = _type_size(ins.type)
            if self._is_ret_slot(ops[0], vmap):
                # promoted return slot: the loaded value IS the return register.
                _ar, eax, _d = self._abi()
                vmap[ins.name] = ("reg", eax, out_sz)
                return anchor
            if (ops[0].name in self._ptr_deref_alias
                    and not _is_ptr_type(ins.type)):
                # *X (deref) of a pointer-alloca slot: the lifter reaches it via a
                # no-op bitcast and a load of the POINTEE type (e.g. `*name` as i8,
                # or a pointer-width `*total_n_read` where total_n_read is a
                # `size_t*`). Read the slot's POINTER value, then ldx through it --
                # native's `mov %X, r; ldx ds, r`. The distinguisher from a slot
                # read is the result's TYPE, not its width: the lifter type-puns
                # BOTH a full pointer-VALUE read (`load ptr, bitcast %X`, ins.type
                # is a pointer -> stays a slot read below) AND `*X` as a non-pointer
                # load (`load i64, bitcast %X` for `*p` where p is `i64*`). A
                # non-pointer result of ANY width is the unambiguous deref.
                poff = self._ptr_deref_off(ops[0], vmap)
                pr = mba.alloc_kreg(8)
                mv = hx.minsn_t(ea)
                mv.opcode = hx.m_mov
                mv.l.make_stkvar(mba, poff)
                mv.l.size = 8
                mv.d.make_reg(pr, 8)
                blk.insert_into_block(mv, anchor)
                mi = hx.minsn_t(ea)
                mi.opcode = hx.m_ldx
                mi.l.make_reg(ds, 2)
                mi.r.make_reg(pr, 8)
                kreg = mba.alloc_kreg(out_sz)
                mi.d.make_reg(kreg, out_sz)
                blk.insert_into_block(mi, mv)
                vmap[ins.name] = ("reg", kreg, out_sz)
                return mi
            slot = self._allocas.get(ops[0].name)
            if slot is not None:
                # %v = load <ty>, ptr %a  (a is a scalar slot) -> mov slot, v
                mi = hx.minsn_t(ea)
                mi.opcode = hx.m_mov
                mi.l.make_reg(slot[0], out_sz)
                kreg = mba.alloc_kreg(out_sz)
                mi.d.make_reg(kreg, out_sz)
                blk.insert_into_block(mi, anchor)
                vmap[ins.name] = ("reg", kreg, out_sz)
                return mi
            stk = self._stkvar_slot(ops[0], vmap)
            if stk is not None:
                # %v = load (frame-slot alloca or GEP-of-alloca field) -> mov stkvar, v
                mi = hx.minsn_t(ea)
                mi.opcode = hx.m_mov
                mi.l.make_stkvar(mba, stk[0])
                mi.l.size = out_sz
                kreg = mba.alloc_kreg(out_sz)
                mi.d.make_reg(kreg, out_sz)
                blk.insert_into_block(mi, anchor)
                vmap[ins.name] = ("reg", kreg, out_sz)
                return mi
            gea = self._global_ea(ops[0])
            if gea is not None:
                # %v = load <ty>, ptr @g  -> mov g, v  (the gvar IS the location)
                mi = hx.minsn_t(ea)
                mi.opcode = hx.m_mov
                mi.l.make_gvar(gea)
                mi.l.size = out_sz
                kreg = mba.alloc_kreg(out_sz)
                mi.d.make_reg(kreg, out_sz)
                blk.insert_into_block(mi, anchor)
                vmap[ins.name] = ("reg", kreg, out_sz)
                return mi
            # %v = load <ty>, <ty>* %p  ->  ldx ds, p, v
            ad = self._desc(ops[0], vmap, 8)
            mi = hx.minsn_t(ea)
            mi.opcode = hx.m_ldx
            mi.l.make_reg(ds, 2)
            self._fill(mi.r, (ad[0], ad[1], 8))
            kreg = mba.alloc_kreg(out_sz)
            mi.d.make_reg(kreg, out_sz)
            blk.insert_into_block(mi, anchor)
            vmap[ins.name] = ("reg", kreg, out_sz)
            return mi
        if op == "store":
            val_sz = _type_size(ops[0].type)
            vd = self._desc(ops[0], vmap, val_sz)
            if self._is_ret_slot(ops[1], vmap):
                # promoted return slot: write the return register directly (each
                # path writes rax, matching native -- no post-merge read block).
                _ar, eax, _d = self._abi()
                mi = hx.minsn_t(ea)
                mi.opcode = hx.m_mov
                self._fill(mi.l, (vd[0], vd[1], val_sz))
                mi.d.make_reg(eax, val_sz)
                blk.insert_into_block(mi, anchor)
                return mi
            if (ops[1].name in self._ptr_deref_alias
                    and not _is_ptr_type(ops[0].type)):
                # *X = v (deref write) of a pointer-alloca slot: a store through
                # the no-op bitcast writes the POINTEE *X (e.g. `oa->style = 10`,
                # or a pointer-width `*total_n_read = 0` where total_n_read is a
                # `size_t*`). Read the slot's POINTER value, then stx through it --
                # native's `mov %X, r; stx v, ds, r`. The distinguisher from a
                # slot DEFINE is the stored value's TYPE, not its width: a store of
                # a POINTER value (`_is_ptr_type(ops[0].type)`) instead DEFINES the
                # pointer and falls through to the slot-write path below (e.g.
                # `bucket = *table`, `oa = &default`). A non-pointer value of ANY
                # width (i8 field OR a full i64 `*p = 0`) is a deref.
                poff = self._ptr_deref_off(ops[1], vmap)
                pr = mba.alloc_kreg(8)
                mv = hx.minsn_t(ea)
                mv.opcode = hx.m_mov
                mv.l.make_stkvar(mba, poff)
                mv.l.size = 8
                mv.d.make_reg(pr, 8)
                blk.insert_into_block(mv, anchor)
                mi = hx.minsn_t(ea)
                mi.opcode = hx.m_stx
                self._fill(mi.l, (vd[0], vd[1], val_sz))
                mi.r.make_reg(ds, 2)
                mi.d.make_reg(pr, 8)
                blk.insert_into_block(mi, mv)
                return mi
            slot = self._allocas.get(ops[1].name)
            if slot is not None:
                # store <ty> %v, ptr %a  (a is a scalar slot) -> mov v, slot
                mi = hx.minsn_t(ea)
                mi.opcode = hx.m_mov
                self._fill(mi.l, (vd[0], vd[1], val_sz))
                mi.d.make_reg(slot[0], val_sz)
                blk.insert_into_block(mi, anchor)
                return mi
            stk = self._stkvar_slot(ops[1], vmap)
            if stk is not None:
                # store (frame-slot alloca or GEP-of-alloca field) -> mov v, stkvar
                mi = hx.minsn_t(ea)
                mi.opcode = hx.m_mov
                self._fill(mi.l, (vd[0], vd[1], val_sz))
                mi.d.make_stkvar(mba, stk[0])
                mi.d.size = val_sz
                blk.insert_into_block(mi, anchor)
                return mi
            gea = self._global_ea(ops[1])
            if gea is not None:
                # store <ty> %v, ptr @g  -> mov v, g  (the gvar IS the location)
                mi = hx.minsn_t(ea)
                mi.opcode = hx.m_mov
                self._fill(mi.l, (vd[0], vd[1], val_sz))
                mi.d.make_gvar(gea)
                mi.d.size = val_sz
                blk.insert_into_block(mi, anchor)
                return mi
            # store <ty> %v, <ty>* %p  ->  stx v, ds, p
            ad = self._desc(ops[1], vmap, 8)
            mi = hx.minsn_t(ea)
            mi.opcode = hx.m_stx
            self._fill(mi.l, (vd[0], vd[1], val_sz))
            mi.r.make_reg(ds, 2)
            self._fill(mi.d, (ad[0], ad[1], 8))
            blk.insert_into_block(mi, anchor)
            return mi
        if op == "getelementptr":
            slot = self._addr_taken.get(ops[0].name)
            if slot is not None:
                # GEP into a frame-slot alloca -> &stkvar(off + field). The result
                # is itself a frame address; record it so a downstream load/store/
                # call resolves to the same stkvar.
                vmap[ins.name] = ("stkaddr",
                                  slot[0] + self._gep_field_offset(ins, ops, vmap),
                                  8)
                return anchor
            # %q = getelementptr <ty>, <ty>* %p, i64 %idx  ->  q = p + idx*sizeof(ty)
            # (single-index array form; the result is an 8-byte address).
            m = re.search(r"getelementptr\s+(?:inbounds\s+)?([\w]+)",
                          str(ins).strip())
            elem_sz = _type_size(m.group(1)) if m else 1
            base = self._desc(ops[0], vmap, 8)
            idx = self._desc(ops[1], vmap, 8)
            if idx[0] == "num":
                off = idx[1] * elem_sz
                if off == 0:
                    # The off==0 decay of a global base yields its ADDRESS, not
                    # the lvalue read at it: a gvar base decays to gvaraddr
                    # (&global). _desc already returns gvaraddr for a ptr-typed
                    # global operand; normalise a gvar base here for safety.
                    if base[0] == "gvar":
                        vmap[ins.name] = ("gvaraddr", base[1], 8)
                    else:
                        vmap[ins.name] = (base[0], base[1], 8)  # alias the base
                    return anchor
                mi = hx.minsn_t(ea)
                mi.opcode = hx.m_add
                self._fill(mi.l, (base[0], base[1], 8))
                mi.r.make_number(off, 8)
            else:
                scaled = idx
                if elem_sz != 1:
                    ml = hx.minsn_t(ea)
                    ml.opcode = hx.m_mul
                    self._fill(ml.l, (idx[0], idx[1], 8))
                    ml.r.make_number(elem_sz, 8)
                    sk = mba.alloc_kreg(8)
                    ml.d.make_reg(sk, 8)
                    blk.insert_into_block(ml, anchor)
                    anchor = ml
                    scaled = ("reg", sk, 8)
                mi = hx.minsn_t(ea)
                mi.opcode = hx.m_add
                self._fill(mi.l, (base[0], base[1], 8))
                self._fill(mi.r, (scaled[0], scaled[1], 8))
            kreg = mba.alloc_kreg(8)
            mi.d.make_reg(kreg, 8)
            blk.insert_into_block(mi, anchor)
            vmap[ins.name] = ("reg", kreg, 8)
            return mi
        if op == "icmp":
            # Folded into the branch (see _build_multiblock); no value emitted.
            return anchor
        if op == "select":
            return self._emit_select(mba, blk, anchor, ea, ins, vmap)
        if op == "call":
            callee = list(ins.operands)[-1].name
            if callee in ("__readfsqword", "__readgsqword"):
                # Stack canary read -> ONE shared (unwritten) kreg for every read,
                # so `saved_canary == reread_canary` is `K == K` -> folds true ->
                # Hex-Rays prunes the __stack_chk_fail branch (the optimizer elides
                # the canary). See memory idavator_drop_canary_gate.
                sz = _type_size(ins.type) or 8
                if self._canary_kreg is None:
                    self._canary_kreg = mba.alloc_kreg(sz)
                vmap[ins.name] = ("reg", self._canary_kreg, sz)
                return anchor
            if callee == "__stack_chk_fail":
                return anchor  # canary fail path -> elided (branch is dead)
            raise RuntimeError("call must be split into its own block "
                               "(handled by the segment splitter, not _emit_value)")
        logger.warning("unhandled LLVM opcode: %s", op)
        return anchor

    def _emit_select(self, mba, blk, anchor, ea, ins, vmap):
        """Lower ``%r = select i1 %c, T %a, T %b`` (a branchless ternary).

        Three forms, all branchless (a 2-way merge can re-trip the noreturn-merge
        INTERR family -- avoid emitting one):

        - SHORT-CIRCUIT boolean (one arm a constant i1): instcombine emits
          ``select i1 %c, i1 true,  i1 %b`` == ``%c | %b`` (or.cond) and
          ``select i1 %c, i1 %a,    i1 false`` == ``%c & %a`` (and.cond). Lower to
          ``m_or``/``m_and`` on the two 1-byte i1 operands (each materialised via
          ``_emit_i1`` -- an icmp arm becomes a setcc). Exact and cheap.
        - BOOLEAN MATERIALISE ``select i1 %c, <N> 1, <N> 0`` == ``zext c to N``:
          the (1-byte) condition widened to the result width.
        - GENERAL ``select i1 %c, T %a, T %b``: a branchless blend
          ``r = b + ((a - b) & mask)`` where ``mask = 0 - (T)c`` is all-ones when
          ``c`` else 0 (so ``r == a`` when ``c``, else ``b``). T is the arm width."""
        ops = list(ins.operands)
        cond, tval, fval = ops[0], ops[1], ops[2]
        out_sz = _type_size(ins.type)
        ts, fs = str(tval).strip(), str(fval).strip()
        t_is_i1 = _type_size(tval.type) == 1 and "i1" in str(tval.type)
        f_is_i1 = _type_size(fval.type) == 1 and "i1" in str(fval.type)

        # SHORT-CIRCUIT: a constant i1 arm collapses to a bitwise op on the i1s.
        if t_is_i1 and ts.split()[-1:] == ["true"]:
            # select c, true, b  ==  c | b
            cd, anchor = self._emit_i1(mba, blk, anchor, ea, cond, vmap)
            bd, anchor = self._emit_i1(mba, blk, anchor, ea, fval, vmap)
            return self._emit_bool_binop(mba, blk, anchor, ea, ins, hx.m_or, cd,
                                         bd, vmap, out_sz)
        if f_is_i1 and fs.split()[-1:] == ["false"]:
            # select c, a, false  ==  c & a
            cd, anchor = self._emit_i1(mba, blk, anchor, ea, cond, vmap)
            ad, anchor = self._emit_i1(mba, blk, anchor, ea, tval, vmap)
            return self._emit_bool_binop(mba, blk, anchor, ea, ins, hx.m_and, cd,
                                         ad, vmap, out_sz)
        if f_is_i1 and fs.split()[-1:] == ["true"]:
            # select c, a, true  ==  !c | a  ==  (c==0) | a
            cd, anchor = self._emit_i1(mba, blk, anchor, ea, cond, vmap)
            ad, anchor = self._emit_i1(mba, blk, anchor, ea, tval, vmap)
            nc = hx.minsn_t(ea)
            nc.opcode = hx.m_setz
            self._fill(nc.l, cd)
            nc.r.make_number(0, 1)
            nk = mba.alloc_kreg(1)
            nc.d.make_reg(nk, 1)
            blk.insert_into_block(nc, anchor)
            anchor = nc
            return self._emit_bool_binop(mba, blk, anchor, ea, ins, hx.m_or,
                                         ("reg", nk, 1), ad, vmap, out_sz)
        if t_is_i1 and ts.split()[-1:] == ["false"]:
            # select c, false, b  ==  !c & b
            cd, anchor = self._emit_i1(mba, blk, anchor, ea, cond, vmap)
            bd, anchor = self._emit_i1(mba, blk, anchor, ea, fval, vmap)
            nc = hx.minsn_t(ea)
            nc.opcode = hx.m_setz
            self._fill(nc.l, cd)
            nc.r.make_number(0, 1)
            nk = mba.alloc_kreg(1)
            nc.d.make_reg(nk, 1)
            blk.insert_into_block(nc, anchor)
            anchor = nc
            return self._emit_bool_binop(mba, blk, anchor, ea, ins, hx.m_and,
                                         ("reg", nk, 1), bd, vmap, out_sz)

        # BOOLEAN MATERIALISE: select c, 1, 0  ==  zext c.
        td = self._desc(tval, vmap, out_sz) if tval.name not in self._icmp_defs \
            else None
        fd = self._desc(fval, vmap, out_sz) if fval.name not in self._icmp_defs \
            else None
        if (td and td[0] == "num" and td[1] == 1
                and fd and fd[0] == "num" and fd[1] == 0):
            cd, anchor = self._emit_i1(mba, blk, anchor, ea, cond, vmap)
            if out_sz == 1:
                vmap[ins.name] = (cd[0], cd[1], 1)
                return anchor
            mi = hx.minsn_t(ea)
            mi.opcode = hx.m_xdu
            self._fill(mi.l, cd)
            kreg = mba.alloc_kreg(out_sz)
            mi.d.make_reg(kreg, out_sz)
            blk.insert_into_block(mi, anchor)
            vmap[ins.name] = ("reg", kreg, out_sz)
            return mi

        # GENERAL branchless blend: r = b + ((a - b) & mask), mask = 0 - (T)c.
        ad = self._desc(tval, vmap, out_sz)
        bd = self._desc(fval, vmap, out_sz)
        cd, anchor = self._emit_i1(mba, blk, anchor, ea, cond, vmap)
        # mask = 0 - zext(c, out_sz)  (all-ones when c, else 0).
        if out_sz != 1:
            zc = hx.minsn_t(ea)
            zc.opcode = hx.m_xdu
            self._fill(zc.l, cd)
            zk = mba.alloc_kreg(out_sz)
            zc.d.make_reg(zk, out_sz)
            blk.insert_into_block(zc, anchor)
            anchor = zc
            cwide = ("reg", zk, out_sz)
        else:
            cwide = cd
        mneg = hx.minsn_t(ea)
        mneg.opcode = hx.m_neg
        self._fill(mneg.l, (cwide[0], cwide[1], out_sz))
        mk = mba.alloc_kreg(out_sz)
        mneg.d.make_reg(mk, out_sz)
        blk.insert_into_block(mneg, anchor)
        anchor = mneg
        # diff = a - b
        sub = hx.minsn_t(ea)
        sub.opcode = hx.m_sub
        self._fill(sub.l, (ad[0], ad[1], out_sz))
        self._fill(sub.r, (bd[0], bd[1], out_sz))
        sk = mba.alloc_kreg(out_sz)
        sub.d.make_reg(sk, out_sz)
        blk.insert_into_block(sub, anchor)
        anchor = sub
        # masked = diff & mask
        msk = hx.minsn_t(ea)
        msk.opcode = hx.m_and
        msk.l.make_reg(sk, out_sz)
        msk.r.make_reg(mk, out_sz)
        ck = mba.alloc_kreg(out_sz)
        msk.d.make_reg(ck, out_sz)
        blk.insert_into_block(msk, anchor)
        anchor = msk
        # result = b + masked
        add = hx.minsn_t(ea)
        add.opcode = hx.m_add
        self._fill(add.l, (bd[0], bd[1], out_sz))
        add.r.make_reg(ck, out_sz)
        rk = mba.alloc_kreg(out_sz)
        add.d.make_reg(rk, out_sz)
        blk.insert_into_block(add, anchor)
        vmap[ins.name] = ("reg", rk, out_sz)
        return add

    def _emit_bool_binop(self, mba, blk, anchor, ea, ins, opcode, ld, rd, vmap,
                         out_sz):
        """Emit ``d = ld <opcode> rd`` (m_or/m_and on two 1-byte i1 values) for a
        short-circuit select; record the 1-byte result in ``vmap`` (widened to the
        select's result width if it is wider than i1)."""
        mi = hx.minsn_t(ea)
        mi.opcode = opcode
        self._fill(mi.l, (ld[0], ld[1], 1))
        self._fill(mi.r, (rd[0], rd[1], 1))
        bk = mba.alloc_kreg(1)
        mi.d.make_reg(bk, 1)
        blk.insert_into_block(mi, anchor)
        anchor = mi
        if out_sz == 1:
            vmap[ins.name] = ("reg", bk, 1)
            return mi
        wi = hx.minsn_t(ea)
        wi.opcode = hx.m_xdu
        wi.l.make_reg(bk, 1)
        wk = mba.alloc_kreg(out_sz)
        wi.d.make_reg(wk, out_sz)
        blk.insert_into_block(wi, anchor)
        vmap[ins.name] = ("reg", wk, out_sz)
        return wi

    def _formal_arg_sizes(self, callee_ea, nargs):
        """``[size_0, ..., size_{nargs-1}]`` of a resolved DIRECT callee's formal
        parameters from its IDB prototype, or ``None`` when no usable widening
        applies -- an unresolved ea, a non-function/vararg type, or a prototype
        whose arity differs from the call (defer rather than mis-pair widths).

        Used by ``_emit_call`` to widen each in-register arg to its real ABI width
        when the lifted IR operand type is narrower than the callee declares (the
        ``memset``/``size_t`` case). A vararg callee is excluded: its surplus-arg
        widths are carried by the explicit mcallinfo path, not here."""
        if callee_ea == ida_idaapi.BADADDR:
            return None
        ctif = ida_typeinf.tinfo_t()
        if not ida_nalt.get_tinfo(ctif, callee_ea) or not ctif.is_func():
            return None
        if ctif.is_vararg_cc() or ctif.get_nargs() != nargs:
            return None
        out = []
        for i in range(nargs):
            sz = ctif.get_nth_arg(i).get_size()
            out.append(sz if sz not in (0, ida_idaapi.BADADDR) else 0)
        return out

    def _emit_call(self, mba, blk, anchor, ea, ins, vmap, argregs):
        """Emit `mov`s into the ABI arg-regs then `m_call l=gvar(callee), d=rax`
        as the block TAIL. The call must terminate its block (it falls through to
        the continuation, BLT_1WAY) -- a call defining rax mid-block is fine for
        50864 but later maturities want calls block-terminal. Returns the call."""
        ops = list(ins.operands)
        callee = ops[-1]
        call_args = ops[:-1]
        _argregs, eax, _ds = self._abi()
        # More integer args than ABI registers: the 7th+ ride on the stack. A
        # direct (named) callee whose prototype is known lets us build an
        # explicit mcallinfo (set_type does the SysV reg/stack classification),
        # so the stack args travel IN the call -- no SP-modeled pushes. Indirect
        # / unresolved callees fall through to their existing handling below.
        if (len(call_args) > len(argregs) and callee.name not in vmap
                and ida_name.get_name_ea(ida_idaapi.BADADDR, callee.name)
                != ida_idaapi.BADADDR):
            return self._emit_call_stackargs(mba, blk, anchor, ea, ins, vmap)
        if len(call_args) > len(argregs):
            raise NotImplementedError(
                "stack-passed call argument (more args than ABI registers)")
        # Callee: a direct named function/global -> gvar. An indirect call through
        # an SSA value (a function pointer, e.g. a loaded struct field) is lowered
        # to Hex-Rays' native m_icall form -- see _emit_call_indirect.
        if callee.name in vmap:
            return self._emit_call_indirect(mba, blk, anchor, ea, ins, vmap)
        callee_ea = ida_name.get_name_ea(ida_idaapi.BADADDR, callee.name)
        if callee_ea == ida_idaapi.BADADDR:
            if callee.name in _HELPER_INTRINSICS:
                arg_descs = [self._desc(a, vmap, _type_size(a.type))
                             for a in call_args]
                return self._emit_helper_call(
                    mba, blk, anchor, ea, ins, callee.name, arg_descs, eax)
            raise ValueError(f"unresolved callee @{callee.name}")
        # Variadic callee (e.g. printf/error) carrying trailing varargs: a bare
        # `m_call gvar` with no mcallinfo leaves Hex-Rays to re-discover the
        # vararg count from its own analysis, and it drops them (rendering only a
        # mis-resolved fmt). Build an EXPLICIT mcallinfo declaring every passed
        # arg (fixed prefix + varargs) so the call carries them deterministically.
        # Only when surplus args are actually present beyond the fixed prototype.
        ctif = ida_typeinf.tinfo_t()
        is_vararg = (ida_nalt.get_tinfo(ctif, callee_ea) and ctif.is_func()
                     and ctif.is_vararg_cc())
        if is_vararg and len(call_args) > ctif.get_nargs():
            return self._emit_call_vararg(
                mba, blk, anchor, ea, ins, vmap, callee_ea, ctif)
        # Variadic callee invoked with EXACTLY its fixed args (no surplus vararg).
        # A bare `m_call gvar` to a variadic prototype still lets Hex-Rays' own
        # vararg recovery invent trailing args -- e.g. `open("/dev/urandom", 0)`
        # mis-renders as `open(a0, a1, a2)`, reading stale incoming-param regs as
        # phantom varargs. Pin the call with an EXPLICIT fixed-arg variadic
        # mcallinfo (FCI_FINAL) so no re-derivation occurs. set_type keeps the
        # ellipsis cc; we fill only the fixed args (no surplus tail).
        if is_vararg and len(call_args) <= ctif.get_nargs():
            return self._emit_call_vararg_fixed(
                mba, blk, anchor, ea, ins, vmap, callee_ea, ctif)
        # Resolve args once; a call that materializes a frame address (&local)
        # must carry the host resting-frame ea so Hex-Rays computes a
        # frame-consistent mcallinfo.call_spd -- else WARN_BAD_CALL_SP ("bad sp
        # value at call"). See memory idavator_sp_gate_call_ea_cracked.
        arg_descs = [self._desc(a, vmap, _type_size(a.type)) for a in call_args]
        passes_stkaddr = any(d[0] == "stkaddr" for d in arg_descs)
        # Argument register WIDTH comes from the callee's REAL prototype, not the
        # lifted IR operand type. ida2llvm may declare an extern narrower than its
        # IDB prototype -- e.g. ``memset`` lifts as ``(i8*, i32, i32)`` so the size
        # arg is an i32 (4 bytes), but IDA's real ``memset`` takes ``size_t n``
        # (8 bytes). A 4-byte ``mov #4, edx`` feeding an 8-byte ``size_t`` use
        # leaves the high dword of rdx undefined: Hex-Rays then can't fold the
        # constant (renders ``LODWORD(v)=4; memset(&ctx,0,v)`` instead of
        # ``memset(&ctx,0,sizeof(ctx))``) and the partial-def trips "local variable
        # allocation has failed". Widen each in-register arg to its formal param
        # size (zero-extend a narrower reg via m_xdu; a number just takes the wider
        # size) -- exactly the formal-size widening _emit_call_stackargs already
        # applies to the 7th+ args. Only for a resolved, non-vararg callee whose
        # prototype arity matches the call (vararg widths ride in the mcallinfo).
        formal_szs = self._formal_arg_sizes(callee_ea, len(call_args))
        for i, (a, d) in enumerate(zip(call_args, arg_descs)):
            asz = _type_size(a.type)
            fsz = formal_szs[i] if formal_szs is not None else asz
            if fsz > asz:
                if d[0] == "num":
                    d, asz = ("num", d[1], fsz), fsz
                elif d[0] == "reg":
                    mi = hx.minsn_t(ea)
                    mi.opcode = hx.m_xdu
                    self._fill(mi.l, (d[0], d[1], asz))
                    kreg = mba.alloc_kreg(fsz)
                    mi.d.make_reg(kreg, fsz)
                    blk.insert_into_block(mi, anchor)
                    anchor = mi
                    d, asz = ("reg", kreg, fsz), fsz
            mv = hx.minsn_t(ea)
            mv.opcode = hx.m_mov
            self._fill(mv.l, d)
            mv.d.make_reg(argregs[i], asz)
            blk.insert_into_block(mv, anchor)
            anchor = mv
        call_ea = (self._call_spd_ea
                   if passes_stkaddr and self._call_spd_ea is not None else ea)
        mc = hx.minsn_t(call_ea)
        mc.opcode = hx.m_call
        mc.l.make_gvar(callee_ea)
        rsz = _type_size(ins.type) if str(ins.type) != "void" else 8
        mc.d.make_reg(eax, rsz)  # call defines rax (the continuation captures it)
        blk.insert_into_block(mc, anchor)
        return mc

    def _emit_call_vararg(self, mba, blk, anchor, ea, ins, vmap, callee_ea, ctif):
        """Emit a DIRECT call to a VARIADIC callee (printf/error/...) carrying
        trailing varargs, with an EXPLICIT variadic ``mcallinfo_t`` so Hex-Rays
        renders every argument (the fixed prefix AND the varargs).

        A bare ``m_call gvar`` with no callinfo leaves the vararg count to
        Hex-Rays' own analysis, which drops the trailing args (it renders only a
        mis-resolved fmt). We MIRROR Hex-Rays' own post-CALLS variadic form
        (native ``mov call $".printf"<...:"const char *format" &fmt, _QWORD v1,
        _QWORD v2> => int .4, eax.4``):

        * ``set_type`` with the callee's REAL variadic tinfo -- this KEEPS the
          ellipsis cc (``is_vararg()`` stays true) and creates the FIXED args with
          their SysV arglocs (fmt in rdi, ...). Do NOT rebuild a concrete
          fastcall prototype: that normalizes ellipsis away and the final
          decompile then drops the call (INTERR 50406/50743).
        * APPEND each surplus vararg to ``fi.args`` with the next SysV integer
          register argloc (rsi, rdx, rcx, r8, r9), exactly as the variadic tail
          appears natively. ``solid_args`` stays the fixed count; the varargs ride
          beyond it.

        The result is DISCARDED (every cp variadic call site discards it), so the
        call is emitted as a bare result-discarded ``m_call`` (``d.size`` 0, no
        retregs) -- a pre-built return register conflicts with Hex-Rays' own
        callinfo re-derivation at MMAT_CALLS (INTERR 50743/50406)."""
        # The result-discarded modeling below has no recipe for a CONSUMED vararg
        # result -- defer (native fallback) rather than silently corrupt. Checked
        # first so no partial microcode is emitted on the unsupported path.
        if self._value_used(ins):
            raise NotImplementedError(
                f"vararg call result consumed for @{ins.operands[-1].name}")
        ops = list(ins.operands)
        call_args = ops[:-1]
        _argregs, eax, _ds = self._abi()
        arg_reg_names = self._arg_reg_names()
        nfixed = ctif.get_nargs()
        # Return width from the callee PROTOTYPE, not the LLVM call type (the lift
        # may type the result narrower than the ABI return reg); void -> 8.
        crettype = ctif.get_rettype()
        rsz = crettype.get_size()
        if rsz in (0, ida_idaapi.BADADDR) or crettype.is_void():
            rsz = 8
        # set_type with the REAL variadic prototype keeps cc=ellipsis + fixed
        # arglocs (is_vararg() stays true).
        fi = hx.mcallinfo_t(callee_ea, 0)
        if not fi.set_type(ctif):
            raise NotImplementedError(
                f"vararg call: set_type failed for @{ins.operands[-1].name}")

        def _fill_arg(arg, a):
            """Fill one mcallarg's mop VALUE from LLVM operand ``a``, widening a
            narrower value into the formal slot (verifier wants mop.size == slot)."""
            nonlocal anchor
            fsz = arg.type.get_size()
            if fsz in (0, ida_idaapi.BADADDR):
                fsz = _type_size(a.type)
            asz = _type_size(a.type)
            d = self._desc(a, vmap, asz)
            if asz < fsz and d[0] in ("reg", "num"):
                if d[0] == "num":
                    d = ("num", d[1], fsz)
                else:
                    mi = hx.minsn_t(ea)
                    mi.opcode = hx.m_xdu
                    self._fill(mi.l, (d[0], d[1], asz))
                    kreg = mba.alloc_kreg(fsz)
                    mi.d.make_reg(kreg, fsz)
                    blk.insert_into_block(mi, anchor)
                    anchor = mi
                    d = ("reg", kreg, fsz)
            tmp = hx.mop_t()
            self._fill(tmp, (d[0], d[1], fsz))
            arg.copy_mop(tmp)
            arg.size = fsz

        # Fixed args: set_type already created their slots + arglocs; fill values.
        for i in range(min(nfixed, fi.args.size())):
            _fill_arg(fi.args[i], call_args[i])
        # Variadic tail: APPEND each surplus arg with the next SysV integer
        # register argloc, mirroring the native variadic tail.
        for i in range(nfixed, len(call_args)):
            a = call_args[i]
            arg = hx.mcallarg_t()
            asz = _type_size(a.type)
            fsz = asz if asz in (1, 2, 4, 8) else 8
            # A pointer operand renders cleanly as a ``void *`` vararg (else the
            # value shows as a ``(signed __int64)`` reinterpret cast).
            if getattr(a.type, "is_pointer", False) or str(a.type) == "ptr":
                pt = ida_typeinf.tinfo_t()
                pt.create_ptr(ida_typeinf.tinfo_t(ida_typeinf.BTF_VOID))
                arg.type = pt
                fsz = 8
            else:
                arg.type = self._int_tinfo(fsz)
            arg.size = fsz
            if i < len(arg_reg_names):
                arg.argloc._set_reg1(ida_idp.str2reg(arg_reg_names[i]))
            _fill_arg(arg, a)
            fi.args.push_back(arg)
        # Result scaffolding. The variadic callees we recover (printf/error/...)
        # DISCARD their result in every cp call site; model the call as a
        # result-discarded one: d.size = 0, NO retregs/return_regs. A pre-built
        # mcallinfo that DECLARES a return register conflicts with Hex-Rays' own
        # re-derivation of the callinfo at MMAT_CALLS (used_retvals vs retval size
        # -> INTERR 50743); a result-discarded call sidesteps that and lets
        # Hex-Rays' printf-format machinery recover the varargs. spoiled keeps rax
        # clobbered (an ABI call destroys it).
        fi.spoiled.add(eax, rsz)
        fi.return_type = self._int_tinfo(rsz)
        fi.flags |= hx.FCI_HASFMT
        call_ea = (self._call_spd_ea
                   if self._call_spd_ea is not None else ea)
        fi.call_spd = ida_frame.get_spd(ida_funcs.get_func(mba.entry_ea),
                                        call_ea) if ida_funcs.get_func(
                                            mba.entry_ea) else 0
        mc = hx.minsn_t(call_ea)
        mc.opcode = hx.m_call
        mc.l.make_gvar(callee_ea)
        mc.d._make_callinfo(fi)
        mc.d.size = 0  # result discarded
        blk.insert_into_block(mc, anchor)
        return mc

    def _emit_call_vararg_fixed(self, mba, blk, anchor, ea, ins, vmap,
                                callee_ea, ctif):
        """Emit a DIRECT call to a VARIADIC callee invoked with EXACTLY its fixed
        args (NO surplus vararg), pinning the arg list so Hex-Rays does not invent
        phantom trailing varargs.

        A bare ``m_call gvar`` to a variadic prototype (e.g. ``open(const char *,
        int, ...)``) lets Hex-Rays' own vararg recovery read stale incoming-param
        registers as phantom varargs -- ``open("/dev/urandom", 0)`` mis-renders as
        ``open(a0, a1, a2)``. We build an EXPLICIT mcallinfo: ``set_type`` with the
        REAL variadic tinfo (KEEPS the ellipsis cc + the fixed-arg SysV arglocs),
        fill ONLY the fixed args, and set ``FCI_FINAL`` so the call list is taken
        as authoritative (no re-derivation).

        Unlike ``_emit_call_vararg`` (which models a result-DISCARDED variadic tail
        call), this handles both the discarded and the CONSUMED result: when the
        result is used we seed retregs/return_regs (mirroring
        ``_emit_call_stackargs``) and wrap the call in a ``mov ... => rax`` so the
        continuation captures it; when discarded we emit a bare result-0 call."""
        _argregs, eax, _ds = self._abi()
        ops = list(ins.operands)
        call_args = ops[:-1]
        crettype = ctif.get_rettype()
        rsz = crettype.get_size()
        if rsz in (0, ida_idaapi.BADADDR) or crettype.is_void():
            rsz = 8
        fi = hx.mcallinfo_t(callee_ea, 0)
        if not fi.set_type(ctif):
            raise NotImplementedError(
                f"fixed-arg vararg call: set_type failed for @{ops[-1].name}")
        # Fill each FIXED formal arg's mop VALUE (set_type fixed its argloc/type);
        # widen a narrower LLVM value into the formal slot (verifier wants
        # mop.size == formal slot size).
        for i in range(min(len(call_args), fi.args.size())):
            arg = fi.args[i]
            a = call_args[i]
            fsz = arg.type.get_size()
            asz = _type_size(a.type)
            if fsz in (0, ida_idaapi.BADADDR):
                fsz = asz
            d = self._desc(a, vmap, asz)
            if asz < fsz and d[0] in ("reg", "num"):
                if d[0] == "num":
                    d = ("num", d[1], fsz)
                else:
                    mi = hx.minsn_t(ea)
                    mi.opcode = hx.m_xdu
                    self._fill(mi.l, (d[0], d[1], asz))
                    kreg = mba.alloc_kreg(fsz)
                    mi.d.make_reg(kreg, fsz)
                    blk.insert_into_block(mi, anchor)
                    anchor = mi
                    d = ("reg", kreg, fsz)
            tmp = hx.mop_t()
            self._fill(tmp, (d[0], d[1], fsz))
            arg.copy_mop(tmp)
            arg.size = fsz
        # FCI_FINAL: take this arg list as authoritative (suppress phantom-vararg
        # re-derivation). spoiled keeps rax clobbered (an ABI call destroys it).
        fi.flags |= hx.FCI_FINAL
        fi.spoiled.add(eax, rsz)
        fi.return_type = (crettype if crettype.get_size() not in
                          (0, ida_idaapi.BADADDR) and not crettype.is_void()
                          else self._int_tinfo(rsz))
        call_ea = (self._call_spd_ea
                   if self._call_spd_ea is not None else ea)
        fi.call_spd = ida_frame.get_spd(ida_funcs.get_func(mba.entry_ea),
                                        call_ea) if ida_funcs.get_func(
                                            mba.entry_ea) else 0
        if not self._value_used(ins):
            # Result discarded -> bare result-0 call (no retregs).
            mc = hx.minsn_t(call_ea)
            mc.opcode = hx.m_call
            mc.l.make_gvar(callee_ea)
            mc.d._make_callinfo(fi)
            mc.d.size = 0
            blk.insert_into_block(mc, anchor)
            return mc
        # Result consumed -> seed retregs/return_regs and wrap in mov => rax so the
        # continuation captures the result (cf. _emit_call_stackargs).
        ret_mop = hx.mop_t()
        ret_mop.make_reg(eax, rsz)
        fi.retregs.push_back(ret_mop)
        fi.return_regs.add(eax, rsz)
        inner = hx.minsn_t(call_ea)
        inner.opcode = hx.m_call
        inner.l.make_gvar(callee_ea)
        inner.d._make_callinfo(fi)
        inner.d.size = rsz
        mov = hx.minsn_t(call_ea)
        mov.opcode = hx.m_mov
        mov.l.make_insn(inner)
        mov.l.size = rsz
        mov.d.make_reg(eax, rsz)
        blk.insert_into_block(mov, anchor)
        return mov

    def _emit_call_stackargs(self, mba, blk, anchor, ea, ins, vmap):
        """Emit a direct call with MORE integer args than ABI registers (the
        7th+ travel on the stack). Build an explicit ``mcallinfo_t`` from the
        callee's known prototype: ``set_type`` does the SysV reg/stack
        classification (regs for 0..5, ALOC_STACK for 6+), so each stack arg
        rides IN the call -- no SP-modeled ``push`` sequence, no WARN_BAD_CALL_SP.

        The emitted shape mirrors Hex-Rays' own post-CALLS form:
        ``mov (call gvar<...> => mop_f(callinfo)) => rax``. We carry the
        host resting-frame ea on the call (frame allocated) and seed
        retregs/return_regs/spoiled so the value-numbering of the result is
        consistent (cf. the indirect-call recipe, memory
        idavator_drop_call_construction)."""
        ops = list(ins.operands)
        callee = ops[-1]
        call_args = ops[:-1]
        _argregs, eax, _ds = self._abi()
        callee_ea = ida_name.get_name_ea(ida_idaapi.BADADDR, callee.name)
        tif = ida_typeinf.tinfo_t()
        if not ida_nalt.get_tinfo(tif, callee_ea) or not tif.is_func():
            raise NotImplementedError(
                f"stack-passed args: no prototype for @{callee.name}")
        nargs = tif.get_nargs()
        if nargs != len(call_args):
            # The lift's arg count must match the prototype for set_type's
            # arglocs to line up; otherwise defer rather than mis-place args.
            raise NotImplementedError(
                f"stack-passed args: prototype arity {nargs} != "
                f"call arity {len(call_args)} for @{callee.name}")

        fi = hx.mcallinfo_t(callee_ea, 0)
        if not fi.set_type(tif):
            raise NotImplementedError(
                f"stack-passed args: set_type failed for @{callee.name}")
        # Return width comes from the PROTOTYPE, not the LLVM call type: the lift
        # may type the result narrower (e.g. `i1` for an `int`-returning fn), but
        # the call's retreg/return_type must match the real ABI return register
        # (else INTERR 50743 -- retreg count vs retval size). void -> 8 (rax def).
        rettype = tif.get_rettype()
        rsz = rettype.get_size()
        if rsz in (0, ida_idaapi.BADADDR) or rettype.is_void():
            rsz = 8
        # Fill each formal arg's mop VALUE (set_type already fixed its argloc +
        # type); copy_mop overwrites only the mop_t base, preserving argloc/type.
        # The verifier requires mop.size == formal type size (INTERR 50735), so
        # a narrower LLVM value (e.g. i1 into an `int` slot) is widened first --
        # exactly what Hex-Rays renders natively (xdu.4(%v.1)).
        for i, a in enumerate(call_args):
            arg = fi.args[i]
            fsz = arg.type.get_size()
            asz = _type_size(a.type)
            d = self._desc(a, vmap, asz)
            if asz < fsz and d[0] in ("reg", "num"):
                if d[0] == "num":
                    d = ("num", d[1], fsz)  # numbers just take the wider size
                else:
                    mi = hx.minsn_t(ea)
                    mi.opcode = hx.m_xdu
                    self._fill(mi.l, (d[0], d[1], asz))
                    kreg = mba.alloc_kreg(fsz)
                    mi.d.make_reg(kreg, fsz)
                    blk.insert_into_block(mi, anchor)
                    anchor = mi
                    d = ("reg", kreg, fsz)
            tmp = hx.mop_t()
            self._fill(tmp, (d[0], d[1], fsz))
            arg.copy_mop(tmp)
            arg.size = fsz
        # Result scaffolding: retregs/return_regs/spoiled keep the rax def
        # consistent under value-numbering (50743/50740 class).
        ret_mop = hx.mop_t()
        ret_mop.make_reg(eax, rsz)
        fi.retregs.push_back(ret_mop)
        fi.return_regs.add(eax, rsz)
        fi.spoiled.add(eax, rsz)
        fi.return_type = (rettype if rettype.get_size() not in
                          (0, ida_idaapi.BADADDR) and not rettype.is_void()
                          else self._int_tinfo(rsz))
        # Frame must be allocated at the call ea (else WARN_BAD_CALL_SP for the
        # stack args / any &local), and record call_spd to match.
        call_ea = (self._call_spd_ea
                   if self._call_spd_ea is not None else ea)
        fi.call_spd = ida_frame.get_spd(ida_funcs.get_func(mba.entry_ea),
                                        call_ea) if ida_funcs.get_func(
                                            mba.entry_ea) else 0
        # Inner m_call: l = callee gvar, d = mop_f(callinfo). Wrap in a mov to
        # rax so the continuation captures the result like any other call.
        inner = hx.minsn_t(call_ea)
        inner.opcode = hx.m_call
        inner.l.make_gvar(callee_ea)
        inner.d._make_callinfo(fi)
        inner.d.size = rsz
        mov = hx.minsn_t(call_ea)
        mov.opcode = hx.m_mov
        mov.l.make_insn(inner)
        mov.l.size = rsz
        mov.d.make_reg(eax, rsz)
        blk.insert_into_block(mov, anchor)
        return mov

    def _emit_call_indirect(self, mba, blk, anchor, ea, ins, vmap):
        """Emit an INDIRECT call -- the callee is an SSA function-pointer VALUE
        (e.g. an ``inttoptr`` of a loaded struct field), not a named symbol -- in
        Hex-Rays' native ``m_icall`` form::

            mov  (icall cs.2, <callee-value>.8 <mcallinfo>) => rax

        This mirrors Hex-Rays' own post-CALLS lowering. Reference (``hash_lookup``
        @ MMAT_CALLS): ``icall cs.2,[ds:(table+0x38)].8<fast:_QWORD a0,_QWORD a1>
        => __int64 .8, rax.8`` with an mcallinfo of ``cc=0x70 callee=BADADDR
        solid_args=2 return_type=__int64``, ``retregs`` EMPTY, ``return_regs=rax``.

        We synthesize a ``__fastcall`` prototype for the callee so ``set_type``
        does the SysV reg/cc classification (regs rdi/rsi/...; cc=0x70) exactly
        like a direct call. The native call carries flags
        ``FCI_PROP|FCI_DEAD|FCI_SPLOK``: ``FCI_PROP`` is REQUIRED -- it lets the
        verifier accept an empty ``retregs`` list (without it the retreg/return
        cross-check fires INTERR 50745 first)."""
        ops = list(ins.operands)
        callee = ops[-1]
        call_args = ops[:-1]
        _argregs, eax, _ds = self._abi()
        rsz = _type_size(ins.type) if str(ins.type) != "void" else 8
        # Synthesize a __fastcall fn-ptr prototype so set_type fixes the SysV
        # arglocs/cc exactly like a direct call (regs rdi/rsi/... ; cc=0x70).
        ft = ida_typeinf.func_type_data_t()
        ft.cc = ida_typeinf.CM_CC_FASTCALL
        ft.rettype = self._int_tinfo(rsz)
        for a in call_args:
            fa = ida_typeinf.funcarg_t()
            fa.type = self._int_tinfo(_type_size(a.type))
            ft.push_back(fa)
        tif = ida_typeinf.tinfo_t()
        tif.create_func(ft)
        fi = hx.mcallinfo_t(ida_idaapi.BADADDR, 0)
        if not fi.set_type(tif):
            raise NotImplementedError("indirect call: set_type failed")
        # Fill each formal arg's mop VALUE (set_type already fixed its argloc +
        # type + size); the verifier requires mop.size == formal type size.
        for i, a in enumerate(call_args):
            arg = fi.args[i]
            fsz = arg.type.get_size()
            d = self._desc(a, vmap, _type_size(a.type))
            tmp = hx.mop_t()
            self._fill(tmp, (d[0], d[1], fsz))
            arg.copy_mop(tmp)
            arg.size = fsz
        # Result scaffolding -- MIRROR NATIVE: retregs EMPTY, return_regs=rax. The
        # empty retregs list is only legal because FCI_PROP is set below.
        fi.return_regs.add(eax, rsz)
        fi.spoiled.add(eax, rsz)
        fi.return_type = self._int_tinfo(rsz)
        fi.solid_args = len(call_args)
        # NATIVE flags (hash_lookup): FCI_PROP|FCI_DEAD|FCI_SPLOK. FCI_PROP makes
        # the verifier SKIP the retreg/return-size cross-check (else 50745/50743).
        fi.flags |= hx.FCI_PROP | hx.FCI_DEAD | hx.FCI_SPLOK
        call_ea = (self._call_spd_ea
                   if self._call_spd_ea is not None else ea)
        fi.call_spd = ida_frame.get_spd(ida_funcs.get_func(mba.entry_ea),
                                        call_ea) if ida_funcs.get_func(
                                            mba.entry_ea) else 0
        # Inner m_icall: l = cs.2 (code segment), r = callee fn-ptr VALUE.8,
        # d = mop_f(callinfo). Wrap in a mov to rax so the continuation captures
        # the result like any other call.
        cs = hx.reg2mreg(ida_idp.str2reg("cs"))
        inner = hx.minsn_t(call_ea)
        inner.opcode = hx.m_icall
        inner.l.make_reg(cs, 2)
        cdesc = self._desc(callee, vmap, 8)
        self._fill(inner.r, (cdesc[0], cdesc[1], 8))
        inner.d._make_callinfo(fi)
        inner.d.size = rsz
        mov = hx.minsn_t(call_ea)
        mov.opcode = hx.m_mov
        mov.l.make_insn(inner)
        mov.l.size = rsz
        mov.d.make_reg(eax, rsz)
        blk.insert_into_block(mov, anchor)
        return mov

    @staticmethod
    def _int_tinfo(size):
        btf = {1: ida_typeinf.BTF_INT8, 2: ida_typeinf.BTF_INT16,
               4: ida_typeinf.BTF_INT32, 8: ida_typeinf.BTF_INT64}.get(
                   size, ida_typeinf.BTF_INT)
        return ida_typeinf.tinfo_t(btf)

    def _emit_helper_call(self, mba, blk, anchor, ea, ins, name, arg_descs, eax):
        """Emit an unresolved rotate intrinsic (`__ROR8__` ...) as a Hex-Rays
        HELPER call. Args ride in the mcallinfo via ``create_helper_call`` (the
        `make_helper` + reg-arg-mov shape crashes the decompiler). ``out`` = rax so
        the segment-split continuation captures the result like any other call."""
        callargs = hx.mcallargs_t()
        for a, d in zip(list(ins.operands)[:-1], arg_descs):
            sz = _type_size(a.type)
            arg = hx.mcallarg_t()
            self._fill(arg, (d[0], d[1], sz))
            arg.size = sz
            arg.type = self._int_tinfo(sz)
            callargs.push_back(arg)
        rsz = _type_size(ins.type) if str(ins.type) != "void" else 8
        out = hx.mop_t()
        out.make_reg(eax, rsz)
        mc = mba.create_helper_call(ea, name, self._int_tinfo(rsz), callargs, out)
        if mc is None:
            raise RuntimeError(f"create_helper_call({name}) returned None")
        blk.insert_into_block(mc, anchor)
        return mc

    def _capture_call_result(self, mba, blk, anchor, ea, call_ins, eax, vmap):
        """At a continuation block's START, copy the call's rax result into a
        scratch kreg so it survives a later call that clobbers rax. Registers
        the kreg in vmap as the call's SSA value."""
        rsz = _type_size(call_ins.type)
        kreg = mba.alloc_kreg(rsz)
        mv = hx.minsn_t(ea)
        mv.opcode = hx.m_mov
        mv.l.make_reg(eax, rsz)
        mv.d.make_reg(kreg, rsz)
        blk.insert_into_block(mv, anchor)
        vmap[call_ins.name] = ("reg", kreg, rsz)
        return mv

    def _emit_ret_value(self, blk, anchor, ea, term, eax, vmap):
        """Emit `mov <retval>, eax` for an LLVM ret (no terminator). A promoted
        return slot is already in the return reg, so no move is emitted."""
        ops = list(term.operands)
        if not ops:
            return anchor
        sz = _type_size(ops[0].type)
        d = self._desc(ops[0], vmap, sz)
        if d[0] == "reg" and d[1] == eax:
            return anchor  # already in the return reg (promoted return slot)
        mi = hx.minsn_t(ea)
        mi.opcode = hx.m_mov
        self._fill(mi.l, d)
        mi.d.make_reg(eax, sz)
        blk.insert_into_block(mi, anchor)
        return mi

    def _alloca_struct_key(self, ins):
        """The ``self._struct_size`` key for a ``%struct`` alloca (quotes stripped,
        llvmlite ``.NN`` disambiguation suffix tolerated), or None for a scalar/ptr
        alloca. Mirrors _array_dims's element-key handling."""
        m = re.search(r"alloca\s+(?:inalloca\s+)?([^,\n]+)", str(ins).strip())
        ty = m.group(1).strip() if m else "i64"
        if "*" in ty or ty == "ptr":
            return None
        key = ty.replace('"', "")
        if key not in self._struct_size:
            key = re.sub(r"\.\d+$", "", key)
        return key if key in self._struct_size else None

    @staticmethod
    def _alloca_is_ptr(ins) -> bool:
        """True if ``ins`` is ``alloca ptr`` (or ``alloca T*``) -- a slot that
        holds a single pointer value. Such a slot is type-punned by the lifter:
        ``bitcast %slot`` reaches both the pointer itself and ``*slot``."""
        m = re.search(r"alloca\s+(?:inalloca\s+)?([^,\n]+)", str(ins).strip())
        ty = m.group(1).strip() if m else ""
        return ty == "ptr" or ty.endswith("*")

    def _alloca_decl_types(self, fn) -> dict:
        """``alloca-name -> declared slot type string`` parsed from the ORIGINAL IR
        text for ``fn``.

        ``llvm.parse_assembly`` normalizes to OPAQUE pointers, so ``str(ins)`` for
        a pointer alloca is the type-erased ``alloca ptr`` -- the pointee struct
        (``%"hash_entry"*``) survives only in the source text. The cursor-typing
        pass needs the pointee, so it reads the declared types straight from
        ``self._ir_text`` (``%"bucket" = alloca %"hash_entry"*``)."""
        body = re.search(
            r'(?ms)^define[^\n]*@"?' + re.escape(fn.name) + r'"?\(.*?\n\}',
            self._ir_text)
        if body is None:
            return {}
        out: dict = {}
        for m in re.finditer(
                r'%"?([\w.$]+)"?\s*=\s*alloca\s+(?:inalloca\s+)?([^,\n]+)',
                body.group(0)):
            out[m.group(1)] = m.group(2).strip()
        return out

    def _pointee_struct_of_type(self, ty: str):
        """The known-struct name of a single-level pointer type string ``T*`` whose
        pointee ``T`` has a computed layout (present in ``self._struct_size``), else
        None. Only ``T*`` qualifies -- ``T**`` is a pointer-to-pointer, not a
        struct cursor; a scalar/opaque pointee (``i8*``, ``ptr``) has no struct."""
        ty = ty.strip()
        if not ty.endswith("*") or ty.endswith("**"):
            return None
        key = ty[:-1].strip()
        if not key.startswith("%"):
            return None
        key = key.replace('"', "")
        if key not in self._struct_size:
            key = re.sub(r"\.\d+$", "", key)
        return key if key in self._struct_size else None

    def _cursor_struct_ptrs(self, fn, names, gepd, escaped, decl_types) -> set:
        """Names of CLEAN (load/store-only -> would otherwise kreg) single-level
        struct-pointer allocas that are genuine CURSORS deserving a typed frame
        slot, not a propagated temporary.

        After the m_ldx pointer-slot-define lift fix, a through-pointer pointer
        member load (``bucket = table->bucket`` at off 0, ``cursor = cursor->next``
        at +8) lowers to a pointer-typed ``store T* v, T** %slot`` -- correct
        SEMANTICS (no spurious deref-write), but the slot is now accessed cleanly
        so ``_scan_allocas`` kregs it and the decompiler renders an UNTYPED
        ``void**`` walk (``i = *(void***)a0; *i; i += 2``) instead of native's typed
        ``for(bucket=table->bucket;;++bucket){if(bucket->data)..}``. Re-anchoring
        such a slot to a real frame slot (like native) lets ``_save_struct_ptr_lvar_
        types`` type it ``T*`` and recover the field-access render.

        A CURSOR must satisfy ALL of:
          * a resolvable single-level struct-pointer pointee (``%hash_entry`` etc.);
          * written at least once from a DERIVED pointer -- a ``load`` from memory
            or a ``getelementptr`` result (the cursor advance / member read). This is
            the pointer-slot-DEFINE the lift fix now emits;
          * its LOADED value is USED beyond a bare ``ret`` -- it is dereferenced,
            advanced, compared, or passed. A walked cursor reads back; a value that
            is only defined-then-returned does not.

        EXCLUSIONS (would otherwise regress self-consistency):
          * the return slot ``funcresult`` -- stored from a ``load`` (``return *p``)
            and loaded only to feed ``ret``; typing it as a frame cursor breaks the
            return-slot promotion (clone_quoting_options, quoting_options_from_style).
            The loaded-beyond-ret test already rejects it; the name guard is cheap
            belt-and-braces.
          * a pure PARAM-COPY slot (its only store value is an incoming argument,
            ``store %".1", %table``): typing it pins a propagatable parameter into a
            stack local and adds ``table = a0`` noise. Its store value is an argument
            (not a load/gep), so it is not in ``derived_into``."""
        clean = (names - gepd - escaped) - {"funcresult"}
        if not clean:
            return set()
        # An operand ValueRef does NOT expose its defining instruction's opcode
        # (llvmlite reports is_instruction=False for an operand), so map each
        # named instruction's result -> opcode and look operands up there.
        defop: dict = {}
        for bb in fn.blocks:
            for ins in bb.instructions:
                if ins.name:
                    defop[ins.name] = ins.opcode
        # name of each load's pointer-source alloca: ``%v = load <ty>, ptr %slot``.
        load_src: dict = {}
        for bb in fn.blocks:
            for ins in bb.instructions:
                if ins.opcode == "load" and ins.name:
                    sops = list(ins.operands)
                    if sops:
                        load_src[ins.name] = sops[0].name
        derived_into: set = set()
        used_loaded: set = set()  # cursor whose loaded value is used beyond `ret`
        for bb in fn.blocks:
            for ins in bb.instructions:
                if ins.opcode == "store":
                    sops = list(ins.operands)
                    val, dst = sops[0], sops[1]
                    if (dst.name in clean
                            and defop.get(val.name) in ("load", "getelementptr")):
                        derived_into.add(dst.name)
                if ins.opcode == "ret":
                    continue
                # any non-ret instruction consuming a load-of-cursor result marks
                # that cursor as genuinely walked (deref / advance / compare / arg).
                for o in ins.operands:
                    base = load_src.get(o.name)
                    if base in clean:
                        used_loaded.add(base)
        out = set()
        for nm in clean:
            if nm not in derived_into or nm not in used_loaded:
                continue
            if self._pointee_struct_of_type(decl_types.get(nm, "")) is not None:
                out.add(nm)
        return out

    def _struct_ptr_alloca_slots(self, fn) -> dict:
        """``name -> (stkoff, struct_name)`` for every ESCAPING single-level
        struct-pointer alloca (and every clean struct-pointer CURSOR -- see
        ``_cursor_struct_ptrs``), using the SAME synthetic frame-offset packing as
        ``_scan_allocas`` so the offsets match the ``make_stkvar(mba, off)`` the
        drop emits -- i.e. the offset by which ``_save_struct_ptr_lvar_types``
        locates the decompiled cursor stkvar.

        The offset bookkeeping here is a faithful replica of ``_scan_allocas``'s
        ``off`` accounting for gepd/escaped/cursor allocas; it needs no ``mba``
        because struct-pointer slots are never re-anchored to a host offset (only
        escaping aggregate structs are) and scalar kreg allocas do not consume
        ``off``. Mirrors the classification exactly to avoid drift."""
        names = {ins.name for bb in fn.blocks for ins in bb.instructions
                 if ins.opcode == "alloca"}
        if not names:
            return {}
        decl_types = self._alloca_decl_types(fn)
        gepd: set = set()
        escaped: set = set()
        for bb in fn.blocks:
            for ins in bb.instructions:
                for idx, o in enumerate(ins.operands):
                    if o.name not in names:
                        continue
                    if ins.opcode == "load" and idx == 0:
                        continue
                    if ins.opcode == "store" and idx == 1:
                        continue
                    if ins.opcode == "getelementptr" and idx == 0:
                        gepd.add(o.name)
                    else:
                        escaped.add(o.name)
        cursors = self._cursor_struct_ptrs(fn, names, gepd, escaped, decl_types)
        out: dict = {}
        off = 0
        for bb in fn.blocks:
            for ins in bb.instructions:
                if ins.opcode != "alloca":
                    continue
                nm = ins.name
                sz = self._alloca_elem_size(ins)
                if nm in gepd:
                    dims = self._array_dims(str(ins))
                    if dims is None:
                        # _scan_allocas will raise on this; nothing to type here.
                        return out
                    off += max(dims[0] * dims[2], 8)
                elif nm in escaped or nm in cursors:
                    # Opaque-pointer normalization erases the pointee from
                    # ``str(ins)``; resolve the struct from the source-text decl.
                    struct = self._pointee_struct_of_type(decl_types.get(nm, ""))
                    if struct is not None:
                        out[nm] = (off, struct)
                    off += max(sz, 8)
                # scalar kreg allocas do not consume a frame offset
        return out

    @staticmethod
    def _struct_ptr_tif(struct: str):
        """A ``T*`` ``tinfo_t`` for the IR struct key ``struct`` (e.g.
        ``%hash_entry``), resolved as the named IDB type ``hash_entry`` (present
        from DWARF on a non-stripped target), or None if it cannot be resolved."""
        type_name = struct.lstrip("%")
        base = ida_typeinf.tinfo_t()
        if not base.get_named_type(None, type_name):
            return None
        ptr = ida_typeinf.tinfo_t()
        ptr.create_ptr(base)
        return ptr if ptr.is_ptr() else None

    def _save_struct_ptr_lvar_types(self, host_ea: int, fn, cf) -> bool:
        """Type each escaping struct-pointer cursor slot as ``T*`` so the decompiler
        renders the faithful cursor walk (``for(bucket=table->bucket;;++bucket){...
        if (bucket->data) break;}``) instead of an untyped ``*(_QWORD*)`` blob walk.

        A pre-decompile ``save_user_lvar_settings`` keyed by a synthetic
        ``vdloc_t().set_stkoff(N)`` does NOT match the real lvar's richer location
        encoding, so it is silently ignored. The mechanism that sticks (and beats
        the decompiler's own ``Hash_table*``/param-propagated inference) is, GIVEN a
        first ``cf``: locate the cursor stkvar by the frame offset the drop assigned
        it, then persist a user type at that lvar's ACTUAL location via
        ``modify_user_lvar_info(MLI_TYPE)``. The caller re-decompiles so the
        structural/type analysis re-runs (a mere ``refresh_func_ctext`` keeps the
        old shape).

        Returns True if any type was applied (caller must re-decompile). The
        pointee struct must resolve as a named IDB type; otherwise the slot is left
        as-is -- identical to prior behaviour, a strict zero-regression
        enhancement."""
        if cf is None:
            return False
        slots = self._struct_ptr_alloca_slots(fn)
        if not slots:
            return False
        # offset -> struct-tif (only resolvable structs)
        want: dict = {}
        for _nm, (stkoff, struct) in slots.items():
            tif = self._struct_ptr_tif(struct)
            if tif is not None:
                want[stkoff] = tif
        if not want:
            return False
        applied = False
        for v in cf.get_lvars():
            if not v.is_stk_var():
                continue
            tif = want.get(v.get_stkoff())
            if tif is None:
                continue
            if v.type() is not None and v.type().dstr() == tif.dstr():
                continue  # already this type (idempotent re-drop)
            info = hx.lvar_saved_info_t()
            info.ll.location = v.location
            info.ll.defea = v.defea
            info.type = tif
            info.size = 8
            if hx.modify_user_lvar_info(host_ea, hx.MLI_TYPE, info):
                applied = True
        return applied

    def _scan_ptr_deref_aliases(self, fn) -> None:
        """SSA names that are a no-op ``bitcast`` chain rooted DIRECTLY at a
        pointer-typed addr-taken alloca (``self._ptr_allocas``), with no ``load``
        or ``getelementptr`` in between.

        The lifter emits ``*X`` (deref) as ``bitcast %X to ptr; load/store <ty>``
        -- the bitcast reinterprets the SLOT but the intent is the pointee. A
        DIRECT slot access uses ``load/store ... ptr %X`` (no bitcast) or
        ``load ptr, bitcast %X`` (the full pointer value). Recording the
        direct-bitcast aliases lets the load/store emit apply the deref-vs-slot
        rule by valtype (see ``_emit_value``); a deref of a pointer LOADED from
        the slot (``load ptr, %X; bitcast; gep``) already works via the generic
        ldx/stx path and is intentionally NOT in this set."""
        self._ptr_deref_alias = set()
        if not self._ptr_allocas:
            return
        changed = True
        while changed:
            changed = False
            for bb in fn.blocks:
                for ins in bb.instructions:
                    if ins.opcode != "bitcast" or ins.name in self._ptr_deref_alias:
                        continue
                    src = list(ins.operands)[0].name
                    if src in self._ptr_allocas or src in self._ptr_deref_alias:
                        self._ptr_deref_alias.add(ins.name)
                        changed = True

    def _alloca_elem_size(self, ins) -> int:
        m = re.search(r"alloca\s+(?:inalloca\s+)?([^,\n]+)", str(ins).strip())
        ty = m.group(1).strip() if m else "i64"
        if "*" in ty or ty == "ptr":
            return 8
        # A ``%struct`` alloca: prefer the already-parsed layout size; _type_size
        # has no struct case and returns 4 (the opaque-name fallback), which would
        # under-size an escaping struct slot and collide with the next alloca.
        key = self._alloca_struct_key(ins)
        if key is not None:
            return self._struct_size[key][0]
        return _type_size(ty)

    def _ptr_deref_off(self, operand, vmap) -> int:
        """Frame offset of the pointer-alloca slot a ``_ptr_deref_alias`` operand
        is rooted at. The alias is a no-op ``bitcast`` chain over the alloca, so
        its resolved stkaddr offset IS the slot offset."""
        d = vmap.get(operand.name)
        if d is not None and d[0] == "stkaddr":
            return d[1]
        s = self._addr_taken.get(operand.name)
        if s is not None:
            return s[0]
        raise ValueError(f"ptr-deref alias {operand.name!r} did not resolve")

    def _stkvar_slot(self, operand, vmap):
        """(stkoff, size) if ``operand`` addresses a frame slot -- an
        address-taken alloca or a GEP-of-alloca field result (stkaddr) -- else
        None."""
        s = self._addr_taken.get(operand.name)
        if s is not None:
            return s
        d = vmap.get(operand.name)
        if d is not None and d[0] == "stkaddr":
            return (d[1], 8)
        return None

    def _array_dims(self, type_str):
        """(count, elem_str, elem_size) for an ``[N x T]`` array. T may be a
        scalar/ptr or a ``%struct`` with a computable layout; else None (e.g. a
        va_list element)."""
        m = re.search(r"\[\s*(\d+)\s+x\s+([^\]]+?)\s*\]", type_str)
        if not m:
            return None
        n, elem = int(m.group(1)), m.group(2).strip()
        if elem == "ptr" or elem in ("i8", "i16", "i32", "i64"):
            return (n, elem, _type_size(elem))
        key = elem.replace('"', "")
        if key not in self._struct_size:
            # llvmlite disambiguates the SAME struct as %foo.NN in some renderings
            # (e.g. the alloca shows [2 x %timespec.15], the GEP/def use %timespec).
            key = re.sub(r"\.\d+$", "", key)
        if key in self._struct_size:
            return (n, elem, self._struct_size[key][0])
        return None

    def _gep_field_offset(self, ins, ops, vmap) -> int:
        """Constant byte offset of ``getelementptr [N x T], ptr %alloca, I0, I1``
        into a frame-slot alloca: ``I0*sizeof([N x T]) + I1*sizeof(T)``."""
        dims = self._array_dims(str(ins))
        if dims is None:
            raise NotImplementedError("GEP-on-stack: struct/va_list element")
        n, _elem, esz = dims
        strides = (n * esz, esz)  # [i0 over whole array, i1 over element]
        total = 0
        for k, o in enumerate(ops[1:]):
            d = self._desc(o, vmap, 8)
            if d[0] != "num":
                raise NotImplementedError("GEP-on-stack: non-constant index")
            total += d[1] * (strides[k] if k < len(strides) else esz)
        return total

    def _scan_allocas(self, mba, fn) -> dict:
        """Classify each alloca; return the SCALAR-SLOT kreg map and populate
        ``self._addr_taken`` (name -> (stkoff, size)) for address-taken ones.

        - scalar slot (used only as a direct load/store pointer) -> kreg;
          Hex-Rays propagates it away.
        - address-taken (its address escapes -- call arg, returned, stored as a
          value) but NOT GEP'd -> an existing host frame slot; &local renders as
          mop_a(stkvar) and the stack-passing call carries the resting-frame ea
          (the SP fix, memory idavator_sp_gate_call_ea_cracked).
        - GEP'd (struct field offsets) -> NotImplementedError (task #3: needs
          struct layout)."""
        names = {ins.name for bb in fn.blocks for ins in bb.instructions
                 if ins.opcode == "alloca"}
        self._addr_taken = {}
        if not names:
            return {}
        gepd: set = set()
        escaped: set = set()
        for bb in fn.blocks:
            for ins in bb.instructions:
                for idx, o in enumerate(ins.operands):
                    if o.name not in names:
                        continue
                    if ins.opcode == "load" and idx == 0:
                        continue
                    if ins.opcode == "store" and idx == 1:
                        continue
                    if ins.opcode == "getelementptr" and idx == 0:
                        gepd.add(o.name)
                    else:
                        escaped.add(o.name)
        # A clean (load/store-only) single-level struct-pointer CURSOR (e.g.
        # ``bucket``/``cursor`` after the m_ldx pointer-slot-define lift fix) would
        # otherwise be a kreg and render as an untyped ``void**`` walk. Re-anchor it
        # to a real frame slot (the escaped path below) so ``_save_struct_ptr_lvar_
        # types`` can type it ``T*`` -- recovering native's typed ``bucket->data`` /
        # ``++bucket``. _cursor_struct_ptrs excludes pure param-copy slots. The set
        # MUST match _struct_ptr_alloca_slots' so the typed stkoff lines up.
        decl_types = self._alloca_decl_types(fn)
        cursors = self._cursor_struct_ptrs(fn, names, gepd, escaped, decl_types)
        escaped |= cursors
        # Host-frame member offsets, keyed by the source name the lifter preserved.
        # An escaping STRUCT alloca must rest at its REAL host offset: the synthetic
        # sequential packing below can land a struct's base on top of a DIFFERENT,
        # independently-materialised host scalar (e.g. ``%storage`` -> synthetic +16
        # == host ``new_size`` in hash_rehash). Hex-Rays then reads/writes the wrong
        # slot -> garbage decompile + the post-noreturn-merge INTERR 50342. Native
        # uses the true frame offsets; matching them by name de-collides the slot.
        host_off = self._host_frame_offsets(mba)
        self._ptr_allocas = {}
        allocas = {}
        # A frame-slot alloca that the HOST FRAME ALSO NAMES must rest at its TRUE
        # host offset, not at a synthetic sequential offset. The old packing (0, 8,
        # 16, ...) ignored the host layout, so a host-named escaped slot routinely
        # got an offset belonging to a DIFFERENT host variable and Hex-Rays then
        # ALIASED the two distinct values onto one stkvar:
        #   create_hole   -- ``punch_holes`` (escaped via ``bitcast i1*``) packed at
        #                    synthetic +8 == host ``size`` (+8): the body read
        #                    ``size = a2`` and gated on ``if (!size)`` (native gates
        #                    on ``punch_holes``);
        #   quotearg_buffer-- ``p`` packed at +8 == host ``o`` (+8);
        #   sparse_copy   -- ``total_n_read`` (+8) / ``last_write_made_hole`` (+16)
        #                    SWAPPED, so ``*total_n_read += n_read`` corrupted
        #                    ``last_write_made_hole``.
        # ANONYMOUS allocas (no anchorable host member -- a deref temp, an SROA
        # leftover, or a host bool the frame byte-packs sub-qword) keep the original
        # LOW synthetic packing from 0. Overlaying a host LOCAL slot is BENIGN
        # (Hex-Rays treats an unnamed scratch slot sharing bytes with a host local as
        # scratch, exactly as the old ``off=0`` packing did) and crucially stays in
        # the LOCAL frame -- it must NOT be pushed UP into the ``__saved_registers`` /
        # ``__return_address`` / outgoing-args region (that corrupts the frame:
        # create_hard_link -> "local variable allocation has failed"; a synthetic slot
        # landing on ``__saved_registers`` renders as ``savedregs``). The ONE thing
        # the low packing must avoid is sharing a slot with a host-NAMED alloca
        # RE-ANCHORED below, so it skips those placed ranges.
        #
        # Re-anchor ONLY to an 8-ALIGNED host offset. The destructive collisions are
        # 8-byte slots at 8-aligned offsets (``punch_holes``@24 vs ``size``@8,
        # ``p``@8 vs ``o``@8, ``total_n_read``@8 / ``last_write_made_hole``@16). A
        # host frame's SUB-8 byte-packed bools (create_hard_link ``dereference``@4,
        # last_component ``saw_slash``@23) are NOT a destructive alias -- the original
        # synthetic packing handled them fine -- and re-anchoring an escaped slot to a
        # mid-qword offset (then sizing it to 8 bytes) OVERLAPS the neighbouring packed
        # member and itself trips the allocation failure. (The earlier struct-only
        # re-anchor under-covered the named 8-aligned collision; scalar/ptr escaped
        # slots need the SAME host-offset de-collision.)
        def _anchorable(name: str) -> bool:
            return name in host_off and host_off[name][0] % 8 == 0
        placed = [
            (host_off[ins.name][0],
             host_off[ins.name][0] + max(self._alloca_elem_size(ins), 8))
            for bb in fn.blocks for ins in bb.instructions
            if ins.opcode == "alloca" and _anchorable(ins.name)
            and (ins.name in gepd or ins.name in escaped)
        ]

        def _synthetic(start: int, size: int) -> int:
            want = max(size, 8)
            o, moved = start, True
            while moved:
                moved = False
                for lo, hi in placed:
                    if o < hi and lo < o + want:
                        o, moved = (hi + 7) & ~7, True
            return o
        off = 0
        for bb in fn.blocks:
            for ins in bb.instructions:
                if ins.opcode != "alloca":
                    continue
                nm = ins.name
                sz = self._alloca_elem_size(ins)
                named = _anchorable(nm)
                if nm in gepd:
                    # GEP'd alloca -> a frame slot sized to the WHOLE array; each GEP
                    # resolves to &stkvar(slot + field). A host-named array rests at
                    # its real offset; an anonymous one keeps a low synthetic slot.
                    dims = self._array_dims(str(ins))
                    if dims is None:
                        raise NotImplementedError(
                            f"GEP-on-stack alloca %{nm} (struct/va_list element "
                            f"needs real layout -- not the scalar-array slice)")
                    arr_sz = dims[0] * dims[2]
                    if named:
                        slot_off = host_off[nm][0]
                    else:
                        slot_off = _synthetic(off, arr_sz)
                        off = slot_off + max(arr_sz, 8)
                    self._addr_taken[nm] = (slot_off, arr_sz)
                elif nm in escaped:
                    # An escaped slot (address used as a value / via a deref bitcast).
                    # Host-named & 8-aligned -> its TRUE host offset (the de-collision
                    # above); else a low synthetic slot skipping the re-anchored ones.
                    # NO frame extension into host territory (the subframe INTERR chain).
                    if named:
                        slot_off = host_off[nm][0]
                    else:
                        slot_off = _synthetic(off, sz)
                        off = slot_off + max(sz, 8)
                    self._addr_taken[nm] = (slot_off, sz)
                    if self._alloca_is_ptr(ins):
                        # A pointer-typed slot: the lifter type-puns it via a no-op
                        # ``bitcast %slot`` for BOTH the pointer value (slot access)
                        # and a deref ``*slot`` (sub-pointer load/store). Record it
                        # so the load/store emit can tell the two apart (the deref
                        # alias set + valtype rule, mirroring native's
                        # ``mov %slot, r; ldx ds, r``).
                        self._ptr_allocas[nm] = slot_off
                else:
                    allocas[nm] = (mba.alloc_kreg(sz), sz)
        return allocas

    def _detect_ret_slot(self, fn) -> None:
        """Promote the lifter's RETURN SLOT to the return register (noreturn fns).

        The lift emits ``%funcresult = alloca T``: every returning path does
        ``store v, funcresult`` and the terminal block ``%r = load funcresult;
        ret %r``. Materialising that slot as an intermediate var (a kreg/stkvar
        READ at a post-merge block) is fine normally, but when a NORETURN call
        prunes an edge into that merge, Hex-Rays' value-numbering INTERRs (50342).
        The real compiler writes the retval straight to rax on each path. So,
        GATED on a noreturn call being present, route the slot's load/store to the
        return reg (the load aliases rax; each store writes rax) -- matching native
        and eliminating the post-merge read block.

        Originally GATED on a noreturn call (the only case that INTERRs 50342). But
        the redundant funcresult routing also makes NON-noreturn + void fns drop
        silent garbage -- an uninit ``return v1`` (the slot kreg is never written),
        or a duplicate post-merge read. Writing the return reg straight on each path
        is what native does universally, so the promotion now fires for ANY fn whose
        return matches the funcresult-slot SHAPE (a single ret-value name loaded from
        a non-escaping alloca). The shape gate below (single ret name + load-of-slot
        + ``_ret_slot_uses_ok``) is the real guard; noreturn is no longer required.
        See memory idavator_drop_noreturn_50342_rootcause.
        """
        ret_names = set()
        for bb in fn.blocks:
            term = list(bb.instructions)[-1]
            if term.opcode == "ret":
                ops = list(term.operands)
                if ops:
                    ret_names.add(ops[0].name)
        if len(ret_names) != 1:
            return
        rv = next(iter(ret_names))
        src = next((list(ins.operands)[0].name
                    for bb in fn.blocks for ins in bb.instructions
                    if ins.opcode == "load" and ins.name == rv), None)
        if src is None or not self._ret_slot_uses_ok(fn, src):
            return
        if src in self._addr_taken:
            self._ret_off = self._addr_taken[src][0]
        elif src in self._allocas:
            self._ret_kreg = self._allocas[src][0]

    def _detect_ret_phi(self, fn) -> None:
        """Promote a RETURN PHI to the return register (the SROA-of-the-slot form).

        SROA collapses the ``%funcresult`` alloca/load/store return slot into a
        ``phi`` at the ret block: ``%fr.0 = phi [%v,@A],[null,@B]; ret %fr.0``.
        Destructuring that phi the normal way (a kreg READ at the post-merge ret
        block) re-fires the SAME value-numbering INTERR 50342 that
        ``_detect_ret_slot`` clears for the alloca form -- because a noreturn call
        still prunes an edge into the merge. The fix is identical in spirit: write
        the return reg straight on each incoming edge instead of materialising a
        merge var. We record the phi NAME; ``_build_multiblock`` then uses eax as
        that phi's "kreg" (PASS A.5 writes each incoming to eax on its edge, PASS
        B's ret skips the now-redundant mov). Originally GATED on a noreturn call;
        generalized alongside ``_detect_ret_slot`` to fire for ANY fn whose ret is a
        single phi result -- writing the return reg per-edge is what native does and
        avoids the materialised post-merge read regardless of noreturn."""
        ret_names = set()
        for bb in fn.blocks:
            term = list(bb.instructions)[-1]
            if term.opcode == "ret":
                ops = list(term.operands)
                if ops:
                    ret_names.add(ops[0].name)
        if len(ret_names) != 1:
            return
        rv = next(iter(ret_names))
        if any(ins.opcode == "phi" and ins.name == rv
               for bb in fn.blocks for ins in bb.instructions):
            self._ret_phi = rv

    def _ret_slot_uses_ok(self, fn, slot_name) -> bool:
        """Promotable only if the slot's address never escapes: every use of it
        (and its bitcast aliases) is a load/store pointer or a bitcast feeding one.
        A call-arg / value use means the address is needed -> don't promote."""
        alias = {slot_name}
        changed = True
        while changed:
            changed = False
            for bb in fn.blocks:
                for ins in bb.instructions:
                    if (ins.opcode == "bitcast"
                            and list(ins.operands)[0].name in alias
                            and ins.name not in alias):
                        alias.add(ins.name)
                        changed = True
        for bb in fn.blocks:
            for ins in bb.instructions:
                for idx, o in enumerate(ins.operands):
                    if o.name not in alias:
                        continue
                    if ins.opcode == "load" and idx == 0:
                        continue
                    if ins.opcode == "store" and idx == 1:
                        continue
                    if ins.opcode == "bitcast" and idx == 0:
                        continue
                    return False
        return True

    def _is_ret_slot(self, operand, vmap) -> bool:
        """True if ``operand`` addresses the promoted return slot -- a direct
        alloca, a bitcast alias resolving to its frame offset, or its scalar
        kreg. Cheap no-op when nothing was promoted."""
        if self._ret_off is not None:
            stk = self._stkvar_slot(operand, vmap)
            if stk is not None and stk[0] == self._ret_off:
                return True
        if self._ret_kreg is not None:
            slot = self._allocas.get(operand.name)
            if slot is not None and slot[0] == self._ret_kreg:
                return True
        return False

    def _synthesize_ret_block(self, mba):
        """A noreturn host has no m_ret block to use as the return sink. Copy a
        valid code block (for a real start/end -- INTERR 50869 otherwise), clear
        it, give it an operandless m_ret (the value flows via rax), and place it
        just before the special STOP block."""
        src = next((b for i in range(mba.qty)
                    if (b := mba.get_mblock(i)) is not None and int(b.type) == 0
                    and b.start != ida_idaapi.BADADDR and b.start < b.end), None)
        if src is None:
            raise RuntimeError("noreturn host: no valid block to copy for m_ret")
        retb = mba.copy_block(src, mba.qty - 1)  # insert before the STOP block
        self._clear(retb)
        for s in [int(x) for x in retb.succset]:
            retb.succset._del(s)
        for p in [int(x) for x in retb.predset]:
            retb.predset._del(p)
        rt = hx.minsn_t(mba.entry_ea)
        rt.opcode = hx.m_ret
        retb.insert_into_block(rt, None)
        retb.mark_lists_dirty()
        return retb

    @staticmethod
    def _host_frame_offsets(mba) -> dict:
        """``{member_name: (frame_offset, size)}`` for the host function's frame
        (the lifter preserves source names, so a ``%storage`` alloca matches the
        host ``storage`` frame member). Empty on any failure / no frame."""
        out: dict = {}
        pfn = ida_funcs.get_func(mba.entry_ea)
        if pfn is None:
            return out
        tif = ida_typeinf.tinfo_t()
        if not ida_frame.get_func_frame(tif, pfn):
            return out
        udt = ida_typeinf.udt_type_data_t()
        if not tif.get_udt_details(udt):
            return out
        for i in range(udt.size()):
            m = udt[i]
            out[m.name] = (m.offset // 8, m.size // 8)
        return out

    @classmethod
    def _incoming_stack_offsets(cls, mba, fn, nregs) -> dict:
        """``{arg_index: host_frame_offset}`` for each incoming param that the
        SysV ABI passed on the CALLER's stack (index >= ``nregs``).

        ``_force_prototype`` ran before the frame was rebuilt, declaring every
        param as the ``a{i}`` member; ``set_type``/the frame builder classified
        params 0..5 into registers and spilled 6+ to the incoming-args region.
        We resolve each stack arg to its real host frame offset so a stkvar mop
        reads it. PRIMARY: match the ``a{i}`` frame member by name. FALLBACK:
        translate the prototype argloc's ``stkoff`` against the frame's
        incoming-args base (first member above ``__return_address``). The
        inverse of ``_emit_call_stackargs`` (the caller-side reg/stack split)."""
        out: dict = {}
        nargs = len(list(fn.arguments))
        if nargs <= nregs:
            return out
        host_off = cls._host_frame_offsets(mba)
        for i in range(nregs, nargs):
            slot = host_off.get(f"a{i}")
            if slot is not None:
                out[i] = slot[0]
        if len(out) == nargs - nregs:
            return out
        # Fallback: derive from the applied prototype's stack arglocs. The argloc
        # stkoff is relative to the incoming-args region; add the frame offset of
        # that region (right after the return address) to get the member offset.
        pfn = ida_funcs.get_func(mba.entry_ea)
        if pfn is None:
            return out
        tif = ida_typeinf.tinfo_t()
        if not ida_nalt.get_tinfo(tif, pfn.start_ea):
            return out
        ftd = ida_typeinf.func_type_data_t()
        if not tif.get_func_details(ftd) or ftd.size() < nargs:
            return out
        ret_member = host_off.get("__return_address")
        args_base = (ret_member[0] + ret_member[1]) if ret_member else None
        for i in range(nregs, nargs):
            if i in out:
                continue
            al = ftd[i].argloc
            if al.atype() == ida_typeinf.ALOC_STACK and args_base is not None:
                out[i] = args_base + al.stkoff()
        return out

    @staticmethod
    def _resting_frame_ea(mba) -> int:
        """The host ea with the deepest (most negative) get_spd -- where the
        frame is fully allocated. A call that materializes a frame address
        (&local) carries this ea so Hex-Rays computes a frame-consistent
        call_spd (else WARN_BAD_CALL_SP). Falls back to entry_ea."""
        pfn = ida_funcs.get_func(mba.entry_ea)
        if pfn is None:
            return int(mba.entry_ea)
        best_ea, best_spd = int(mba.entry_ea), 0
        ea = pfn.start_ea
        while ea < pfn.end_ea and ea != ida_idaapi.BADADDR:
            spd = ida_frame.get_spd(pfn, ea)
            if spd < best_spd:
                best_spd, best_ea = spd, int(ea)
            ea = ida_bytes.next_head(ea, pfn.end_ea)
        return best_ea

    # -- build dispatch --------------------------------------------------
    def _build(self, mba, fn) -> None:
        self._cur_mba = mba
        self._call_spd_ea = self._resting_frame_ea(mba)
        self._canary_kreg = None
        self._ret_off = None
        self._ret_kreg = None
        self._ret_phi = None
        argregs, eax, ds = self._abi()
        retb = next((mba.get_mblock(i) for i in range(mba.qty)
                     if (b := mba.get_mblock(i)) is not None and b.tail is not None
                     and int(b.tail.opcode) == hx.m_ret), None)
        if retb is None:
            retb = self._synthesize_ret_block(mba)

        # icmp defs (name -> (pred, operands)): a `select` consuming an icmp result
        # as a value materialises it via setcc (icmp itself is otherwise folded into
        # branches and emits nothing). _build_multiblock reuses this for the br fold.
        self._icmp_defs = {}
        for bb in fn.blocks:
            for ins in bb.instructions:
                if ins.opcode == "icmp":
                    pred = re.search(r"icmp\s+(\w+)\s", str(ins).strip())
                    self._icmp_defs[ins.name] = (
                        pred.group(1) if pred else "ne", list(ins.operands))

        self._allocas = self._scan_allocas(mba, fn)
        self._scan_ptr_deref_aliases(fn)
        self._detect_ret_slot(fn)
        self._detect_ret_phi(fn)
        vmap: dict[str, tuple] = {}
        recv_stkoffs = self._incoming_stack_offsets(mba, fn, len(argregs))
        for i, a in enumerate(fn.arguments):
            if i < len(argregs):
                vmap[a.name] = ("reg", argregs[i], _type_size(a.type))
                continue
            # Incoming param 7+ (SysV): the caller spilled it to its stack; after
            # the standard prologue it rests in the host frame's incoming-args
            # region. _force_prototype declared it as the `a{i}` frame member, so
            # read its VALUE straight from that slot (a stkvar mop) -- the inverse
            # of _emit_call_stackargs' caller-side reg/stack split.
            off = recv_stkoffs.get(i)
            if off is None:
                raise NotImplementedError(
                    f"incoming stack arg #{i}: no host frame slot "
                    "(set_type/frame layout did not materialise it)")
            vmap[a.name] = ("stkvar", off, _type_size(a.type))

        # A call must terminate its block, so a function with any call needs the
        # multi-block (segment-splitting) path even if it has one LLVM block.
        has_call = any(ins.opcode == "call"
                       for bb in fn.blocks for ins in bb.instructions)
        if len(list(fn.blocks)) == 1 and not has_call:
            self._build_singleblock(mba, fn, retb, eax, ds, vmap)
        else:
            self._build_multiblock(mba, fn, retb, eax, ds, vmap)
        mba.mark_chains_dirty()

    def _build_singleblock(self, mba, fn, retb, eax, ds, vmap) -> None:
        ea = mba.entry_ea
        # full-body replace: drop every non-terminator instruction.
        for i in range(mba.qty):
            b = mba.get_mblock(i)
            if b is None:
                continue
            ins = b.head
            while ins is not None:
                nxt = ins.next
                if int(ins.opcode) not in _TERM:
                    b.remove_from_block(ins)
                ins = nxt
            b.mark_lists_dirty()

        anchor = retb.tail.prev
        for ins in list(fn.blocks)[0].instructions:
            if ins.opcode == "ret":
                anchor = self._emit_ret_value(retb, anchor, ea, ins, eax, vmap)
            else:
                anchor = self._emit_value(mba, retb, anchor, ea, ins, vmap, ds)
        retb.mark_lists_dirty()

    @staticmethod
    def _phi_incomings(ins):
        """[(value_operand, pred_block_name)] for a phi. The incoming labels are
        text-only (not in .operands), so zip the value operands (textual order)
        with the labels parsed from str(ins)."""
        ops = list(ins.operands)
        preds = re.findall(r"\[\s*[^,\[\]]+,\s*%([\w.]+)\s*\]", str(ins))
        return list(zip(ops, preds))

    @staticmethod
    def _switch_arms(ins):
        """(value_operand, default_name, [(case_int, target_name), ...]) for an
        LLVM ``switch``. llvmlite exposes the operands as
        ``[cond, default_label, caseval0, label0, caseval1, label1, ...]`` -- the
        case integers are unnamed constants (parse the literal from str), and the
        labels are text-only (parse the block names from str, textual order)."""
        ops = list(ins.operands)
        cond = ops[0]
        # Block names in textual order: the default first, then each case target.
        names = re.findall(r'label %"?([\w.@]+)"?', str(ins))
        default_name = names[0]
        target_names = names[1:]
        case_vals = []
        for o in ops[2::2]:
            m = re.search(r"(-?\d+)\s*$", str(o).strip())
            case_vals.append(int(m.group(1)) if m else 0)
        return cond, default_name, list(zip(case_vals, target_names))

    @staticmethod
    def _is_canary_call(ins) -> bool:
        """A stack-protector call -- __readfsqword/__readgsqword (canary read) or
        __stack_chk_fail (the fail path). These are ELIDED (not split into their
        own block), matching the optimizer, which drops the canary from faithful
        output. See memory idavator_drop_canary_gate."""
        if ins.opcode != "call":
            return False
        return list(ins.operands)[-1].name in (
            "__readfsqword", "__readgsqword", "__stack_chk_fail")

    @staticmethod
    def _value_used(ins) -> bool:
        """True if the SSA result of ``ins`` is referenced as an operand by any
        instruction in its function (llvmlite ValueRef exposes no use list, so we
        scan). A void/unnamed result is never used."""
        name = getattr(ins, "name", "")
        if not name:
            return False
        for blk in ins.function.blocks:
            for other in blk.instructions:
                for op in other.operands:
                    if op.name == name:
                        return True
        return False

    @staticmethod
    def _callee_is_noreturn(ins) -> bool:
        """True if the call's DIRECT callee is __noreturn (xalloc_die/abort/...).
        Such a call ends its block with NO successor (BLT_0WAY) and everything
        after it is dead. Indirect/unresolved -> False."""
        callee_ea = ida_name.get_name_ea(ida_idaapi.BADADDR,
                                         list(ins.operands)[-1].name)
        if callee_ea == ida_idaapi.BADADDR:
            return False
        f = ida_funcs.get_func(callee_ea)
        return bool(f is not None and (f.flags & ida_funcs.FUNC_NORET))

    @staticmethod
    def _segment_block(bb):
        """Split an LLVM block's instruction stream into SEGMENTS at calls. A
        call must end its microcode block (BLT_1WAY, falls through), so each call
        closes a segment; the next segment captures that call's result at its
        start. Returns [seg,...]; only the LAST seg carries the terminator. A
        noreturn call closes the FINAL segment (BLT_0WAY) -- the rest is dead."""
        insns = list(bb.instructions)
        term = insns[-1]
        segs, cur = [], {"values": [], "call": None, "term": None,
                         "prev_call": None}
        # NB: llvmlite returns a FRESH ValueRef wrapper per iteration, so an
        # identity (`is`) check against `term` never matches -- stop by INDEX.
        for idx, ins in enumerate(insns):
            if idx == len(insns) - 1:
                break
            if ins.opcode == "phi":
                continue
            if ins.opcode == "call" and not LLVMDropConverter._is_canary_call(ins):
                cur["call"] = ins
                if LLVMDropConverter._callee_is_noreturn(ins):
                    cur["noreturn"] = True
                    segs.append(cur)
                    return segs  # noreturn -> no continuation; drop the dead tail
                segs.append(cur)
                cur = {"values": [], "call": None, "term": None,
                       "prev_call": ins}
            else:
                # canary calls stay in-segment -> _emit_value elides them.
                cur["values"].append(ins)
        cur["term"] = term
        segs.append(cur)
        return segs

    def _build_multiblock(self, mba, fn, retb, eax, ds, vmap) -> None:
        """One microcode block per SEGMENT (an LLVM block split at calls).
        A call-segment is a BLT_1WAY tail that falls through to its continuation;
        the terminal segment carries the LLVM terminator. Conditional ``br``
        lowers to a 2-way jump with a FALSE-arm trampoline at serial+1 (+ a lazy
        TRUE-arm trampoline when a phi needs that edge); phi is destructed out of
        SSA by copying each incoming value into the phi's kreg on its edge block.
        Hex-Rays rebuilds the CFG + lvars from these terminators."""
        ea = mba.entry_ea
        argregs, _eax, _ds = self._abi()
        llvm_blocks = list(fn.blocks)

        # icmp defs (for a `br %c` to fold its compare into the jump) were scanned
        # in _build (shared with the select->setcc materialisation).
        icmp_map = self._icmp_defs

        # Pre-scan phi nodes -> per-block list + the set of edges needing a copy.
        phis: dict[str, list] = {}
        edges_need_copy: set[tuple] = set()
        for bb in llvm_blocks:
            bps = [(ins.name, ins, self._phi_incomings(ins))
                   for ins in bb.instructions if ins.opcode == "phi"]
            if bps:
                phis[bb.name] = bps
                for _, _, incs in bps:
                    for _val, pred_name in incs:
                        edges_need_copy.add((pred_name, bb.name))

        # Plan the segment layout. name_serial[bb] = its FIRST segment; the
        # terminal segment of a conditional br gets a FALSE trampoline (always)
        # and a TRUE trampoline (only if a phi drops a copy on that edge).
        plan, serial = [], 1
        name_serial: dict[str, int] = {}
        term_entry: dict[str, dict] = {}
        for bb in llvm_blocks:
            segs = self._segment_block(bb)
            for si, seg in enumerate(segs):
                e = {**seg, "bb": bb, "code": serial,
                     "ftramp": None, "ttramp": None,
                     "cmps": None, "dtramp": None}
                if si == 0:
                    name_serial[bb.name] = serial
                serial += 1
                if seg["term"] is not None:
                    term_entry[bb.name] = e
                    tops = list(seg["term"].operands)
                    if seg["term"].opcode == "br" and len(tops) == 3:
                        e["ftramp"] = serial
                        serial += 1
                        if (bb.name, tops[2].name) in edges_need_copy:
                            e["ttramp"] = serial
                            serial += 1
                    elif seg["term"].opcode == "switch":
                        # Lower to a chain of equality tests: the switch block is
                        # the FIRST compare (case 0); reserve one extra comparison
                        # block per remaining case. Each compare is a 2-way whose
                        # FALSE arm falls through (serial+1) to the next compare;
                        # the last compare's FALSE arm falls through to a DEFAULT
                        # trampoline (a 1-way `goto default`), mirroring the br
                        # ftramp -- a 2-way false arm must be the next serial.
                        _, _, arms = self._switch_arms(seg["term"])
                        e["cmps"] = [serial + k
                                     for k in range(max(0, len(arms) - 1))]
                        serial += len(e["cmps"])
                        e["dtramp"] = serial
                        serial += 1
                plan.append(e)
        needed = serial - 1

        # Mint enough code blocks before retb (copy_block inherits a valid
        # start/end; INTERR 50869 otherwise), then clear serials 1..needed.
        avail = retb.serial - 1
        for _ in range(max(0, needed - avail)):
            mba.copy_block(retb, retb.serial)
        for s in range(1, needed + 1):
            self._clear(mba.get_mblock(s))
        # The found host m_ret block keeps its ORIGINAL body, but the drop re-emits
        # every computation in the minted LLVM blocks and only needs retb as the
        # bare return sink. A side-effecting leftover (e.g. a leaf fn whose whole
        # body -- ``*__errno_location()=0x5F; return -1`` -- lives in the ret block)
        # is NOT dead-code-eliminated and survives as a DUPLICATE store + stale rax
        # write. Strip retb to its m_ret tail (singleblock does its own full wipe;
        # only the multiblock sink retb is missed by the 1..needed clear above).
        while retb.head is not None and retb.head is not retb.tail:
            retb.remove_from_block(retb.head)
        retb.mark_lists_dirty()
        for s in range(needed + 1, retb.serial):  # leftover host blocks -> dead
            lb = mba.get_mblock(s)
            self._clear(lb)
            g = hx.minsn_t(ea)
            g.opcode = hx.m_goto
            g.l.make_blkref(retb.serial)
            lb.insert_into_block(g, None)
            lb.type = hx.BLT_1WAY
            self._wire(lb, [retb.serial])

        # phi result kregs must exist before pass A (the phi's own block reads
        # them); register them in vmap up front. The RETURN PHI (its result feeds
        # `ret`, downstream of a noreturn) is promoted: its "kreg" is the return
        # reg (eax), so PASS A.5 writes each incoming straight to eax and PASS B's
        # ret elides the redundant mov -- matching native, no post-merge read
        # block (clears INTERR 50342). See _detect_ret_phi.
        phi_kreg: dict[str, tuple] = {}
        for bps in phis.values():
            for pname, pins, _ in bps:
                sz = _type_size(pins.type)
                if pname == self._ret_phi:
                    phi_kreg[pname] = (eax, sz)
                    vmap[pname] = ("reg", eax, sz)
                else:
                    kreg = mba.alloc_kreg(sz)
                    phi_kreg[pname] = (kreg, sz)
                    vmap[pname] = ("reg", kreg, sz)

        # PASS A: per segment, capture the previous call's result, emit value
        # instructions, then (for a call-segment) the call tail + fall-through.
        for e in plan:
            blk = mba.get_mblock(e["code"])
            anchor = None
            if (e["prev_call"] is not None
                    and str(e["prev_call"].type) != "void"
                    and self._value_used(e["prev_call"])):
                # Skip capturing a result no instruction consumes: the kreg copy
                # would be dead, and for a result-discarded call (e.g. a variadic
                # printf modeled with d.size 0) it would read an undefined rax.
                anchor = self._capture_call_result(
                    mba, blk, anchor, ea, e["prev_call"], eax, vmap)
            for ins in e["values"]:
                anchor = self._emit_value(mba, blk, anchor, ea, ins, vmap, ds)
            if e["call"] is not None:
                self._emit_call(mba, blk, anchor, ea, e["call"], vmap, argregs)
                if e.get("noreturn"):
                    blk.type = hx.BLT_0WAY  # noreturn tail: control never leaves
                else:
                    blk.type = hx.BLT_1WAY  # call tail -> continuation (serial+1)
                    self._wire(blk, [e["code"] + 1])

        # PASS A.5: out-of-SSA phi copies onto each incoming edge block (append
        # after the values; pass B appends the terminator after them, so the copy
        # always precedes the branch on that edge).
        def edge_serial(pred_name, target_name):
            pe = term_entry[pred_name]
            t = pe["term"]
            tops = list(t.operands)
            if t.opcode == "br" and len(tops) == 1:
                return pe["code"]
            if t.opcode == "br":
                if tops[1].name == target_name:
                    return pe["ftramp"]
                if tops[2].name == target_name:
                    return pe["ttramp"]
            raise ValueError(f"no edge {pred_name}->{target_name}")

        for bname, bps in phis.items():
            for pname, _pins, incs in bps:
                kreg, sz = phi_kreg[pname]
                for val_op, pred_name in incs:
                    eb = mba.get_mblock(edge_serial(pred_name, bname))
                    mi = hx.minsn_t(ea)
                    mi.opcode = hx.m_mov
                    self._fill(mi.l, self._desc(val_op, vmap, sz))
                    mi.d.make_reg(kreg, sz)
                    eb.insert_into_block(mi, eb.tail)

        # PASS B: terminators on TERMINAL segments only (call-segments were
        # already wired BLT_1WAY in pass A).
        for e in plan:
            if e["term"] is None:
                continue
            blk = mba.get_mblock(e["code"])
            term, ops = e["term"], list(e["term"].operands)
            anchor = blk.tail
            if term.opcode == "ret":
                anchor = self._emit_ret_value(blk, anchor, ea, term, eax, vmap)
                g = hx.minsn_t(ea)
                g.opcode = hx.m_goto
                g.l.make_blkref(retb.serial)
                blk.insert_into_block(g, anchor)
                blk.type = hx.BLT_1WAY
                self._wire(blk, [retb.serial])
            elif term.opcode == "br" and len(ops) == 1:
                tgt = name_serial[ops[0].name]
                g = hx.minsn_t(ea)
                g.opcode = hx.m_goto
                g.l.make_blkref(tgt)
                blk.insert_into_block(g, anchor)
                blk.type = hx.BLT_1WAY
                self._wire(blk, [tgt])
            elif term.opcode == "br":
                # br %cond, %false, %true  (llvmlite operand order!)
                cond = ops[0]
                false_s = name_serial[ops[1].name]
                true_s = name_serial[ops[2].name]
                mi = hx.minsn_t(ea)
                if cond.name in icmp_map:
                    pred, iops = icmp_map[cond.name]
                    sz = _type_size(iops[0].type)
                    mi.opcode = _ICMP_JMP.get(pred, hx.m_jnz)
                    self._fill(mi.l, self._desc(iops[0], vmap, sz))
                    self._fill(mi.r, self._desc(iops[1], vmap, sz))
                else:
                    sz = _type_size(cond.type)
                    mi.opcode = hx.m_jnz  # jump to TRUE when cond != 0
                    self._fill(mi.l, self._desc(cond, vmap, sz))
                    mi.r.make_number(0, sz)
                # taken arm = true trampoline (if any) else the true block.
                taken = e["ttramp"] if e["ttramp"] else true_s
                mi.d.make_blkref(taken)
                blk.insert_into_block(mi, anchor)
                blk.type = hx.BLT_2WAY
                self._wire(blk, [e["ftramp"], taken])  # [fall-through, taken]
                ft = mba.get_mblock(e["ftramp"])
                gf = hx.minsn_t(ea)
                gf.opcode = hx.m_goto
                gf.l.make_blkref(false_s)
                ft.insert_into_block(gf, ft.tail)
                ft.type = hx.BLT_1WAY
                self._wire(ft, [false_s])
                if e["ttramp"]:
                    tt = mba.get_mblock(e["ttramp"])
                    gt = hx.minsn_t(ea)
                    gt.opcode = hx.m_goto
                    gt.l.make_blkref(true_s)
                    tt.insert_into_block(gt, tt.tail)
                    tt.type = hx.BLT_1WAY
                    self._wire(tt, [true_s])
            elif term.opcode == "switch":
                # switch %v, default [ vi -> Bi ]  ->  a chain of equality tests.
                # Each comparison block does `jz %v, vi -> Bi` (2-way) and falls
                # through (serial+1) to the next compare; the last compare falls
                # through to a DEFAULT trampoline (1-way goto default). Hex-Rays
                # re-folds the chain back into a switch/if-ladder.
                cond, default_name, arms = self._switch_arms(term)
                default_s = name_serial[default_name]
                csz = _type_size(cond.type)
                cdesc = self._desc(cond, vmap, csz)
                # A switch-edge phi copy needs a per-case trampoline (not built);
                # defer rather than silently drop the out-of-SSA copy.
                for _v, tgt in arms:
                    if (e["bb"].name, tgt) in edges_need_copy:
                        raise NotImplementedError(
                            "switch edge needs a phi copy (per-case trampoline)")
                if not arms:
                    # degenerate switch (default only) -> unconditional goto.
                    g = hx.minsn_t(ea)
                    g.opcode = hx.m_goto
                    g.l.make_blkref(default_s)
                    blk.insert_into_block(g, anchor)
                    blk.type = hx.BLT_1WAY
                    self._wire(blk, [default_s])
                    continue
                cmp_blocks = [blk] + [mba.get_mblock(s) for s in e["cmps"]]
                for k, (cval, tgt) in enumerate(arms):
                    cblk = cmp_blocks[k]
                    target_s = name_serial[tgt]
                    # FALSE/fall-through arm: the next compare, or the default
                    # trampoline after the last compare (must be serial+1).
                    fall_s = (cmp_blocks[k + 1].serial
                              if k + 1 < len(cmp_blocks) else e["dtramp"])
                    mi = hx.minsn_t(ea)
                    mi.opcode = hx.m_jz
                    self._fill(mi.l, (cdesc[0], cdesc[1], csz))
                    mi.r.make_number(cval & ((1 << (8 * csz)) - 1), csz)
                    mi.d.make_blkref(target_s)
                    cblk.insert_into_block(mi, cblk.tail)
                    cblk.type = hx.BLT_2WAY
                    self._wire(cblk, [fall_s, target_s])  # [fall-through, taken]
                dt = mba.get_mblock(e["dtramp"])
                gd = hx.minsn_t(ea)
                gd.opcode = hx.m_goto
                gd.l.make_blkref(default_s)
                dt.insert_into_block(gd, dt.tail)
                dt.type = hx.BLT_1WAY
                self._wire(dt, [default_s])
            elif term.opcode == "unreachable":
                # After a noreturn call or the elided canary fail path: a dead
                # goto to the ret block (the block is unreachable once the canary
                # compare folds, so Hex-Rays prunes it).
                g = hx.minsn_t(ea)
                g.opcode = hx.m_goto
                g.l.make_blkref(retb.serial)
                blk.insert_into_block(g, anchor)
                blk.type = hx.BLT_1WAY
                self._wire(blk, [retb.serial])
            else:
                raise NotImplementedError(
                    f"unhandled terminator {term.opcode!r}")
        retb.mark_lists_dirty()


def drop_llvm_function(ir_text: str, host_ea: int, llvm_fn_name: str):
    """Convenience: drop ``@llvm_fn_name`` from ``ir_text`` into ``host_ea``."""
    return LLVMDropConverter(ir_text).drop(host_ea, llvm_fn_name)
