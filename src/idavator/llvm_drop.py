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

import ida_funcs
import ida_hexrays as hx
import ida_idp
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
}
# zext/sext/trunc -> microcode widening/narrowing.
_CAST = {"zext": hx.m_xdu, "sext": hx.m_xds, "trunc": hx.m_low}


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


class LLVMDropConverter:
    """Drop a (straight-line) LLVM function into a host's decompiled output."""

    def __init__(self, ir_text: str):
        self.module = llvm.parse_assembly(ir_text)

    # -- public API ------------------------------------------------------
    def drop(self, host_ea: int, llvm_fn_name: str):
        """Rebuild ``host_ea``'s microcode from the named LLVM function and return
        the resulting ``cfunc_t`` (or None on failure)."""
        fn = next((g for g in self.module.functions
                   if g.name == llvm_fn_name and not g.is_declaration), None)
        if fn is None:
            raise ValueError(f"no definition for @{llvm_fn_name}")
        self._force_prototype(host_ea, fn)

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
        finally:
            hook.unhook()
        if box["err"]:
            logger.error("drop build failed:\n%s", box["err"])
        self.last_interr = box["interr"]
        self.last_error = box["err"]
        return cf

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

    @classmethod
    def _force_prototype(cls, host_ea: int, fn) -> None:
        args = list(fn.arguments)
        params = ", ".join(f"{cls._arg_ctype(a, fn)} a{i}"
                           for i, a in enumerate(args)) or "void"
        tif = ida_typeinf.tinfo_t()
        ida_typeinf.parse_decl(tif, None, f"int __fastcall f({params});", 0)
        ida_typeinf.apply_tinfo(host_ea, tif, ida_typeinf.TINFO_DEFINITE)

    @staticmethod
    def _abi():
        argregs = [hx.reg2mreg(ida_idp.str2reg(r))
                   for r in ("rdi", "rsi", "rdx", "rcx", "r8", "r9")]
        return (argregs, hx.reg2mreg(ida_idp.str2reg("rax")),
                hx.reg2mreg(ida_idp.str2reg("ds")))

    def _build(self, mba, fn) -> None:
        argregs, eax, ds = self._abi()
        # ret block (kept; block 0 + last must stay empty -- INTERR 51814).
        retb = next((mba.get_mblock(i) for i in range(mba.qty)
                     if (b := mba.get_mblock(i)) is not None and b.tail is not None
                     and int(b.tail.opcode) == hx.m_ret), None)
        if retb is None:
            raise RuntimeError("host has no m_ret block at preoptimized")
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

        # value descriptor map: name -> ("reg", mreg, size) | ("num", val, size)
        vmap: dict[str, tuple] = {}
        for i, a in enumerate(fn.arguments):
            vmap[a.name] = ("reg", argregs[i], _type_size(a.type))

        def fill(mop, d):
            kind, val, size = d
            if kind == "reg":
                mop.make_reg(val, size)
            else:
                mop.make_number(val & ((1 << (8 * size)) - 1), size)

        def desc(operand, default_size):
            nm = operand.name
            if nm and nm in vmap:
                return vmap[nm]
            s = str(operand).strip()
            num = re.search(r"(-?\d+)\s*$", s)
            if num:
                return ("num", int(num.group(1)), _type_size(s) or default_size)
            raise ValueError(f"unhandled operand {s!r}")

        blocks = list(fn.blocks)
        if len(blocks) != 1:
            raise NotImplementedError(
                "multi-block control flow not yet wired into the module "
                "(mechanic proven in tests/test_drop_controlflow.py)")

        anchor = retb.tail.prev
        for ins in blocks[0].instructions:
            op = ins.opcode
            ops = list(ins.operands)
            if op in _BINOP:
                size = _type_size(ins.type)
                mi = hx.minsn_t(ea)
                mi.opcode = _BINOP[op]
                fill(mi.l, desc(ops[0], size))
                r_desc = desc(ops[1], size)
                if op in ("shl", "lshr", "ashr"):
                    # shift-amount operand must be size 1 (INTERR 50835).
                    r_desc = (r_desc[0], r_desc[1], 1)
                fill(mi.r, r_desc)
                kreg = mba.alloc_kreg(size)
                mi.d.make_reg(kreg, size)
                retb.insert_into_block(mi, anchor)
                anchor = mi
                vmap[ins.name] = ("reg", kreg, size)
            elif op in _CAST:
                in_sz = _type_size(ops[0].type)
                out_sz = _type_size(ins.type)
                mi = hx.minsn_t(ea)
                mi.opcode = _CAST[op]
                fill(mi.l, desc(ops[0], in_sz))
                kreg = mba.alloc_kreg(out_sz)
                mi.d.make_reg(kreg, out_sz)
                retb.insert_into_block(mi, anchor)
                anchor = mi
                vmap[ins.name] = ("reg", kreg, out_sz)
            elif op == "load":
                # %v = load <ty>, <ty>* %p  ->  ldx ds, p, v
                out_sz = _type_size(ins.type)
                ad = desc(ops[0], 8)
                mi = hx.minsn_t(ea)
                mi.opcode = hx.m_ldx
                mi.l.make_reg(ds, 2)
                fill(mi.r, (ad[0], ad[1], 8))
                kreg = mba.alloc_kreg(out_sz)
                mi.d.make_reg(kreg, out_sz)
                retb.insert_into_block(mi, anchor)
                anchor = mi
                vmap[ins.name] = ("reg", kreg, out_sz)
            elif op == "store":
                # store <ty> %v, <ty>* %p  ->  stx v, ds, p
                val_sz = _type_size(ops[0].type)
                vd = desc(ops[0], val_sz)
                ad = desc(ops[1], 8)
                mi = hx.minsn_t(ea)
                mi.opcode = hx.m_stx
                fill(mi.l, (vd[0], vd[1], val_sz))
                mi.r.make_reg(ds, 2)
                fill(mi.d, (ad[0], ad[1], 8))
                retb.insert_into_block(mi, anchor)
                anchor = mi
            elif op == "ret":
                if ops:
                    mi = hx.minsn_t(ea)
                    mi.opcode = hx.m_mov
                    fill(mi.l, desc(ops[0], 4))
                    mi.d.make_reg(eax, 4)
                    retb.insert_into_block(mi, anchor)
                    anchor = mi
            else:
                logger.warning("unhandled LLVM opcode: %s", op)
        retb.mark_lists_dirty()
        mba.mark_chains_dirty()


def drop_llvm_function(ir_text: str, host_ea: int, llvm_fn_name: str):
    """Convenience: drop ``@llvm_fn_name`` from ``ir_text`` into ``host_ea``."""
    return LLVMDropConverter(ir_text).drop(host_ea, llvm_fn_name)
