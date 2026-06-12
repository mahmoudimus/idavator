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

# icmp predicate -> a 2-way conditional jump that branches to ``d`` when the
# predicate holds (the fall-through, serial+1, takes the FALSE arm).
_ICMP_JMP = {
    "eq": hx.m_jz, "ne": hx.m_jnz,
    "ugt": hx.m_ja, "uge": hx.m_jae, "ult": hx.m_jb, "ule": hx.m_jbe,
    "sgt": hx.m_jg, "sge": hx.m_jge, "slt": hx.m_jl, "sle": hx.m_jle,
}


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

    # -- microcode emit helpers -----------------------------------------
    @staticmethod
    def _fill(mop, d) -> None:
        kind, val, size = d
        if kind == "reg":
            mop.make_reg(val, size)
        else:
            mop.make_number(val & ((1 << (8 * size)) - 1), size)

    @staticmethod
    def _desc(operand, vmap, default_size):
        """Resolve an LLVM operand to a value descriptor (reg kreg / numeric)."""
        nm = operand.name
        if nm and nm in vmap:
            return vmap[nm]
        s = str(operand).strip()
        num = re.search(r"(-?\d+)\s*$", s)
        if num:
            return ("num", int(num.group(1)), _type_size(s) or default_size)
        raise ValueError(f"unhandled operand {s!r}")

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
        blk.mark_lists_dirty()

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
        if op in _CAST:
            in_sz = _type_size(ops[0].type)
            out_sz = _type_size(ins.type)
            mi = hx.minsn_t(ea)
            mi.opcode = _CAST[op]
            self._fill(mi.l, self._desc(ops[0], vmap, in_sz))
            kreg = mba.alloc_kreg(out_sz)
            mi.d.make_reg(kreg, out_sz)
            blk.insert_into_block(mi, anchor)
            vmap[ins.name] = ("reg", kreg, out_sz)
            return mi
        if op == "load":
            # %v = load <ty>, <ty>* %p  ->  ldx ds, p, v
            out_sz = _type_size(ins.type)
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
            # store <ty> %v, <ty>* %p  ->  stx v, ds, p
            val_sz = _type_size(ops[0].type)
            vd = self._desc(ops[0], vmap, val_sz)
            ad = self._desc(ops[1], vmap, 8)
            mi = hx.minsn_t(ea)
            mi.opcode = hx.m_stx
            self._fill(mi.l, (vd[0], vd[1], val_sz))
            mi.r.make_reg(ds, 2)
            self._fill(mi.d, (ad[0], ad[1], 8))
            blk.insert_into_block(mi, anchor)
            return mi
        if op == "icmp":
            # Folded into the branch (see _build_multiblock); no value emitted.
            return anchor
        logger.warning("unhandled LLVM opcode: %s", op)
        return anchor

    def _emit_ret_value(self, blk, anchor, ea, term, eax, vmap):
        """Emit `mov <retval>, eax` for an LLVM ret (no terminator)."""
        ops = list(term.operands)
        if not ops:
            return anchor
        sz = _type_size(ops[0].type)
        mi = hx.minsn_t(ea)
        mi.opcode = hx.m_mov
        self._fill(mi.l, self._desc(ops[0], vmap, sz))
        mi.d.make_reg(eax, sz)
        blk.insert_into_block(mi, anchor)
        return mi

    # -- build dispatch --------------------------------------------------
    def _build(self, mba, fn) -> None:
        argregs, eax, ds = self._abi()
        retb = next((mba.get_mblock(i) for i in range(mba.qty)
                     if (b := mba.get_mblock(i)) is not None and b.tail is not None
                     and int(b.tail.opcode) == hx.m_ret), None)
        if retb is None:
            raise RuntimeError("host has no m_ret block at preoptimized")

        vmap: dict[str, tuple] = {}
        for i, a in enumerate(fn.arguments):
            vmap[a.name] = ("reg", argregs[i], _type_size(a.type))

        if len(list(fn.blocks)) == 1:
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

    def _build_multiblock(self, mba, fn, retb, eax, ds, vmap) -> None:
        """One microcode block per LLVM block. Conditional ``br`` lowers to a
        2-way jump whose FALSE arm is a trampoline placed at serial+1 (the jcc
        fall-through is structurally the next block). Hex-Rays rebuilds the CFG
        + lvars from these terminators after PREOPTIMIZED."""
        ea = mba.entry_ea
        llvm_blocks = list(fn.blocks)

        # Pre-scan icmp defs so a `br %c` can fold its compare into the jump.
        icmp_map: dict[str, tuple] = {}
        for bb in llvm_blocks:
            for ins in bb.instructions:
                if ins.opcode == "icmp":
                    pred = re.search(r"icmp\s+(\w+)\s", str(ins).strip())
                    icmp_map[ins.name] = (pred.group(1) if pred else "ne",
                                          list(ins.operands))

        # Plan the physical serial layout: code block per LLVM block, plus a
        # trampoline immediately after each conditional branch.
        plan, serial = [], 1
        for bb in llvm_blocks:
            term = list(bb.instructions)[-1]
            is_cond = term.opcode == "br" and len(list(term.operands)) == 3
            entry = {"bb": bb, "term": term, "code": serial, "tramp": None}
            serial += 1
            if is_cond:
                entry["tramp"] = serial
                serial += 1
            plan.append(entry)
        needed = serial - 1
        name_serial = {e["bb"].name: e["code"] for e in plan}

        # Mint enough code blocks before retb (copy_block inherits a valid
        # start/end; INTERR 50869 otherwise). Existing host code blocks are
        # serials 1..retb.serial-1; mint the remainder right before retb.
        avail = retb.serial - 1
        for _ in range(max(0, needed - avail)):
            mba.copy_block(retb, retb.serial)
        # Now serials 1..needed are code blocks; retb at needed+1. Clear them
        # all (drops any copied m_ret) before emitting.
        for s in range(1, needed + 1):
            self._clear(mba.get_mblock(s))
        # Any leftover host blocks between our code and retb -> dead goto retb.
        for s in range(needed + 1, retb.serial):
            lb = mba.get_mblock(s)
            self._clear(lb)
            g = hx.minsn_t(ea)
            g.opcode = hx.m_goto
            g.l.make_blkref(retb.serial)
            lb.insert_into_block(g, None)
            lb.type = hx.BLT_1WAY
            self._wire(lb, [retb.serial])

        for e in plan:
            blk = mba.get_mblock(e["code"])
            term, ops = e["term"], list(e["term"].operands)
            anchor = None
            for ins in e["bb"].instructions:
                if ins is term:
                    break
                anchor = self._emit_value(mba, blk, anchor, ea, ins, vmap, ds)

            if term.opcode == "ret":
                anchor = self._emit_ret_value(blk, anchor, ea, term, eax, vmap)
                g = hx.minsn_t(ea)
                g.opcode = hx.m_goto
                g.l.make_blkref(retb.serial)
                blk.insert_into_block(g, anchor)
                blk.type = hx.BLT_1WAY
                self._wire(blk, [retb.serial])
            elif term.opcode == "br" and len(ops) == 1:
                # unconditional br %T
                g = hx.minsn_t(ea)
                g.opcode = hx.m_goto
                g.l.make_blkref(name_serial[ops[0].name])
                blk.insert_into_block(g, anchor)
                blk.type = hx.BLT_1WAY
                self._wire(blk, [name_serial[ops[0].name]])
            elif term.opcode == "br":
                # br %cond, %false, %true  (llvmlite operand order!)
                cond, false_s = ops[0], name_serial[ops[1].name]
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
                mi.d.make_blkref(true_s)
                blk.insert_into_block(mi, anchor)
                blk.type = hx.BLT_2WAY
                # succset = [fall-through (trampoline), taken (true)].
                self._wire(blk, [e["tramp"], true_s])
                tramp = mba.get_mblock(e["tramp"])
                g = hx.minsn_t(ea)
                g.opcode = hx.m_goto
                g.l.make_blkref(false_s)
                tramp.insert_into_block(g, None)
                tramp.type = hx.BLT_1WAY
                self._wire(tramp, [false_s])
            else:
                raise NotImplementedError(
                    f"unhandled terminator {term.opcode!r}")
        retb.mark_lists_dirty()


def drop_llvm_function(ir_text: str, host_ea: int, llvm_fn_name: str):
    """Convenience: drop ``@llvm_fn_name`` from ``ir_text`` into ``host_ea``."""
    return LLVMDropConverter(ir_text).drop(host_ea, llvm_fn_name)
