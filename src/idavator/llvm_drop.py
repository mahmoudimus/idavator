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
import ida_idaapi
import ida_idp
import ida_name
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
        if op == "getelementptr":
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
        if op == "call":
            raise RuntimeError("call must be split into its own block "
                               "(handled by the segment splitter, not _emit_value)")
        logger.warning("unhandled LLVM opcode: %s", op)
        return anchor

    def _emit_call(self, mba, blk, anchor, ea, ins, vmap, argregs):
        """Emit `mov`s into the ABI arg-regs then `m_call l=gvar(callee), d=rax`
        as the block TAIL. The call must terminate its block (it falls through to
        the continuation, BLT_1WAY) -- a call defining rax mid-block is fine for
        50864 but later maturities want calls block-terminal. Returns the call."""
        ops = list(ins.operands)
        callee = ops[-1]
        call_args = ops[:-1]
        callee_ea = ida_name.get_name_ea(ida_idaapi.BADADDR, callee.name)
        if callee_ea == ida_idaapi.BADADDR:
            raise ValueError(f"unresolved callee @{callee.name}")
        _argregs, eax, _ds = self._abi()
        for i, a in enumerate(call_args):
            asz = _type_size(a.type)
            mv = hx.minsn_t(ea)
            mv.opcode = hx.m_mov
            self._fill(mv.l, self._desc(a, vmap, asz))
            mv.d.make_reg(argregs[i], asz)
            blk.insert_into_block(mv, anchor)
            anchor = mv
        mc = hx.minsn_t(ea)
        mc.opcode = hx.m_call
        mc.l.make_gvar(callee_ea)
        rsz = _type_size(ins.type) if str(ins.type) != "void" else 8
        mc.d.make_reg(eax, rsz)  # call defines rax (the continuation captures it)
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
    def _segment_block(bb):
        """Split an LLVM block's instruction stream into SEGMENTS at calls. A
        call must end its microcode block (BLT_1WAY, falls through), so each call
        closes a segment; the next segment captures that call's result at its
        start. Returns [seg,...]; only the LAST seg carries the terminator."""
        term = list(bb.instructions)[-1]
        segs, cur = [], {"values": [], "call": None, "term": None,
                         "prev_call": None}
        for ins in bb.instructions:
            if ins is term:
                break
            if ins.opcode == "phi":
                continue
            if ins.opcode == "call":
                cur["call"] = ins
                segs.append(cur)
                cur = {"values": [], "call": None, "term": None,
                       "prev_call": ins}
            else:
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

        # Pre-scan icmp defs so a `br %c` can fold its compare into the jump.
        icmp_map: dict[str, tuple] = {}
        for bb in llvm_blocks:
            for ins in bb.instructions:
                if ins.opcode == "icmp":
                    pred = re.search(r"icmp\s+(\w+)\s", str(ins).strip())
                    icmp_map[ins.name] = (pred.group(1) if pred else "ne",
                                          list(ins.operands))

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
                     "ftramp": None, "ttramp": None}
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
                plan.append(e)
        needed = serial - 1

        # Mint enough code blocks before retb (copy_block inherits a valid
        # start/end; INTERR 50869 otherwise), then clear serials 1..needed.
        avail = retb.serial - 1
        for _ in range(max(0, needed - avail)):
            mba.copy_block(retb, retb.serial)
        for s in range(1, needed + 1):
            self._clear(mba.get_mblock(s))
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
        # them); register them in vmap up front.
        phi_kreg: dict[str, tuple] = {}
        for bps in phis.values():
            for pname, pins, _ in bps:
                sz = _type_size(pins.type)
                kreg = mba.alloc_kreg(sz)
                phi_kreg[pname] = (kreg, sz)
                vmap[pname] = ("reg", kreg, sz)

        # PASS A: per segment, capture the previous call's result, emit value
        # instructions, then (for a call-segment) the call tail + fall-through.
        for e in plan:
            blk = mba.get_mblock(e["code"])
            anchor = None
            if e["prev_call"] is not None and str(e["prev_call"].type) != "void":
                anchor = self._capture_call_result(
                    mba, blk, anchor, ea, e["prev_call"], eax, vmap)
            for ins in e["values"]:
                anchor = self._emit_value(mba, blk, anchor, ea, ins, vmap, ds)
            if e["call"] is not None:
                self._emit_call(mba, blk, anchor, ea, e["call"], vmap, argregs)
                blk.type = hx.BLT_1WAY      # call tail -> continuation (serial+1)
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
            else:
                raise NotImplementedError(
                    f"unhandled terminator {term.opcode!r}")
        retb.mark_lists_dirty()


def drop_llvm_function(ir_text: str, host_ea: int, llvm_fn_name: str):
    """Convenience: drop ``@llvm_fn_name`` from ``ir_text`` into ``host_ea``."""
    return LLVMDropConverter(ir_text).drop(host_ea, llvm_fn_name)
