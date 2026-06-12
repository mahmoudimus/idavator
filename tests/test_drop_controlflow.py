"""Control-flow drop foundation: build NEW blocks + wire goto/jcnd, orphan-sweep.

Step 1 (this file): prove block insertion + edge wiring works -- insert one block,
redirect the entry to it, let Hex-Rays sweep the orphaned host blocks. Then the
2-way if/else. Builds with block-ref gotos (l.make_blkref) + manual succset/predset
wiring + mba.entry_ea EAs (the  patterns), gated by try_verify.

Run:  PYTHONPATH=src pytest -m ida tests/test_drop_controlflow.py -s
"""
from __future__ import annotations

from pathlib import Path

import pytest


def _idalib() -> bool:
    try:
        import idapro  # noqa: F401

        return True
    except ImportError:
        return False


def _rewire(blk, new_succs):
    """Replace blk's successors with new_succs, fixing predsets ( pattern)."""
    mba = blk.mba
    for old in [int(s) for s in blk.succset]:
        blk.succset._del(old)
        ob = mba.get_mblock(old)
        if ob is not None:
            with __import__("contextlib").suppress(Exception):
                ob.predset._del(blk.serial)
            ob.mark_lists_dirty()
    for ns in new_succs:
        blk.succset.push_back(ns)
        nb = mba.get_mblock(ns)
        if nb is not None and blk.serial not in [int(p) for p in nb.predset]:
            nb.predset.push_back(blk.serial)
            nb.mark_lists_dirty()
    blk.mark_lists_dirty()


@pytest.mark.ida
class TestControlFlowFoundation:
    def test_insert_block_and_redirect(self, examples_dir: Path) -> None:
        if not _idalib():
            pytest.skip("idalib unavailable")
        import idapro
        import ida_funcs
        import ida_hexrays as hx
        import ida_idp
        import ida_typeinf
        import idautils

        from idavator.cfg_verify import try_verify

        if not (examples_dir / "cp").exists():
            pytest.skip("missing example binary")

        idapro.open_database(str(examples_dir / "cp"), True)
        try:
            assert hx.init_hexrays_plugin()
            EAX = hx.reg2mreg(ida_idp.str2reg("rax"))

            # Linear host with an m_ret at LOCOPT (blocks are WIRED there:
            # block-ref jumps + succset/predset + types -- the shape we build).
            host = None
            for ea in idautils.Functions():
                f = ida_funcs.get_func(ea)
                if f is None or not (8 <= f.end_ea - f.start_ea <= 200):
                    continue
                if hx.decompile(ea) is None:
                    continue
                hf = hx.hexrays_failure_t()
                mbr = hx.mba_ranges_t()
                mbr.ranges.push_back(f)
                m = hx.gen_microcode(mbr, hf, None, hx.DECOMP_NO_WAIT, hx.MMAT_PREOPTIMIZED)
                if m is None:
                    continue
                tails = {int(b.tail.opcode) for i in range(m.qty)
                         if (b := m.get_mblock(i)) is not None and b.tail is not None}
                conds = {hx.m_jcnd, hx.m_jz, hx.m_jnz, hx.m_jtbl}
                if hx.m_ret in tails and not (tails & conds):
                    host = ea
                    break
            assert host is not None, "no linear host found"
            tif = ida_typeinf.tinfo_t()
            ida_typeinf.parse_decl(tif, None, "int __fastcall f();", 0)
            ida_typeinf.apply_tinfo(host, tif, ida_typeinf.TINFO_DEFINITE)

            box = {"interr": None, "err": None, "qty0": 0}

            class _Hook(hx.Hexrays_Hooks):
                def preoptimized(self, mba):  # m_ret exists; we wire explicitly
                    try:
                        box["qty0"] = mba.qty
                        TERM = {hx.m_goto, hx.m_ret, hx.m_jcnd, hx.m_jz,
                                hx.m_jnz, hx.m_jtbl}
                        # ret block R (keep it). Block 0 + the last block must stay
                        # EMPTY -- INTERR 51814 if a special block is non-empty.
                        R = None
                        for i in range(mba.qty):
                            b = mba.get_mblock(i)
                            if b is not None and b.tail is not None and int(b.tail.opcode) == hx.m_ret:
                                R = b
                                break
                        b1 = mba.get_mblock(1)
                        if R is None or b1 is None or b1.serial == R.serial:
                            return 0
                        ea = mba.entry_ea
                        # block 1: clear, emit `mov #42, eax ; goto R` (block-ref).
                        ins = b1.head
                        while ins is not None:
                            nxt = ins.next
                            b1.remove_from_block(ins)
                            ins = nxt
                        mov = hx.minsn_t(ea)
                        mov.opcode = hx.m_mov
                        mov.l.make_number(42, 4)
                        mov.d.make_reg(EAX, 4)
                        b1.insert_into_block(mov, None)
                        goto = hx.minsn_t(ea)
                        goto.opcode = hx.m_goto
                        goto.l.make_blkref(R.serial)
                        b1.insert_into_block(goto, b1.tail)
                        b1.type = hx.BLT_1WAY
                        _rewire(b1, [R.serial])
                        # R: drop non-terminators, keep the ret (returns eax).
                        ins = R.head
                        while ins is not None:
                            nxt = ins.next
                            if int(ins.opcode) not in TERM:
                                R.remove_from_block(ins)
                            ins = nxt
                        R.mark_lists_dirty()
                        mba.mark_chains_dirty()
                        ok, code = try_verify(mba, "after rewire")
                        box["interr"] = code
                    except Exception:  # noqa: BLE001
                        import traceback
                        box["err"] = traceback.format_exc()
                    return 0

            hook = _Hook()
            assert hook.hook()
            try:
                hx.mark_cfunc_dirty(host)
                cf = hx.decompile(host)
            finally:
                hook.unhook()
            text = str(cf) if cf is not None else "<None>"
            print(f"\n=== insert+redirect (host={host:#x} qty0={box['qty0']} "
                  f"interr={box['interr']}) ===\n{text}\nerr={box['err']}")
            assert box["err"] is None, box["err"]
            assert text != "<None>", "decompile failed"
            assert "0X2A" in text.upper() or "42" in text, f"expected 42 in:\n{text}"
        finally:
            idapro.close_database()


@pytest.mark.ida
class TestIfElse:
    """2-way if/else: icmp+br -> a decompiled conditional. Reuses block 1 (cond,
    m_jnz) + block 2 (els, already fall-through-adjacent to block 1), inserts a
    THEN block, both arms converge to the ret block."""

    def test_if_else_drops(self, examples_dir: Path) -> None:
        if not _idalib():
            pytest.skip("idalib unavailable")
        import idapro
        import ida_funcs
        import ida_hexrays as hx
        import ida_idp
        import ida_typeinf
        import idautils

        from idavator.cfg_verify import try_verify

        if not (examples_dir / "cp").exists():
            pytest.skip("missing example binary")
        idapro.open_database(str(examples_dir / "cp"), True)
        try:
            assert hx.init_hexrays_plugin()
            EAX = hx.reg2mreg(ida_idp.str2reg("rax"))
            EDI = hx.reg2mreg(ida_idp.str2reg("rdi"))
            TERM = {hx.m_goto, hx.m_ret, hx.m_jcnd, hx.m_jz, hx.m_jnz, hx.m_jtbl}

            host = None
            for ea in idautils.Functions():
                f = ida_funcs.get_func(ea)
                if f is None or not (8 <= f.end_ea - f.start_ea <= 200):
                    continue
                if hx.decompile(ea) is None:
                    continue
                hf = hx.hexrays_failure_t()
                mbr = hx.mba_ranges_t()
                mbr.ranges.push_back(f)
                m = hx.gen_microcode(mbr, hf, None, hx.DECOMP_NO_WAIT, hx.MMAT_PREOPTIMIZED)
                if m is None:
                    continue
                # linear host (no conds -> simple orphans), blocks 1 & 2 free,
                # m_ret block at serial >= 3.
                tails = {int(b.tail.opcode) for i in range(m.qty)
                         if (b := m.get_mblock(i)) is not None and b.tail is not None}
                conds = {hx.m_jcnd, hx.m_jz, hx.m_jnz, hx.m_jtbl}
                rser = next((m.get_mblock(i).serial for i in range(m.qty)
                             if (b := m.get_mblock(i)) is not None and b.tail is not None
                             and int(b.tail.opcode) == hx.m_ret), -1)
                if rser >= 3 and m.qty >= 5 and not (tails & conds):
                    host = ea
                    break
            assert host is not None, "no suitable host"
            tif = ida_typeinf.tinfo_t()
            ida_typeinf.parse_decl(tif, None, "int __fastcall f(int x);", 0)
            ida_typeinf.apply_tinfo(host, tif, ida_typeinf.TINFO_DEFINITE)

            box = {"interr": None, "err": None}

            def clear(blk):
                ins = blk.head
                while ins is not None:
                    nxt = ins.next
                    blk.remove_from_block(ins)
                    ins = nxt

            class _Hook(hx.Hexrays_Hooks):
                def preoptimized(self, mba):
                    try:
                        R = next((mba.get_mblock(i) for i in range(mba.qty)
                                  if (b := mba.get_mblock(i)) is not None and b.tail is not None
                                  and int(b.tail.opcode) == hx.m_ret), None)
                        b1 = mba.get_mblock(1)
                        b2 = mba.get_mblock(2)
                        if R is None or b1 is None or b2 is None or R.serial <= 2:
                            return 0
                        ea = mba.entry_ea
                        r_old = R.serial
                        # copy_block (not insert_block) -> inherits a valid
                        # start/end (INTERR 50869 if start>=end); insert before R.
                        then_blk = mba.copy_block(R, r_old)
                        then_s = r_old
                        r_s = r_old + 1
                        for s in [int(x) for x in then_blk.succset]:
                            then_blk.succset._del(s)
                        for p in [int(x) for x in then_blk.predset]:
                            then_blk.predset._del(p)

                        def emit_mov(blk, val):
                            mi = hx.minsn_t(ea)
                            mi.opcode = hx.m_mov
                            mi.l.make_number(val, 4)
                            mi.d.make_reg(EAX, 4)
                            blk.insert_into_block(mi, None)
                            g = hx.minsn_t(ea)
                            g.opcode = hx.m_goto
                            g.l.make_blkref(r_s)
                            blk.insert_into_block(g, blk.tail)
                            blk.type = hx.BLT_1WAY
                            _rewire(blk, [r_s])

                        # COND (block 1): jnz edi,#0 -> THEN ; fall-through -> ELS(2)
                        clear(b1)
                        jnz = hx.minsn_t(ea)
                        jnz.opcode = hx.m_jnz
                        jnz.l.make_reg(EDI, 4)
                        jnz.r.make_number(0, 4)
                        jnz.d.make_blkref(then_s)
                        b1.insert_into_block(jnz, None)
                        b1.type = hx.BLT_2WAY
                        _rewire(b1, [2, then_s])   # [fall-through, taken]

                        clear(b2)
                        emit_mov(b2, 20)           # ELS: x == 0
                        clear(then_blk)
                        emit_mov(then_blk, 10)     # THEN: x != 0

                        Rblk = mba.get_mblock(r_s)
                        ins = Rblk.head
                        while ins is not None:
                            nxt = ins.next
                            if int(ins.opcode) not in TERM:
                                Rblk.remove_from_block(ins)
                            ins = nxt
                        Rblk.mark_lists_dirty()
                        mba.mark_chains_dirty()
                        ok, code = try_verify(mba, "after if/else build")
                        box["interr"] = code
                    except Exception:  # noqa: BLE001
                        import traceback
                        box["err"] = traceback.format_exc()
                    return 0

            hook = _Hook()
            assert hook.hook()
            try:
                hx.mark_cfunc_dirty(host)
                cf = hx.decompile(host)
            finally:
                hook.unhook()
            text = str(cf) if cf is not None else "<None>"
            print(f"\n=== if/else (host={host:#x} interr={box['interr']}) ===\n"
                  f"{text}\nerr={box['err']}")
            assert box["err"] is None, box["err"]
            assert text != "<None>", "decompile failed"
            up = text.upper()
            assert "0XA" in up or "10" in text, f"then(10) missing:\n{text}"
            assert "0X14" in up or "20" in text, f"els(20) missing:\n{text}"
        finally:
            idapro.close_database()
