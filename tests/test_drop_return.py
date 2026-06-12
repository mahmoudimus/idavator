"""First real LLVM->microcode drop: lower `ret add(i32 %x, 1)` to `return x + 1`.

Model 2 (proven in test_drop_spike.py): hook hxe_microcode, emit the simplest
register-based microcode, let decompile() do wiring/lvars/maturity/ctree. Here we
drive a single `add` from PARSED LLVM IR (read the addend from the module) and
inject `add edi, <addend> -> eax` before the ret (x86-64 SysV: arg0=edi, ret=eax).
Hex-Rays DCEs the host's original return and renders our computation.

Run:  PYTHONPATH=src pytest -m ida tests/test_drop_return.py -s
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


def _idalib() -> bool:
    try:
        import idapro  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.ida
class TestDropReturnXPlus1:
    def test_llvm_add_one_drops_to_return_x_plus_1(self, examples_dir: Path) -> None:
        if not _idalib():
            pytest.skip("idalib unavailable")
        import idapro
        import ida_funcs
        import ida_hexrays
        import ida_idp
        import ida_typeinf
        import idautils
        import llvmlite.binding as llvm

        from idavator.cfg_verify import try_verify

        binary = examples_dir / "cp"
        if not binary.exists():
            pytest.skip("missing example binary")

        idapro.open_database(str(binary), True)
        try:
            assert ida_hexrays.init_hexrays_plugin()

            # Host function: small, decompilable, and with an m_ret block at
            # MMAT_GENERATED (skip stubs like _init whose exit is a tail-jump).
            host = None
            for ea in idautils.Functions():
                f = ida_funcs.get_func(ea)
                if f is None or not (8 <= f.end_ea - f.start_ea <= 160):
                    continue
                if ida_hexrays.decompile(ea) is None:
                    continue
                hf = ida_hexrays.hexrays_failure_t()
                mbr = ida_hexrays.mba_ranges_t()
                mbr.ranges.push_back(f)
                m = ida_hexrays.gen_microcode(
                    mbr, hf, None, ida_hexrays.DECOMP_NO_WAIT, ida_hexrays.MMAT_PREOPTIMIZED)
                if m is None:
                    continue
                has_ret = any(
                    (b := m.get_mblock(i)) is not None and b.tail is not None
                    and int(b.tail.opcode) == ida_hexrays.m_ret
                    for i in range(m.qty)
                )
                if has_ret:
                    host = ea
                    break
            assert host is not None, "no host function with an m_ret block found"

            # Force prototype int __fastcall f(int x) so edi renders as int `x`.
            tif = ida_typeinf.tinfo_t()
            ida_typeinf.parse_decl(tif, None, "int __fastcall f(int x);", 0)
            ida_typeinf.apply_tinfo(host, tif, ida_typeinf.TINFO_DEFINITE)
            ida_hexrays.mark_cfunc_dirty(host)
            print(f"\n=== host {host:#x} baseline (typed) ===\n{ida_hexrays.decompile(host)}")

            # Parse trivial LLVM and DRIVE the addend from it.
            mod = llvm.parse_assembly(
                "define i32 @f(i32 %x) {\nentry:\n"
                "  %r = add i32 %x, 1\n  ret i32 %r\n}\n")
            fn = next(g for g in mod.functions if not g.is_declaration)
            addend = None
            for bb in fn.blocks:
                for ins in bb.instructions:
                    if ins.opcode == "add":
                        for op in ins.operands:
                            m = re.search(r"i\d+\s+(-?\d+)\s*$", str(op).strip())
                            if m:
                                addend = int(m.group(1))
            assert addend is not None, "could not read add constant from LLVM"
            print(f"LLVM-driven addend = {addend}")

            EAX = ida_hexrays.reg2mreg(ida_idp.str2reg("rax"))
            EDI = ida_hexrays.reg2mreg(ida_idp.str2reg("rdi"))
            print(f"mregs: eax={EAX} edi={EDI}")

            box = {"fired": False, "interr": None, "rets": 0}

            class _DropHook(ida_hexrays.Hexrays_Hooks):
                def preoptimized(self, mba):  # hxe_preoptimized (MMAT_PREOPTIMIZED; m_ret exists)
                    if box["fired"]:
                        return 0
                    box["fired"] = True
                    for i in range(mba.qty):
                        blk = mba.get_mblock(i)
                        if blk is None or blk.tail is None:
                            continue
                        if int(blk.tail.opcode) != ida_hexrays.m_ret:
                            continue
                        box["rets"] += 1
                        ins = ida_hexrays.minsn_t(int(blk.tail.ea))
                        ins.opcode = ida_hexrays.m_add
                        ins.l.make_reg(EDI, 4)
                        ins.r.make_number(addend & 0xFFFFFFFF, 4)
                        ins.d.make_reg(EAX, 4)
                        blk.insert_into_block(ins, blk.tail.prev)
                        # INTERR 50873 fix: the block's use/def lists are now stale.
                        blk.mark_lists_dirty()
                    ok, code = try_verify(mba, "after inject")
                    box["interr"] = code
                    return 0

            hook = _DropHook()
            assert hook.hook()
            try:
                ida_hexrays.mark_cfunc_dirty(host)
                cf = ida_hexrays.decompile(host)
            finally:
                hook.unhook()
            text = str(cf) if cf is not None else "<None>"
            print(f"\n=== DROPPED (fired={box['fired']} rets={box['rets']} "
                  f"verify_interr={box['interr']}) ===\n{text}")
            assert cf is not None, "decompile returned None"
            assert box["rets"] >= 1, "no m_ret block found to inject into"
            assert "+ 1" in text, f"expected `+ 1` in the drop:\n{text}"
        finally:
            idapro.close_database()
