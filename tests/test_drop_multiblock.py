"""Drive LLVMDropConverter on MULTI-BLOCK LLVM (branches) via the module.

The 2-way if/else mechanic was proven by hand in test_drop_controlflow.py; this
exercises the *general* layout engine in src/idavator/llvm_drop.py: one microcode
block per LLVM block, icmp folded into a 2-way jump, the FALSE arm a trampoline at
serial+1, Hex-Rays rebuilding the CFG/lvars from the terminators.

Run:  PYTHONPATH=src pytest -m ida tests/test_drop_multiblock.py -s
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


def _find_linear_host(ida_funcs, hx, idautils):
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
            return ea
    return None


# Each case: (ir, fn, [needles that MUST appear], [needles that must NOT]).
_CASES = [
    # if/else: sgt 5 -> 100 / 200. Hex-Rays may invert the test (<= 5) and swap
    # arms, so assert on both constants + a conditional, not exact arm order.
    ("define i32 @ife(i32 %x) {\n"
     "entry:\n  %c = icmp sgt i32 %x, 5\n"
     "  br i1 %c, label %big, label %small\n"
     "big:\n  ret i32 100\n"
     "small:\n  ret i32 200\n}\n",
     # Hex-Rays renders the return constants in hex (100=0x64, 200=0xC8).
     "ife", ["0x64", "0xC8", "if"], []),
    # unconditional-br chain: entry -> mid -> exit (straight line across blocks).
    ("define i32 @chain(i32 %x) {\n"
     "entry:\n  %a = add i32 %x, 1\n  br label %mid\n"
     "mid:\n  %b = mul i32 %a, 2\n  br label %exit\n"
     "exit:\n  ret i32 %b\n}\n",
     "chain", ["2", "* "], []),
    # if with a computed then-arm: returns x*x when x>0 else 0.
    ("define i32 @sqpos(i32 %x) {\n"
     "entry:\n  %c = icmp sgt i32 %x, 0\n"
     "  br i1 %c, label %pos, label %zero\n"
     "pos:\n  %s = mul i32 %x, %x\n  ret i32 %s\n"
     "zero:\n  ret i32 0\n}\n",
     "sqpos", ["return 0", "*"], []),
]


@pytest.mark.ida
class TestMultiBlockDrop:
    @pytest.mark.parametrize("ir, fn, must, must_not", _CASES)
    def test_branches(self, examples_dir: Path, ir, fn, must, must_not):
        if not _idalib():
            pytest.skip("idalib unavailable")
        import idapro
        import ida_funcs
        import ida_hexrays as hx
        import idautils

        if not (examples_dir / "cp").exists():
            pytest.skip("missing example binary")

        idapro.open_database(str(examples_dir / "cp"), True)
        try:
            assert hx.init_hexrays_plugin()
            from idavator.llvm_drop import LLVMDropConverter

            host = _find_linear_host(ida_funcs, hx, idautils)
            assert host is not None, "no linear host found"

            conv = LLVMDropConverter(ir)
            cf = conv.drop(host, fn)
            text = str(cf) if cf is not None else "<None>"
            print(f"\n=== drop @{fn} (host={host:#x} interr={conv.last_interr} "
                  f"err={'yes' if conv.last_error else None}) ===\n{text}")
            assert conv.last_error is None, conv.last_error
            assert conv.last_interr is None, f"INTERR {conv.last_interr}"
            assert cf is not None, "decompile failed"
            assert "local variable allocation has failed" not in text, text
            for needle in must:
                assert needle in text, f"missing {needle!r} in:\n{text}"
            for needle in must_not:
                assert needle not in text, f"unexpected {needle!r} in:\n{text}"
        finally:
            idapro.close_database()
