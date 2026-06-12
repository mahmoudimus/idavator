"""Drive the consolidated LLVMDropConverter on straight-line LLVM programs."""
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


@pytest.mark.ida
class TestLLVMDropModule:
    @pytest.mark.parametrize(
        "ir, fn, must_contain",
        [
            ("define i32 @f(i32 %x) {\n  %a = mul i32 %x, 3\n"
             "  %b = add i32 %a, 7\n  ret i32 %b\n}\n",
             "f", ["3", "7"]),
            # Hex-Rays folds `x << 2` into `4 * x` (semantically identical).
            ("define i32 @g(i32 %x) {\n  %a = shl i32 %x, 2\n"
             "  %b = or i32 %a, 1\n  ret i32 %b\n}\n",
             "g", ["4 * a0", "| 1"]),
            ("define i32 @h(i32 %x, i32 %y) {\n  %a = xor i32 %x, %y\n"
             "  ret i32 %a\n}\n",
             "h", ["^"]),
        ],
    )
    def test_straight_line_drops(self, examples_dir: Path, ir, fn, must_contain):
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
            assert cf is not None, "decompile failed"
            for needle in must_contain:
                assert needle in text, f"missing {needle!r} in:\n{text}"
        finally:
            idapro.close_database()
