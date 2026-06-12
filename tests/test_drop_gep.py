"""Drive LLVMDropConverter on getelementptr (address arithmetic feeding ld/st).

GEP lowers to `base + index*sizeof(elem)`; the result is an 8-byte address a load
or store consumes. Single-index array form.

Run:  PYTHONPATH=src pytest -m ida tests/test_drop_gep.py -s
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


_CASES = [
    # p[i]: variable index, i32 stride 4.
    ("define i32 @idx(i32* %p, i64 %i) {\n"
     "entry:\n  %q = getelementptr i32, i32* %p, i64 %i\n"
     "  %v = load i32, i32* %q\n  ret i32 %v\n}\n",
     "idx", ["a0[a1]"]),
    # p[2]: constant index -> a0[2] / *(a0 + 2).
    ("define i32 @two(i32* %p) {\n"
     "entry:\n  %q = getelementptr i32, i32* %p, i64 2\n"
     "  %v = load i32, i32* %q\n  ret i32 %v\n}\n",
     "two", ["a0[2]"]),
    # store p[i] = v.
    ("define void @st(i32* %p, i64 %i, i32 %v) {\n"
     "entry:\n  %q = getelementptr i32, i32* %p, i64 %i\n"
     "  store i32 %v, i32* %q\n  ret void\n}\n",
     "st", ["a0[a1] = "]),
]


@pytest.mark.ida
class TestGepDrop:
    @pytest.mark.parametrize("ir, fn, must", _CASES)
    def test_gep(self, examples_dir: Path, ir, fn, must):
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
            assert host is not None
            conv = LLVMDropConverter(ir)
            cf = conv.drop(host, fn)
            text = str(cf) if cf is not None else "<None>"
            print(f"\n=== drop @{fn} (interr={conv.last_interr} "
                  f"err={'yes' if conv.last_error else None}) ===\n{text}")
            assert conv.last_error is None, conv.last_error
            assert conv.last_interr is None, f"INTERR {conv.last_interr}"
            assert cf is not None, "decompile failed"
            assert "allocation has failed" not in text, text
            for needle in must:
                assert needle in text, f"missing {needle!r} in:\n{text}"
        finally:
            idapro.close_database()
