"""End-to-end: the public apply_llvm_ir() folds onto LLVMDropConverter.

apply_llvm_ir resolves each defined LLVM function to the IDB function of the same
name and drops it (Model-2 rebuild). This proves the llvm2ida.py public entry now
runs the working converter instead of the old create_empty_mba path.

Run:  PYTHONPATH=src pytest -m ida tests/test_apply_llvm_ir.py -s
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


def _find_named_linear_host(ida_funcs, ida_name, hx, idautils):
    """Return (ea, name) of a linear host with a usable name."""
    for ea in idautils.Functions():
        f = ida_funcs.get_func(ea)
        if f is None or not (8 <= f.end_ea - f.start_ea <= 200):
            continue
        name = ida_funcs.get_func_name(ea)
        if not name or not name.isidentifier():
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
            return ea, name
    return None, None


@pytest.mark.ida
class TestApplyLLVMIR:
    def test_apply_by_name(self, examples_dir: Path) -> None:
        if not _idalib():
            pytest.skip("idalib unavailable")
        import idapro
        import ida_funcs
        import ida_hexrays as hx
        import ida_name
        import idautils

        if not (examples_dir / "cp").exists():
            pytest.skip("missing example binary")

        idapro.open_database(str(examples_dir / "cp"), True)
        try:
            assert hx.init_hexrays_plugin()
            from idavator.llvm2ida import apply_llvm_ir

            host, name = _find_named_linear_host(
                ida_funcs, ida_name, hx, idautils)
            assert host is not None, "no named linear host found"

            # Drop `(x * 3) + 7` into the host resolved BY NAME.
            ir = (f"define i32 @{name}(i32 %x) {{\n"
                  f"  %a = mul i32 %x, 3\n  %b = add i32 %a, 7\n"
                  f"  ret i32 %b\n}}\n")
            assert apply_llvm_ir(ir) is True

            cf = hx.decompile(host)
            text = str(cf) if cf is not None else "<None>"
            print(f"\n=== apply @{name} ({host:#x}) ===\n{text}")
            assert "* 3" in text or "3 *" in text, text
            assert "+ 7" in text, text
        finally:
            idapro.close_database()

    def test_apply_unknown_name_returns_false(self, examples_dir: Path) -> None:
        if not _idalib():
            pytest.skip("idalib unavailable")
        import idapro
        import ida_hexrays as hx

        if not (examples_dir / "cp").exists():
            pytest.skip("missing example binary")

        idapro.open_database(str(examples_dir / "cp"), True)
        try:
            assert hx.init_hexrays_plugin()
            from idavator.llvm2ida import apply_llvm_ir

            ir = ("define i32 @no_such_function_zzz(i32 %x) {\n"
                  "  ret i32 %x\n}\n")
            # No IDB function of that name -> nothing applied -> False.
            assert apply_llvm_ir(ir) is False
        finally:
            idapro.close_database()
