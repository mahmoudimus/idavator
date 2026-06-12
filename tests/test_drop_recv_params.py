"""Callee-side reception of >6 incoming params (SysV stack args).

The SysV AMD64 ABI passes integer/pointer params 1-6 in registers and spills
params 7+ to the CALLER's stack; after the standard prologue they rest in the
host frame's incoming-args region. ``_build`` previously hard-capped at the 6
ABI registers (``NotImplementedError: stack-passed argument``), so any function
RECEIVING more than 6 such params failed before any body was built.

This regresses ``force_linkat`` (7 params: the 7th, ``a6``, is the lone stack
arg). The dropped body must read ``a6`` from its real incoming slot -- the
native renders ``v12 = a6; if ( a6 < 0 ) ...`` -- and must NOT fall back to a
bogus fixed address (``byte_<addr>``) for the missing register.

Guard for ``feat(drop): read callee-side >6 incoming params from the caller
stack`` (ticket ida-khup). Proven to FAIL without the fix: stashing
``llvm_drop.py`` to the pre-fix gate makes ``drop`` raise at the 7th argument,
``decompile`` then returns the stale native cfunc, and the
``v12 = a6`` stack-read assertions still pass on native -- so the test instead
pins the exact pre-fix failure: ``conv.last_error`` carries the
``stack-passed argument`` ``NotImplementedError`` (None after the fix).
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


@pytest.mark.ida
class TestRecvParams:
    def test_force_linkat_reads_seventh_stack_param(
            self, examples_dir: Path) -> None:
        if not _idalib():
            pytest.skip("idalib unavailable")
        import idapro
        import ida_hexrays
        import ida_idaapi
        import ida_name

        binary = examples_dir / "cp"
        ir_path = examples_dir / "cp.ll"
        if not (binary.exists() and ir_path.exists()):
            pytest.skip("missing cp / cp.ll")

        from idavator.llvm_drop import LLVMDropConverter

        idapro.open_database(str(binary), True)
        try:
            assert ida_hexrays.init_hexrays_plugin()
            ea = ida_name.get_name_ea(ida_idaapi.BADADDR, "force_linkat")
            if ea == ida_idaapi.BADADDR:
                pytest.skip("force_linkat not in this binary")

            conv = LLVMDropConverter(ir_path.read_text())
            cf = conv.drop(ea, "force_linkat")
            # The pre-fix gate raises NotImplementedError at the 7th arg, which is
            # captured in last_error; the fix clears it.
            assert conv.last_error is None, conv.last_error
            assert cf is not None, "decompile returned None"
            txt = str(cf)
            # The 7th param (the lone SysV stack arg) must be read from its real
            # incoming slot -- native: `v12 = a6;` guarded by `if ( a6 < 0 )`.
            assert "= a6;" in txt, f"7th stack param not read as a6:\n{txt}"
            assert "if ( a6 < 0 )" in txt, f"a6 mis-valued:\n{txt}"
            # No register-arg fallback to a bogus fixed address.
            assert "byte_" not in txt, f"garbage stack-arg address:\n{txt}"
            assert "write access to const memory" not in txt, txt
        finally:
            idapro.close_database()
