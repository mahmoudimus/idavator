"""Rotate intrinsics (__ROR8__/__ROL8__/...) -> Hex-Rays HELPER calls.

The lift emits IDA rotate intrinsics as calls to unresolved `@__ROR8__` etc.
These survive into FAITHFUL pseudocode (e.g. cp's rotr_sz -> `return
__ROR8__(a0, a1)`), so the drop emits them as helper calls via
``create_helper_call`` -- args ride in the mcallinfo. The naive `mop.make_helper`
+ register-arg-mov shape CRASHES the decompiler, so this guards that regression
(and the deliberate exclusion of the canary `__readfsqword`, which the optimizer
elides from faithful output).
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


PROBE = """
define i64 @probe(i64 %x, i64 %n) {{
entry:
  %r = call i64 @{intr}(i64 %x, i64 %n)
  ret i64 %r
}}
declare i64 @{intr}(i64, i64)
"""


@pytest.mark.ida
class TestHelperIntrinsic:
    def test_rotate_renders_as_helper_no_crash(self, examples_dir: Path) -> None:
        if not _idalib():
            pytest.skip("idalib unavailable")
        import idapro
        import ida_funcs
        import ida_hexrays
        import idautils

        binary = examples_dir / "cp"
        if not binary.exists():
            pytest.skip("missing example binary: cp")

        from idavator.llvm_drop import LLVMDropConverter

        idapro.open_database(str(binary), True)
        try:
            assert ida_hexrays.init_hexrays_plugin()
            host = next((ea for ea in idautils.Functions()
                         if ida_funcs.get_func(ea) is not None
                         and ida_hexrays.decompile(ea) is not None), None)
            assert host is not None, "no decompilable host"

            for intr in ("__ROR8__", "__ROL8__"):
                conv = LLVMDropConverter(PROBE.format(intr=intr))
                cf = conv.drop(host, "probe")
                # the make_helper-crash regression: must not crash / INTERR.
                assert conv.last_error is None, f"[{intr}] {conv.last_error}"
                assert conv.last_interr is None, f"[{intr}] INTERR {conv.last_interr}"
                assert cf is not None, f"[{intr}] decompile returned None"
                txt = str(cf)
                assert intr in txt, f"[{intr}] helper not rendered:\n{txt}"
        finally:
            idapro.close_database()
