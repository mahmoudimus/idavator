"""GEP-on-stack: ``getelementptr [N x T], ptr %alloca, i32 0, i32 IDX`` into a
frame-slot alloca lowers to ``&stkvar(off + IDX*sizeof(T))``; a downstream
load/store/call resolves to that stkvar. Scalar/ptr element arrays only -- a
struct/va_list element still needs real struct layout (deliberately unsupported).
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


# arr[0]=x, arr[1]=y, return arr[0]+arr[1]  (a [2 x i64] stack array, two GEP
# fields at offsets 0 and 8, store + load through each).
PROBE = """
define i64 @probe(i64 %x, i64 %y) {
entry:
  %arr = alloca [2 x i64], align 8
  %p0 = getelementptr [2 x i64], ptr %arr, i32 0, i32 0
  %p1 = getelementptr [2 x i64], ptr %arr, i32 0, i32 1
  store i64 %x, ptr %p0
  store i64 %y, ptr %p1
  %a = load i64, ptr %p0
  %b = load i64, ptr %p1
  %s = add i64 %a, %b
  ret i64 %s
}
"""


@pytest.mark.ida
class TestGepOnStack:
    def test_array_field_access(self, examples_dir: Path) -> None:
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
                         if (f := ida_funcs.get_func(ea)) is not None
                         and int(getattr(f, "frsize", 0)) >= 16
                         and not (f.flags & ida_funcs.FUNC_NORET)
                         and ida_hexrays.decompile(ea) is not None), None)
            assert host is not None, "no host with frsize >= 16"

            conv = LLVMDropConverter(PROBE)
            cf = conv.drop(host, "probe")
            assert conv.last_error is None, conv.last_error
            assert conv.last_interr is None, f"INTERR {conv.last_interr}"
            assert cf is not None, "decompile returned None"
            txt = str(cf)
            assert "bad sp value" not in txt, f"WARN_BAD_CALL_SP:\n{txt}"
            # both stack fields flow into the returned sum.
            assert "return" in txt, txt
        finally:
            idapro.close_database()
