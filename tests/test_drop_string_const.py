"""Private string-constant VALUE operand resolution.

A private string-constant global (``@x = private constant [N x i8] c"..."``)
passed as a VALUE (e.g. an error message handed to a call, decayed via
``getelementptr``) must resolve to the address of the matching IDB string
literal. The LLVM symbol is truncated (``aInvalidKindInG``) while IDA auto-names
the literal from its longer content (``aInvalidKindInGenTempname``), so a plain
``get_name_ea`` on the LLVM name misses -- the converter matches the decoded
``c"..."`` body against the IDB string table by exact content instead.

Without that fix, ``_desc`` raised ``ValueError: unhandled operand
'@x = private constant ...'`` and the whole drop failed (cf is None).
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


# A private string constant decayed (getelementptr) and passed as the pointer
# VALUE arg of a real callee. The c"..." content ("valid_options (options)")
# exists verbatim in the cp IDB string table (auto-named aValidOptionsOptions),
# so it must resolve by content -- the LLVM name @msg never appears in the IDB.
PROBE = """
@msg = private constant [24 x i8] c"valid_options (options)\\00"
declare i64 @strlen(ptr)
define i64 @probe(i64 %n) {
entry:
  %p = getelementptr [24 x i8], ptr @msg, i32 0, i32 0
  %r = call i64 @strlen(ptr %p)
  ret i64 %r
}
"""

# The exact string literal content the probe must reference in the dropped C.
_STRING = "valid_options (options)"


@pytest.mark.ida
class TestDropStringConst:
    def test_private_string_constant_value_resolves(
        self, examples_dir: Path
    ) -> None:
        if not _idalib():
            pytest.skip("idalib unavailable")
        import idapro
        import ida_funcs
        import ida_hexrays
        import ida_idaapi
        import ida_name
        import idautils

        binary = examples_dir / "cp"
        if not binary.exists():
            pytest.skip("missing example binary: cp")

        from idavator.llvm_drop import LLVMDropConverter

        idapro.open_database(str(binary), True)
        try:
            assert ida_hexrays.init_hexrays_plugin()
            # The IDB literal the probe's c"..." must resolve to (by content);
            # its IDB auto-name is NOT the LLVM symbol @msg.
            lit_ea = ida_name.get_name_ea(ida_idaapi.BADADDR, "aValidOptionsOptions")
            assert lit_ea != ida_idaapi.BADADDR, "expected aValidOptionsOptions literal"
            assert (
                ida_name.get_name_ea(ida_idaapi.BADADDR, "msg") == ida_idaapi.BADADDR
            ), "the LLVM symbol must NOT exist in the IDB (proves content match)"

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
            # The string constant must be REFERENCED (its content rendered), and
            # the operand must NOT collapse to a placeholder (byte_/nullptr/
            # const-memory write) where the real string belongs.
            assert _STRING in txt, f"string not referenced:\n{txt}"
            assert "byte_" not in txt, f"placeholder operand:\n{txt}"
            assert "const memory" not in txt, f"const-memory write:\n{txt}"
            assert "nullptr" not in txt, f"nullptr operand:\n{txt}"
        finally:
            idapro.close_database()

    def test_strconst_ea_matches_by_content(self, examples_dir: Path) -> None:
        # Directly: the resolver maps the LLVM private-constant operand text to
        # the IDB literal address by decoded content (LLVM name lookup misses).
        if not _idalib():
            pytest.skip("idalib unavailable")
        import idapro
        import ida_idaapi
        import ida_name

        binary = examples_dir / "cp"
        if not binary.exists():
            pytest.skip("missing example binary: cp")

        from idavator.llvm_drop import LLVMDropConverter

        idapro.open_database(str(binary), True)
        try:
            lit_ea = ida_name.get_name_ea(ida_idaapi.BADADDR, "aValidOptionsOptions")
            assert lit_ea != ida_idaapi.BADADDR
            conv = LLVMDropConverter(PROBE)
            operand = '@msg = private constant [24 x i8] c"valid_options (options)\\00"'
            assert conv._strconst_ea(operand) == lit_ea
            # A constant whose content is not in the IDB resolves to None.
            absent = '@z = private constant [6 x i8] c"zzzzz\\00"'
            assert conv._strconst_ea(absent) is None
        finally:
            idapro.close_database()
