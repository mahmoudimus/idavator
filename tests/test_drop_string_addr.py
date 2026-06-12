"""Global-as-value resolves to its ADDRESS (``gvaraddr``), not the lvalue read.

A global used as a POINTER VALUE (an array/string constant that decays to a
``ptr``, e.g. a message handed to ``gettext``/``error``/``__assert_fail``) must
materialise as ``&global`` -- the symbol's ADDRESS -- not as the value stored at
that address. Before the ``gvaraddr`` descriptor (``_desc`` resolved a ptr-typed
named global to ``("gvar", ea, n)`` and ``_fill`` did ``make_gvar`` = an lvalue
read), the dropped C rendered the bug signature ``gettext(*(const char **)"write
error")`` instead of ``gettext("write error")``.

This guard drops a real cp function (``close_stdout``) whose dropped C exhibited
that signature and asserts the string literal is now referenced directly, with no
``*(const char **)"..."`` deref of a string literal anywhere in the output.

PROVEN to fail without the fix: with the ``gvaraddr`` branch reverted, the dropped
text contains ``gettext(*(const char **)"write error")`` and this test's
``_STRLIT_DEREF`` assertion fires.
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


# A real cp function: close_stdout calls gettext("write error"). Its IR decays
# the named global @aWriteError ([12 x i8]) via `getelementptr [12 x i8], ptr
# @aWriteError, i32 0, i32 0` and hands the result to gettext as a VALUE -- the
# canonical global-as-pointer-value case the gvaraddr fix targets.
_FN = "close_stdout"
_STRING = "write error"

# The bug signature: a deref of a string LITERAL (NOT *(const char **)&param,
# which is a legitimate varargs-param cast that also appears in native).
_STRLIT_DEREF = re.compile(r'\(const char \*\*\)"')


@pytest.mark.ida
class TestDropStringAddr:
    def test_global_value_renders_as_address_not_lvalue_read(
        self, examples_dir: Path
    ) -> None:
        if not _idalib():
            pytest.skip("idalib unavailable")
        import idapro
        import ida_hexrays
        import ida_idaapi
        import ida_name

        binary = examples_dir / "cp"
        if not binary.exists():
            pytest.skip("missing example binary: cp")

        from idavator.llvm_drop import LLVMDropConverter

        ll = (examples_dir / "cp.ll").read_text()
        idapro.open_database(str(binary), True)
        try:
            assert ida_hexrays.init_hexrays_plugin()
            ea = ida_name.get_name_ea(ida_idaapi.BADADDR, _FN)
            assert ea != ida_idaapi.BADADDR, f"missing function {_FN}"

            conv = LLVMDropConverter(ll)
            cf = conv.drop(ea, _FN)
            assert conv.last_error is None, conv.last_error
            assert conv.last_interr is None, f"INTERR {conv.last_interr}"
            assert cf is not None, "decompile returned None"
            txt = str(cf)

            # The string constant must be referenced directly (gvaraddr renders
            # the bare literal), and NO string-literal deref may remain -- the
            # exact bug the fix removes.
            assert _STRING in txt, f"string not referenced:\n{txt}"
            m = _STRLIT_DEREF.search(txt)
            assert m is None, (
                "global-as-value rendered as an lvalue read "
                f"(*(const char **)\"...\") -- gvaraddr fix regressed:\n{txt}"
            )
            # Specifically, gettext must take the bare string, not the deref.
            assert '(const char **)"write error"' not in txt, (
                f"gettext arg is a string-literal deref:\n{txt}"
            )
        finally:
            idapro.close_database()

    def test_desc_resolves_ptr_typed_global_to_gvaraddr(
        self, examples_dir: Path
    ) -> None:
        # Directly: a ptr-typed named-global operand resolves to ("gvaraddr",
        # ea, 8) -- the symbol's ADDRESS -- not ("gvar", ...) (the lvalue read).
        if not _idalib():
            pytest.skip("idalib unavailable")
        import idapro
        import ida_idaapi
        import ida_name
        import llvmlite.binding as llvm

        binary = examples_dir / "cp"
        if not binary.exists():
            pytest.skip("missing example binary: cp")

        from idavator.llvm_drop import LLVMDropConverter

        ll = (examples_dir / "cp.ll").read_text()
        idapro.open_database(str(binary), True)
        try:
            mod = llvm.parse_assembly(ll)
            fn = next(g for g in mod.functions if g.name == _FN)
            # Find the [N x i8] GEP whose base is the named string global, and
            # resolve that base operand through _desc.
            conv = LLVMDropConverter(ll)
            base_op = None
            for blk in fn.blocks:
                for ins in blk.instructions:
                    s = str(ins).strip()
                    if "getelementptr" in s and "i8]" in s:
                        op0 = list(ins.operands)[0]
                        if op0.name and ida_name.get_name_ea(
                            ida_idaapi.BADADDR, op0.name
                        ) != ida_idaapi.BADADDR:
                            base_op = op0
                            break
                if base_op is not None:
                    break
            assert base_op is not None, "no named-global [N x i8] GEP base found"
            assert str(base_op.type) == "ptr", base_op.type
            d = conv._desc(base_op, {}, 8)
            assert d[0] == "gvaraddr", f"expected gvaraddr, got {d!r}"
            assert d[2] == 8, d
        finally:
            idapro.close_database()
