"""Variadic call sites (printf/error/fprintf/...) must carry their trailing
varargs through the LLVM->microcode drop.

The lifter used to TRUNCATE every variadic call to its FIXED parameters
(``args = args[:len(l_pointee.args)]`` in ``ida2llvm._llvm_inst_from_minsn``),
dropping ALL trailing varargs; the values were computed (dead loads) then
discarded. Even with the IR carrying the varargs, a bare ``m_call gvar`` left
Hex-Rays to re-discover the vararg count and it dropped them (rendering only a
mis-resolved fmt). ``LLVMDropConverter._emit_call_vararg`` now builds an explicit
variadic ``mcallinfo`` (callee's real ellipsis prototype + appended register
varargs, result-discarded so it survives Hex-Rays' callinfo re-derivation), so
the drop emits the full call.

Ground truth is the PRISTINE-NATIVE decompile (a throwaway IDB copy, no
``_force_prototype`` persistence). ``emit_verbose`` natively renders
``printf("%s -> %s", v4, v3)`` -- the drop must carry both varargs after the
format string, not collapse to ``printf(a0)``.

Proven fail-without-fix: on tree 217c8dc the drop renders
``printf((const char *)a0)`` (zero varargs).
"""
from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path

import pytest


def _idalib() -> bool:
    try:
        import idapro  # noqa: F401

        return True
    except ImportError:
        return False


def _fresh_cp(binary: Path):
    """A throwaway copy of the cp binary so a drop's _force_prototype never
    persists into the shared IDB (pristine-oracle hygiene)."""
    tmp = Path(tempfile.mkdtemp(prefix="vararg_guard_"))
    dst = tmp / "cp"
    shutil.copy(binary, dst)
    return tmp, dst


def _printf_arg_count(text: str, fmt_substr: str) -> int:
    """Number of comma-separated arguments of the printf/error call whose first
    rendered argument contains ``fmt_substr`` (the format string)."""
    for line in text.splitlines():
        if fmt_substr in line and re.search(r"\b(printf|error|fprintf)\s*\(", line):
            inner = line[line.index("(") + 1: line.rindex(")")]
            depth = 0
            args = 1
            for ch in inner:
                if ch in "([":
                    depth += 1
                elif ch in ")]":
                    depth -= 1
                elif ch == "," and depth == 0:
                    args += 1
            return args
    return -1


@pytest.mark.ida
class TestDropCallVarargs:
    def test_emit_verbose_printf_carries_varargs(self, examples_dir: Path) -> None:
        """emit_verbose's first printf must drop as printf("%s -> %s", v4, v3) --
        the format string PLUS two varargs -- matching pristine-native."""
        if not _idalib():
            pytest.skip("idalib unavailable")
        import idapro
        import ida_hexrays
        import ida_idaapi
        import ida_name

        from idavator.llvm_drop import LLVMDropConverter

        binary = examples_dir / "cp"
        if not binary.exists():
            pytest.skip("missing example binary: cp")
        ll = examples_dir / "cp.ll"
        if not ll.exists():
            pytest.skip("missing example IR: cp.ll")

        tmp, dst = _fresh_cp(binary)
        idapro.open_database(str(dst), True)
        try:
            assert ida_hexrays.init_hexrays_plugin()
            ea = ida_name.get_name_ea(ida_idaapi.BADADDR, "emit_verbose")
            assert ea != ida_idaapi.BADADDR, "emit_verbose missing from cp"

            conv = LLVMDropConverter(ll.read_text())
            cf = conv.drop(ea, "emit_verbose")
            assert conv.last_error is None, conv.last_error
            assert conv.last_interr is None, f"INTERR {conv.last_interr}"
            assert cf is not None, "drop returned None"
            txt = str(cf)

            # The format string is recovered ...
            assert "%s -> %s" in txt, f"format string lost:\n{txt}"
            # ... and the printf carries it PLUS two varargs (3 args total). The
            # buggy drop renders printf(a0) -- a single argument.
            nargs = _printf_arg_count(txt, "%s -> %s")
            assert nargs == 3, (
                f"expected printf(fmt, v4, v3) = 3 args, got {nargs}:\n{txt}")
        finally:
            idapro.close_database()
            shutil.rmtree(tmp, ignore_errors=True)

    def test_close_stdout_error_carries_varargs(self, examples_dir: Path) -> None:
        """close_stdout's error must drop as error(0, *errno, "%s: %s", v0,
        write_error) -- the format string PLUS two varargs."""
        if not _idalib():
            pytest.skip("idalib unavailable")
        import idapro
        import ida_hexrays
        import ida_idaapi
        import ida_name

        from idavator.llvm_drop import LLVMDropConverter

        binary = examples_dir / "cp"
        if not binary.exists():
            pytest.skip("missing example binary: cp")
        ll = examples_dir / "cp.ll"
        if not ll.exists():
            pytest.skip("missing example IR: cp.ll")

        tmp, dst = _fresh_cp(binary)
        idapro.open_database(str(dst), True)
        try:
            assert ida_hexrays.init_hexrays_plugin()
            ea = ida_name.get_name_ea(ida_idaapi.BADADDR, "close_stdout")
            assert ea != ida_idaapi.BADADDR, "close_stdout missing from cp"

            conv = LLVMDropConverter(ll.read_text())
            cf = conv.drop(ea, "close_stdout")
            assert conv.last_error is None, conv.last_error
            assert conv.last_interr is None, f"INTERR {conv.last_interr}"
            assert cf is not None, "drop returned None"
            txt = str(cf)

            assert '"%s: %s"' in txt, f"format string lost:\n{txt}"
            # error(status, errnum, "%s: %s", arg1, arg2) = 5 args.
            nargs = _printf_arg_count(txt, "%s: %s")
            assert nargs == 5, (
                f'expected error(0, *e, "%s: %s", v0, write_error) = 5 args, '
                f"got {nargs}:\n{txt}")
        finally:
            idapro.close_database()
            shutil.rmtree(tmp, ignore_errors=True)
