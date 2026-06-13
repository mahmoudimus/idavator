"""A variadic function's body va_list machine must drop as the Hex-Rays
``va_start``/``va_arg``/``va_end`` MACROS over ``__va_list_tag`` -- NOT fall back
to a native decompile, and NOT leak the lifter's redundant synth scaffold.

Two coupled defects produced the va_start drop failure:

  1. LIFTER (ida2llvm): on top of the body's correct ``@va_start``/``@va_arg``
     over the real ``[1 x %__va_list_tag]`` storage, the lifter ALSO bolts a
     REDUNDANT scaffold -- ``%ArgList = alloca i8*`` (uninit) + ``load`` +
     ``call @llvm.va_start`` / ``@llvm.va_end`` -- that has NO native counterpart.
  2. DROP (llvm_drop): ``_emit_call`` could not resolve ``@llvm.va_start`` /
     ``@va_start`` / ``@va_arg`` / ``@va_end`` (unresolved callees) ->
     ``ValueError: unresolved callee @llvm.va_start.p0`` -> NATIVE FALLBACK.

The fix is DROP-ONLY (plus a narrow STALE-cp.ll re-splice of the 3 vararg bodies
so the variadic ``openat``/``open`` call carries its ``mode`` tail arg, which the
ida-23as lifter fix already emits but the committed cp.ll predated):
``_emit_value`` NO-OPs the dead ``llvm.va_start``/``llvm.va_end`` scaffold and
the dead ``@IDA_QWORD`` type-marker bitcast, and routes ``va_start``/``va_arg``/
``va_end`` to ``_emit_helper_call`` (the same helper-call shape as ``__ROR8__``).
``_segment_block`` keeps these calls IN-segment (like the canary) so they render
INLINE, not block-terminal.

Ground truth = PRISTINE-NATIVE (throwaway IDB, no ``_force_prototype``
persistence). Native renders ``va_start(authors, version)`` /
``v4 = va_arg(authors, _QWORD)`` / ``authors[0].gp_offset = 0x20`` -- helper-call
macros, NOT the ``!va_start`` IR intrinsic.

Proven fail-without-fix (base 90c5c31, before the drop fix): version_etc and
openat_safer drop returns ``conv.last_error ==
'ValueError: unresolved callee @llvm.va_start.p0'`` (a native fallback, NOT a
real drop); openat_safer's ``openat`` call also loses its ``mode`` tail arg.
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
    tmp = Path(tempfile.mkdtemp(prefix="vaintr_guard_"))
    dst = tmp / "cp"
    shutil.copy(binary, dst)
    return tmp, dst


def _call_arg_count(text: str, callee: str) -> int:
    """Number of comma-separated top-level arguments of the FIRST call to
    ``callee`` (a bare name, not a substring of a longer identifier)."""
    pat = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(callee) + r"\s*\(")
    for line in text.splitlines():
        m = pat.search(line)
        if not m:
            continue
        inner = line[line.index("(", m.start()) + 1: line.rindex(")")]
        depth = 0
        args = 1 if inner.strip() else 0
        for ch in inner:
            if ch in "([":
                depth += 1
            elif ch in ")]":
                depth -= 1
            elif ch == "," and depth == 0:
                args += 1
        return args
    return -1


def _drop(examples_dir: Path, name: str):
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
        ea = ida_name.get_name_ea(ida_idaapi.BADADDR, name)
        assert ea != ida_idaapi.BADADDR, f"{name} missing from cp"
        conv = LLVMDropConverter(ll.read_text())
        cf = conv.drop(ea, name)
        return conv, (str(cf) if cf is not None else None)
    finally:
        idapro.close_database()
        shutil.rmtree(tmp, ignore_errors=True)


@pytest.mark.ida
class TestDropVarargIntrinsic:
    def test_version_etc_va_machine_is_real_drop(self, examples_dir: Path) -> None:
        """version_etc must be a REAL drop (not a native fallback) rendering the
        va_list machine as helper macros, with NO scaffold leakage."""
        if not _idalib():
            pytest.skip("idalib unavailable")
        conv, txt = _drop(examples_dir, "version_etc")

        # REAL drop: cf built AND no native-fallback (last_error/last_interr None).
        # Without the fix this is `ValueError: unresolved callee @llvm.va_start.p0`.
        assert conv.last_error is None, conv.last_error
        assert conv.last_interr is None, f"INTERR {conv.last_interr}"
        assert txt is not None, "drop returned None"

        # The body va_list machine renders as macros (like native).
        assert "va_start(" in txt, f"va_start macro missing:\n{txt}"
        assert "va_arg(" in txt, f"va_arg macro missing:\n{txt}"
        # The gp_offset write and the real consumer call survive.
        assert "gp_offset" in txt, f"gp_offset write lost:\n{txt}"
        assert "version_etc_va(" in txt, f"version_etc_va consumer lost:\n{txt}"

        # The lifter's redundant synth scaffold must NOT leak into output.
        assert "ArgList" not in txt, f"synth %ArgList leaked:\n{txt}"
        assert "llvm.va_start" not in txt, f"scaffold llvm.va_start leaked:\n{txt}"
        assert "IDA_QWORD" not in txt, f"IDA_QWORD type-marker leaked:\n{txt}"

    def test_openat_safer_carries_mode_tail_arg(self, examples_dir: Path) -> None:
        """openat_safer must be a REAL drop whose ``openat`` carries its variadic
        ``mode`` tail arg (4 args: fd, file, flags, mode) -- matching native. The
        va_arg result flows ``v = va_arg(...); if (flags & 0x40) mode = v;`` into
        the call's 4th argument."""
        if not _idalib():
            pytest.skip("idalib unavailable")
        conv, txt = _drop(examples_dir, "openat_safer")

        assert conv.last_error is None, conv.last_error
        assert conv.last_interr is None, f"INTERR {conv.last_interr}"
        assert txt is not None, "drop returned None"

        assert "va_start(" in txt, f"va_start macro missing:\n{txt}"
        assert "va_arg(" in txt, f"va_arg macro missing:\n{txt}"

        # openat carries the mode tail arg (4 args), not the truncated 3-arg call
        # the stale cp.ll produced.
        nargs = _call_arg_count(txt, "openat")
        assert nargs == 4, (
            f"expected openat(fd, file, flags, mode) = 4 args, got {nargs}:\n{txt}")

        assert "ArgList" not in txt, f"synth %ArgList leaked:\n{txt}"
        assert "llvm.va_start" not in txt, f"scaffold llvm.va_start leaked:\n{txt}"
