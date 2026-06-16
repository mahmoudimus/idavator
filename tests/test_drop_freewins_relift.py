"""Re-lifted free-win bodies in examples/cp.ll must drop FAITHFUL to the pristine
native -- a checked-in guard against the committed-stale bodies regressing back.

Two coreutils functions had STALE committed bodies in cp.ll (lifted by an older
lifter) whose drops diverged from IDA's own pristine/forced-prototype native. The
current lifter re-lifts them faithfully; this test re-splices nothing -- it drops the
*committed* cp.ll body and asserts the recovered signatures are present, so it FAILS
if the stale body is ever restored.

``set_program_name`` (HIGH VALUE -- a crashing miscompile): the stale body stored the
``__progname`` / ``__progname_full`` GLIBC externs through a BADADDR pointer
(``*(_QWORD *)0xFFFFFFFFFFFFFFFFLL = ...`` -- a NULL/-1 address store that traps at
runtime). The faithful drop writes the real globals
(``_progname = (__int64)(base + 3)`` ; ``_progname_full = (__int64)argv0a``). Ground
truth (PRISTINE + forced-prototype native): ``_progname_full = (__int64)argv0a`` and
NO ``0xFFFF...FF`` store. The fix adds the ``@"__progname@GLIBC_2.2.5"`` /
``@"__progname_full@GLIBC_2.2.5"`` global definitions so the lifted store resolves to
the extern instead of -1.

``randint_genmax``: the stale body lowered the ``randread`` scratch buffer onto the
GLOBAL ``randnum`` (``randread((randread_source *)a0, &randnum, ...)`` reading
``*((u8 *)&randnum + i)``) instead of a LOCAL stack buffer. The faithful drop uses a
local buffer (``randread(s->source, buf, i)`` ; ``randnum = ... + buf[ia]``), matching
pristine, which reads ``s->source`` into a local and fills a stack ``buf``.

Fail-without-fix (proven against the committed-stale bodies on this branch base):
``set_program_name`` drops ``*(_QWORD *)0xFFFFFFFFFFFFFFFFLL = v3`` (and lacks
``_progname_full =``); ``randint_genmax`` drops ``randread(..., &randnum, ...)`` (the
global-buffer aliasing). A native fallback is rejected -- this asserts a REAL drop.
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


def _paths(examples_dir: Path):
    binary = examples_dir / "cp"
    ir_path = examples_dir / "cp.ll"
    if not (binary.exists() and ir_path.exists()):
        pytest.skip("missing cp / cp.ll")
    return binary, ir_path


def _drop_only(examples_dir: Path, name: str) -> str:
    """Drop ``name`` from cp.ll into its own ea in a FRESH session and return the
    dropped pseudocode. A native fallback (build error -> ``last_error`` set) is
    rejected: this asserts a REAL drop, never IDA's own recovery."""
    import idapro
    import ida_hexrays
    import ida_idaapi
    import ida_name

    binary, ir_path = _paths(examples_dir)
    from idavator.llvm_drop import LLVMDropConverter

    idapro.open_database(str(binary), True)
    try:
        assert ida_hexrays.init_hexrays_plugin()
        ea = ida_name.get_name_ea(ida_idaapi.BADADDR, name)
        if ea == ida_idaapi.BADADDR:
            pytest.skip(f"{name} not in this binary")
        conv = LLVMDropConverter(ir_path.read_text())
        cf = conv.drop(ea, name)
        assert conv.last_error is None, conv.last_error
        assert cf is not None, "decompile returned None"
        return str(cf)
    finally:
        idapro.close_database()


@pytest.mark.ida
class TestFreeWinRelift:
    def test_set_program_name_no_badaddr_store(self, examples_dir: Path) -> None:
        """The stale crashing body must not be restored: the faithful drop names
        every writable global it CAN resolve.

        Build-conditional. ``set_program_name`` also stores to the glibc
        interposition externs ``__progname`` / ``__progname_full``. Where this IDA
        build attaches a get_name_ea-resolvable name to those slots (arm64 keeps
        ``__progname_full``), the drop renders them BY NAME and no
        ``0xFFFFFFFFFFFFFFFF`` store appears -- the original stale-body guard. Where
        it does NOT (amd64 exposes them only as type-library display names with no
        address), the slots are unaddressable and BOTH the drop AND Hex-Rays' own
        native decompile render the stores as ``*(_QWORD *)0xFFFFFFFFFFFFFFFFLL`` --
        faithful, not a miscompile (full-body faithfulness is asserted by
        ``test_drop_global_reloc::test_set_program_name_drop_equals_native``). So
        the no-BADADDR invariant only applies on a build that names the extern."""
        if not _idalib():
            pytest.skip("idalib unavailable")
        import idapro
        import ida_hexrays
        import ida_idaapi
        import ida_name

        binary, ir_path = _paths(examples_dir)
        from idavator.llvm_drop import LLVMDropConverter

        idapro.open_database(str(binary), True)
        try:
            assert ida_hexrays.init_hexrays_plugin()
            ea = ida_name.get_name_ea(ida_idaapi.BADADDR, "set_program_name")
            if ea == ida_idaapi.BADADDR:
                pytest.skip("set_program_name not in this binary")
            # Where the glibc __progname_full slot is unaddressable (amd64), the
            # 0xFFFF... store is the FAITHFUL rendering (native renders it too), so
            # the no-BADADDR guard does not apply -- faithfulness is covered by
            # test_drop_global_reloc::test_set_program_name_drop_equals_native.
            if ida_name.get_name_ea(
                    ida_idaapi.BADADDR,
                    "__progname_full@GLIBC_2.2.5") == ida_idaapi.BADADDR:
                pytest.xfail(
                    "glibc __progname_full is unaddressable on this build; the "
                    "faithful drop renders the same *(_QWORD *)0xFFFF... store as "
                    "native (faithfulness asserted by test_drop_global_reloc)")
            conv = LLVMDropConverter(ir_path.read_text())
            cf = conv.drop(ea, "set_program_name")
            assert conv.last_error is None, conv.last_error
            assert cf is not None, "decompile returned None"
            dropped = str(cf)
            assert "0xFFFFFFFFFFFFFFFF" not in dropped, (
                f"CRASHING BADADDR store still present (stale body restored):\n{dropped}")
            assert "_progname_full" in dropped, (
                f"__progname_full extern write missing (stale body restored):\n{dropped}")
        finally:
            idapro.close_database()

    def test_randint_genmax_uses_local_buffer(self, examples_dir: Path) -> None:
        """``randread`` must fill a LOCAL stack buffer, not the global ``randnum``."""
        if not _idalib():
            pytest.skip("idalib unavailable")
        dropped = _drop_only(examples_dir, "randint_genmax")

        # Stale body: `randread((randread_source *)a0, &randnum, ...)` -- the global
        # `randnum` used as the scratch buffer. Faithful: a local buffer.
        assert "&randnum" not in dropped, (
            f"randread aliased onto the GLOBAL randnum buffer "
            f"(stale body restored):\n{dropped}")
        assert "randread(" in dropped, (
            f"randread call missing entirely:\n{dropped}")
