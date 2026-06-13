"""A struct-member compare the native does must survive the lift+drop with its
true byte offset -- two compares of two DIFFERENT fields must NOT collapse onto a
single field, silently dropping the other field's liveness.

``source_is_dst_backup`` (gnulib ``backupfile.c``) ends with the equality test
``src_st->st_ino == dst_back_sb->st_ino && src_st->st_dev == dst_back_sb->st_dev``
(``st_ino`` at ``stat`` offset 8, ``st_dev`` at offset 0). IDA's PRISTINE native
renders exactly that pair of field compares.

The bug was a STALE ``examples/cp.ll`` body: an older lifter emitted the second
comparand of the ``st_ino`` block as ``bitcast %dst_back_sb to i64*`` at offset 0
(``dst_back_sb->st_dev``) instead of ``getelementptr i8, %dst_back_sb, 8`` followed
by the i64 load (``dst_back_sb->st_ino``). The ``src_st`` side carried offset 8
correctly, so the drop emitted
``*((_QWORD*)a1 + 1) == dst_back_sb.st_dev && *(_QWORD*)a1 == dst_back_sb.st_dev``
-- ``dst_back_sb.st_dev`` compared TWICE and ``dst_back_sb.st_ino`` LOST. On a
backup with the same ``st_dev`` but a different ``st_ino`` this wrongly reports the
file as its own backup.

Ground truth (the current ida2llvm lifter re-lifting this binary, and IDA's own
PRISTINE native): the ``dst_back_sb`` comparand of the first equality is a
``getelementptr i8, ptr %dst_back_sb, 8`` -> i64 load = ``st_ino``. The fix
re-splices that offset-8 GEP into the stale ``@6`` block.

Fail-without-fix: against the stale ``@6`` (no GEP), the drop renders
``dst_back_sb.st_dev`` twice and never emits ``dst_back_sb.st_ino`` (proven by
reverting the four edited ``@6`` lines to the bare offset-0 bitcast+load).
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
    dropped pseudocode. Nothing decompiles the ea first (idalib non-determinism).
    A native fallback (build error) is rejected: this asserts a REAL drop."""
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
        # A real drop (not a native fallback): build succeeded with no late error.
        assert conv.last_error is None, conv.last_error
        assert cf is not None, "decompile returned None"
        return str(cf)
    finally:
        idapro.close_database()


@pytest.mark.ida
class TestValueLivenessSurvives:
    def test_source_is_dst_backup_compares_both_ino_and_dev(
            self, examples_dir: Path) -> None:
        """``source_is_dst_backup`` must compare BOTH ``dst_back_sb.st_ino`` (off 8)
        and ``dst_back_sb.st_dev`` (off 0) against the source stat -- the two field
        compares the native does.

        Fail-without-fix: the stale ``@6`` block read ``dst_back_sb`` at offset 0
        for BOTH compares, so the drop rendered ``dst_back_sb.st_dev`` twice and
        DROPPED ``dst_back_sb.st_ino``."""
        if not _idalib():
            pytest.skip("idalib unavailable")
        dropped = _drop_only(examples_dir, "source_is_dst_backup")

        # st_ino (stat offset 8) must be a live comparand -- the field the stale
        # body collapsed onto st_dev.
        assert "dst_back_sb.st_ino" in dropped, (
            "dst_back_sb.st_ino compare DROPPED (offset 8 collapsed onto 0):\n"
            f"{dropped}")
        # st_dev (stat offset 0) is the other comparand and must remain.
        assert "dst_back_sb.st_dev" in dropped, (
            f"dst_back_sb.st_dev compare missing:\n{dropped}")
        # The defining regression signature: st_dev must NOT be compared twice in
        # place of st_ino.
        assert dropped.count("dst_back_sb.st_dev") == 1, (
            "dst_back_sb.st_dev compared more than once -- st_ino liveness lost:\n"
            f"{dropped}")
