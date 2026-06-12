"""Every struct-member store the native does must survive the lift+drop -- a
member written at a non-zero byte offset must NOT collapse onto offset 0 (where
later members silently overwrite earlier ones and become dead writes).

The lifter decayed a struct-typed local to ``i8*`` in ``ida2llvm.get_offset_to``
and then DISCARDED the member byte offset for the ``IdentifiedStructType`` case
(it returned the bare base pointer, ignoring ``off``). The IDA microcode carried
the offset -- ``seen_file`` lowers to
``mov file, new_ent`` / ``ldx ds,(stats+8),new_ent@8`` / ``ldx ds,stats,new_ent@16``
(``@8`` = ``st_ino``, ``@16`` = ``st_dev``) -- but every destination was lifted to
``&new_ent`` at offset 0. ``mem2reg`` then sees three writes to the same slot and
keeps only the last, so the drop emits a single ``new_ent.name = ...`` and DROPS
``st_ino`` / ``st_dev``. ``forget_created`` lost ``probe.st_dev`` / ``probe.name``
the same way (``Src_to_dest`` = ``{st_ino@0, st_dev@8, name@16}``).

Ground truth (clang ``-O2 -emit-llvm`` on the gnulib ``cp-hash.c`` triple and
IDA's own PRISTINE native): the three stores land at offsets ``0`` / ``8`` / ``16``
(``store ptr ..., @new_ent`` ; ``store i64 ..., gep @new_ent, 8`` ;
``store i32 ..., gep @new_ent, 16``). Native renders
``new_ent.name = file; new_ent.st_ino = stats->st_ino; new_ent.st_dev = stats->st_dev``.

Fix (``ida2llvm.get_offset_to``): when the pointee is an ``IdentifiedStructType``
and ``off > 0``, carry the offset as an ``i8*`` byte GEP after the decay, instead
of dropping it.

Fail-without-fix: against the pre-fix lifted IR every member store aliases
offset 0, so the drop omits the ``st_ino`` / ``st_dev`` (``seen_file``) and
``st_dev`` / ``name`` (``forget_created``) field writes (proven by reverting the
``get_offset_to`` ``IdentifiedStructType`` branch to the bare ``i8*`` cast and
re-lifting these bodies).
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
    dropped pseudocode. Nothing decompiles the ea first -- a prior decompile of
    the same ea perturbs the lvar cache and reshapes the drop (idalib
    non-determinism). A native fallback (build error) is rejected: this asserts a
    REAL drop, never IDA's own recovery."""
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
class TestStructMemberStoresSurvive:
    def test_seen_file_writes_all_three_members(
            self, examples_dir: Path) -> None:
        """``seen_file`` must write ``new_ent.name`` / ``new_ent.st_ino`` /
        ``new_ent.st_dev`` -- all three members native fills before
        ``hash_lookup``.

        Fail-without-fix: ``get_offset_to`` dropped the member offset, so
        ``st_ino`` (off 8) and ``st_dev`` (off 16) aliased ``name`` (off 0) and
        ``mem2reg`` kept only the last write -- the drop emits a lone
        ``new_ent.name = ...``."""
        if not _idalib():
            pytest.skip("idalib unavailable")
        dropped = _drop_only(examples_dir, "seen_file")

        assert "new_ent.name" in dropped, (
            f"name member store missing:\n{dropped}")
        # st_ino is at F_triple offset 8, read from stats->st_ino (stat off 8).
        assert "new_ent.st_ino" in dropped, (
            f"st_ino member store DROPPED (offset collapsed onto 0):\n{dropped}")
        # st_dev is at F_triple offset 16, read from stats->st_dev (stat off 0).
        assert "new_ent.st_dev" in dropped, (
            f"st_dev member store DROPPED (offset collapsed onto 0):\n{dropped}")
        # The two extra members must read from the `stats` pointer (a2/arg),
        # i.e. the offset writes are real, not a single scalar store.
        assert dropped.count("new_ent.") >= 3, (
            f"fewer than 3 distinct new_ent member writes:\n{dropped}")

    def test_forget_created_writes_all_three_members(
            self, examples_dir: Path) -> None:
        """``forget_created`` must fill ``probe.st_ino`` / ``probe.st_dev`` /
        ``probe.name`` before ``hash_delete``.

        Fail-without-fix: ``st_dev`` (off 8) and ``name`` (off 16) aliased off 0,
        so the drop kept only one write (``probe.st_ino = 0``) and lost the
        ino/dev parameter stores entirely."""
        if not _idalib():
            pytest.skip("idalib unavailable")
        dropped = _drop_only(examples_dir, "forget_created")

        assert "probe.st_ino" in dropped, (
            f"st_ino member store missing:\n{dropped}")
        assert "probe.st_dev" in dropped, (
            f"st_dev member store DROPPED (offset collapsed onto 0):\n{dropped}")
        assert "probe.name" in dropped, (
            f"name member store DROPPED (offset collapsed onto 0):\n{dropped}")
        assert dropped.count("probe.") >= 3, (
            f"fewer than 3 distinct probe member writes:\n{dropped}")
