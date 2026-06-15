"""Return-slot / return-phi promotion generalized to NON-noreturn + void fns.

The lifter emits a ``%funcresult`` return slot: every returning path does
``store v, funcresult`` and the terminal block ``%r = load funcresult; ret %r``.
Promoting that slot to the return register (each path writes rax directly, the
post-merge ``load`` collapses) was originally GATED on a noreturn call being
present -- the only shape that INTERRs 50342. But the redundant funcresult routing
also makes NON-noreturn + void fns drop SILENT GARBAGE:

- ``cp_options_default`` rendered each struct field store TWICE (once via the
  ``a0`` arg, once via a duplicate ``x->...`` alias) and returned the uninit slot
  kreg ``return v2`` -- the ``%x`` slot was mis-escaped by a no-op ``bitcast``;
- ``setfscreatecon`` dropped a DUPLICATE ``*__errno_location() = 0x5F`` plus a
  stale ``return v2`` -- the host m_ret block's own body (a leaf fn whose whole
  computation lives in the ret block) was reused as the bare return sink WITHOUT
  being cleared, so its side-effecting leftover survived past the re-emitted body.

The fix widens the promotion gate (it now fires for any funcresult-slot SHAPE, not
just noreturn) AND strips the reused host m_ret block down to its bare ``ret`` on
the multiblock path. Regression guard for ticket ida-ajb2 (csweep RETSLOT bucket).
See memory idavator_drop_correctness_coverage / idavator_drop_noreturn_50342_rootcause.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tests.render_tolerance import contains_int, count_int, search_with_ints


def _idalib() -> bool:
    try:
        import idapro  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.ida
class TestRetSlotGeneralize:
    def _drop(self, examples_dir: Path, name: str) -> str:
        """Drop ``name`` into its OWN ea and return the decompiled C text."""
        import ida_hexrays
        import ida_idaapi
        import ida_name
        import idapro

        binary = examples_dir / "cp"
        ir_path = examples_dir / "cp.ll"
        if not (binary.exists() and ir_path.exists()):
            pytest.skip("missing cp / cp.ll")

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
            assert cf is not None, f"{name}: decompile returned None"
            return str(cf)
        finally:
            idapro.close_database()

    def test_cp_options_default_no_dual_alias_no_uninit(
            self, examples_dir: Path) -> None:
        """The csweep RETSLOT exemplar: a non-noreturn ptr-returning fn whose ``%x``
        slot was mis-classified as address-taken (a no-op ``bitcast %x`` escaped it)
        rendered the SAME field stores TWICE -- once via ``a0[..]`` and again via a
        duplicate ``x->...`` alias -- and returned the uninit slot kreg ``v1``/``v2``.
        Stripping the reused host m_ret block kills the duplicate alias rendering and
        the promotion stops the un-promoted-slot uninit read.

        The field-store offset (27) is asserted by VALUE: IDA 9.3 Linux renders
        ``*((_BYTE *)a0 + 27)``, dev macOS IDA ``*((_BYTE *)a0 + 0x1B)`` -- cosmetic
        render divergence; the recovered store is faithful either way."""
        if not _idalib():
            pytest.skip("idalib unavailable")
        txt = self._drop(examples_dir, "cp_options_default")
        # the field store must appear, through the ``a0`` arg pointer (offset 27 in
        # any base).
        assert search_with_ints(r"\*\(\(_BYTE \*\)a0 \+ {off}\)", txt,
                                {"off": 0x1B}) is not None, (
            f"field store at offset 27 (0x1B) lost:\n{txt}")
        # the duplicate ``x->`` alias block (the csweep-flagged dual-alias) is gone.
        assert "x->owner_privileges" not in txt, (
            f"duplicate dual-alias var still rendered:\n{txt}")
        # no uninit slot-kreg return (the pre-fix ``return v1``/``return v2``).
        for bad in ("return v1;", "return v2;", "return v0;"):
            assert bad not in txt, f"uninit return-slot kreg ({bad}):\n{txt}"

    def test_setfscreatecon_no_duplicate_or_uninit(
            self, examples_dir: Path) -> None:
        """A leaf fn whose body lives in the host m_ret block must not leak that
        block's side effects (duplicate ``*errno=95``) nor return uninit.

        The errno constant (95) and the -1 return are asserted by VALUE: IDA 9.3
        Linux renders them in decimal (``95`` / ``-1``), dev macOS IDA in hex
        (``0x5F`` / ``0xFFFFFFFF``) -- cosmetic render divergence; the leaf body is
        faithful either way."""
        if not _idalib():
            pytest.skip("idalib unavailable")
        txt = self._drop(examples_dir, "setfscreatecon")
        # native: ``*__errno_location() = 95; return -1;`` -- exactly ONE errno
        # store (value 95, any base), the constant -1 return (any base).
        assert count_int(txt, 0x5F) == 1, (
            f"duplicate errno store (uncleared host m_ret block):\n{txt}")
        assert contains_int(txt, 0xFFFFFFFF, width=32), (
            f"constant -1 return lost:\n{txt}")
        # no dereference-of-slot-kreg garbage (``*v1 = 95`` etc.) and no
        # raw byte_ global write-through.
        assert "byte_" not in txt, f"garbage global operand:\n{txt}"
