"""Struct-POINTER parameter must be written THROUGH the caller's pointer, not a
local value-copy.

The lifter (``ida2llvm._store_as``) modelled a plain pointer copy ``p = q`` --
where ``q`` is a ``struct *`` and the destination is a POINTER SLOT (``struct **``)
-- as a ``memcpy`` of the whole pointee into the slot. For ``set_char_quoting``
that emitted ``memcpy(local, a0, 0x38)`` followed by field writes to ``local``,
so the function mutated a 56-byte STACK COPY instead of the caller's
``quoting_options`` (native does ``o->quote_these_too[...] ^= ...`` straight
through the pointer ``a0``). ``set_custom_quoting`` was the same shape on
``->style`` / ``->left_quote`` / ``->right_quote``.

Ground truth (clang ``-O2``/``-O0`` on gnulib ``quotearg.c`` and IDA's own
pristine native): the ``o ? o : &default`` select yields a POINTER and every
field write goes through it (``store ptr``, an 8-byte pointer store -- never a
56-byte ``memcpy`` of the pointee). The fix gates ``_store_as``'s memcpy on the
DESTINATION addressing the aggregate (``d_pointee`` is the struct), not merely on
the stored VALUE being a struct pointer; a pointer slot (``d_pointee`` is a
pointer) gets a plain ``store``.

Fail-without-fix: against the pre-fix lifted IR the drop reintroduces the
``memcpy(..., 0x38u)`` clobber and writes the fields of the local copy, so the
through-pointer field write is ABSENT (proven by reverting ida2llvm._store_as).
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
    non-determinism), so the through-pointer check must run clean."""
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
class TestStructParamWriteThroughPointer:
    def test_set_char_quoting_writes_through_pointer_not_local_copy(
            self, examples_dir: Path) -> None:
        """``set_char_quoting`` must toggle ``o->quote_these_too[..]`` through the
        caller's pointer, with NO 56-byte copy of the pointee into a local.

        Fail-without-fix: ``ida2llvm._store_as`` memcpy'd the pointee into the
        pointer slot, so the drop emits ``memcpy(local, a0, 0x38u)`` and mutates
        the local copy -- the caller's struct is never written."""
        if not _idalib():
            pytest.skip("idalib unavailable")
        dropped = _drop_only(examples_dir, "set_char_quoting")

        # The 56-byte (0x38) pointee copy that severs the write-through is GONE.
        assert "0x38" not in dropped, (
            f"struct-pointer param value-copied into a local (0x38 blob):\n{dropped}")
        assert "memcpy(" not in dropped, (
            f"memcpy reintroduces the local-copy clobber:\n{dropped}")
        # The field is read+written through the pointer (the `quote_these_too`
        # bitset toggle); native does the same `^=` through `v3`.
        assert "quote_these_too" in dropped, (
            f"field access scalarised away (no through-pointer write):\n{dropped}")
        assert "^=" in dropped, (
            f"the bitset toggle (write-through) is missing:\n{dropped}")

    def test_set_custom_quoting_writes_fields_through_pointer(
            self, examples_dir: Path) -> None:
        """``set_custom_quoting`` must write ``->style`` / ``->left_quote`` /
        ``->right_quote`` through the caller's pointer, with no pointee copy.

        Fail-without-fix: the pre-fix ``memcpy(local, a0, 0x38)`` makes the three
        field stores land in the local copy, not the caller's struct."""
        if not _idalib():
            pytest.skip("idalib unavailable")
        dropped = _drop_only(examples_dir, "set_custom_quoting")

        assert "0x38" not in dropped, (
            f"struct-pointer param value-copied into a local (0x38 blob):\n{dropped}")
        assert "memcpy(" not in dropped, (
            f"memcpy reintroduces the local-copy clobber:\n{dropped}")
        # The custom-quoting fields are written through the selected pointer.
        assert "->style" in dropped, (
            f"`->style` field write missing (scalarised?):\n{dropped}")
        assert "->left_quote" in dropped and "->right_quote" in dropped, (
            f"left/right quote field writes missing:\n{dropped}")
