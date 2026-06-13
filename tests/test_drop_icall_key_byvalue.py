"""Indirect-call key passed BY VALUE, not over-dereferenced one level.

``safe_hasher`` calls ``table->hasher(key, table->n_buckets)`` -- the ``key``
PARAMETER (a pointer) is handed to the callback BY VALUE. The lifter materialises
``key`` into a pointer-typed alloca slot and reads the pointer back for the call
as ``load i64, bitcast i8** %key to i64*`` -- a pointer-VALUE read type-punned to
``i64`` (the callback's first formal is opaque ``i64`` in cp.ll).

DROP BUG (ticket ida-ypfw): the drop's pointer-alloca deref-vs-slot classifier
keyed only on the load's RESULT TYPE (``not _is_ptr_type``), so the punned
``load i64`` looked like a ``*key`` deref and the drop emitted an extra ``ldx``
through the slot -- ``hasher(*(_QWORD *)a1, ...)`` instead of ``hasher(key, ...)``.

clang ``-O2 -emit-llvm`` of the gnulib ``safe_hasher`` source confirms the native
semantics: ``call i64 %hasher(ptr %key, i64 %n_buckets)`` -- the key is passed
directly, with NO load before the call. ``hash_lookup``'s ``table->comparator(
entry, cursor->data)`` carries the same pattern (entry by value).

FIX: a pointer-width read of a SUB-pointer-width pointee slot (``i8*`` -> pointee
``i8``) is a punned pointer-VALUE read, not a deref -- it falls through to the
slot read (returns the pointer value). A read whose width MATCHES the pointee
(``*p`` on an 8-byte pointee) stays a deref.

Fail-without-fix: against the pre-fix classifier, ``safe_hasher`` drops
``hasher(*(_QWORD *)a1, ...)`` and ``hash_lookup`` drops the comparator call as
``(...)(*(_QWORD *)a1, ...)`` -- the over-deref the asserts below pin out.
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


def _drop_only(examples_dir: Path, name: str) -> str:
    """Drop ``name`` from cp.ll into its own ea in a FRESH session; return the
    dropped pseudocode. A native fallback (build error) is rejected -- this
    asserts a REAL drop."""
    import idapro
    import ida_hexrays
    import ida_idaapi
    import ida_name

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
        assert cf is not None, "decompile returned None"
        return str(cf)
    finally:
        idapro.close_database()


@pytest.mark.ida
class TestIcallKeyByValue:
    def test_safe_hasher_passes_key_by_value(self, examples_dir: Path) -> None:
        """``safe_hasher`` hands the ``key`` POINTER to the indirect hasher BY
        VALUE -- the call argument is the pointer itself (``a1``), NOT the
        over-dereferenced ``*(_QWORD *)a1``."""
        if not _idalib():
            pytest.skip("idalib unavailable")
        dropped = _drop_only(examples_dir, "safe_hasher")

        # The indirect hasher call must NOT dereference the key pointer.
        assert "*(_QWORD *)a1" not in dropped, (
            f"key over-dereferenced one level (`hasher(*(_QWORD *)a1, ...)` "
            f"instead of `hasher(key, ...)`):\n{dropped}")
        # The call still dispatches indirectly through table->hasher (offset +6).
        assert "+ 6)" in dropped or "+ 6 )" in dropped, (
            f"indirect hasher dispatch (table->hasher at +6) lost:\n{dropped}")

    def test_hash_lookup_comparator_key_by_value(
            self, examples_dir: Path) -> None:
        """``hash_lookup`` hands ``entry`` to ``table->comparator`` BY VALUE -- the
        comparator's first argument is the entry pointer (``a1``), NOT
        ``*(_QWORD *)a1``. (The surrounding loop body has a separate, pre-existing
        SROA-residual divergence; this pins only the call-dispatch argument.)"""
        if not _idalib():
            pytest.skip("idalib unavailable")
        dropped = _drop_only(examples_dir, "hash_lookup")

        # The indirect comparator call must NOT dereference the entry pointer.
        assert "*(_QWORD *)a1" not in dropped, (
            f"entry over-dereferenced one level in the comparator call "
            f"(`(...)(*(_QWORD *)a1, ...)`):\n{dropped}")
        # safe_hasher is still called with the entry pointer directly.
        assert "safe_hasher(" in dropped, (
            f"safe_hasher dispatch lost:\n{dropped}")
