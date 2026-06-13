"""Zero-offset through-pointer cursor walk: lift fidelity + drop-side lvar typing.

Two coupled defects produced a wrong/blob ``hash_get_first`` (and the other hash
cursor-walks ``transfer_entries`` / ``hash_get_max_bucket_length`` /
``hash_insert_if_absent``):

LIFT (ticket ida-th87, fixed by 57e7c90 + 217c8dc; the stale ``cp.ll`` bodies were
re-spliced through the fixed lifter): a struct-pointer member access at OFFSET 0
collapsed one indirection --

  * ``bucket = table->bucket`` (field 0) lifted as ``bucket = table`` (the table
    SLOT was read, not ``*table``) -> one deref short.
  * ``bucket->data`` (field 0) lifted as ``bucket == 0`` (no deref through bucket).
  * ``++bucket`` (a struct-pointer +sizeof increment) lifted as a 16-byte
    ``memcpy(&bucket, bucket+16, 16)`` clobbering the 8-byte slot, not a pointer
    add.

  clang ``-O0``/``-O2`` and the native ``MMAT_GLBOPT2`` microcode
  (``ldx ds,rdi,%bucket`` / ``jz [ds:%bucket],0`` / ``add %bucket,0x10``) all show
  these ARE real derefs / a pointer add. Offset>0 fields (``->next`` at +8) already
  lifted correctly.

DROP (ticket ida-1o1r): even with correct IR, the cursor stkvar is untyped, so the
decompiler renders an ``*(_QWORD*)`` blob walk (or picks up the parameter's
``Hash_table*`` by propagation). Typing the cursor slot ``hash_entry*`` (via a
persistent user-lvar type at the slot's real location, applied between two
decompiles) recovers the faithful ``for(bucket=...;;++bucket){if(bucket->data)..}``.

Fail-without-fix: against the pre-fix lifted bodies the drop emits the memcpy
increment + the ``bucket == 0`` collapse (no ``->data`` deref, no ``++``); without
the typing the cursor is an untyped blob / ``Hash_table*`` and never shows
``hash_entry`` + ``->data``. The asserts below pin BOTH halves.
"""
from __future__ import annotations

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


def _paths(examples_dir: Path):
    binary = examples_dir / "cp"
    ir_path = examples_dir / "cp.ll"
    if not (binary.exists() and ir_path.exists()):
        pytest.skip("missing cp / cp.ll")
    return binary, ir_path


def _drop_only(examples_dir: Path, name: str) -> str:
    """Drop ``name`` from cp.ll into its own ea in a FRESH session; return the
    dropped pseudocode. A native fallback (build error) is rejected -- this asserts
    a REAL drop."""
    import idapro
    import ida_hexrays
    import ida_idaapi
    import ida_name

    binary, ir_path = _paths(examples_dir)
    from idavator.llvm_drop import LLVMDropConverter

    # PRISTINE per-drop IDB: copy the binary to a throwaway dir so the drop's
    # _force_prototype set_types (saved by close_database) never persists into the
    # shared examples/cp.i64 -- forced-prototype writes accumulate across runs and
    # poison the native baseline for later cases. cp.ll stays the real read-only IR.
    tmp = Path(tempfile.mkdtemp(prefix="cursor_zerooff_"))
    dst = tmp / "cp"
    shutil.copy(binary, dst)
    idapro.open_database(str(dst), True)
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
        shutil.rmtree(tmp, ignore_errors=True)


@pytest.mark.ida
class TestZeroOffsetCursorWalk:
    def test_hash_get_first_typed_cursor_walk(self, examples_dir: Path) -> None:
        """``hash_get_first`` drops the faithful typed cursor walk:
        ``for(bucket=table->bucket;;++bucket){ if(bucket->data) break; }`` with the
        cursor typed ``hash_entry*`` -- no memcpy increment, no ``*(_QWORD*)`` blob,
        no zero-offset collapse."""
        if not _idalib():
            pytest.skip("idalib unavailable")
        dropped = _drop_only(examples_dir, "hash_get_first")

        # the cursor is typed as the struct (drop renders `hash_entry *` and
        # `->data`), NOT a raw _QWORD blob or the param's Hash_table*.
        assert "hash_entry" in dropped, (
            f"cursor not typed hash_entry* (untyped blob / Hash_table*):\n"
            f"{dropped}")
        assert "->data" in dropped, (
            f"zero-offset field deref lost (`bucket->data` collapsed):\n"
            f"{dropped}")
        # the ++ increment is a pointer add (`++x`), NOT a 16-byte struct memcpy.
        assert "memcpy" not in dropped, (
            f"struct-pointer ++ lowered as a memcpy:\n{dropped}")
        assert "++" in dropped, (
            f"cursor increment collapsed (no pointer ++):\n{dropped}")
        # the "allocation has failed" blob banner must be gone.
        assert "allocation has failed" not in dropped, (
            f"cursor still an unallocated blob:\n{dropped}")
        # The SPECIFIC zero-offset pointer-slot-DEFINE regression: the cursor init
        # `bucket = table->bucket` (an off-0 POINTER-field load into a pointer slot)
        # must lift pointer-typed -> render as a DEFINE, NOT the deref-WRITE
        # `bucket->data = *(void **)a0` the pre-fix i64-typed load collapsed to.
        assert "bucket->data = *(" not in dropped, (
            f"zero-offset pointer-slot DEFINE collapsed to a deref-write "
            f"(`bucket = table->bucket` rendered as `bucket->data = *...`):\n"
            f"{dropped}")

    def test_hash_get_max_bucket_length_typed_inner_walk(
            self, examples_dir: Path) -> None:
        """``hash_get_max_bucket_length`` walks an inner ``cursor=cursor->next``
        list off the bucket; both the outer (``++``) and inner (``->next``) cursors
        must be typed ``hash_entry*`` -- the +8 ``->next`` deref and the +16 ``++``
        both survive."""
        if not _idalib():
            pytest.skip("idalib unavailable")
        dropped = _drop_only(examples_dir, "hash_get_max_bucket_length")

        assert "hash_entry" in dropped, f"cursor not typed:\n{dropped}"
        assert "->next" in dropped, (
            f"inner list-walk `->next` deref lost:\n{dropped}")
        assert "->data" in dropped, f"`->data` deref lost:\n{dropped}"
        assert "memcpy" not in dropped, (
            f"struct-pointer ++ lowered as a memcpy:\n{dropped}")

    def test_hash_insert_if_absent_typed_bucket(
            self, examples_dir: Path) -> None:
        """``hash_insert_if_absent`` derefs ``bucket->data`` / ``bucket->next`` and
        links ``new_entry`` (``new_entry->data`` / ``new_entry->next``) -- the
        zero-offset deref + the typed cursor make these ``->`` field accesses, not
        ``*(_QWORD*)`` blobs.

        ``bucket`` here is the ``hash_find_entry(&bucket)`` out-param and ``new_entry``
        is the ``allocate_entry`` result (a register, not a stack cursor), so the
        struct TYPE NAME does not appear in the rendered BODY -- native itself renders
        only the ``->`` field accesses (the type shows in the collapsed declarations).
        The faithful signature is therefore the field-deref structure: at least four
        ``->data``/``->next`` accesses and NO raw ``*(_QWORD *)`` bucket blob."""
        if not _idalib():
            pytest.skip("idalib unavailable")
        dropped = _drop_only(examples_dir, "hash_insert_if_absent")

        assert "->data" in dropped, f"`->data` deref lost:\n{dropped}"
        assert "->next" in dropped, f"`->next` deref lost:\n{dropped}"
        # bucket->{data,next} + new_entry->{data,next}: the typed field accesses
        # native renders. A blob walk would collapse these to *(_QWORD *) offsets.
        assert dropped.count("->") >= 4, (
            f"field-deref cursor walk collapsed (typed `->` accesses lost):\n"
            f"{dropped}")
        assert "memcpy" not in dropped, f"spurious memcpy:\n{dropped}"
