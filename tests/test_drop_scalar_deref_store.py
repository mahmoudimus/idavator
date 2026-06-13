"""A pointer-WIDTH deref-store/load through a pointer-typed alloca slot must
DEREFERENCE the pointer, not overwrite the slot.

``sparse_copy``'s 10th parameter ``total_n_read`` is a ``size_t*`` (``i64*``).
The body does ``*total_n_read = 0`` then ``*total_n_read += n_read`` -- a
running counter written THROUGH the caller's pointer. The lifter emits each as
a no-op ``bitcast i64** %total_n_read to i64*`` followed by a pointer-WIDTH
``store i64`` / ``load i64`` of the POINTEE (an ``i64``, NOT a pointer):

    %.33 = bitcast i64** %total_n_read to i64*
    store i64 0, i64* %.33                 ; *total_n_read = 0

The drop's ``_ptr_deref_alias`` deref rule was gated on ``val_sz < 8`` (it only
fired for SUB-pointer fields like ``oa->style`` as ``i32``). A pointer-width
store of a non-pointer value fell through to the slot-write path and clobbered
the LOCAL pointer variable instead of the pointee -- the drop rendered
``total_n_read = nullptr;`` / ``total_n_read = (off_t*)((char*)total_n_read + n)``,
SILENTLY DROPPING the side-effect on the caller's ``*total_n_read``.

Ground truth (gnulib ``copy.c`` ``sparse_copy`` + IDA's own PRISTINE native):
``*total_n_read = 0;`` then ``*total_n_read += n_read;`` -- a write THROUGH the
pointer. ``clang -O2`` emits ``store i64 ..., ptr %total_n_read`` (a real store
to the pointee), never an assignment to the pointer slot.

Fix (``llvm_drop`` load/store of a ``_ptr_deref_alias``): the deref-vs-slot
distinguisher is the value's TYPE, not its width. A NON-pointer value of ANY
width (``i8`` field OR a pointer-width ``i64`` ``*p = 0``) is a deref; only a
POINTER value DEFINES the slot (``oa = &default``, ``bucket = *table``). The
``val_sz < 8`` / ``out_sz < 8`` width gate was dropped, keeping the
``_is_ptr_type`` type guard.

Fail-without-fix: with the ``val_sz < 8`` gate the pointer-width
``*total_n_read = 0`` writes the SLOT, so the drop renders the bare-name slot
assignment ``total_n_read = ...`` and NO ``*(_QWORD *)a9`` deref-store (proven
by restoring the width gate and re-dropping ``sparse_copy``).
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


def _paths(examples_dir: Path):
    binary = examples_dir / "cp"
    ir_path = examples_dir / "cp.ll"
    if not (binary.exists() and ir_path.exists()):
        pytest.skip("missing cp / cp.ll")
    return binary, ir_path


def _drop_only(examples_dir: Path, name: str) -> str:
    """Drop ``name`` from cp.ll at its own ea in a FRESH session and return the
    dropped pseudocode. Nothing decompiles the ea first (a prior decompile
    perturbs the lvar cache and reshapes the drop -- idalib non-determinism). A
    native fallback (build error) is rejected: this asserts a REAL drop."""
    import ida_hexrays
    import ida_idaapi
    import ida_name
    import idapro

    binary, ir_path = _paths(examples_dir)
    from idavator.llvm_drop import LLVMDropConverter

    # PRISTINE per-drop IDB: copy the binary to a throwaway dir so the drop's
    # _force_prototype set_types (saved by close_database) never persists into the
    # shared examples/cp.i64 -- forced-prototype writes accumulate across runs and
    # poison the native baseline for later cases. cp.ll stays the real read-only IR.
    tmp = Path(tempfile.mkdtemp(prefix="scalar_deref_"))
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
class TestScalarDerefStoreThroughPointer:
    def test_sparse_copy_writes_through_total_n_read(
            self, examples_dir: Path) -> None:
        """``sparse_copy`` writes the running counter THROUGH the ``size_t*``
        param (``*total_n_read = 0`` / ``+= n_read``), not into the local
        pointer slot.

        Fail-without-fix: the ``val_sz < 8`` deref gate skips the pointer-width
        store, so the drop assigns the slot -- ``total_n_read = nullptr`` /
        ``total_n_read = (off_t *)(... + n)`` -- and there is NO deref-store of
        the pointee."""
        if not _idalib():
            pytest.skip("idalib unavailable")
        dropped = _drop_only(examples_dir, "sparse_copy")

        # Strip the signature/declaration lines so the param NAME in the prototype
        # (``off_t *total_n_read``) is not mistaken for a body slot-assignment.
        body = "\n".join(
            ln
            for ln in dropped.splitlines()
            if "(" not in ln or "=" in ln or ")" in ln.split("(")[0]
        )

        # The pointee is written through the pointer at pointer width: the 10th
        # arg ``a9`` (the ``size_t* total_n_read``) is dereferenced and zeroed,
        # then accumulated. The pristine native renders this as ``*total_n_read``;
        # the drop (no recovered param name) renders ``*(_QWORD *)a9`` -- both are
        # a deref-STORE of the pointee.
        m_zero = re.search(r"\*\(_[A-Z]+ \*\)a9\s*=\s*0\s*;", dropped)
        assert m_zero is not None, (
            "pointer-width deref-store lost -- expected `*(_QWORD *)a9 = 0;` "
            f"(i.e. `*total_n_read = 0`):\n{dropped}")
        m_acc = re.search(r"\*\(_[A-Z]+ \*\)a9\s*\+=", dropped)
        assert m_acc is not None, (
            "running-counter deref-store lost -- expected `*(_QWORD *)a9 += ...;` "
            f"(i.e. `*total_n_read += n_read`):\n{dropped}")

        # The bug clobbers the LOCAL pointer instead: a bare slot assignment to a
        # name rooted at the pointer (``total_n_read = nullptr`` /
        # ``total_n_read = (off_t *)(...)``). A correct drop NEVER assigns such a
        # slot for these two statements -- it always derefs.
        assert not re.search(r"^\s*total_n_read\s*=", body, re.MULTILINE), (
            "slot clobbered -- the pointer-width store wrote the local pointer "
            f"`total_n_read = ...` instead of the pointee:\n{dropped}")
        assert "= nullptr;" not in body or "*(_" in dropped, (
            "the deref-store collapsed to a null slot assignment "
            f"(`= nullptr`):\n{dropped}")
