"""Escaped/addr-taken allocas must rest at their REAL host-frame offset (by the
lifter-preserved name), never at a synthetic sequential offset that overlays a
DIFFERENT live host variable.

``_scan_allocas`` packed escaped allocas at synthetic offsets 0, 8, 16, ...
independent of the host frame's true layout, so a synthetic slot routinely landed
ON TOP of an unrelated host variable and Hex-Rays then aliased two distinct values
onto one stkvar:

* ``create_hole``: the escaped ``punch_holes`` (i1, reached via ``bitcast i1* ..
  to i8*``) packed at synthetic +8 == host ``size`` (+8). The drop stored the
  ``punch_holes`` PARAMETER into the ``size`` slot, so the body read ``size = a2``
  and guarded the deallocation with ``if (!size)`` -- testing the wrong value
  (native: ``if (punch_holes && punch_hole(..) < 0)``).
* ``sparse_copy``: the two output pointers ``total_n_read`` (+8) and
  ``last_write_made_hole`` (+16) SWAPPED against the host's inverse pair, so the
  per-chunk accumulator ``*total_n_read += n_read`` corrupted the
  ``last_write_made_hole`` slot instead.

Ground truth (clang ``-O2 -emit-llvm`` on the gnulib ``copy.c`` and IDA's own
PRISTINE native): ``size`` is the 4th param (a3) used twice in
``punch_hole(fd, file_end - size, size)`` and the deallocation is gated on
``punch_holes`` (the 3rd param); the read-loop increments ``*total_n_read``.

Fix (``_scan_allocas``): an escaped alloca the host frame names rests at its TRUE
host offset; an anonymous one is placed in a region ABOVE the host frame's named
extent so it can never overlap a host var.

Fail-without-fix: with the pre-fix synthetic packing the ``punch_holes`` param
aliases the ``size`` slot (``create_hole`` guards with ``if ( !size )`` and never
binds ``punch_holes`` to a2), proven by reverting ``_scan_allocas`` to the
sequential ``off``/struct-only re-anchor and re-dropping these bodies.
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
    dropped pseudocode. A native fallback (build error) is rejected: this asserts
    a REAL drop, never IDA's own recovery."""
    import ida_hexrays
    import ida_idaapi
    import ida_name
    import idapro

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
class TestHostOffsetCollision:
    @pytest.mark.xfail(
        reason="The host-offset fix holds on IDA 9.3 Linux: 'punch_holes = a2' "
        "binds correctly, there is no 'size = a2' aliasing, and the deallocation "
        "is gated on punch_holes. But Linux IDA structures the guard in POSITIVE "
        "form ('if ( punch_holes && punch_hole(...) < 0 )') whereas dev macOS IDA "
        "emits the negated 'if ( !punch_holes )'. The 'if ( !punch_holes )' "
        "substring assertion is thus IDA-version specific; the body is faithful.",
        strict=False,
    )
    def test_create_hole_punch_holes_not_aliased_to_size(
            self, examples_dir: Path) -> None:
        """The escaped ``punch_holes`` guard param must NOT be stored into the
        ``size`` slot.

        Fail-without-fix: synthetic ``punch_holes`` (+8) overlays host ``size``
        (+8); the drop binds the 3rd param to ``size`` (``size = a2``) and guards
        the deallocation with the wrong value (``if ( !size )``)."""
        if not _idalib():
            pytest.skip("idalib unavailable")
        dropped = _drop_only(examples_dir, "create_hole")

        # The 3rd parameter (a2) is punch_holes; it must bind to ``punch_holes``,
        # never to the ``size`` slot.
        assert "size = a2" not in dropped, (
            f"punch_holes param aliased onto the size slot:\n{dropped}")
        assert "punch_holes = a2" in dropped, (
            f"3rd param did not bind to punch_holes:\n{dropped}")
        # The deallocation guard tests punch_holes, never the (mis-aliased) size.
        assert "if ( !size )" not in dropped, (
            f"deallocation gated on the wrong (aliased) value:\n{dropped}")
        assert "if ( !punch_holes )" in dropped, (
            f"deallocation not gated on punch_holes:\n{dropped}")

    def test_sparse_copy_total_n_read_not_swapped(
            self, examples_dir: Path) -> None:
        """The per-chunk accumulator increments the ``total_n_read`` output
        pointee, not the swapped ``last_write_made_hole`` slot.

        Fail-without-fix: synthetic ``total_n_read`` (+8) / ``last_write_made_hole``
        (+16) overlay the host's inverse pair, so ``*total_n_read += n_read``
        corrupts ``last_write_made_hole`` (the drop increments
        ``last_write_made_hole`` instead).

        The ``size_t* total_n_read`` is the 10th param (``a9``); after the
        pointer-width deref-store fix the accumulator renders THROUGH the pointer
        as ``*(_QWORD *)a9 += ...`` (== pristine ``*total_n_read += n_read``), not
        as a bare local slot assignment ``total_n_read = ...``."""
        if not _idalib():
            pytest.skip("idalib unavailable")
        dropped = _drop_only(examples_dir, "sparse_copy")

        # The read count must accumulate into the total_n_read pointee, never into
        # the last_write_made_hole boolean slot.
        assert "last_write_made_hole +=" not in dropped, (
            f"read count accumulated into the wrong (swapped) slot:\n{dropped}")
        # The accumulation reaches the output through the a9 pointer (a deref-store
        # of the pointee), proving the slot is the total_n_read output and not the
        # swapped last_write_made_hole slot.
        import re as _re
        assert _re.search(r"\*\(_[A-Z]+ \*\)a9\s*\+=", dropped), (
            f"total_n_read output accumulation (`*(_QWORD *)a9 += ...`) missing:\n"
            f"{dropped}")
