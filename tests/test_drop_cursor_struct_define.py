"""A direct-bitcast pointer-WIDTH store into a STRUCT-pointer cursor slot must
DEFINE the pointer (advance the cursor), not write the struct pointee at offset 0.

``extent_scan_read`` (gnulib ``lib/extent-scan.c``) advances its ``last_ei``
cursor (``struct extent_info *``) several times by POINTER ARITHMETIC:

  last_ei = scan->ext_info + prev_idx;     // after xnrealloc
  last_ei = &scan->ext_info[si_0];         // new-extent arm
  last_ei = &scan->ext_info[--si_0 - 1];   // coalesce arm

The lifter lowers each to ``store i64 <ptrtoint(base)+24*idx>, bitcast(%last_ei
to i64*)`` -- the value is a NON-pointer i64 (pointer arithmetic that lost its
type through ``ptrtoint``+``add``), but it DEFINES the cursor. The drop's
``_ptr_deref_alias`` rule keyed deref-vs-define on the stored value's TYPE
(``_is_ptr_type``), so a non-pointer pointer-width value looked like a deref and
the drop emitted ``last_ei->ext_logical = <arith>`` -- corrupting the first field
of whatever ``last_ei`` happened to point at, and never advancing the cursor.

Ground truth (``clang -O2`` of the gnulib source + IDA's PRISTINE native): each is
a cursor pointer ASSIGN ``last_ei = scan->ext_info + idx``. A genuine first-field
deref (``last_ei->ext_logical = v``) LOADs the pointer from the slot FIRST, so its
bitcast is rooted at the load -- NOT at the alloca -- and is handled by the generic
stx path, never entering ``_ptr_deref_alias``. Thus a DIRECT-bitcast store into a
struct-pointer slot is unambiguously a DEFINE.

Fix (``llvm_drop``): a direct-bitcast pointer-WIDTH (8B) store of a non-pointer
value into a ``_ptr_alloca_pointee_struct`` slot DEFINES the pointer (falls
through to the slot-write path). A SCALAR-pointee slot (``size_t*`` ->
``*total_n_read = 0``) is NOT struct -> stays a deref (guarded by
``test_drop_scalar_deref_store``); a sub-pointer-width field store
(``oa->style = 10`` as i32) stays a deref (width != 8).

This also re-splices ``extent_scan_read`` through the current lifter (its stale
``cp.ll`` body lost the 3-arg ``ioctl(fd, FS_IOC_FIEMAP, &fiemap_buf)``, collapsed
the ``fm_flags``/``fm_extent_count``/``fm_length`` struct stores to offset 0, and
deref-wrote the first cursor anchor).

Fail-without-fix: the cursor-define stores drop as ``last_ei->ext_logical =
<arith>`` (a first-field deref-write) and ``last_ei`` is never reassigned by
arithmetic -- the cursor advance is silently lost.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.render_tolerance import search_with_ints


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
    perturbs the lvar cache -- idalib non-determinism). A native fallback (build
    error) is rejected: this asserts a REAL drop."""
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
class TestCursorStructDefine:
    def test_extent_scan_read_cursor_defines_not_derefs(
            self, examples_dir: Path) -> None:
        """``extent_scan_read`` advances the ``last_ei`` cursor by pointer
        arithmetic (``last_ei = scan->ext_info + idx``), a pointer DEFINE -- not a
        deref-write of the first field.

        Fail-without-fix: the pointer-width arithmetic store renders as
        ``last_ei->ext_logical = <arith>`` (first-field deref) and the cursor is
        never reassigned by arithmetic.

        The stride (24) is asserted by VALUE: IDA 9.3 Linux renders ``+ 24 * idx``,
        dev macOS IDA ``+ 0x18 * idx`` -- cosmetic render divergence; the
        cursor-define recovery is faithful either way."""
        if not _idalib():
            pytest.skip("idalib unavailable")
        dropped = _drop_only(examples_dir, "extent_scan_read")

        # The cursor is ADVANCED by an arithmetic pointer DEFINE: at least one
        # statement assigns the bare `last_ei` to (base + 24 * idx). The drop has no
        # recovered struct name for `a0`, so the base renders as
        # `*((_QWORD *)a0 + 5)` (== scan->ext_info) and the assign as
        # `last_ei = (extent_info *)(... + 24 * idx)` (24 in any base).
        assert search_with_ints(
            r"\blast_ei\s*=\s*\(extent_info \*\)\([^;]*\+\s*{stride}\s*\*",
            dropped, {"stride": 0x18}) is not None, (
            "cursor advance lost -- no `last_ei = (extent_info *)(base + 24 * "
            f"idx)` pointer define:\n{dropped}")

        # The bug wrote the first field instead of defining the pointer: the
        # cursor anchor / advances must NOT collapse to a deref-write of
        # ext_logical sourced from the ext_info base (`last_ei->ext_logical =
        # *((_QWORD *)a0 + 5)` was the pre-fix anchor corruption).
        assert "last_ei->ext_logical = *((_QWORD *)a0 + 5);" not in dropped, (
            "cursor anchor deref-wrote ext_logical instead of assigning the "
            f"pointer:\n{dropped}")

    def test_extent_scan_read_ioctl_has_buffer_arg(
            self, examples_dir: Path) -> None:
        """``ioctl`` carries its 3rd arg ``&fiemap_buf`` (the stale cp.ll lost it).

        Fail-without-fix: drops the 2-arg ``ioctl(*a0, 0xC020660B)`` -- the
        FIEMAP buffer pointer is never passed and the kernel writes nowhere."""
        if not _idalib():
            pytest.skip("idalib unavailable")
        dropped = _drop_only(examples_dir, "extent_scan_read")

        m = re.search(r"ioctl\(([^;]*)\);", dropped)
        assert m is not None, f"no ioctl call in extent_scan_read:\n{dropped}"
        assert "&fiemap_buf" in m.group(1), (
            "ioctl lost its 3rd arg `&fiemap_buf` (2-arg ioctl writes nowhere):\n"
            f"{dropped}")

    def test_extent_scan_read_fiemap_struct_stores(
            self, examples_dir: Path) -> None:
        """The fiemap request fields ``fm_flags`` / ``fm_extent_count`` /
        ``fm_length`` are all stored (the stale cp.ll collapsed them to offset 0,
        leaving only one survivor).

        Fail-without-fix: only a single ``fiemap_buf.f.fm_*`` store survives (the
        rest overwrote each other at offset 0).

        ``fm_extent_count``'s value (72) is asserted by VALUE: IDA 9.3 Linux renders
        ``= 72``, dev macOS IDA ``= 0x48`` -- cosmetic render divergence; the store
        is faithful either way."""
        if not _idalib():
            pytest.skip("idalib unavailable")
        dropped = _drop_only(examples_dir, "extent_scan_read")

        assert "fiemap_buf.f.fm_flags = " in dropped, (
            f"fm_flags store lost (collapsed to offset 0):\n{dropped}")
        assert search_with_ints(
            r"fiemap_buf\.f\.fm_extent_count = {n};", dropped,
            {"n": 0x48}) is not None, (
            f"fm_extent_count = 72 (0x48) store lost:\n{dropped}")
        assert "fiemap_buf.f.fm_length = " in dropped, (
            f"fm_length store lost:\n{dropped}")
