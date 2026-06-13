"""Whole-struct COPY must not scalarize to its first field.

The frontend lowers a C struct assignment ``options = default_quoting_options``
(a 56-byte ``struct quoting_options``) as a WIDE-INTEGER load/store pair::

    %v = load  i448, ptr @default_quoting_options
         store i448 %v, ptr %options

``_type_size`` had no exact ``iN`` case: the substring scan missed ``i448`` and
fell through to the 4-byte default, so the copy SCALARIZED to ``options.style =
default_quoting_options.style`` -- the .style field (first i32) only. The other
fields (flags + the ``quote_these`` table + the quote pointers) were LOST, leaving
``options`` partially uninitialized when handed to ``set_char_quoting(&options,..)``
/ ``quotearg_n_options(..,&options)`` -> a miscompile.

Fix (drop-side struct-copy lowering): ``_type_size`` sizes any ``iN`` as
``ceil(N/8)``; an over-16-byte aggregate load/store (too wide for a kreg --
``alloc_kreg`` rejects >16) lowers as one mem-to-mem ``m_mov`` SRC_lvalue ->
DST_lvalue with both operands flagged ``set_udt()`` (so the verifier's
``is_valid_size`` check, INTERR 50757, is skipped for the struct-sized mov). This
is the microcode model of native's whole-struct assignment.

The exoneration gate held: ``clang -O2 -emit-llvm`` on the gnulib-faithful source
emits ``llvm.memcpy(... i64 56 ...)`` -- the native copy IS a whole-struct memcpy,
so the .style-only render was a real, recoverable DROP defect. Ticket ida-45rg
(goal ida-dvb4); see memory idavator_drop_correctness_coverage.
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


@pytest.mark.ida
class TestWholeStructCopy:
    def test_quotearg_char_mem_copies_whole_struct(
            self, examples_dir: Path) -> None:
        if not _idalib():
            pytest.skip("idalib unavailable")
        import idapro
        import ida_hexrays
        import ida_idaapi
        import ida_name

        binary = examples_dir / "cp"
        ir_path = examples_dir / "cp.ll"
        if not (binary.exists() and ir_path.exists()):
            pytest.skip("missing cp / cp.ll")

        from idavator.llvm_drop import LLVMDropConverter

        # PRISTINE per-drop IDB: copy the binary to a throwaway dir so the drop's
        # _force_prototype set_types (saved by close_database) never persists into
        # the shared examples/cp.i64 -- forced-prototype writes accumulate across
        # runs and poison the native baseline for later cases. cp.ll stays the real
        # read-only IR.
        tmp = Path(tempfile.mkdtemp(prefix="struct_copy_"))
        dst = tmp / "cp"
        shutil.copy(binary, dst)
        idapro.open_database(str(dst), True)
        try:
            assert ida_hexrays.init_hexrays_plugin()
            ea = ida_name.get_name_ea(
                ida_idaapi.BADADDR, "quotearg_char_mem")
            if ea == ida_idaapi.BADADDR:
                pytest.skip("quotearg_char_mem not in this binary")

            conv = LLVMDropConverter(ir_path.read_text())
            cf = conv.drop(ea, "quotearg_char_mem")
            # real_drop: a genuine idavator drop, NOT a native fallback.
            assert conv.last_error is None, conv.last_error
            assert cf is not None, "decompile returned None"
            txt = str(cf)
            # The whole-struct copy must be present...
            assert "options = default_quoting_options" in txt, (
                f"struct copy not rendered:\n{txt}")
            # ...and must NOT have scalarized to the .style field only
            # (the proven fail-without-fix signature).
            assert "options.style = default_quoting_options.style" not in txt, (
                f"struct copy SCALARIZED to .style:\n{txt}")
            # The downstream uses of the (now fully-initialised) struct survive.
            assert "set_char_quoting(&options" in txt, txt
            assert "&options)" in txt, txt
        finally:
            idapro.close_database()
            shutil.rmtree(tmp, ignore_errors=True)
