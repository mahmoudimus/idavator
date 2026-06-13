"""In-register call argument WIDTH comes from the callee's IDB formal, not the
lifted IR operand type (ticket ida-k0ug).

``ida2llvm`` declares ``memset`` with a NARROW prototype --
``declare i8* @memset(i8*, i32, i32)`` -- so the size operand lifts as an ``i32``
(4 bytes). IDA's real ``memset`` takes ``size_t n`` (8 bytes). The drop's
``_emit_call`` used to size the size-arg register from the IR operand type, emitting
a 4-byte ``mov #0x50, edx`` into the 8-byte ``size_t`` use. The high dword of rdx is
then undefined, Hex-Rays cannot fold the constant, and it materializes the partial
def as a separate temp:

    LODWORD(v1) = 0x50;        // <- the truncation pathology
    memset(a0, 0, v1);

which also trips "local variable allocation has failed" on the larger callers.

The fix widens each in-register arg to its callee IDB formal-param size (m_xdu a
narrower reg / widen a number) -- mirroring ``_emit_call_stackargs``' existing
formal-size widening of the 7th+ args. With it, the constant folds and ``memset``
renders with the full size argument inline:

    memset(a0, 0, 0x50u);      // folded, matches native sizeof(cp_options)

Fail-without-fix: against the pre-fix narrow-width emission ``cp_options_default``'s
drop carries the ``LODWORD(v1) = 0x50; memset(a0, 0, v1)`` split (the constant is NOT
folded into the call). The asserts pin the folded form and reject the split.
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
    """Drop ``name`` from cp.ll into its own ea in a FRESH session; return the
    dropped pseudocode. A native fallback (build error) is rejected -- this asserts
    a REAL drop, not a fallback that would render native's already-folded memset."""
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
    tmp = Path(tempfile.mkdtemp(prefix="call_argwidth_"))
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
class TestInRegisterCallArgWidth:
    def test_memset_size_arg_folds_to_formal_width(
            self, examples_dir: Path) -> None:
        """``cp_options_default`` zeroes its struct with a SINGLE folded call --
        ``memset(a0, 0, 0x50u)`` -- because the size arg is widened to the
        ``size_t`` formal. The pre-fix narrow ``edx`` left the high dword undef,
        forcing the unfoldable ``LODWORD(v1) = 0x50; memset(a0, 0, v1)`` split."""
        if not _idalib():
            pytest.skip("idalib unavailable")
        dropped = _drop_only(examples_dir, "cp_options_default")

        # The struct size (0x50 == sizeof(cp_options) == 80) is folded INLINE into
        # the memset call -- this is the post-fix faithful form.
        assert re.search(r"memset\(\s*a0\s*,\s*0\s*,\s*0x50u?\s*\)", dropped), (
            f"memset size constant not folded into the call (expected "
            f"`memset(a0, 0, 0x50u)`):\n{dropped}")

        # Fail-without-fix signature: the size constant materialized as a separate
        # truncated temp threaded into the call via a partial-width register.
        assert not re.search(r"LODWORD\(\w+\)\s*=\s*0x50\s*;", dropped), (
            f"size arg still emitted at narrow IR width -- the constant is split "
            f"into `LODWORD(v) = 0x50;` instead of folding (call-arg width not "
            f"widened to the size_t formal):\n{dropped}")

        # The partial-def of the size register must not trip the allocation banner.
        assert "allocation has failed" not in dropped, (
            f"narrow size-arg partial-def still trips the local-variable "
            f"allocation failure:\n{dropped}")
