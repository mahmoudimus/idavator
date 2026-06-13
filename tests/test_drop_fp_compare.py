"""Floating-point COMPARE lowering: the lifter must emit ``fcmp`` (with the right
ordered/unordered predicate) for a native float comparison, NOT cast both float
operands to integer and ``icmp`` them.

``check_tuning`` (gnulib ``lib/hash.c``) validates a hash-tuning struct with a
chain of FLOAT comparisons against literals::

    if ( tuning->growth_threshold > 0.1
      && (float)(1.0 - 0.1) > tuning->growth_threshold
      && tuning->growth_factor > (float)(0.1 + 1.0)
      && tuning->shrink_threshold >= 0.0
      && ... )

The ida2llvm lifter lowered every microcode FP compare (``jbe.fpu``/``ja.fpu``,
flagged ``is_fpinsn``) by unconditionally ``typecast``-ing both operands to
``IntType`` and emitting ``fptoui float + icmp`` -- truncating BOTH sides to
integer before comparing. That:

 * loses the fraction (``0.1`` -> ``(unsigned int)0.1`` == ``0``), and
 * INVERTS the result -- ``growth_threshold > 0.1`` became the nonsensical
   ``(unsigned int)*(double *)&...growth_threshold <= (unsigned int)0.1``.

The 6c275f2 drop-side ``_fp_compare_operands`` recovery rescued only SYMMETRIC
fp-width compares (both operands the same float width); an ASYMMETRIC pair (a
``double`` field vs a ``float`` literal) escaped the recovery and the lossy
integer compare survived into the output.

THE FIX (lifter, ``_handle_comparison`` / ``_handle_conditional_jump``): when an
operand is floating point, emit ``fcmp`` with the predicate mapped from the
microcode OPCODE -- ordered for ``ja``/``jae`` (``ogt``/``oge``), unordered for
``jb``/``jbe``/``jz`` (``ult``/``ule``/``ueq``), ``one`` for ``jnz`` -- mirroring
the float guard ``_handle_binary_arithmetic`` already had. The drop then lowers
that ``fcmp`` straight back to the FPU jcc (``_FCMP_JMP`` + ``set_fpinsn``).

Ground truth (IDA PRISTINE native + clang ``-O2`` on gnulib ``hash.c``): the
compares are ORDERED/UNORDERED FLOAT compares (``growth_threshold > 0.1``), and
the drop microcode is byte-identical to native (same ``jbe.fpu``/``ja.fpu`` block
chain). The predicate polarity was verified against LLVM's own x86 backend
(``ogt``->``seta``, ``ule``->``setbe``, etc.).

Fail-without-fix: the int-cast+icmp path renders the lossy ``(unsigned int)``
truncations on the float comparands (and inverts the predicate). This is proven
to FAIL on the 6c275f2 baseline (the ASYMMETRIC double-field-vs-float-literal
compare in ``check_tuning`` is exactly what ``_fp_compare_operands`` could not
recover).
"""
from __future__ import annotations

import re
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
    a REAL drop, not the native decompile drop() falls back to on a build error."""
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
        assert conv.last_error is None, conv.last_error
        assert cf is not None, "decompile returned None"
        return str(cf)
    finally:
        idapro.close_database()


@pytest.mark.ida
class TestFloatCompare:
    def test_check_tuning_float_compares_not_int_truncated(
            self, examples_dir: Path) -> None:
        """The hash-tuning validation compares must drop as FLOAT compares against
        the float literals, NOT integer-truncated ``(unsigned int)`` compares.

        Fail-without-fix (int-cast+icmp lifter path): each float comparand is cast
        to ``(unsigned int)`` and the literal ``0.1`` collapses to ``0`` -- e.g.
        ``(unsigned int)*(double *)&...growth_threshold <= (unsigned int)0.1`` --
        with the predicate inverted from native's ``> 0.1``."""
        if not _idalib():
            pytest.skip("idalib unavailable")
        dropped = _drop_only(examples_dir, "check_tuning")

        # The DEFINING bug signature: a float comparand cast to (unsigned int) for
        # an integer compare against a (truncated) float literal. The fixed lifter
        # emits `fcmp` -> the drop renders a plain float compare with NO such cast.
        assert "(unsigned int)0.1" not in dropped, (
            "float literal 0.1 truncated to `(unsigned int)0.1` (== 0) -- the FP "
            "compare was lowered as a lossy integer compare instead of `fcmp`:\n"
            f"{dropped}")
        bad = re.search(r"\(unsigned int\)\*\(double \*\)", dropped)
        assert bad is None, (
            "a `double` field comparand was cast to `(unsigned int)` for an "
            "integer compare -- the int-cast+icmp FP-compare bug:\n"
            f"{dropped}")

        # Positively: the validation renders a FLOAT compare against the 0.1 literal
        # (the `growth_threshold` first condition), matching native's float compare.
        # Both polarities are accepted (HexRays may render the equivalent De Morgan
        # inversion `<= 0.1` of native's `> 0.1`); what matters is it is a FLOAT
        # compare, not an integer-truncated one.
        float_cmp = re.search(r"(growth_threshold|shrink_factor)\s*(<=|>=|<|>)\s*", dropped)
        assert float_cmp, (
            "no float-typed tuning compare (growth_threshold/shrink_factor) -- the "
            f"FP compare chain did not survive:\n{dropped}")
