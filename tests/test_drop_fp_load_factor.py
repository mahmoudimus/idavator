"""Floating-point load-factor compute: FP-arithmetic lowering + the SSA-merge
store of an ``i2f`` conversion into the merge slot.

``hash_insert_if_absent`` (gnulib ``lib/hash.c``) gates a rehash on a floating
load factor::

    if ( (float)n_buckets_used > (float)(n_buckets) * tuning->growth_threshold )

IDA lifts ``(float)(int)n`` two ways depending on the sign of ``n`` (a signed->
float widening the compiler splits on the high bit): the non-negative arm is a
plain ``i2f`` (``sitofp i32 -> float``); the negative arm reconstructs the value
``(n&1)|(n>>1)`` and does ``(float)(int)v + (float)(int)v``. Both arms write a
single merge slot (``v5``/``v8``) that the load-factor compare then reads.

Two coupled defects produced a divergent drop:

 1. DROP FP-arith lowering (the proven ``_emit_value`` work): the drop had no
    ``sitofp``/``fadd``/``fmul``/``fptoui`` lowering, so it either failed to build
    the load-factor block or rendered the lossy integer surrogate.
 2. LIFT SSA-merge store (ticket ida-n2ja): the ida2llvm lifter's ``m_i2f`` (and
    ``m_f2i``/``m_f2u``/``m_u2f``) returned the conversion VALUE but never stored
    it to the destination ``d`` -- so the non-negative arm's ``sitofp`` result was
    never written to the merge slot. The stale ``cp.ll`` body therefore had a store
    ONLY in the negative arm; the merge slot held the doubled ``fadd`` value on
    BOTH paths and the simple ``(float)(int)n`` non-negative arm was absent. The fix
    stores the conversion result via ``_store_as`` (mirroring ``m_f2f``/``m_xdu``);
    ``cp.ll`` was re-spliced through the fixed lifter.

Ground truth (IDA PRISTINE native + clang ``-O2`` on gnulib ``hash.c``): the
load-factor block has BOTH merge arms -- ``v5 = (float)(int)n_buckets_used`` on the
non-negative path -- and the compare is an ORDERED FLOAT compare
``v5 > (float)(v8 * ...)``, not the lossy ``(unsigned int)v5 > (unsigned int)...``.

Fail-without-fix:

 * Without the FP-arith lowering the drop is a native fallback (``last_error`` set)
   or renders no ``(float)`` load-factor at all.
 * Without the ``m_i2f`` SSA-merge store (against the stale ``cp.ll``), the drop
   renders the doubled ``(float)(int)(..) + (float)(int)(..)`` on the merge slot but
   NEVER the simple non-negative ``(float)(int)n`` arm -- proven by reverting the
   ``m_i2f``/``m_f2i``/``m_f2u``/``m_u2f`` ``_store_as`` calls and re-splicing.
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
    a REAL drop, not the native decompile drop() falls back to on a build error."""
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
    tmp = Path(tempfile.mkdtemp(prefix="fp_load_factor_"))
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
class TestFloatLoadFactor:
    def test_hash_insert_if_absent_load_factor_both_arms(
            self, examples_dir: Path) -> None:
        """The float load-factor compute must drop BOTH SSA-merge arms and a real
        float compare.

        Fail-without-fix (DROP FP lowering): the block is a native fallback or has
        no ``(float)`` load-factor.
        Fail-without-fix (LIFT m_i2f store): the simple non-negative
        ``(float)(int)n`` arm is absent -- only the doubled fadd survives."""
        if not _idalib():
            pytest.skip("idalib unavailable")
        dropped = _drop_only(examples_dir, "hash_insert_if_absent")

        # FP-arith lowering reached the load-factor block: a float-typed compare
        # `... > (float)(...)` exists. (Native renders the ordered float compare
        # this way; the lossy integer surrogate would show `(unsigned int)` casts.)
        float_cmps = re.findall(r">\s*\(float\)\(", dropped)
        assert float_cmps, (
            "no `> (float)(...)` load-factor compare -- FP-arith lowering missing "
            f"or the compare fell back to the lossy integer surrogate:\n{dropped}")

        # The defining DEFECT-1 signature: the NON-NEGATIVE merge arm is a SIMPLE
        # `(float)(int)<value>` -- NOT a doubled `(float)(int)(..) + (float)(int)(..)`
        # sum. Find a `(float)(int)EXPR` that is not immediately summed with another
        # `(float)(int)`. The stale body (no m_i2f store) had ONLY the doubled form.
        simple_arm = re.search(
            r"\(float\)\(int\)(\w+|\([^()]*\))\s*;", dropped)
        assert simple_arm, (
            "no SIMPLE non-negative `(float)(int)n` merge arm -- the m_i2f "
            "conversion result was never stored to the merge slot, so only the "
            f"doubled negative-arm fadd survived:\n{dropped}")

        # The lossy integer-compare surrogate must NOT appear on the load factor:
        # a float comparand cast to `(unsigned int)` for an integer icmp is the
        # DEFECT-2 signature the FP float-compare recovery removes.
        assert "(unsigned int)v" not in dropped or float_cmps, (
            f"load-factor compare rendered as a lossy integer compare:\n{dropped}")

    def test_hash_insert_if_absent_rehash_block_present(
            self, examples_dir: Path) -> None:
        """The rehash trigger guarded by the load factor must drop -- ``check_tuning``
        + ``hash_rehash`` + the ``hash_find_entry``-then-``abort`` re-probe. Without the
        FP load-factor compute building, this whole block was absent from the drop."""
        if not _idalib():
            pytest.skip("idalib unavailable")
        dropped = _drop_only(examples_dir, "hash_insert_if_absent")

        assert "check_tuning" in dropped, (
            f"check_tuning call (load-factor rehash trigger) absent:\n{dropped}")
        assert "hash_rehash" in dropped, (
            f"hash_rehash call absent -- the rehash block was dropped:\n{dropped}")
        # the float->u64 size conversion idiom feeding hash_rehash (native's
        # `(unsigned int)(int)(float)(v - 9.22e18) ^ 0x8000000000000000`).
        assert "0x8000000000000000" in dropped, (
            f"float->u64 rehash-size conversion idiom absent:\n{dropped}")
