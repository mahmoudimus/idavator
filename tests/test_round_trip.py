"""Round-trip harness: drop lifted-shaped IR and oracle-compare to the reference,
plus a drop-coverage report over the real lifted module (examples/cp.ll).

The coverage test is pure (llvmlite only) and documents the frontier; the drop
round-trips need idalib + libclang (skip otherwise).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from idavator.oracle import clang_available
from idavator.round_trip import module_coverage, round_trip


def _idalib() -> bool:
    try:
        import idapro  # noqa: F401

        return True
    except ImportError:
        return False


def _find_linear_host(ida_funcs, hx, idautils):
    for ea in idautils.Functions():
        f = ida_funcs.get_func(ea)
        if f is None or not (8 <= f.end_ea - f.start_ea <= 200):
            continue
        if hx.decompile(ea) is None:
            continue
        hf = hx.hexrays_failure_t()
        mbr = hx.mba_ranges_t()
        mbr.ranges.push_back(f)
        m = hx.gen_microcode(mbr, hf, None, hx.DECOMP_NO_WAIT, hx.MMAT_PREOPTIMIZED)
        if m is None:
            continue
        tails = {int(b.tail.opcode) for i in range(m.qty)
                 if (b := m.get_mblock(i)) is not None and b.tail is not None}
        conds = {hx.m_jcnd, hx.m_jz, hx.m_jnz, hx.m_jtbl}
        if hx.m_ret in tails and not (tails & conds):
            return ea
    return None


class TestCpLlCoverage:
    """Drop-coverage over the real lifted module -- a checked-in frontier fact."""

    def test_coverage_frontier(self, examples_dir: Path):
        ll = examples_dir / "cp.ll"
        if not ll.exists():
            pytest.skip("examples/cp.ll missing")
        cov = module_coverage(ll.read_text())
        assert cov.total > 300, cov.total
        # Scalar-slot and most address-taken allocas now drop, so well over half
        # the module is in the supported subset. The unsupported frontier has
        # shifted OFF allocas onto GEP shapes (struct/array indexing) and a tail
        # of unhandled opcodes.
        assert len(cov.supported) >= 200, len(cov.supported)
        hist = cov.reason_histogram()
        assert hist.get("gep", 0) > 50, hist          # GEP is now the top blocker
        assert hist.get("alloca", 0) < 25, hist        # allocas largely supported


# (ir, fn, reference_c) -- lifted-shape IR that stays in the supported subset and
# renders structurally close to the reference (no Hex-Rays temp explosion).
_ROUNDTRIP = [
    # GEP + load + arithmetic: p[2] + 7.
    ("define i32 @gep7(i32* %p) {\n"
     "entry:\n  %q = getelementptr i32, i32* %p, i64 2\n"
     "  %v = load i32, i32* %q\n  %r = add i32 %v, 7\n  ret i32 %r\n}\n",
     "gep7", "int gep7(int *p){ return p[2] + 7; }"),
    # call + consumed result: strlen(s) + 1 (segment split + result capture).
    ("declare i64 @strlen(i8*)\n"
     "define i64 @lp1(i8* %s) {\n"
     "entry:\n  %n = call i64 @strlen(i8* %s)\n"
     "  %r = add i64 %n, 1\n  ret i64 %r\n}\n",
     "lp1", "long long lp1(char *s){ return strlen(s) + 1; }"),
    # no-op casts (ptrtoint/inttoptr) + and: (((long)p) & 0xF).
    ("define i64 @lowbits(i8* %p) {\n"
     "entry:\n  %i = ptrtoint i8* %p to i64\n"
     "  %m = and i64 %i, 15\n  ret i64 %m\n}\n",
     "lowbits", "long long lowbits(char *p){ return ((long long)p) & 15; }"),
]


@pytest.mark.skipif(not clang_available(),
                    reason="IDA libclang /  clang_loader unavailable")
@pytest.mark.ida
class TestRealCpFunctionRoundTrip:
    """A TRUE round trip on real lifted IR: drop a cp.ll function's IR back into
    its own EA and assert the result is semantically identical to the original."""

    # Real coreutils functions whose lifted IR is in the supported subset and
    # round-trips byte-faithful through the oracle (globals + alloca + calls).
    # rpl_fclose drops VERBATIM (goto/label body) via the exact-text fast path.
    FAITHFUL = ["c_tolower", "quotearg_char", "rpl_fclose", "rpl_fflush",
                "xstrdup", "base_len"]

    @pytest.mark.parametrize("name", FAITHFUL)
    def test_real_function_round_trips(self, examples_dir: Path, name):
        if not _idalib():
            pytest.skip("idalib unavailable")
        import idapro
        import ida_hexrays as hx
        import ida_idaapi
        import ida_name

        if not (examples_dir / "cp").exists() or not (examples_dir / "cp.ll").exists():
            pytest.skip("missing example binary / cp.ll")

        idapro.open_database(str(examples_dir / "cp"), True)
        try:
            assert hx.init_hexrays_plugin()
            ea = ida_name.get_name_ea(ida_idaapi.BADADDR, name)
            assert ea != ida_idaapi.BADADDR, f"no EA for {name}"
            original = hx.decompile(ea)
            assert original is not None, f"baseline decompile of {name} failed"
            ir = (examples_dir / "cp.ll").read_text()
            res = round_trip(ir, name, ea, str(original))
            print(f"\n=== real round_trip @{name} ok={res.ok} "
                  f"err={res.error} ===\nledger={res.ledger}\n{res.dropped_c}")
            assert res.error is None, res.error
            assert res.ok, f"diverged: {res.ledger}"
        finally:
            idapro.close_database()

    def test_faithful_round_trip_count(self, examples_dir: Path):
        """Sweep the supported subset in one session and assert a large fraction
        round-trips faithfully -- the headline drop-fidelity metric over cp.ll."""
        if not _idalib():
            pytest.skip("idalib unavailable")
        import idapro
        import ida_hexrays as hx
        import ida_idaapi
        import ida_name

        if not (examples_dir / "cp").exists() or not (examples_dir / "cp.ll").exists():
            pytest.skip("missing example binary / cp.ll")

        idapro.open_database(str(examples_dir / "cp"), True)
        try:
            assert hx.init_hexrays_plugin()
            ir = (examples_dir / "cp.ll").read_text()
            cov = module_coverage(ir)
            faithful = 0
            for name in cov.supported:
                ea = ida_name.get_name_ea(ida_idaapi.BADADDR, name)
                if ea == ida_idaapi.BADADDR:
                    continue
                original = hx.decompile(ea)
                if original is None:
                    continue
                try:
                    res = round_trip(ir, name, ea, str(original))
                except Exception:  # noqa: BLE001 - unsupported construct
                    continue
                if res.ok and not res.error:
                    faithful += 1
            print(f"\nfaithful round-trips: {faithful}/{len(cov.supported)}")
            assert faithful >= 70, faithful
        finally:
            idapro.close_database()


@pytest.mark.skipif(not clang_available(),
                    reason="IDA libclang /  clang_loader unavailable")
@pytest.mark.ida
class TestDropRoundTrip:
    @pytest.mark.parametrize("ir, fn, ref_c", _ROUNDTRIP)
    def test_round_trip_is_faithful(self, examples_dir: Path, ir, fn, ref_c):
        if not _idalib():
            pytest.skip("idalib unavailable")
        import idapro
        import ida_funcs
        import ida_hexrays as hx
        import idautils

        if not (examples_dir / "cp").exists():
            pytest.skip("missing example binary")

        idapro.open_database(str(examples_dir / "cp"), True)
        try:
            assert hx.init_hexrays_plugin()
            host = _find_linear_host(ida_funcs, hx, idautils)
            assert host is not None
            res = round_trip(ir, fn, host, ref_c)
            print(f"\n=== round_trip @{fn} ok={res.ok} interr={res.interr} "
                  f"err={res.error} ===\n{res.dropped_c}\nledger={res.ledger}")
            assert res.error is None, res.error
            assert res.ok, f"round-trip diverged: {res.ledger}\n{res.dropped_c}"
        finally:
            idapro.close_database()
