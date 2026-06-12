"""Tests for the libclang AST-canonicalization oracle + a drop round-trip.

The oracle parses C with IDA's libclang (via 's clang_loader); these tests
skip unless that is importable. The round-trip test additionally needs idalib.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from idavator.oracle import (
    clang_available,
    fidelity_ledger,
    matches,
)

requires_clang = pytest.mark.skipif(
    not clang_available(), reason="IDA libclang /  clang_loader unavailable")


@requires_clang
class TestAstEquivalence:
    def test_local_renaming_is_invariant(self):
        assert matches("int f(int x){ return x + 1; }",
                       "int g(int a0){ return a0 + 1; }")

    def test_constant_base_is_invariant(self):
        assert matches("int f(){ return 100; }", "int f(){ return 0x64; }")

    def test_commutativity_is_invariant(self):
        assert matches("int f(int x,int y){ return x + y; }",
                       "int f(int a,int b){ return b + a; }")

    def test_shift_folds_to_multiply(self):
        assert matches("int f(int x){ return 4 * x; }",
                       "int f(int a0){ return a0 << 2; }")

    def test_rshift_folds_to_divide(self):
        assert matches("int f(int x){ return x / 8; }",
                       "int f(int a0){ return a0 >> 3; }")

    def test_inverted_test_and_swapped_arms_match(self):
        a = "int f(int x){ if (x > 5) return 100; else return 200; }"
        b = "int g(int a0){ if (a0 <= 5) return 0xC8; else return 0x64; }"
        assert matches(a, b)

    def test_cosmetic_cast_is_transparent(self):
        assert matches("__int64 f(char *p){ return strlen(p); }",
                       "__int64 g(char *a0){ return strlen((const char *)a0); }")

    def test_different_operations_do_not_match(self):
        assert not matches("int f(int a,int b){ return a + b; }",
                           "int f(int a,int b){ return a - b; }")

    def test_different_callees_do_not_match(self):
        assert not matches("__int64 f(char *p){ return strlen(p); }",
                           "__int64 f(char *p){ return strnlen(p); }")

    def test_missing_branch_is_detected(self):
        full = "int f(int x){ if (x) return 1; else return 2; }"
        flat = "int f(int x){ return 2; }"
        assert not matches(full, flat)


@requires_clang
class TestFidelityLedger:
    def test_faithful_drop_has_empty_ledger(self):
        assert fidelity_ledger(
            "int f(int x){ if (x > 0) return x; else return 0; }",
            "int g(int a0){ if (a0 <= 0) return 0; else return a0; }") == {}

    def test_ledger_reports_divergence(self):
        led = fidelity_ledger("__int64 f(char *p){ return strlen(p); }",
                              "__int64 f(char *p){ return strnlen(p); }")
        assert led != {}
        assert "expected" in led and "actual" in led


# --- round-trip: drop LLVM IR, assert the rendered body matches the oracle -----

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


# (ir, fn, expected_c) -- expected_c is the human-written intent of the IR.
_ROUNDTRIP = [
    ("define i32 @f(i32 %x) {\n  %a = mul i32 %x, 3\n"
     "  %b = add i32 %a, 7\n  ret i32 %b\n}\n",
     "f", "int f(int x){ return 3 * x + 7; }"),
    ("define i32 @ife(i32 %x) {\nentry:\n  %c = icmp sgt i32 %x, 5\n"
     "  br i1 %c, label %big, label %small\n"
     "big:\n  ret i32 100\nsmall:\n  ret i32 200\n}\n",
     "ife", "int ife(int x){ if (x > 5) return 100; else return 200; }"),
]


@requires_clang
@pytest.mark.ida
class TestDropRoundTripOracle:
    @pytest.mark.parametrize("ir, fn, expected_c", _ROUNDTRIP)
    def test_drop_matches_oracle(self, examples_dir: Path, ir, fn, expected_c):
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
            from idavator.llvm_drop import LLVMDropConverter

            host = _find_linear_host(ida_funcs, hx, idautils)
            assert host is not None
            cf = LLVMDropConverter(ir).drop(host, fn)
            assert cf is not None
            actual = str(cf)
            led = fidelity_ledger(expected_c, actual)
            assert led == {}, f"drop diverged from oracle: {led}\n{actual}"
        finally:
            idapro.close_database()
