"""Unit tests for the integer-render tolerance helpers (no IDA needed)."""
from __future__ import annotations

from tests.render_tolerance import (
    contains_int,
    count_int,
    int_renderings,
    search_with_ints,
    structural_equiv,
    structural_norm,
)


class TestIntRenderings:
    def test_small_positive_decimal_and_hex(self):
        r = int_renderings(27)
        assert "27" in r and "0x1B" in r

    def test_minus_one_32bit(self):
        r = int_renderings(-1, width=32)
        assert "-1" in r and "0xFFFFFFFF" in r

    def test_minus_one_64bit(self):
        r = int_renderings(-1, width=64)
        assert "-1" in r and "0xFFFFFFFFFFFFFFFF" in r

    def test_unsigned_hex_maps_back_to_signed(self):
        # 0xFFFFFFFF given as the value should also offer -1 (signed view).
        r = int_renderings(0xFFFFFFFF, width=32)
        assert "-1" in r and "0xFFFFFFFF" in r and "4294967295" in r


class TestContainsInt:
    def test_decimal_render_found_for_hex_query(self):
        assert contains_int("x = *((_BYTE *)a0 + 27);", 0x1B)

    def test_hex_render_found_for_hex_query(self):
        assert contains_int("x = *((_BYTE *)a0 + 0x1B);", 0x1B)

    def test_suffixed_literal_matches(self):
        assert contains_int("fm_extent_count = 72u;", 0x48)
        assert contains_int("y = 24LL * i;", 24)

    def test_minus_one_either_base(self):
        assert contains_int("return -1;", 0xFFFFFFFF, width=32)
        assert contains_int("return 0xFFFFFFFF;", 0xFFFFFFFF, width=32)
        assert contains_int("return 0xFFFFFFFFFFFFFFFF;", -1, width=64)

    def test_not_a_substring_of_larger_number(self):
        # 27 must NOT match inside 127 or 270 or 0x27B.
        assert not contains_int("x = 127;", 0x1B)
        assert not contains_int("x = 270;", 0x1B)
        assert not contains_int("x = 0x12C;", 0x2C)  # 0x2C inside 0x12C? token-safe

    def test_absent_value_not_found(self):
        assert not contains_int("return 0;", 0x1B)


class TestCountInt:
    def test_counts_each_occurrence(self):
        assert count_int("*_errno_location() = 95; if (95) {}", 0x5F) == 2

    def test_zero_when_absent(self):
        assert count_int("return 0;", 0x5F) == 0


class TestSearchWithInts:
    def test_positional_constant_tolerant(self):
        m = search_with_ints(
            r"fm_extent_count = {n};", "fm_extent_count = 72;", {"n": 0x48})
        assert m is not None

    def test_positional_constant_hex(self):
        m = search_with_ints(
            r"fm_extent_count = {n};", "fm_extent_count = 0x48;", {"n": 0x48})
        assert m is not None

    def test_stride_in_pointer_arith(self):
        # base + 0x18 * idx  OR  base + 24 * idx
        for txt in ("last_ei = (extent_info *)(x + 0x18 * v4);",
                    "last_ei = (extent_info *)(x + 24 * v4);",
                    "last_ei = (extent_info *)(x + 24LL * v4);"):
            m = search_with_ints(
                r"last_ei = \(extent_info \*\)\([^;]*\+\s*{s}\s*\*", txt,
                {"s": 0x18})
            assert m is not None, txt

    def test_wrong_constant_rejected(self):
        m = search_with_ints(
            r"fm_extent_count = {n};", "fm_extent_count = 99;", {"n": 0x48})
        assert m is None


# A faithful weakly-typed (IR-drop) body and its richly-typed amd64-native twin,
# differing ONLY on the benign axes structural_equiv tolerates.
_DROP = (
    "int __fastcall f(_DWORD *a0, _DWORD *a1)\n{\n"
    "  int result; // eax\n"
    "  if ( !valid(a0) )\n"
    "    _assert_fail(\"m\", \"s.c\", 0xBC3u, \"f\");\n"
    "  g = (const char *)a0;\n"
    "  LOBYTE(result) = h((const char *)a0, (const char *)a1, a1);\n"
    "  return result;\n}")
_NATIVE = (
    "bool __cdecl f(const char *src, const char *dst)\n{\n"
    "  unsigned __int64 v3; // [rsp+8h]\n\n"
    "  v3 = __readfsqword(0x28u);\n"
    "  if ( !valid(src) )\n"
    "    __assert_fail(\"m\", \"s.c\", 0xBC3u, \"f\");\n"
    "  g = src;\n"
    "  return h(src, dst, dst);\n}")


class TestStructuralNorm:
    def test_identical_bodies_match(self):
        assert structural_norm(_DROP) == structural_norm(_DROP)

    def test_norm_strips_canary_and_is_stable(self):
        # The normalized native has no canary read and no BYREF decl noise.
        assert "__readfsqword" not in structural_norm(_NATIVE)
        assert "// " not in structural_norm(_NATIVE)


class TestStructuralEquiv:
    def test_benign_type_canary_underscore_split_tolerated(self):
        # The faithful drop and its richly-typed amd64 twin are equivalent: the
        # only deltas are the canary, casts/types, the __assert_fail underscore,
        # the LOBYTE-materialized return, and the a0/a1-vs-src/dst value split.
        assert structural_equiv(_DROP, _NATIVE)

    def test_reflexive(self):
        assert structural_equiv(_NATIVE, _NATIVE)

    def test_wrong_callee_rejected(self):
        bad = _DROP.replace("h((const char *)a0", "WRONG((const char *)a0")
        assert not structural_equiv(bad, _NATIVE)

    def test_missing_statement_rejected(self):
        bad = _DROP.replace("  g = (const char *)a0;\n", "")
        assert not structural_equiv(bad, _NATIVE)

    def test_extra_statement_rejected(self):
        bad = _DROP.replace("  return result;",
                            "  extra(a0);\n  return result;")
        assert not structural_equiv(bad, _NATIVE)

    def test_wrong_constant_rejected(self):
        bad = _DROP.replace("0xBC3u", "0xBC4u")
        assert not structural_equiv(bad, _NATIVE)

    def test_wrong_string_rejected(self):
        bad = _DROP.replace('"m"', '"DIFFERENT"')
        assert not structural_equiv(bad, _NATIVE)

    def test_value_merge_rejected(self):
        # The drop MERGING two distinct native values into one name is the UNSAFE
        # direction and must be rejected (weak typing only ever SPLITS).
        merge_native = "int f(int x, int y){ return p(x) + q(y); }"
        merge_drop = "int f(int z){ return p(z) + q(z); }"
        assert not structural_equiv(merge_drop, merge_native)

    def test_value_split_tolerated(self):
        # The drop SPLITTING one native value into two typed names is benign.
        split_native = "int f(int x){ return p(x) + q(x); }"
        split_drop = "int f(int x){ return p(x) + q(y); }"
        assert structural_equiv(split_drop, split_native)

    def test_field_access_vs_raw_offset_rejected(self):
        # A struct-field access vs raw pointer-offset arithmetic (the
        # transfer_entries amd64 divergence) is structural, not cosmetic.
        drop = "int f(_DWORD *a0){ return *((_QWORD *)a0 + 3); }"
        native = "int f(Hash_table *dst){ return dst->n_buckets_used; }"
        assert not structural_equiv(drop, native)

    def test_enum_scope_constant_rejected(self):
        a = "int f(){ return g(style::shell_always); }"
        b = "int f(){ return g(style::shell_never); }"
        assert not structural_equiv(a, b)

    def test_combined_vs_nested_branch_rejected(self):
        # `if (a < 0 && b)` vs nested `if (a<0){ if (b) {...} }` (the
        # create_hard_link amd64 divergence) is a real control-flow reshape.
        combined = "int f(int a,int b){ if (a < 0 && b) p(); return 0; }"
        nested = "int f(int a,int b){ if (a < 0){ if (b) p(); } return 0; }"
        assert not structural_equiv(combined, nested)
