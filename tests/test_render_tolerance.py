"""Unit tests for the integer-render tolerance helpers (no IDA needed)."""
from __future__ import annotations

from tests.render_tolerance import (
    contains_int,
    count_int,
    int_renderings,
    search_with_ints,
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
