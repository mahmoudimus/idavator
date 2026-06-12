"""``select`` operand lowering (branchless ternary).

``%r = select i1 %c, T %a, T %b`` is a branchless ternary. The lifter's SROA
fallback (and instcombine) introduces three shapes, all lowered branchlessly so a
2-way merge never re-trips the noreturn-merge INTERR family:

- SHORT-CIRCUIT boolean: ``select i1 %c, i1 %a, i1 false`` == ``%c & %a``
  (and.cond) and ``select i1 %c, i1 true, i1 %b`` == ``%c | %b`` (or.cond) ->
  ``m_and`` / ``m_or`` on the 1-byte i1 operands (an icmp arm becomes a setcc).
- BOOLEAN MATERIALISE ``select i1 %c, <N> 1, <N> 0`` == ``zext c to N``.
- GENERAL ``select i1 %c, T %a, T %b`` -> ``b + ((a - b) & (0 - (T)c))``.

Without the fix, ``_emit_value`` had no ``select`` case and ``_desc`` raised
``ValueError: unhandled operand '%or.cond = select i1 ...'`` -- the drop failed
(``last_error`` set, ``cf`` is None). These guards FAIL when the fix is stashed.
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


# SHORT-CIRCUIT and.cond: select i1 %c1, i1 %c2, i1 false  ==  %c1 & %c2, where
# both conditions are icmp results (so each is materialised via a setcc). The
# result feeds the return -> the dropped C must show a bitwise `&` of the two
# comparisons, exactly as a native and.cond renders.
PROBE_SHORTCIRCUIT = """
define i32 @probe(i32 %a, i32 %b) {
entry:
  %c1 = icmp ne i32 %a, 0
  %c2 = icmp eq i32 %a, %b
  %and = select i1 %c1, i1 %c2, i1 false
  %r = zext i1 %and to i32
  ret i32 %r
}
"""

# GENERAL mux: select i1 %c, i64 %a, i64 %b with BOTH arms non-constant ->
# branchless blend b + ((a-b) & mask). The result feeds the return.
PROBE_GENERAL = """
define i64 @probe(i64 %a, i64 %b, i64 %x) {
entry:
  %c = icmp ult i64 %x, 10
  %sel = select i1 %c, i64 %a, i64 %b
  ret i64 %sel
}
"""


def _linear_host(ida_funcs, ida_hexrays):
    import idautils

    for ea in idautils.Functions():
        f = ida_funcs.get_func(ea)
        if f is None or (f.flags & ida_funcs.FUNC_NORET):
            continue
        if not (8 <= f.end_ea - f.start_ea <= 400):
            continue
        if ida_hexrays.decompile(ea) is not None:
            return ea
    return None


@pytest.mark.ida
class TestDropSelect:
    def _drop_probe(self, examples_dir: Path, probe: str):
        import idapro
        import ida_funcs
        import ida_hexrays

        from idavator.llvm_drop import LLVMDropConverter

        binary = examples_dir / "cp"
        if not binary.exists():
            pytest.skip("missing example binary: cp")
        idapro.open_database(str(binary), True)
        try:
            assert ida_hexrays.init_hexrays_plugin()
            host = _linear_host(ida_funcs, ida_hexrays)
            assert host is not None, "no linear host found"
            conv = LLVMDropConverter(probe)
            cf = conv.drop(host, "probe")
            return conv, (str(cf) if cf is not None else None)
        finally:
            idapro.close_database()

    def test_select_shortcircuit_and_cond(self, examples_dir: Path) -> None:
        if not _idalib():
            pytest.skip("idalib unavailable")
        conv, txt = self._drop_probe(examples_dir, PROBE_SHORTCIRCUIT)
        # The operand must be HANDLED (no 'unhandled operand select' build error).
        assert conv.last_error is None, conv.last_error
        assert conv.last_interr is None, f"INTERR {conv.last_interr}"
        assert txt is not None, "decompile returned None"
        # and.cond renders as a bitwise '&' of the two conditions (the lowered
        # m_and on the i1 setcc values) -- never a literal `select`.
        assert "&" in txt, f"expected bitwise-and short-circuit:\n{txt}"
        assert "select" not in txt, f"select leaked into output:\n{txt}"

    def test_select_general_branchless_mux(self, examples_dir: Path) -> None:
        if not _idalib():
            pytest.skip("idalib unavailable")
        conv, txt = self._drop_probe(examples_dir, PROBE_GENERAL)
        assert conv.last_error is None, conv.last_error
        assert conv.last_interr is None, f"INTERR {conv.last_interr}"
        assert txt is not None, "decompile returned None"
        # Hex-Rays re-folds the branchless blend back to a ternary `?:` (or a
        # masked form) -- the point is it BUILDS and renders, with no `select`.
        assert "select" not in txt, f"select leaked into output:\n{txt}"
