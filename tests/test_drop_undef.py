"""``undef`` operand resolution (don't-care -> zero).

An LLVM ``undef`` value is a don't-care. The lifter's SROA fallback leaves them
on dead phi incomings, masked-insert leftovers (``and i32 undef, 255``), and a
``ret <ty> undef`` whose path is pruned. ``_desc`` resolves an ``undef`` operand
to ``("num", 0, size)`` of the operand's width -- any concrete value is correct,
and 0 keeps the value-numbering trivial.

Without the fix, ``_desc`` raised ``ValueError: unhandled operand 'i32 undef'``
(/``ptr undef``/``i64 undef``) and the drop failed. This guard FAILS when the fix
is stashed.
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


# `undef` consumed as a real operand: `%r = or i32 %a, undef` -> with the fix,
# undef resolves to 0, so r = a | 0 = a (the `or` is emitted; no build error).
# Without the fix, `_desc(undef)` raises -> last_error set, cf is None.
PROBE_OR_UNDEF = """
define i32 @probe(i32 %a) {
entry:
  %r = or i32 %a, undef
  ret i32 %r
}
"""

# `undef` as the returned value directly (mirrors `ret ptr undef` in the SROA'd
# version_etc_va / get_nonce). With the fix the return slot is 0; without it the
# operand raises.
PROBE_RET_UNDEF = """
define i64 @probe(i64 %a) {
entry:
  ret i64 undef
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
class TestDropUndef:
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

    def test_undef_as_binop_operand(self, examples_dir: Path) -> None:
        if not _idalib():
            pytest.skip("idalib unavailable")
        conv, txt = self._drop_probe(examples_dir, PROBE_OR_UNDEF)
        # The operand must be HANDLED (no 'unhandled operand undef' build error).
        assert conv.last_error is None, conv.last_error
        assert conv.last_interr is None, f"INTERR {conv.last_interr}"
        assert txt is not None, "decompile returned None"
        assert "undef" not in txt, f"undef leaked into output:\n{txt}"

    def test_undef_as_return_value(self, examples_dir: Path) -> None:
        if not _idalib():
            pytest.skip("idalib unavailable")
        conv, _txt = self._drop_probe(examples_dir, PROBE_RET_UNDEF)
        assert conv.last_error is None, conv.last_error
        assert conv.last_interr is None, f"INTERR {conv.last_interr}"

    def test_desc_resolves_undef_to_zero(self, examples_dir: Path) -> None:
        # Direct unit-ish check: _desc maps an `undef` operand to zero of width.
        if not _idalib():
            pytest.skip("idalib unavailable")
        import idapro

        from idavator.llvm_drop import LLVMDropConverter

        binary = examples_dir / "cp"
        if not binary.exists():
            pytest.skip("missing example binary: cp")
        idapro.open_database(str(binary), True)
        try:
            conv = LLVMDropConverter(PROBE_OR_UNDEF)
            fn = next(g for g in conv.module.functions if g.name == "probe")
            orins = next(i for bb in fn.blocks for i in bb.instructions
                         if i.opcode == "or")
            undef_op = list(orins.operands)[1]
            assert "undef" in str(undef_op)
            kind, val, size = conv._desc(undef_op, {}, 4)
            assert (kind, val) == ("num", 0), (kind, val, size)
            assert size == 4
        finally:
            idapro.close_database()
