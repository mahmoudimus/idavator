"""Noreturn-tail calls. A `__noreturn` callee (xalloc_die/abort/...) ends its
block with NO successor (BLT_0WAY); Hex-Rays INTERRs 50174-style ("should be
BLT_0WAY", 51774) if such a block is wired BLT_1WAY. The segment splitter stops
at a noreturn call -- the rest of the LLVM block is dead -- so no continuation is
built to orphan. See memory idavator_drop_call_construction / canary_gate.
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


# if (x == 0) abort();  return x;   -- abort() is the noreturn tail.
PROBE = """
define i64 @probe(i64 %x) {
entry:
  %c = icmp eq i64 %x, 0
  br i1 %c, label %bad, label %ok
bad:
  call void @xalloc_die()
  unreachable
ok:
  ret i64 %x
}
declare void @xalloc_die()
"""


@pytest.mark.ida
class TestNoreturnTail:
    def test_noreturn_call_is_blt0way(self, examples_dir: Path) -> None:
        if not _idalib():
            pytest.skip("idalib unavailable")
        import idapro
        import ida_funcs
        import ida_hexrays
        import ida_idaapi
        import ida_name
        import idautils

        binary = examples_dir / "cp"
        if not binary.exists():
            pytest.skip("missing example binary: cp")

        from idavator.llvm_drop import LLVMDropConverter

        idapro.open_database(str(binary), True)
        try:
            assert ida_hexrays.init_hexrays_plugin()
            if ida_name.get_name_ea(
                    ida_idaapi.BADADDR, "xalloc_die") == ida_idaapi.BADADDR:
                pytest.skip("xalloc_die not in this binary")
            host = next((ea for ea in idautils.Functions()
                         if (f := ida_funcs.get_func(ea)) is not None
                         and int(getattr(f, "frsize", 0)) >= 16
                         and not (f.flags & ida_funcs.FUNC_NORET)
                         and ida_hexrays.decompile(ea) is not None), None)
            assert host is not None

            conv = LLVMDropConverter(PROBE)
            cf = conv.drop(host, "probe")
            assert conv.last_error is None, conv.last_error
            # the BLT_0WAY regression: must NOT INTERR (51774 "should be BLT_0WAY").
            assert conv.last_interr is None, f"INTERR {conv.last_interr}"
            assert cf is not None, "decompile returned None"
            txt = str(cf)
            assert "xalloc_die" in txt, f"noreturn call dropped:\n{txt}"
            assert "return" in txt, txt
        finally:
            idapro.close_database()
