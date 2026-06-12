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

# The lifter's RETURN SLOT shape: a `funcresult` alloca written on multiple paths
# (incl. the `bitcast funcresult to i64*; store 0` early-return) and read at a
# post-merge `ret` block, with a NORETURN call pruning one edge into that merge.
# This used to INTERR 50342 (the slot was materialised as a var read at the merge,
# instead of writing the return reg straight on each path). Modelled on cp!xrealloc.
PROBE_RSLOT = """
define i8* @rslot_noret(i8* %p, i64 %n) {
e:
  %fr = alloca i8*
  %pa = alloca i8*
  %nz = icmp ne i64 %n, 0
  br i1 %nz, label %doit, label %chkp
chkp:
  %pi = ptrtoint i8* %p to i64
  %pz = icmp eq i64 %pi, 0
  br i1 %pz, label %doit, label %early
early:
  %frz = bitcast i8** %fr to i64*
  store i64 0, i64* %frz
  br label %ret
doit:
  %r = call i8* @realloc(i8* %p, i64 %n)
  store i8* %r, i8** %pa
  %t = ptrtoint i8* %r to i64
  %ok = icmp ne i64 %t, 0
  br i1 %ok, label %merge, label %chkn
chkn:
  %z2 = icmp eq i64 %n, 0
  br i1 %z2, label %merge, label %die
die:
  call void @xalloc_die()
  br label %merge
merge:
  %m = load i8*, i8** %pa
  store i8* %m, i8** %fr
  br label %ret
ret:
  %v = load i8*, i8** %fr
  ret i8* %v
}
declare i8* @realloc(i8*, i64)
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

    def test_return_slot_promotion_no_interr_50342(
            self, examples_dir: Path) -> None:
        """A `funcresult` return slot read at a post-merge ret block, downstream
        of a noreturn BLT_0WAY block, used to INTERR 50342 (cf=None). Return-slot
        promotion writes the retval straight to the return reg on each path
        (matching native), clearing it. Regression for
        memory idavator_drop_noreturn_50342_rootcause."""
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
            for needed in ("xalloc_die", "realloc"):
                if ida_name.get_name_ea(
                        ida_idaapi.BADADDR, needed) == ida_idaapi.BADADDR:
                    pytest.skip(f"{needed} not in this binary")
            host = next((ea for ea in idautils.Functions()
                         if (f := ida_funcs.get_func(ea)) is not None
                         and int(getattr(f, "frsize", 0)) >= 16
                         and not (f.flags & ida_funcs.FUNC_NORET)
                         and ida_hexrays.decompile(ea) is not None), None)
            assert host is not None

            conv = LLVMDropConverter(PROBE_RSLOT)
            cf = conv.drop(host, "rslot_noret")
            assert conv.last_error is None, conv.last_error
            # 50342 fires late (after preoptimized), so it surfaces as cf=None.
            assert cf is not None, "decompile returned None (INTERR 50342?)"
            txt = str(cf)
            assert "xalloc_die" in txt, f"noreturn call dropped:\n{txt}"
        finally:
            idapro.close_database()
