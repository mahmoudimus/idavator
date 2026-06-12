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

    def test_remember_copied_sroa_fallback_clears_50342(
            self, examples_dir: Path) -> None:
        """The REAL cp!remember_copied. Its lifted IR returns a `funcresult`
        alloca slot read at a post-merge `ret` block downstream of a noreturn
        (xalloc_die) -- the alloca-form return-slot promotion does NOT clear it
        (the slot is a struct-field load, not a clean scalar store), so the plain
        drop INTERRs 50342 (cf=None). The SROA fallback in drop() re-lifts from an
        SROA-optimized copy (the slot collapses to a return phi), and the return-
        phi promotion writes the retval straight to the return reg -> 50342 gone.

        Drops @remember_copied into its OWN ea so the result round-trips against
        the genuine reference. Asserts the real calls survive and NO `byte_`
        (the 50342 corruption / a wrong const-memory drop would surface as those).
        Proven to FAIL without the fix: `git stash` llvm_drop.py -> cf is None."""
        if not _idalib():
            pytest.skip("idalib unavailable")
        import idapro
        import ida_hexrays
        import ida_idaapi
        import ida_name

        binary = examples_dir / "cp"
        ir = examples_dir / "cp.ll"
        if not binary.exists() or not ir.exists():
            pytest.skip("missing example: cp / cp.ll")

        from idavator.llvm_drop import LLVMDropConverter

        idapro.open_database(str(binary), True)
        try:
            assert ida_hexrays.init_hexrays_plugin()
            for needed in ("xalloc_die", "xmalloc", "hash_insert"):
                if ida_name.get_name_ea(
                        ida_idaapi.BADADDR, needed) == ida_idaapi.BADADDR:
                    pytest.skip(f"{needed} not in this binary")
            ea = ida_name.get_name_ea(ida_idaapi.BADADDR, "remember_copied")
            if ea == ida_idaapi.BADADDR:
                pytest.skip("remember_copied not in this binary")

            conv = LLVMDropConverter(ir.read_text())
            cf = conv.drop(ea, "remember_copied")
            assert conv.last_error is None, conv.last_error
            # The 50342 surfaces late as cf=None; the SROA fallback must clear it.
            assert cf is not None, "decompile returned None (INTERR 50342?)"
            assert conv.last_interr is None, f"INTERR {conv.last_interr}"
            txt = str(cf)
            # Real calls must survive (a corrupt/garbage drop would lose them).
            for call in ("xmalloc", "hash_insert", "xalloc_die"):
                assert call in txt, f"missing call {call!r}:\n{txt}"
            # 50342 corruption / a wrong const-memory drop renders as byte_<addr>.
            assert "byte_" not in txt, f"corrupt drop (byte_):\n{txt}"
        finally:
            idapro.close_database()
