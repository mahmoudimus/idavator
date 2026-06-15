"""INTERR 50342 -- converging return-value defs at a body-less ``ret`` merge.

A multi-block body is synthesized at the SHARED function-entry ea. Hex-Rays
value-numbers a definition by (location, size, ea); when several return paths
produce the return value with same-ea defs (``mov #-1, eax``; ``mov call
errno_fail(), eax``; a promoted-slot store) and CONVERGE -- directly or via
propagation -- at the lifter's single body-less ``%r=load funcresult; ret %r``
block, those same-(loc,ea) defs COLLIDE at the fake STOP. The def's (loc,size) is
not uniquely registered, so the verifier asserts INTERR 50342 (at MMAT_GLBOPT2 /
GLBOPT3) and the plain drop fails LATE.

The FIX is a scoped, faithful retry (``LLVMDropConverter.drop``): on a late 50342
failure, the SAME module is rebuilt with ``_distinct_segment_eas`` so every
segment is anchored at its block's real start ea (mirroring native, which never
reuses one ea for the body). The converging defs are then individually
numberable and the body reaches MMAT_LVARS. The retry runs BEFORE the SROA
fallback (so a faithful same-IR body beats SROA's coarser reshape) and ONLY when
the plain build already failed -- every currently-passing function is untouched.

``clone_quoting_options`` is the canonical clean win: a converging-return 50342
case the retry recovers FAITHFULLY (canonically == native). ``renameatu`` and
``do_copy`` also reach MMAT_LVARS under the retry (50342 cleared) but carry an
INDEPENDENT lifter divergence -- a 0-arg ``renameat2()`` syscall whose incoming
fd args are spilled by arg-preservation (rendered ``renameat2(0, ...)``), and a
``_DWORD*``-typed pointer arg walked at 2x stride / dropped ``__readfsqword`` --
so the B5 decline gate correctly routes THEM to native fallback. Those declines
are not 50342 regressions: the structure is recovered; the gate vetoes the body
on the orthogonal divergence. See memory ida_optimize_global_cfg_kill.
"""
from __future__ import annotations

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


@pytest.mark.ida
class TestDistinctEa50342:
    def _drop(self, examples_dir: Path, name: str):
        """Drop ``name`` into a FRESH copy of ``cp``; return (conv, cfunc-or-None)."""
        import ida_hexrays
        import ida_idaapi
        import ida_name
        import idapro

        binary = examples_dir / "cp"
        ir_path = examples_dir / "cp.ll"
        if not (binary.exists() and ir_path.exists()):
            pytest.skip("missing cp / cp.ll")

        from idavator.llvm_drop import LLVMDropConverter

        tmp = Path(tempfile.mkdtemp(prefix="distinct_ea_"))
        try:
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
                # Surface a fresh decompile desc so a LATE INTERR is visible to the
                # asserts (cf is None on a build failure; desc tells 50342 apart).
                hf = ida_hexrays.hexrays_failure_t()
                ida_hexrays.decompile(ea, hf)
                return conv, (str(cf) if cf is not None else None), hf.desc()
            finally:
                idapro.close_database()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_clone_quoting_options_distinct_ea_faithful(
            self, examples_dir: Path) -> None:
        """The converging-return 50342 exemplar recovers FAITHFULLY via the
        distinct-ea retry: a real body ships (not declined) through the
        DISTINCT-EA-RETRY path, and it is the ``xmemdup(o ? o : &default, 0x38)``
        clone -- not an INTERR, not a native fallback."""
        if not _idalib():
            pytest.skip("idalib unavailable")
        from idavator import oracle

        conv, body, _desc = self._drop(examples_dir, "clone_quoting_options")
        assert body is not None, "clone_quoting_options: no body (INTERR/decline)"
        assert conv.last_build_path == "DISTINCT-EA-RETRY", (
            f"expected the distinct-ea retry path, got {conv.last_build_path}")
        assert conv.last_primary_late_interr == 50342, (
            "expected the PRIMARY path to fail LATE on 50342 "
            f"(got {conv.last_primary_late_interr})")
        assert not conv.last_declined_divergent, "faithful body must not decline"
        assert "xmemdup(" in body, f"clone body lost xmemdup:\n{body}"
        # Canonically equivalent to the native decompile (variable names aside).
        import ida_hexrays
        import ida_idaapi
        import ida_name
        import idapro

        binary = examples_dir / "cp"
        tmp = Path(tempfile.mkdtemp(prefix="clone_native_"))
        try:
            dst = tmp / "cp"
            shutil.copy(binary, dst)
            idapro.open_database(str(dst), True)
            try:
                assert ida_hexrays.init_hexrays_plugin()
                ea = ida_name.get_name_ea(
                    ida_idaapi.BADADDR, "clone_quoting_options")
                native = str(ida_hexrays.decompile(ea))
            finally:
                idapro.close_database()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        if oracle.clang_available():
            assert oracle.matches(native, body), (
                f"distinct-ea body diverges from native:\n{body}")

    def test_renameatu_no_late_interr_50342(self, examples_dir: Path) -> None:
        """``renameatu`` (an 11-pred body-less return merge) must NO LONGER fail
        LATE with an unhandled 50342 surfacing on the host: the distinct-ea retry
        clears the value-number collision (the body reaches MMAT_LVARS). The B5
        gate then DECLINES it to native fallback on the orthogonal 0-arg
        ``renameat2`` divergence -- that decline is correct, not a 50342 leak."""
        if not _idalib():
            pytest.skip("idalib unavailable")
        conv, _body, desc = self._drop(examples_dir, "renameatu")
        assert conv.last_primary_late_interr == 50342, (
            "renameatu's PRIMARY path should fail LATE on 50342 "
            f"(else the fixture no longer exercises the shape): {desc}")
        # The retry built a body (50342 cleared); the decline gate then vetoed it.
        assert conv.last_build_path == "DISTINCT-EA-RETRY", (
            f"expected distinct-ea retry to reach a body, got "
            f"{conv.last_build_path}")
        assert conv.last_declined_divergent, (
            "renameatu's recovered body diverges (0-arg renameat2) -> must DECLINE")

    def test_do_copy_no_late_interr_50342(self, examples_dir: Path) -> None:
        """``do_copy`` (an inner body-less merge, not the ret-block) likewise has
        its 50342 collision cleared by the distinct-ea retry (reaches MMAT_LVARS);
        the recovered body declines on the orthogonal pointer-stride / dropped-
        canary divergence. Regression guard that the retry handles an inner merge,
        not only the ret-block convergence."""
        if not _idalib():
            pytest.skip("idalib unavailable")
        conv, _body, desc = self._drop(examples_dir, "do_copy")
        assert conv.last_primary_late_interr == 50342, (
            f"do_copy's PRIMARY path should fail LATE on 50342: {desc}")
        assert conv.last_build_path == "DISTINCT-EA-RETRY", (
            f"expected distinct-ea retry to reach a body, got "
            f"{conv.last_build_path}")
        assert conv.last_declined_divergent, (
            "do_copy's recovered body diverges -> must DECLINE to native")
