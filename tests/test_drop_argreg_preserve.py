"""Incoming arg-register VALUE preserved across a clobbering call.

A SysV integer argument lives in a caller-saved register (rdi/rsi/rdx/...). A
``call`` clobbers every such register -- its OWN argument setup overwrites them.
When a function reads an argument's value AFTER a call, the raw arg register is
stale. The lifter's plain IR spills each param to an alloca and reloads it (so
the value survives), but the SROA fallback promotes those allocas away, leaving
the param SSA value read directly from the now-clobbered register.

DROP BUG (ticket ida-k83m): the drop seeded ``vmap[arg] = (reg, rdi)`` and used
the raw register everywhere. For SROA-promoted ``remember_copied``,
``mov 0x18, rdi`` (the ``xmalloc(0x18)`` arg setup) clobbered ``rdi`` (the
``name`` param) BEFORE ``xstrdup`` read it -- and the decompiler rendered the
stale rdi (== 0x18) as ``xstrdup(&off_18)`` (a bogus global), not
``xstrdup(name)``. clang ``-O2`` of the gnulib ``remember_copied`` source shows
``call ptr @xstrdup(ptr %name)`` -- the param passed directly.

FIX: copy each incoming arg register whose value is read across a call into a
stable kreg at the entry block (before any call clobbers it); the decompiler
places the kreg in a callee-saved register / stack. Args used only before any
call keep the raw register (inert).

Fail-without-fix (all SROA-promoted, all read a reg arg across a call):
  * remember_copied -- ``xstrdup(&off_18)`` instead of ``xstrdup(name)``;
  * __xargmatch_internal -- ``argmatch_invalid(arg, arglist, ...)`` (the clobbered
    rdi/rsi) instead of ``argmatch_invalid(context, arg, ...)``;
  * setlocale_null_unlocked -- ``memcpy(buf, <stale>, ...)`` -- the ``result``
    pointer (from ``setlocale_null_androidfix``) lost across ``strlen``.
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


def _drop_only(examples_dir: Path, name: str) -> str:
    """Drop ``name`` from cp.ll into its own ea in a FRESH session; return the
    dropped pseudocode. A native fallback (build error) is rejected -- this
    asserts a REAL drop."""
    import idapro
    import ida_hexrays
    import ida_idaapi
    import ida_name

    binary = examples_dir / "cp"
    ir_path = examples_dir / "cp.ll"
    if not (binary.exists() and ir_path.exists()):
        pytest.skip("missing cp / cp.ll")
    from idavator.llvm_drop import LLVMDropConverter

    # PRISTINE per-drop IDB: copy the binary to a throwaway dir so the drop's
    # _force_prototype set_types (saved by close_database) never persists into the
    # shared examples/cp.i64 -- forced-prototype writes accumulate across runs and
    # poison the native baseline for later cases. cp.ll stays the real read-only IR.
    tmp = Path(tempfile.mkdtemp(prefix="argreg_preserve_"))
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
        assert conv.last_error is None, conv.last_error
        assert cf is not None, "decompile returned None"
        return str(cf)
    finally:
        idapro.close_database()
        shutil.rmtree(tmp, ignore_errors=True)


@pytest.mark.ida
class TestArgRegPreserve:
    def test_remember_copied_xstrdup_gets_name_param(
            self, examples_dir: Path) -> None:
        """``remember_copied`` passes the ``name`` PARAM (rdi, ``a0``) to
        ``xstrdup`` -- NOT the stale-rdi global ``&off_18`` left by the
        ``xmalloc(0x18)`` arg setup."""
        if not _idalib():
            pytest.skip("idalib unavailable")
        dropped = _drop_only(examples_dir, "remember_copied")

        # the xstrdup argument is NOT a global (`off_18` is the stale 0x18 from
        # the clobbered rdi rendered as an address).
        assert "off_18" not in dropped, (
            f"xstrdup arg is a stale-register global (`xstrdup(&off_18)`) "
            f"instead of the name param:\n{dropped}")
        # xstrdup is called with the first param (a0 == name).
        assert "xstrdup(a0)" in dropped or "xstrdup((const char *)a0)" in dropped, (
            f"xstrdup not called with the name param a0:\n{dropped}")

    def test_xargmatch_internal_invalid_gets_context_and_arg(
            self, examples_dir: Path) -> None:
        """``__xargmatch_internal`` passes ``context`` (a0) and ``arg`` (a1) to
        ``argmatch_invalid`` -- NOT the rdi/rsi clobbered by the preceding
        ``argmatch`` call (which left a1/a2 there)."""
        if not _idalib():
            pytest.skip("idalib unavailable")
        dropped = _drop_only(examples_dir, "__xargmatch_internal")

        # the first two argmatch_invalid args are a0 (context) and a1 (arg).
        assert "argmatch_invalid((const char *)a0, (const char *)a1," in dropped, (
            f"argmatch_invalid args clobbered (not context=a0, arg=a1):\n"
            f"{dropped}")

    def test_setlocale_null_unlocked_declines_divergent_body(
            self, examples_dir: Path) -> None:
        """``setlocale_null_unlocked`` only builds via the SROA fallback (the
        faithful path INTERRs 50342 on the return slot), and that degraded body
        DIVERGES: it collapses the three distinct returns (0 / 0x22 / 0x16) into
        the ``strlen`` result and drops the ``*buf = 0`` error-path store. A
        divergent body is worse than a clean native fallback, so the B5 decline
        gate now REFUSES it -> ``cf is None`` (the caller uses the correct native
        decompilation). (Previously this asserted the memcpy-source sub-property of
        the divergent body; that narrow property was faithful but the whole body
        was not -- B5 correctness over coverage.)"""
        if not _idalib():
            pytest.skip("idalib unavailable")
        import idapro
        import ida_hexrays
        import ida_idaapi
        import ida_name

        binary = examples_dir / "cp"
        ir_path = examples_dir / "cp.ll"
        if not (binary.exists() and ir_path.exists()):
            pytest.skip("missing cp / cp.ll")
        from idavator.llvm_drop import LLVMDropConverter

        tmp = Path(tempfile.mkdtemp(prefix="argreg_preserve_"))
        dst = tmp / "cp"
        shutil.copy(binary, dst)
        idapro.open_database(str(dst), True)
        try:
            assert ida_hexrays.init_hexrays_plugin()
            ea = ida_name.get_name_ea(ida_idaapi.BADADDR, "setlocale_null_unlocked")
            if ea == ida_idaapi.BADADDR:
                pytest.skip("setlocale_null_unlocked not in this binary")
            conv = LLVMDropConverter(ir_path.read_text())
            cf = conv.drop(ea, "setlocale_null_unlocked")
            assert conv.last_error is None, conv.last_error
            assert cf is None, (
                "expected a native fallback (divergent SROA body declined), got a "
                f"shipped body:\n{cf}")
            assert conv.last_declined_divergent is True, (
                "body should have been declined by the B5 divergence gate")
        finally:
            idapro.close_database()
            shutil.rmtree(tmp, ignore_errors=True)
