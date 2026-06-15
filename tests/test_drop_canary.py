"""Stack-canary elision. `-fstack-protector` makes the lift emit
`__readfsqword` (canary read) + a `__stack_chk_fail` fail branch ending in
`unreachable`. The optimizer ELIDES all of that from faithful output, so the
drop models every `__readfsqword` as ONE shared kreg (the `saved == reread`
compare folds `K == K` -> true), skips `__stack_chk_fail`, and routes
`unreachable` to the ret block -- Hex-Rays then prunes the dead fail branch.
See memory idavator_drop_canary_gate.

CANARY IS NOT THE COPY-FAMILY (do_copy/copy/copy_internal) BLOCKER -- PROVEN, no
code change. A 2026-06-14 investigation hypothesised that the copy family declines
because native RENDERS ``v59 = __readfsqword(0x28u)`` as a visible stack-local while
the drop folds the read away, and tried to "recover" it by emitting the read as a
helper call + escaping its store-target alloca to a real frame slot. That hypothesis
is FALSE and the fix REGRESSES 12 functions for 0 gains. The falsifying evidence:

* The round-trip oracle (``idavator.oracle.matches``, the B5 decline gate's
  ground truth) is INVARIANT to the canary statement. On ``main``, native
  ``quotearg_char_mem`` carries ``v5 = __readfsqword(0x28u);`` (an assignment to an
  UNUSED local), the drop OMITS it entirely, and ``matches`` returns True. The
  canary local is dead, so its assign-statement is oracle-transparent whether kept
  or dropped. Verified across all 12 faithful keep-cohort fns (quotearg_char_mem,
  quotearg_n_style{,_mem}, quotearg_n_custom_mem, emit_ancillary_info, version_etc,
  version_etc_va, src_to_dest_lookup, forget_created, qset_acl, qcopy_acl).
* ``oracle.matches(native, drop)`` and ``oracle.matches(canary_stripped_native,
  drop)`` are IDENTICAL (both False) for do_copy/copy/copy_internal -- the canary is
  provably NOT the (or even *a*) divergence; the first-diff is elsewhere every time.
* Emitting the read (helper-call form) makes a DEAD store collapse to a bare
  ``__readfsqword(0x28);`` expression-statement (the kreg store is DCE'd, the
  side-effecting call orphaned), which the oracle sees as ``("stmt",(call,))`` !=
  native's ``("assign", v, call)`` -> NEW divergence, turning 12 faithful PRIMARY
  bodies divergent. The shared-kreg FOLD (omit the canary) is therefore OPTIMAL:
  omission already matches, so no canary rendering can be a net gain.

The REAL, canary-independent copy-family gaps (deferred; recovery paths noted):

* ``copy`` (real_drop=True, ships a PRIMARY body that already diverges on main):
  two non-canary divergences -- (1) the assert callee renders ``_assert_fail`` vs
  native's ``__assert_fail`` (an underscore-count callee-name mismatch the oracle
  treats as significant), and (2) the tail ``LOBYTE(result) = copy_internal(...);
  return result;`` vs native's direct ``return copy_internal(...)`` (the dead-rax
  return-materialisation; the same shape ``wvfull.py`` norm-folds as benign).
  Recovery: resolve the assert callee to the ``__assert_fail`` symbol and route the
  tail-call return through the return reg directly (the ret-promotion path), not via
  a captured ``result`` kreg.
* ``copy_internal`` (real_drop=False, clean native fallback): ``_scan_allocas``
  raises ``NotImplementedError: GEP-on-stack alloca %v114`` (a scalar byte-pun /
  va_list / anonymous-numeric IR alloca, not a laid-out struct field). This is the
  documented GEP-on-stack decline boundary (struct layout needed); fallback by
  design, NOT a regression.
* ``do_copy`` (real_drop=False, degraded body DECLINED by the B5 gate): PRIMARY
  fails LATE with INTERR 50342 (return-slot value-number family); the DISTINCT-EA
  retry produces a body the gate correctly declines because do_copy's VLA stack-probe
  loops (``while (&xa != (cp_options **)((char *)&xa - (v14 & ...)))`` + ``alloca(v14
  & 0xFFF)``) render as ``0 < var0`` -- a genuine alloca/VLA-rendering divergence, the
  ledger first-diff. Recovery: model the explicit-VLA stack-probe construct (distinct
  from the dead-result ``alloca`` already in ``_HELPER_INTRINSICS``).
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


# Hex-Rays' COLLAPSED-declaration rendering: the dead canary local (``v5``) is hidden
# behind ``// [COLLAPSED LOCAL DECLARATIONS]`` and never declared. These three minimal
# bodies reproduce the exact round-trip oracle behaviour that makes the shared-kreg
# canary FOLD (omit the read) optimal and any "emit the read" change a regression.
_CANARY_KEEP = (  # native: keeps ``v5 = __readfsqword(0x28u)`` (v5 undeclared, dead)
    "int f(int a)\n{\n"
    "  v5 = __readfsqword(0x28u);\n"
    "  return a + 1;\n}\n"
)
_CANARY_OMIT = (  # the drop's shared-kreg fold: no canary statement at all
    "int f(int a)\n{\n"
    "  return a + 1;\n}\n"
)
_CANARY_BARE = (  # a helper-call render whose dead result is DCE'd: a bare call-stmt
    "int f(int a)\n{\n"
    "  __readfsqword(0x28u);\n"
    "  return a + 1;\n}\n"
)


class TestCanaryOracleInvariance:
    """The B5 decline gate's ground truth (``idavator.oracle.matches``) is INVARIANT
    to the dead canary statement in Hex-Rays' COLLAPSED-declaration form: an
    undeclared-LHS dead assignment is transparent in the canonical form. So OMITTING
    the canary (the shared-kreg fold) already round-trips faithfully, and no canary
    rendering can be a net gain. A BARE-call render (the dead store DCE'd) is NOT
    transparent -- it adds a visible call-statement -> divergence. This is why the
    canary is NOT the copy-family blocker and why "recover the read" regresses 12
    keep-cohort functions. Pure oracle, no IDA; guards against re-deriving that
    falsified hypothesis. See the module docstring + memory idavator_drop_canary_gate.
    """

    def test_oracle_tolerates_canary_omission(self) -> None:
        from idavator import oracle

        if not oracle.clang_available():
            pytest.skip("libclang unavailable (oracle needs IDA's clang_loader)")
        # native KEEPS the dead canary, drop OMITS it -> still faithful.
        assert oracle.matches(_CANARY_KEEP, _CANARY_OMIT), (
            "oracle must tolerate canary omission (the shared-kreg fold) -- if this "
            "fails the fold is no longer safe and the gate would falsely decline")

    def test_oracle_rejects_bare_canary_call(self) -> None:
        from idavator import oracle

        if not oracle.clang_available():
            pytest.skip("libclang unavailable (oracle needs IDA's clang_loader)")
        # a bare ``__readfsqword(...)`` call-statement (dead store DCE'd) DIVERGES
        # from native's ``v5 = __readfsqword(...)`` -- emitting the read this way
        # turns faithful PRIMARY bodies divergent (the 12-function regression).
        assert not oracle.matches(_CANARY_KEEP, _CANARY_BARE), (
            "a bare canary call-statement must NOT match native's assign form -- "
            "this asymmetry is why 'emit the read' is a regression, not a fix")


# canary read -> store -> body -> reread -> compare -> fail(__stack_chk_fail) / ok.
PROBE = """
define i64 @probe(i64 %x) {
entry:
  %c1 = call i64 @__readfsqword(i32 40)
  %slot = alloca i64, align 8
  store i64 %c1, ptr %slot
  %r = add i64 %x, 1
  %c2 = load i64, ptr %slot
  %c3 = call i64 @__readfsqword(i32 40)
  %cmp = icmp eq i64 %c2, %c3
  br i1 %cmp, label %ok, label %fail
fail:
  call void @__stack_chk_fail()
  unreachable
ok:
  ret i64 %r
}
declare i64 @__readfsqword(i32)
declare void @__stack_chk_fail()
"""


@pytest.mark.ida
class TestCanaryElision:
    def test_canary_is_elided(self, examples_dir: Path) -> None:
        if not _idalib():
            pytest.skip("idalib unavailable")
        import idapro
        import ida_funcs
        import ida_hexrays
        import idautils

        binary = examples_dir / "cp"
        if not binary.exists():
            pytest.skip("missing example binary: cp")

        from idavator.llvm_drop import LLVMDropConverter

        idapro.open_database(str(binary), True)
        try:
            assert ida_hexrays.init_hexrays_plugin()
            host = next((ea for ea in idautils.Functions()
                         if (f := ida_funcs.get_func(ea)) is not None
                         and int(getattr(f, "frsize", 0)) >= 16
                         and not (f.flags & ida_funcs.FUNC_NORET)
                         and ida_hexrays.decompile(ea) is not None), None)
            assert host is not None, "no host with frsize >= 16"

            conv = LLVMDropConverter(PROBE)
            cf = conv.drop(host, "probe")
            assert conv.last_error is None, conv.last_error
            assert conv.last_interr is None, f"INTERR {conv.last_interr}"
            assert cf is not None, "decompile returned None"
            txt = str(cf)
            # the whole canary must be gone -- no read, no fail call, no warning.
            assert "__readfsqword" not in txt, f"canary read survived:\n{txt}"
            assert "__stack_chk_fail" not in txt, f"fail branch survived:\n{txt}"
            assert "bad sp value" not in txt, txt
            assert "return" in txt, txt
        finally:
            idapro.close_database()
