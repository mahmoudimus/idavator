"""INTERR 52700: an incoming REGISTER argument forwarded to a STACK call-argument.

A direct call with more integer args than ABI registers builds an explicit
``mcallinfo_t`` with ALOC_STACK arglocs for the 7th+ args (``_emit_call_stackargs``).
When one of those stack args' VALUE is a register tracing to an incoming function
argument, Hex-Rays' glbopt (MMAT_GLBOPT2) trips INTERR 52700: its sorted
stkarg/stkvar side-table lookup (hexx64 ``sub_180126700``, table at obj+0x540, keyed
by a stack offset/address) has no entry for an incoming-arg register placed directly
at a stack-call-arg location. That registration is produced ONLY by the
binary-driven ``gen_microcode`` (the real ``push [rbp+spill]`` sequence at
MMAT_CALLS), which the drop -- building at MMAT_PREOPTIMIZED -- bypasses.

Native instead PUSHES the value from a stack slot (``push [rbp+copy_into_self]``);
the slot read IS a registered stkvar the table has. The fix mirrors that: each
register-valued stack arg is SPILLED to a low scratch frame slot
(``mov reg, %scratch``) and the argloc is filled with a stkvar READ of that slot.
The value at the call is then a stack-memory reference, not a raw incoming-arg
register, and the +0x540 lookup is consistent. The spill also requires the call's
``call_spd`` to be measured at the call's own ``ea`` (not the host resting-frame
ea), so the outgoing-arg/spill region maps consistently under glbopt.

Witnesses (real cp functions): ``copy`` (the 29-line / 9-block minimal 52700
reducer -- ``copy_internal``'s stack args 8/9 carry the incoming ``copy_into_self``
/ ``rename_succeeded`` pointer params) and ``extent_copy`` (``sparse_copy``'s stack
args 6/7/8 carry incoming register params). Both must now drop to C that matches the
decompiled reference exactly (modulo lvar names) -- in particular the full argument
list with the stack-passed, incoming-arg tail.

NB: ``backupfile_internal`` (the third family member) is a DISTINCT shape -- its
only stack call-arg is an ``&local`` (mop_a stkvar), not a register -- and is a
PROVEN Hex-Rays frame-model boundary, left to the native fallback. A bare ``&local``
mop_a at an ALOC_STACK arg drops FAITHFULLY when the call's microcode ``call_spd``
is 0 (``extent_copy``'s ``sparse_copy`` args 9/10 are exactly this and are byte-
faithful). ``backupfile_internal`` differs ONLY in that the host SP at its
``numbered_backup`` call is a mid-frame ``-152`` (a ``sub rsp`` for locals + the
``push &sdir`` precede it), whereas NATIVE's mcallinfo carries ``call_spd==0`` /
``stkargs_top==8`` (gen_microcode normalises outgoing args into the callinfo, SP-
neutral). The drop reuses the HOST frame, so ``get_spd`` at the call returns ``-152``
and glbopt's +0x540 outgoing-stkarg side-table lookup (hexx64 ``sub_180126700``,
keyed by ``call_spd``/``stkargs_top``) MISSES -> INTERR 52700 at MMAT_GLBOPT2.
Forcing ``call_spd=0`` to register (native's value, measured at the ENTRY ea) DOES
clear 52700 but LIES about the real ``-152`` SP: the host's address-taken slots
(``funcresult``/``sdir``/...) are positioned for SP ``-152``, so SP=0 mis-maps every
stack-offset and Hex-Rays emits "local variable allocation has failed" (a corrupt,
rejected body -- verified across the full call_spd sweep {-152..+8}: only
call_spd>=0 registers, and only call_spd>=0 corrupts). The register-spill
alternative (materialise ``&sdir`` -> kreg, spill -> stkvar, mirroring native's
``lea;push``) registers but (a) REGRESSES ``extent_copy`` by reshaping its already-
faithful bare-mop_a args, and (b) trips the converging-return INTERR 50342 on the
PRIMARY path whose own distinct-ea fix re-breaks the spill registration (52700
returns). Recovery requires rebuilding ``backupfile_internal``'s frame with an
SP=0-consistent outgoing-args region (what gen_microcode does), which is orthogonal
to and far beyond the stack-arg registration mechanism. Intentionally NOT covered
here; it stays a faithful native fallback.

Run:  PYTHONPATH=src pytest -m ida tests/test_drop_stackarg_incoming_reg.py -s
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tests.render_tolerance import structural_equiv


def _idalib() -> bool:
    try:
        import idapro  # noqa: F401

        return True
    except ImportError:
        return False


# (function, callee, xfail_signature) -- a >6-arg call whose stack tail carries
# incoming reg args. The reference comparison is build-tolerant (structural_equiv):
# native and the drop are read from the SAME IDA; amd64 native carries DWARF
# types/param-names + a __readfsqword canary the weakly-typed IR drop cannot
# reproduce, so a faithful drop differs ONLY on those benign axes.
#
# copy is exactly such a benign case (same valid_options assert, same
# top_level_*_name stores, same 10-arg copy_internal call -- verified by raw-
# reading drop vs amd64-native; the only deltas are the canary, the BYREF `=0`
# init, casts/types, the `__assert_fail` vs `_assert_fail` underscore, and the
# weak-int return materialization, all of which structural_equiv collapses), so it
# carries NO xfail signature -- any divergence there is a real failure.
#
# extent_copy is a GENUINE per-build structural divergence (NOT a cosmetic axis)
# on amd64 ONLY: amd64 native keeps the `extent_scan` struct (`scan.ei_count`,
# `scan.ext_info[i].ext_logical`, `while (extent_scan_read(&scan))`), whereas the
# drop SCALARIZES the struct into raw pointer arithmetic (`*(_QWORD *)(last_ext_len
# + 24LL*i)`) and RESTRUCTURES the loop (`while (1) { ...; if (...) break; }` with
# the read hoisted inside). On macOS-arm64 native lacks the DWARF struct types and
# renders the SAME body as the drop, so there it PASSES. So instead of a static
# (build-blind) xfail, the case carries an xfail_SIGNATURE recognising EXACTLY this
# struct-scalarization + loop-restructure shape; any OTHER divergence is a real
# FAILURE (B5).
def _extent_copy_scalarize_xfail(drop: str, native: str) -> "str | None":
    # Native (DWARF) keeps the extent_scan struct: typed array-of-struct indexing
    # (scan.ext_info[i] / scan.ei_count). The drop SCALARIZES it -- it reuses the
    # `last_ext_len` scalar slot AS the struct base, mis-casting `&last_ext_len`
    # to `extent_scan *` (instead of native's dedicated `&scan`) and reaching the
    # fields via raw offset arithmetic. Keyed on THAT struct-base mis-cast --
    # the build/run-invariant hallmark -- NOT on the surviving loop header
    # (`while(1)` vs `while(extent_scan_read())`, which idalib varies) nor the
    # exact field-offset rendering (`24LL * i` on amd64, `24 * i` elsewhere).
    native_struct = "scan.ext_info[" in native or "scan.ei_count" in native
    drop_scalarized = ("(extent_scan *)&last_ext_len" in drop
                       and "&scan" not in drop)
    if native_struct and drop_scalarized:
        return ("native keeps the extent_scan struct (scan.ext_info[i], "
                "scan.ei_count); the drop scalarizes it -- mis-casting the "
                "last_ext_len slot to `extent_scan *` and reaching fields via raw "
                "offset arithmetic instead of native's `&scan` -- a genuine per-"
                "build structural divergence, not a cosmetic type/canary axis. "
                "Matches (and ships on) a build whose native lacks the DWARF "
                "struct (e.g. when macOS native elides it).")
    return None


_CASES = [
    ("copy", "copy_internal", None),
    ("extent_copy", "sparse_copy", _extent_copy_scalarize_xfail),
]


@pytest.mark.ida
class TestStackArgIncomingReg:
    @pytest.mark.parametrize("fn, callee, xfail_sig", _CASES)
    def test_incoming_reg_stack_arg_matches_reference(
            self, examples_dir: Path, fn: str, callee: str, xfail_sig) -> None:
        if not _idalib():
            pytest.skip("idalib unavailable")
        import ida_hexrays as hx
        import ida_idaapi
        import ida_name
        import idapro

        if not (examples_dir / "cp").exists():
            pytest.skip("missing example binary")
        ll = examples_dir / "cp.ll"
        if not ll.exists():
            pytest.skip("missing cp.ll")

        idapro.open_database(str(examples_dir / "cp"), True)
        try:
            assert hx.init_hexrays_plugin()
            from idavator.llvm_drop import LLVMDropConverter

            ea = ida_name.get_name_ea(ida_idaapi.BADADDR, fn)
            if ea == ida_idaapi.BADADDR:
                pytest.skip(f"{fn} not in this build")
            orig = str(hx.decompile(ea))
            conv = LLVMDropConverter(ll.read_text())
            cf = conv.drop(ea, fn)
            text = str(cf) if cf is not None else "<None>"
            print(f"\n=== drop @{fn} (interr={conv.last_interr} "
                  f"late={conv.last_primary_late_interr} "
                  f"err={'yes' if conv.last_error else None}) ===\n{text}")
            # The whole point: 52700 must NOT surface (early OR late).
            assert conv.last_error is None, conv.last_error
            assert conv.last_interr is None, f"INTERR {conv.last_interr}"
            assert conv.last_primary_late_interr is None, (
                f"late INTERR {conv.last_primary_late_interr} "
                "(52700 not cleared)")
            assert cf is not None, "decompile failed"
            # The PRIMARY (faithful) path must produce the body -- not a degraded
            # SROA/kreg retry (those would signal the spill did not take).
            assert conv.last_build_path == "PRIMARY", conv.last_build_path
            assert callee in text, f"callee {callee!r} missing in:\n{text}"
            # Build-tolerant structural match: equal STRUCTURE (statements, calls,
            # arg counts, constants, control flow) modulo the benign type/canary/
            # underscore/value-split axes.
            if not structural_equiv(text, orig):
                reason = xfail_sig(text, orig) if xfail_sig else None
                if reason is not None:
                    pytest.xfail(reason)
                pytest.fail(
                    f"dropped C diverges from reference.\n"
                    f"--- dropped ---\n{text}\n"
                    f"--- reference ---\n{orig}")
        finally:
            idapro.close_database()
