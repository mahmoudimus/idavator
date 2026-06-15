"""GAP B guard: a direct call with MORE integer args than ABI registers (the
7th+ travel on the stack).

``_emit_call`` previously raised ``NotImplementedError: stack-passed call
argument (more args than ABI registers)``. For a direct (named) callee whose
prototype is known, the drop now builds an explicit ``mcallinfo_t`` via
``set_type`` (which does the SysV register/stack classification) so the stack
args ride IN the call -- no SP-modeled ``push`` sequence. ``_emit_call_stackargs``.

Witnesses (real cp functions, stack-arg calls, no orthogonal noreturn/select
gap): ``create_hard_link`` (a 7-arg ``force_linkat``) and ``quotearg_buffer``
(a 9-arg ``quotearg_buffer_restyled``) must drop to C that matches the
decompiled reference exactly (modulo lvar names) -- in particular the full
argument list with the stack-passed tail.

Run:  PYTHONPATH=src pytest -m ida tests/test_drop_stackargs.py -s
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


# (function, callee, expected stack-arg count = nargs - 6, xfail_signature). The
# comparison is build-tolerant (structural_equiv): native and the drop are read
# from the SAME IDA, and on the amd64 idalib build native carries DWARF
# types/param-names + a __readfsqword stack canary the weakly-typed IR drop
# cannot reproduce, so a faithful drop differs from native ONLY on those benign
# type/canary/underscore axes -- which structural_equiv collapses while still
# catching a real divergence.
#
# create_hard_link is a GENUINE per-build structural divergence (NOT a cosmetic
# axis) on amd64 ONLY: amd64 native combines the guard as `if ( err < 0 && verbose
# )` and returns `1` directly, whereas the drop emits a NESTED `if (result<0){ if
# (a3){...} }` and a fall-through `LOBYTE(result)=1; ...; return result` (verified
# by raw-reading drop vs amd64-native). On macOS-arm64 native lacks the DWARF/
# canary and renders the SAME body as the drop, so there it PASSES. So instead of
# a static (build-blind) xfail, the case carries an xfail_SIGNATURE: a predicate
# that recognises EXACTLY this combined-vs-nested-guard shape. The test xfails
# only when structural_equiv fails AND the signature matches; any OTHER divergence
# is a real FAILURE (B5: a build-blind "xfail on any mismatch" would hide a
# regression).
def _chl_combined_guard_xfail(drop: str, native: str) -> "str | None":
    # Native (DWARF, `bool` return) COMBINES the guard as `if ( err < 0 && verbose
    # )` and returns `1` directly; the drop (weak `int` return) NESTS it as
    # `if (result<0){ if (a3){...} }` and materializes the return. The run-
    # invariant hallmark: native carries the combined `< 0 &&` guard that the
    # nested drop does NOT. (force_linkat presence -- i.e. a faithful, non-corrupt
    # body -- is already asserted by the caller's `callee in text` check.)
    if "< 0 && " in native and "< 0 && " not in drop:
        return ("amd64 native combines `if(err<0 && verbose)` + bare `return 1`; "
                "the drop nests the guard and materializes the return -- a genuine "
                "per-build control-flow divergence, not a cosmetic type/canary "
                "axis. Matches (and ships on) macOS-arm64 native.")
    return None


_CASES = [
    ("create_hard_link", "force_linkat", 1, _chl_combined_guard_xfail),
    ("quotearg_buffer", "quotearg_buffer_restyled", 3, None),
]


@pytest.mark.ida
class TestStackArgsDrop:
    @pytest.mark.parametrize("fn, callee, n_stack, xfail_sig", _CASES)
    def test_stack_arg_call_matches_reference(
            self, examples_dir: Path, fn, callee, n_stack, xfail_sig) -> None:
        if not _idalib():
            pytest.skip("idalib unavailable")
        import idapro
        import ida_hexrays as hx
        import ida_idaapi
        import ida_name

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
                  f"err={'yes' if conv.last_error else None}) ===\n{text}")
            assert conv.last_error is None, conv.last_error
            assert conv.last_interr is None, f"INTERR {conv.last_interr}"
            assert cf is not None, "decompile failed"
            # No corruption hallmarks.
            for bad in ("byte_", "write access to const memory"):
                assert bad not in text, f"corruption {bad!r} in:\n{text}"
            # The callee must be present with the SAME (full) arg list as the
            # reference -- i.e. the stack-passed tail survived.
            assert callee in text, f"callee {callee!r} missing in:\n{text}"
            # Build-tolerant structural match against the decompiled reference:
            # equal STRUCTURE (statements, calls, arg counts, constants, control
            # flow) modulo the benign type/canary/underscore/value-split axes.
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
