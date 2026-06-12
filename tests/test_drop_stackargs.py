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

import re
from pathlib import Path

import pytest


def _idalib() -> bool:
    try:
        import idapro  # noqa: F401

        return True
    except ImportError:
        return False


def _norm(text: str) -> str:
    """Normalise pseudocode for comparison: drop the COLLAPSED-decls banner and
    rename every local (vN / aN) to a positional placeholder so the comparison
    is name-agnostic but structure/constant/callee exact."""
    out = []
    for line in text.splitlines():
        if "COLLAPSED LOCAL DECLARATIONS" in line:
            continue
        line = re.sub(r"\bv\d+\b", "V", line)
        line = re.sub(r"\ba\d+\b", "A", line)
        out.append(line.rstrip())
    return "\n".join(out).strip()


# (function, callee, expected stack-arg count = nargs - 6)
_CASES = [
    ("create_hard_link", "force_linkat", 1),
    ("quotearg_buffer", "quotearg_buffer_restyled", 3),
]


@pytest.mark.ida
class TestStackArgsDrop:
    @pytest.mark.parametrize("fn, callee, n_stack", _CASES)
    def test_stack_arg_call_matches_reference(
            self, examples_dir: Path, fn, callee, n_stack) -> None:
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
            # Name-agnostic structural match against the decompiled reference.
            assert _norm(text) == _norm(orig), (
                f"dropped C diverges from reference.\n"
                f"--- dropped ---\n{_norm(text)}\n"
                f"--- reference ---\n{_norm(orig)}")
        finally:
            idapro.close_database()
