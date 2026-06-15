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
only stack call-arg is an ``&local`` (mop_a stkvar), not a register -- and is left
to the native fallback (spilling an address value is unfaithful); it is intentionally
NOT covered here.

Run:  PYTHONPATH=src pytest -m ida tests/test_drop_stackarg_incoming_reg.py -s
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
    """Normalise pseudocode: drop the COLLAPSED-decls banner and rename every local
    (vN / aN) to a positional placeholder -- name-agnostic but structure/constant/
    callee exact (same convention as test_drop_stackargs)."""
    out = []
    for line in text.splitlines():
        if "COLLAPSED LOCAL DECLARATIONS" in line:
            continue
        line = re.sub(r"\bv\d+\b", "V", line)
        line = re.sub(r"\ba\d+\b", "A", line)
        out.append(line.rstrip())
    return "\n".join(out).strip()


# (function, callee) -- a >6-arg call whose stack tail carries incoming reg args.
_CASES = [
    ("copy", "copy_internal"),
    ("extent_copy", "sparse_copy"),
]


@pytest.mark.ida
class TestStackArgIncomingReg:
    @pytest.mark.parametrize("fn, callee", _CASES)
    def test_incoming_reg_stack_arg_matches_reference(
            self, examples_dir: Path, fn: str, callee: str) -> None:
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
            assert _norm(text) == _norm(orig), (
                f"dropped C diverges from reference.\n"
                f"--- dropped ---\n{_norm(text)}\n"
                f"--- reference ---\n{_norm(orig)}")
        finally:
            idapro.close_database()
