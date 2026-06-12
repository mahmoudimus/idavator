"""Indirect (function-pointer) call lowering via Hex-Rays' native ``m_icall``.

When the callee of an LLVM ``call`` is an SSA VALUE (a function pointer loaded
from memory / produced by ``inttoptr``) rather than a named symbol, there is no
gvar to target. The lifter emits ``%r = call <ty> %fp(args...)``; the drop lowers
it to Hex-Rays' own post-CALLS form::

    mov  (icall cs.2, <callee-value>.8 <mcallinfo>) => rax

The mcallinfo is built from a synthesized ``__fastcall`` prototype (``set_type``
does the SysV reg/cc classification) and carries the native flag set
``FCI_PROP|FCI_DEAD|FCI_SPLOK``. ``FCI_PROP`` is REQUIRED: it lets the verifier
accept an EMPTY ``retregs`` list (mirroring native), which it otherwise rejects
with INTERR 50745 (verify.cpp:131) before any later check.

Reference: ``hash_lookup`` / ``safe_hasher`` @ MMAT_CALLS call their comparator /
hasher through ``(*((... __fastcall **)a0 + N))(...)`` -- a function pointer at a
struct-field offset. The indirect-dispatch RENDER produced by this lowering is
byte-identical to native (the remaining hash_lookup/safe_hasher divergence is the
separate struct-blob-vs-typed-field SROA gap, not the call). Ticket ida-poz6.

Before this lowering the drop raised ``NotImplementedError`` for any
function-pointer callee, so these guards build only with the feature present.
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


# An indirect call through a function pointer LOADED from a struct-like slot --
# the hash_lookup/safe_hasher shape (a fn-ptr at an object-field offset). The
# callee `%fp` is an SSA value, so there is no gvar; the result feeds the return.
PROBE_INDIRECT = """
define i64 @probe(i8** %obj, i64 %x) {
entry:
  %slot = getelementptr i8*, i8** %obj, i64 3
  %fp = load i8*, i8** %slot, align 8
  %callee = bitcast i8* %fp to i64 (i64)*
  %r = call i64 %callee(i64 %x)
  ret i64 %r
}
"""


def _linear_host(ida_funcs, ida_hexrays):
    import idautils

    for ea in idautils.Functions():
        f = ida_funcs.get_func(ea)
        if f is None or (f.flags & ida_funcs.FUNC_NORET):
            continue
        if not (8 <= f.end_ea - f.start_ea <= 400):
            continue
        if ida_hexrays.decompile(ea) is not None:
            return ea
    return None


@pytest.mark.ida
class TestDropIndirectCall:
    def _drop_probe(self, examples_dir: Path, probe: str):
        import idapro
        import ida_funcs
        import ida_hexrays

        from idavator.llvm_drop import LLVMDropConverter

        binary = examples_dir / "cp"
        if not binary.exists():
            pytest.skip("missing example binary: cp")
        idapro.open_database(str(binary), True)
        try:
            assert ida_hexrays.init_hexrays_plugin()
            host = _linear_host(ida_funcs, ida_hexrays)
            assert host is not None, "no linear host found"
            conv = LLVMDropConverter(probe)
            cf = conv.drop(host, "probe")
            return conv, (str(cf) if cf is not None else None)
        finally:
            idapro.close_database()

    def test_indirect_call_builds_via_m_icall(
            self, examples_dir: Path) -> None:
        """A function-pointer callee builds (no NotImplementedError, no INTERR)
        and renders as an INDIRECT call -- a dereference-through-fn-ptr-VALUE,
        never a named callee. The FCI_PROP-flagged empty-retregs mcallinfo must
        clear the 50745 retreg cross-check that previously blocked it."""
        if not _idalib():
            pytest.skip("idalib unavailable")
        conv, txt = self._drop_probe(examples_dir, PROBE_INDIRECT)
        assert conv.last_error is None, conv.last_error
        # specifically the retreg/return cross-check that FCI_PROP suppresses.
        assert conv.last_interr not in (50745, 50743), (
            f"indirect-call mcallinfo tripped retreg INTERR {conv.last_interr}")
        assert conv.last_interr is None, f"INTERR {conv.last_interr}"
        assert txt is not None, "decompile returned None (indirect call failed)"
        # The call must render as an indirect dispatch through a function-pointer
        # VALUE: Hex-Rays prints a `(*(... (__fastcall *...))...)(...)` callee, NOT
        # a bare named symbol. The fn-ptr cast token is the faithful marker (it is
        # what native hash_lookup/safe_hasher print for the same shape).
        assert "__fastcall" in txt and "(*(" in txt, (
            f"indirect call did not render as a fn-ptr dispatch:\n{txt}")
