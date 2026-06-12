"""Size-changing ``ptrtoint`` / ``inttoptr`` must NARROW / WIDEN, not alias.

``ptrtoint``/``inttoptr``/``bitcast`` are usually bit-identical reinterpretations
of the SAME width (ptr<->int, ptr<->ptr) -- the lifter aliases the operand with no
microcode. But ``ptrtoint``/``inttoptr`` can also CHANGE width:

- ``ptrtoint i8* %p to i8``  truncates an 8-byte pointer to 1 byte;
- ``inttoptr i32 %x to ptr``  widens a 4-byte int to an 8-byte pointer.

Aliasing those (the old ``_NOOP_CAST`` behavior) leaks an 8-byte operand where the
consumer expects 1 byte. The canonical consumer is an ``icmp`` folded to a
conditional jump: the branch then reads ``m_jz l=kr.8 r=#0.1`` and the verifier
rejects the size mismatch with INTERR 50831 (verify.cpp conditional-branch
operand-size check requires ``l.size == r.size``). This is the SAME shape carried
by the ``ptrtoint i8* %p to i8`` sites in hash_lookup / hash_find_entry /
hash_do_for_each (ticket ida-28mi).

The fix lowers a width-changing ptrtoint/inttoptr like trunc/zext (``m_low`` to
narrow, ``m_xdu`` to widen) into a fresh kreg. With the fix stashed, the narrowing
probe trips INTERR 50831 (or fails to build); with it, the comparison is on a
1-byte operand and the function builds + renders.
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


# ptrtoint i8* %p to i8 -> the low byte of a pointer, COMPARED to a 1-byte 0 and
# branched on. Without narrowing, the alias keeps size 8 and the icmp->jz emits
# `jz kr.8, #0.1` -> INTERR 50831. With it, `m_low` produces a 1-byte value and
# the branch operands match.
PROBE_PTRTOINT_NARROW = """
define i64 @probe(i8* %p) {
entry:
  %lo = ptrtoint i8* %p to i8
  %c = icmp eq i8 %lo, 0
  br i1 %c, label %zero, label %nonzero
zero:
  ret i64 0
nonzero:
  ret i64 1
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
class TestDropCastNarrow:
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

    def test_ptrtoint_to_i8_narrows_not_aliases(
            self, examples_dir: Path) -> None:
        """`ptrtoint i8* %p to i8` feeding an icmp->branch: the size-changing cast
        must narrow (m_low) so the conditional jump's operands match. With the
        cast aliased (pre-fix), the branch is `jz kr.8, #0.1` and verify trips
        INTERR 50831; the build then fails (cf is None / last_error set)."""
        if not _idalib():
            pytest.skip("idalib unavailable")
        conv, txt = self._drop_probe(examples_dir, PROBE_PTRTOINT_NARROW)
        # The width-changing cast must NOT trip the conditional-branch size check.
        assert conv.last_interr != 50831, (
            "ptrtoint i8*->i8 still aliased size-8 into a 1-byte compare "
            "(INTERR 50831 -- conditional-branch operand-size mismatch)")
        assert conv.last_error is None, conv.last_error
        assert txt is not None, "decompile returned None (cast narrow failed)"
        # The narrow is observable in the rendered C: the comparison is on the LOW
        # BYTE of the pointer (`(_BYTE)`/`(char)`/`BYTE` view), not the full 8-byte
        # value. Hex-Rays folds the if/else into the equivalent boolean
        # `return (_BYTE)a0 != 0;` -- the byte cast is the proof the m_low ran
        # (pre-fix the operand stayed 8 bytes and tripped 50831 before any render).
        assert any(tok in txt for tok in ("(_BYTE)", "(char)", "BYTE", "LOBYTE")), (
            f"narrow not reflected -- comparison not on the low byte:\n{txt}")
