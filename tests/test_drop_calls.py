"""Drive LLVMDropConverter on LLVM `call` instructions.

At PREOPTIMIZED a call is just `m_call l=gvar(callee)`; Hex-Rays reconstructs the
arguments from the ABI registers (we set them with movs) + the callee's prototype.
Callees are resolved by name against the open IDB (examples/cp ships strlen etc.).

Run:  PYTHONPATH=src pytest -m ida tests/test_drop_calls.py -s
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


def _find_linear_host(ida_funcs, hx, idautils):
    for ea in idautils.Functions():
        f = ida_funcs.get_func(ea)
        if f is None or not (8 <= f.end_ea - f.start_ea <= 200):
            continue
        if hx.decompile(ea) is None:
            continue
        hf = hx.hexrays_failure_t()
        mbr = hx.mba_ranges_t()
        mbr.ranges.push_back(f)
        m = hx.gen_microcode(mbr, hf, None, hx.DECOMP_NO_WAIT, hx.MMAT_PREOPTIMIZED)
        if m is None:
            continue
        tails = {int(b.tail.opcode) for i in range(m.qty)
                 if (b := m.get_mblock(i)) is not None and b.tail is not None}
        conds = {hx.m_jcnd, hx.m_jz, hx.m_jnz, hx.m_jtbl}
        if hx.m_ret in tails and not (tails & conds):
            return ea
    return None


# Proven-safe subset: a call result is only kept in rax when unused (void) or
# returned directly (tail call). At PREOPTIMIZED an m_call needs a non-empty `d`
# (INTERR 50864 -- see the decode in llvm_drop.py); we use d=rax.
_CASES = [
    # tail call: strlen(s).
    ("declare i64 @strlen(i8*)\n"
     "define i64 @mylen(i8* %s) {\n"
     "entry:\n  %n = call i64 @strlen(i8* %s)\n  ret i64 %n\n}\n",
     "mylen", ["strlen"]),
    # void call, result unused.
    ("declare void @free(i8*)\n"
     "define void @vfree(i8* %p) {\n"
     "entry:\n  call void @free(i8* %p)\n  ret void\n}\n",
     "vfree", ["free("]),
]


@pytest.mark.ida
class TestCallDrop:
    @pytest.mark.parametrize("ir, fn, must", _CASES)
    def test_calls(self, examples_dir: Path, ir, fn, must):
        if not _idalib():
            pytest.skip("idalib unavailable")
        import idapro
        import ida_funcs
        import ida_hexrays as hx
        import idautils

        if not (examples_dir / "cp").exists():
            pytest.skip("missing example binary")

        idapro.open_database(str(examples_dir / "cp"), True)
        try:
            assert hx.init_hexrays_plugin()
            from idavator.llvm_drop import LLVMDropConverter

            host = _find_linear_host(ida_funcs, hx, idautils)
            assert host is not None, "no linear host found"

            conv = LLVMDropConverter(ir)
            cf = conv.drop(host, fn)
            text = str(cf) if cf is not None else "<None>"
            print(f"\n=== drop @{fn} (host={host:#x} interr={conv.last_interr} "
                  f"err={'yes' if conv.last_error else None}) ===\n{text}")
            assert conv.last_error is None, conv.last_error
            assert conv.last_interr is None, f"INTERR {conv.last_interr}"
            assert cf is not None, "decompile failed"
            assert "allocation has failed" not in text, text
            for needle in must:
                assert needle in text, f"missing {needle!r} in:\n{text}"
        finally:
            idapro.close_database()

    def test_consumed_call_result_raises_cleanly(self, examples_dir: Path):
        """A call result consumed by arithmetic would segfault a later maturity
        pass -- the converter must raise NotImplementedError, not crash."""
        if not _idalib():
            pytest.skip("idalib unavailable")
        import idapro
        import ida_funcs
        import ida_hexrays as hx
        import idautils

        if not (examples_dir / "cp").exists():
            pytest.skip("missing example binary")

        idapro.open_database(str(examples_dir / "cp"), True)
        try:
            assert hx.init_hexrays_plugin()
            from idavator.llvm_drop import LLVMDropConverter

            host = _find_linear_host(ida_funcs, hx, idautils)
            assert host is not None
            ir = ("declare i64 @strlen(i8*)\n"
                  "define i64 @lenp1(i8* %s) {\n"
                  "entry:\n  %n = call i64 @strlen(i8* %s)\n"
                  "  %r = add i64 %n, 1\n  ret i64 %r\n}\n")
            conv = LLVMDropConverter(ir)
            with pytest.raises(NotImplementedError):
                conv.drop(host, "lenp1")
        finally:
            idapro.close_database()
