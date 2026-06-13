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


# A call must terminate its microcode block (INTERR 50864), so the converter
# SPLITS each LLVM block at calls: the call is a BLT_1WAY tail that falls through
# to a continuation block which captures the rax result into a kreg. This lifts
# the old "consumed result" restriction.
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
    # consumed result: strlen(s) + 1 (was the old crashing/guarded case).
    ("declare i64 @strlen(i8*)\n"
     "define i64 @lenp1(i8* %s) {\n"
     "entry:\n  %n = call i64 @strlen(i8* %s)\n"
     "  %r = add i64 %n, 1\n  ret i64 %r\n}\n",
     "lenp1", ["strlen", "+ 1"]),
    # chained calls: strlen(a) + strlen(b) -- the first result must survive the
    # second call (kreg capture), proving the spill across a clobbering call.
    ("declare i64 @strlen(i8*)\n"
     "define i64 @two(i8* %a, i8* %b) {\n"
     "entry:\n  %x = call i64 @strlen(i8* %a)\n"
     "  %y = call i64 @strlen(i8* %b)\n"
     "  %r = add i64 %x, %y\n  ret i64 %r\n}\n",
     "two", ["strlen"]),
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

    def test_consumed_result_renders_arithmetic(self, examples_dir: Path):
        """The consumed call result `strlen(s) + 1` now renders (block split)."""
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
            cf = conv.drop(host, "lenp1")
            text = str(cf) if cf is not None else "<None>"
            assert conv.last_error is None, conv.last_error
            assert cf is not None and "strlen" in text, text
            assert "+ 1" in text, text
        finally:
            idapro.close_database()

    def test_fixed_arg_vararg_call_pins_args(self, examples_dir: Path):
        """A variadic callee invoked with EXACTLY its fixed args (no surplus
        vararg) must carry those args faithfully. Real get_nonce calls
        ``open("/dev/urandom", 0)`` -- ``open`` is variadic ``int(const char *,
        int, ...)``. Without an explicit fixed-arg variadic mcallinfo, Hex-Rays'
        own vararg recovery invents phantom trailing args off stale incoming-param
        registers and mis-renders the call as ``open(a0, a1, a2)`` (the path
        literal lost). The fix pins the call with FCI_FINAL; this guard fails
        without it (the literal vanishes / phantom params appear).

        Pristine-based: drops the real cp ``get_nonce`` against ``examples/cp``."""
        if not _idalib():
            pytest.skip("idalib unavailable")
        import idapro
        import ida_hexrays as hx
        import ida_idaapi
        import ida_name

        if not (examples_dir / "cp").exists() or not (
                examples_dir / "cp.ll").exists():
            pytest.skip("missing example binary / IR")

        ll = (examples_dir / "cp.ll").read_text()
        idapro.open_database(str(examples_dir / "cp"), True)
        try:
            assert hx.init_hexrays_plugin()
            from idavator.llvm_drop import LLVMDropConverter

            ea = ida_name.get_name_ea(ida_idaapi.BADADDR, "get_nonce")
            if ea == ida_idaapi.BADADDR:
                pytest.skip("get_nonce not in this cp build")
            conv = LLVMDropConverter(ll)
            cf = conv.drop(ea, "get_nonce")
            text = str(cf) if cf is not None else "<None>"
            # A real drop (not a native fallback): no build error.
            assert conv.last_error is None, conv.last_error
            assert cf is not None, "decompile failed"
            open_lines = [ln.strip() for ln in text.splitlines()
                          if "open(" in ln]
            assert open_lines, f"no open() call rendered in:\n{text}"
            joined = "\n".join(open_lines)
            # The /dev/urandom path literal must survive as open's first arg.
            assert "/dev/urandom" in joined, (
                f"open() lost its path literal (phantom-vararg "
                f"mis-render):\n{joined}")
            # And the phantom 3-arg form reading the function's own params must
            # NOT appear (the pre-fix signature).
            assert "open((const char *)a0" not in joined, (
                f"open() mis-rendered with phantom params:\n{joined}")
        finally:
            idapro.close_database()
