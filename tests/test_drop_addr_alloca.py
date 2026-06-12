"""Address-taken scalar alloca (drop task #2).

An alloca whose address escapes (passed to a call, returned, stored as a value)
but is NOT GEP'd lands in an existing host frame slot; ``&local`` renders as
``mop_a(stkvar)``. The stack-passing call carries the host resting-frame ea so
Hex-Rays computes a frame-consistent ``mcallinfo.call_spd`` -- so the output is
free of WARN_BAD_CALL_SP ("bad sp value at call has been detected, the output
may be wrong"). This is the regression guard for the SP gate fix
(memory ``idavator_sp_gate_call_ea_cracked``): the drop must NEVER emit that
warning.
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


_CALLEES = ("memset", "_memset", "memcpy", "free")

# &local passed to a call; the call's result is returned (keeps both the call
# and &local live -- otherwise Hex-Rays DCEs the write-only local).
PROBE_CALL = """
define i64 @probe() {{
entry:
  %l = alloca i64, align 8
  %c = call i8* @{callee}(i8* %l, i32 0, i64 8)
  %r = ptrtoint i8* %c to i64
  ret i64 %r
}}
declare i8* @{callee}(i8*, i32, i64)
"""

# store-to-stkvar + &local-to-call + load-from-stkvar (all three slot paths).
PROBE_SCL = """
define i64 @probe(i64 %x) {{
entry:
  %l = alloca i64, align 8
  store i64 %x, i64* %l
  %c = call i8* @{callee}(i8* %l, i32 0, i64 8)
  %v = load i64, i64* %l
  ret i64 %v
}}
declare i8* @{callee}(i8*, i32, i64)
"""


def _resolve_callee():
    import ida_idaapi
    import ida_name

    for nm in _CALLEES:
        if ida_name.get_name_ea(ida_idaapi.BADADDR, nm) != ida_idaapi.BADADDR:
            return nm
    return None


def _hosts(n: int):
    """Decompilable, non-noreturn functions with a real frame (frsize >= 16)."""
    import ida_funcs
    import ida_hexrays
    import idautils

    out = []
    for ea in idautils.Functions():
        f = ida_funcs.get_func(ea)
        if f is None or int(getattr(f, "frsize", 0)) < 16:
            continue
        if f.flags & ida_funcs.FUNC_NORET:
            continue
        if ida_hexrays.decompile(ea) is not None:
            out.append(ea)
        if len(out) >= n:
            break
    return out


@pytest.mark.ida
class TestAddressTakenAllocaSP:
    def test_amp_local_to_call_is_warning_free(self, examples_dir: Path) -> None:
        if not _idalib():
            pytest.skip("idalib unavailable")
        import idapro
        import ida_hexrays

        binary = examples_dir / "cp"
        if not binary.exists():
            pytest.skip("missing example binary: cp")

        from idavator.llvm_drop import LLVMDropConverter

        idapro.open_database(str(binary), True)
        try:
            assert ida_hexrays.init_hexrays_plugin()
            callee = _resolve_callee()
            assert callee, "no resolvable libc callee in cp"
            hosts = _hosts(2)
            assert len(hosts) >= 2, "need two frsize>=16 hosts"

            for label, ir, host in (
                ("call-only", PROBE_CALL, hosts[0]),
                ("store+call+load", PROBE_SCL, hosts[1]),
            ):
                conv = LLVMDropConverter(ir.format(callee=callee))
                cf = conv.drop(host, "probe")
                assert conv.last_error is None, f"[{label}] {conv.last_error}"
                assert conv.last_interr is None, f"[{label}] INTERR {conv.last_interr}"
                assert cf is not None, f"[{label}] decompile returned None"
                txt = str(cf)
                # THE regression guard: the SP warning must be gone.
                assert "bad sp value" not in txt, (
                    f"[{label}] WARN_BAD_CALL_SP regressed:\n{txt}")
                # the call (and &local) must have survived + rendered.
                assert callee in txt, f"[{label}] call DCE'd:\n{txt}"
        finally:
            idapro.close_database()
