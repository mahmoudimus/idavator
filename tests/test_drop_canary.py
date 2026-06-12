"""Stack-canary elision. `-fstack-protector` makes the lift emit
`__readfsqword` (canary read) + a `__stack_chk_fail` fail branch ending in
`unreachable`. The optimizer ELIDES all of that from faithful output, so the
drop models every `__readfsqword` as ONE shared kreg (the `saved == reread`
compare folds `K == K` -> true), skips `__stack_chk_fail`, and routes
`unreachable` to the ret block -- Hex-Rays then prunes the dead fail branch.
See memory idavator_drop_canary_gate.
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
