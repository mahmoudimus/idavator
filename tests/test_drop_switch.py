"""GAP A guard: the LLVM ``switch`` terminator lowers to a chain of equality
tests that Hex-Rays re-folds into a switch / if-ladder.

``_build_multiblock`` PASS B previously raised ``NotImplementedError: unhandled
terminator 'switch'``. The drop now emits one ``jz %v, Ci -> Bi`` comparison
block per case (each 2-way, falling through to the next compare; the last falls
through to a default trampoline). This mirrors the conditional-``br`` machinery
(``_ICMP_JMP`` / ``m_jz`` / the ftramp planning / ``_wire``).

Two checks:
- a SYNTHETIC clean switch dropped into a linear host renders the per-case
  constants + a conditional, with no INTERR / build error;
- the real cp!``version_etc_arn`` (a 10-case ``switch``) drops to an if-ladder
  that matches its decompiled reference (modulo names).

Run:  PYTHONPATH=src pytest -m ida tests/test_drop_switch.py -s
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
    """A small function whose preoptimized microcode ends in m_ret with NO
    conditional tail -- a clean linear host to drop synthetic IR into."""
    for ea in idautils.Functions():
        f = ida_funcs.get_func(ea)
        if f is None or not (8 <= f.end_ea - f.start_ea <= 200):
            continue
        if hx.decompile(ea) is None:
            continue
        hf = hx.hexrays_failure_t()
        mbr = hx.mba_ranges_t()
        mbr.ranges.push_back(f)
        m = hx.gen_microcode(mbr, hf, None, hx.DECOMP_NO_WAIT,
                             hx.MMAT_PREOPTIMIZED)
        if m is None:
            continue
        tails = {int(b.tail.opcode) for i in range(m.qty)
                 if (b := m.get_mblock(i)) is not None and b.tail is not None}
        conds = {hx.m_jcnd, hx.m_jz, hx.m_jnz, hx.m_jtbl}
        if hx.m_ret in tails and not (tails & conds):
            return ea
    return None


# A clean 3-case switch (distinct return per arm; no phi/loop/noreturn) so the
# only thing under test is the switch lowering itself. 100=0x64, 200=0xC8,
# 300=0x12C. Hex-Rays renders the dispatch as a switch or an equivalent
# if-ladder over the case constants.
_SWITCH_IR = (
    "define i32 @sw(i32 %x) {\n"
    "entry:\n"
    "  switch i32 %x, label %def [ i32 1, label %a  i32 2, label %b  "
    "i32 3, label %c ]\n"
    "a:\n  ret i32 100\n"
    "b:\n  ret i32 200\n"
    "c:\n  ret i32 300\n"
    "def:\n  ret i32 0\n}\n"
)


@pytest.mark.ida
class TestSwitchDrop:
    def test_synthetic_switch_lowers(self, examples_dir: Path) -> None:
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

            conv = LLVMDropConverter(_SWITCH_IR)
            cf = conv.drop(host, "sw")
            text = str(cf) if cf is not None else "<None>"
            print(f"\n=== drop @sw (host={host:#x} "
                  f"interr={conv.last_interr} "
                  f"err={'yes' if conv.last_error else None}) ===\n{text}")
            assert conv.last_error is None, conv.last_error
            assert conv.last_interr is None, f"INTERR {conv.last_interr}"
            assert cf is not None, "decompile failed"
            assert "local variable allocation has failed" not in text, text
            # Each case constant must survive, and a dispatch (switch/if) over x.
            for needle in ("0x64", "0xC8", "0x12C"):
                assert needle in text, f"missing case {needle!r} in:\n{text}"
            assert ("switch" in text or "if" in text), \
                f"no switch/if dispatch in:\n{text}"
        finally:
            idapro.close_database()

    def test_real_version_etc_switch(self, examples_dir: Path) -> None:
        """cp!version_etc_arn has a 10-case switch; its dropped if-ladder must
        match the decompiled reference (modulo lvar names)."""
        if not _idalib():
            pytest.skip("idalib unavailable")
        import idapro
        import ida_hexrays as hx
        import ida_idaapi
        import ida_name

        if not (examples_dir / "cp").exists():
            pytest.skip("missing example binary")
        ll = (examples_dir / "cp.ll")
        if not ll.exists():
            pytest.skip("missing cp.ll")

        idapro.open_database(str(examples_dir / "cp"), True)
        try:
            assert hx.init_hexrays_plugin()
            from idavator.llvm_drop import LLVMDropConverter

            ea = ida_name.get_name_ea(ida_idaapi.BADADDR, "version_etc_arn")
            if ea == ida_idaapi.BADADDR:
                pytest.skip("version_etc_arn not in this build")
            orig = str(hx.decompile(ea))
            conv = LLVMDropConverter(ll.read_text())
            cf = conv.drop(ea, "version_etc_arn")
            text = str(cf) if cf is not None else "<None>"
            assert conv.last_error is None, conv.last_error
            assert conv.last_interr is None, f"INTERR {conv.last_interr}"
            assert cf is not None, "decompile failed"
            # The reference renders the switch as an if-ladder; the drop must
            # produce the SAME case ladder (==N comparisons over the same vals).
            ladder = [n for n in (1, 2, 3, 4, 5, 6, 7, 8, 9)
                      if f"== {n}" in orig]
            assert len(ladder) >= 5, f"reference ladder too small: {ladder}"
            for n in ladder:
                assert f"== {n}" in text, (
                    f"dropped C missing case '== {n}' present in reference:\n"
                    f"{text}")
        finally:
            idapro.close_database()
