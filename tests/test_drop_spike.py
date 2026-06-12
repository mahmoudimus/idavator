"""Drop-path spike: learn the TARGET microcode shape before rewriting llvm2ida.

The LLVM->microcode converter must produce a valid ``mba_t``. Before building that,
we need ground truth: what does valid microcode look like for a trivial function at
a low maturity, and does Hex-Rays decompile a hand-touched mba? This spike dumps the
reference shape from a real ``examples/cp`` function and probes ``create_empty_mba``.

Run:  pytest -m ida tests/test_drop_spike.py -s
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


def _dump_mba(mba, ida_hexrays, label: str) -> None:
    print(f"\n=== {label}: qty={mba.qty} maturity={int(mba.maturity)} "
          f"entry_ea={int(getattr(mba, 'entry_ea', 0)):#x} ===")
    # lvars
    try:
        nvars = mba.vars.size()
        print(f"  lvars: {nvars}")
        for i in range(nvars):
            v = mba.vars[i]
            flags = []
            for attr in ("is_arg_var", "is_result_var", "is_stk_var", "is_reg_var"):
                fn = getattr(v, attr, None)
                with_val = None
                try:
                    with_val = bool(fn()) if callable(fn) else bool(fn)
                except Exception:
                    with_val = None
                if with_val:
                    flags.append(attr)
            print(f"    [{i}] name={v.name!r} width={getattr(v, 'width', '?')} "
                  f"{'|'.join(flags)}")
    except Exception as exc:  # noqa: BLE001
        print(f"  lvars: <error {exc!r}>")
    # blocks
    for i in range(mba.qty):
        blk = mba.get_mblock(i)
        if blk is None:
            continue
        succs = [int(s) for s in blk.succset]
        preds = [int(p) for p in blk.predset]
        print(f"  blk[{i}] type={int(blk.type)} nsucc={blk.nsucc()} "
              f"succs={succs} preds={preds}")
        ins = blk.head
        n = 0
        while ins is not None and n < 12:
            print(f"        op={int(ins.opcode):>3}  {ins.dstr()}")
            ins = ins.next
            n += 1


@pytest.mark.ida
class TestDropSpike:
    def test_reference_shape_and_empty_mba(self, examples_dir: Path) -> None:
        if not _idalib():
            pytest.skip("idalib unavailable")
        import idapro
        import ida_funcs
        import ida_hexrays
        import idautils

        binary = examples_dir / "cp"
        if not binary.exists():
            pytest.skip(f"missing example binary: {binary}")

        from idavator.cfg_verify import try_verify

        idapro.open_database(str(binary), True)
        try:
            assert ida_hexrays.init_hexrays_plugin(), "hexrays unavailable"

            # 1) Find the smallest non-trivial function that decompiles -> the
            #    reference "trivial function" shape.
            ref_ea = None
            for ea in idautils.Functions():
                f = ida_funcs.get_func(ea)
                if f is None:
                    continue
                size = f.end_ea - f.start_ea
                if not (8 <= size <= 64):
                    continue
                if ida_hexrays.decompile(ea) is not None:
                    ref_ea = ea
                    break
            assert ref_ea is not None, "no small decompilable function found"
            print(f"\nreference function @ {ref_ea:#x} "
                  f"(size={ida_funcs.get_func(ref_ea).end_ea - ida_funcs.get_func(ref_ea).start_ea})")
            print("baseline pseudocode:\n", str(ida_hexrays.decompile(ref_ea)))

            # 2) Dump its microcode at a LOW maturity (the build target shape).
            for matname in ("MMAT_GENERATED", "MMAT_PREOPTIMIZED", "MMAT_LOCOPT"):
                mat = getattr(ida_hexrays, matname, None)
                if mat is None:
                    continue
                hf = ida_hexrays.hexrays_failure_t()
                mbr = ida_hexrays.mba_ranges_t()
                mbr.ranges.push_back(ida_funcs.get_func(ref_ea))
                mba = ida_hexrays.gen_microcode(
                    mbr, hf, None, ida_hexrays.DECOMP_NO_WAIT, mat)
                if mba is None:
                    print(f"\n{matname}: gen_microcode -> None ({hf.code}/{hf.desc()!r})")
                    continue
                ok, code = try_verify(mba, matname)
                _dump_mba(mba, ida_hexrays, f"{matname} (verify ok={ok} interr={code})")

            # (create_cfunc / edit-flow checks moved to a dedicated test below.)
            # 3) Probe create_empty_mba (the build-from-scratch entry point).
            hf2 = ida_hexrays.hexrays_failure_t()
            mbr2 = ida_hexrays.mba_ranges_t()
            mbr2.ranges.push_back(ida_funcs.get_func(ref_ea))
            empty = ida_hexrays.create_empty_mba(mbr2, hf2)
            if empty is None:
                print(f"\ncreate_empty_mba -> None ({hf2.code}/{hf2.desc()!r})")
            else:
                ok, code = try_verify(empty, "create_empty_mba")
                print(f"\n=== create_empty_mba: qty={empty.qty} "
                      f"maturity={int(empty.maturity)} verify_ok={ok} interr={code} ===")
        finally:
            idapro.close_database()


def _find_small_decompilable(ida_funcs, ida_hexrays, idautils, lo=8, hi=80):
    for ea in idautils.Functions():
        f = ida_funcs.get_func(ea)
        if f is None:
            continue
        if not (lo <= f.end_ea - f.start_ea <= hi):
            continue
        if ida_hexrays.decompile(ea) is not None:
            return ea
    return None


@pytest.mark.ida
class TestCreateCfuncMechanism:
    """Prove the drop MECHANISM: build/edit an mba, then create_cfunc -> pseudocode.
    This is how an LLVM-derived mba becomes decompiler output (the decompile path
    re-gens microcode and would ignore our edits; create_cfunc consumes OUR mba)."""

    def _gen(self, ida_hexrays, ida_funcs, ea, reqmat=None):
        hf = ida_hexrays.hexrays_failure_t()
        mbr = ida_hexrays.mba_ranges_t()
        mbr.ranges.push_back(ida_funcs.get_func(ea))
        if reqmat is None:
            return ida_hexrays.gen_microcode(mbr, hf, None, ida_hexrays.DECOMP_NO_WAIT), hf
        return ida_hexrays.gen_microcode(
            mbr, hf, None, ida_hexrays.DECOMP_NO_WAIT, reqmat), hf

    def test_create_cfunc_reproduces_and_edit_flows(self, examples_dir: Path) -> None:
        if not _idalib():
            pytest.skip("idalib unavailable")
        import idapro
        import ida_funcs
        import ida_hexrays
        import idautils

        binary = examples_dir / "cp"
        if not binary.exists():
            pytest.skip(f"missing example binary: {binary}")
        from idavator.cfg_verify import try_verify

        idapro.open_database(str(binary), True)
        try:
            assert ida_hexrays.init_hexrays_plugin()
            ea = _find_small_decompilable(ida_funcs, ida_hexrays, idautils)
            assert ea is not None
            baseline = str(ida_hexrays.decompile(ea))
            print(f"\n=== ref {ea:#x} baseline ===\n{baseline}")

            # (1) create_cfunc on a freshly-gen'd mba -> should reproduce baseline.
            #     MUST gen at the FINAL maturity MMAT_LVARS (8); at GLBOPT3 (7)
            #     create_cfunc yields a degenerate empty function.
            mba, hf = self._gen(ida_hexrays, ida_funcs, ea, reqmat=ida_hexrays.MMAT_LVARS)
            print(f"gen: mba={mba is not None} mat={int(mba.maturity) if mba else '-'} "
                  f"qty={mba.qty if mba else '-'} hf={hf.code}/{hf.desc()!r}")
            ok, code = try_verify(mba, "gen default")
            print(f"verify ok={ok} interr={code}")
            try:
                cf = ida_hexrays.create_cfunc(mba)
                print(f"=== create_cfunc reproduced ===\n{cf}")
            except Exception as exc:  # noqa: BLE001
                print(f"create_cfunc FAILED: {exc!r}")
                cf = None

            # (2) edit-flow: find a function WITH a numeric const, swap it,
            #     create_cfunc, and confirm the edit appears in pseudocode.
            cea = None
            for cand in idautils.Functions():
                cf2_func = ida_funcs.get_func(cand)
                if cf2_func is None or not (8 <= cf2_func.end_ea - cf2_func.start_ea <= 200):
                    continue
                m, _h = self._gen(ida_hexrays, ida_funcs, cand, reqmat=ida_hexrays.MMAT_LVARS)
                if m is None:
                    continue
                has_const = False
                for i in range(m.qty):
                    b = m.get_mblock(i)
                    ins = b.head if b else None
                    while ins is not None:
                        if (int(ins.opcode) == ida_hexrays.m_mov and ins.l is not None
                                and ins.l.t == ida_hexrays.mop_n
                                and 1 <= int(ins.l.nnn.value) <= 0xFFFF):
                            has_const = True
                            break
                        ins = ins.next
                    if has_const:
                        break
                if has_const:
                    cea = cand
                    break
            print(f"edit-target function: {cea:#x}" if cea else "edit-target: none")
            mba2, _ = self._gen(ida_hexrays, ida_funcs, cea or ea, reqmat=ida_hexrays.MMAT_LVARS)
            swapped = False
            for i in range(mba2.qty):
                blk = mba2.get_mblock(i)
                ins = blk.head if blk else None
                while ins is not None:
                    if (ins.l is not None and ins.l.t == ida_hexrays.mop_n
                            and int(ins.opcode) == ida_hexrays.m_mov):
                        old = int(ins.l.nnn.value)
                        ins.l.nnn.update_value(0x539)
                        print(f"  swapped const {old:#x} -> 0x539 in blk[{i}] ({ins.dstr()})")
                        swapped = True
                        break
                    ins = ins.next
                if swapped:
                    break
            print(f"swapped_any={swapped}")
            if swapped:
                ok2, code2 = try_verify(mba2, "after const swap")
                print(f"verify-after-edit ok={ok2} interr={code2}")
                try:
                    cf2 = ida_hexrays.create_cfunc(mba2)
                    txt = str(cf2)
                    print(f"=== create_cfunc after edit ===\n{txt}")
                    print(f"EDIT VISIBLE (0x539): {'0x539' in txt or '1337' in txt}")
                except Exception as exc:  # noqa: BLE001
                    print(f"create_cfunc(edited) FAILED: {exc!r}")
        finally:
            idapro.close_database()


@pytest.mark.ida
class TestModel2Hook:
    """Model 2 (the right drop architecture): hook the decompile at the EARLIEST
    microcode maturity (hxe_microcode), edit/rebuild the mba, and let the NORMAL
    decompile() pipeline run to completion -- it builds the ctree -> FULL
    pseudocode (unlike create_cfunc, which gave an empty body). Editing the
    UNWIRED, register-based MMAT_GENERATED microcode means Hex-Rays does the CFG
    wiring + lvar allocation + optimization for us."""

    def test_edit_at_microcode_flows_to_full_pseudocode(self, examples_dir: Path) -> None:
        if not _idalib():
            pytest.skip("idalib unavailable")
        import idapro
        import ida_funcs
        import ida_hexrays
        import idautils

        binary = examples_dir / "cp"
        if not binary.exists():
            pytest.skip("missing example binary")

        idapro.open_database(str(binary), True)
        try:
            assert ida_hexrays.init_hexrays_plugin()

            # Find a function with a size>=4 numeric const at MMAT_GENERATED.
            SWAP_TO = 0xCAFEBABE
            target = None
            for ea in idautils.Functions():
                f = ida_funcs.get_func(ea)
                if f is None or not (8 <= f.end_ea - f.start_ea <= 400):
                    continue
                hf = ida_hexrays.hexrays_failure_t()
                mbr = ida_hexrays.mba_ranges_t()
                mbr.ranges.push_back(f)
                m = ida_hexrays.gen_microcode(
                    mbr, hf, None, ida_hexrays.DECOMP_NO_WAIT, ida_hexrays.MMAT_GENERATED)
                if m is None:
                    continue
                found = False
                for i in range(m.qty):
                    b = m.get_mblock(i)
                    ins = b.head if b else None
                    while ins is not None:
                        if (int(ins.opcode) == ida_hexrays.m_mov and ins.l is not None
                                and ins.l.t == ida_hexrays.mop_n and ins.l.size >= 4
                                and int(ins.l.nnn.value) not in (0, SWAP_TO)):
                            found = True
                            break
                        ins = ins.next
                    if found:
                        break
                if found:
                    target = ea
                    break
            assert target is not None, "no function with a size>=4 const found"
            baseline = str(ida_hexrays.decompile(target))
            print(f"\n=== target {target:#x} baseline ===\n{baseline}")

            box = {"fired": False, "swapped": None}

            class _MicrocodeHook(ida_hexrays.Hexrays_Hooks):
                def microcode(self, mba):  # hxe_microcode (MMAT_GENERATED)
                    if box["fired"]:
                        return 0
                    box["fired"] = True
                    for i in range(mba.qty):
                        b = mba.get_mblock(i)
                        ins = b.head if b else None
                        while ins is not None:
                            if (int(ins.opcode) == ida_hexrays.m_mov and ins.l is not None
                                    and ins.l.t == ida_hexrays.mop_n and ins.l.size >= 4
                                    and int(ins.l.nnn.value) not in (0, SWAP_TO)):
                                box["swapped"] = int(ins.l.nnn.value)
                                ins.l.nnn.update_value(SWAP_TO)
                                return 0
                            ins = ins.next
                    return 0

            hook = _MicrocodeHook()
            assert hook.hook()
            try:
                ida_hexrays.mark_cfunc_dirty(target)
                cf = ida_hexrays.decompile(target)
            finally:
                hook.unhook()
            text = str(cf) if cf is not None else "<None>"
            print(f"\n=== after microcode-stage edit (fired={box['fired']} "
                  f"swapped={hex(box['swapped']) if box['swapped'] else None} -> {hex(SWAP_TO)}) ===")
            print(text)
            assert box["fired"], "microcode hook never fired"
            assert cf is not None, "decompile returned None after edit"
            body_nonempty = text.count(";") > 1 or "return" in text
            print(f"\nBODY NON-EMPTY: {body_nonempty}")
            print(f"EDIT VISIBLE (CAFEBABE/3405691582): "
                  f"{'CAFEBABE' in text.upper() or '3405691582' in text}")
        finally:
            idapro.close_database()
