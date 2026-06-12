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
