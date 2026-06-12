"""Full-body LLVM->microcode drop: clear the host body, emit ONLY the LLVM.

Generalizes the kernel (test_drop_return.py) to: (a) a clean full-body replace --
clear the host's instructions so no residue survives; (b) a real op map; (c)
multi-instruction functions via scratch registers (alloc_kreg) for SSA results.
Lowers `(x*3)+7` to a clean `return 3 * x + 7`.

Run:  PYTHONPATH=src pytest -m ida tests/test_drop_fullbody.py -s
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


def _type_size(type_str: str) -> int:
    s = str(type_str)
    if "i1" in s and "i16" not in s and "i1" != s.strip():
        pass
    for tok, sz in (("i64", 8), ("i32", 4), ("i16", 2), ("i8", 1), ("i1", 1),
                    ("double", 8), ("float", 4)):
        if tok in s:
            return sz
    if "*" in s:
        return 8
    return 4


def _drop(examples_dir: Path, ir_text: str, llvm_fn_name: str):
    """Drop a single-block LLVM function into a linear host; return the pseudocode."""
    import idapro
    import ida_funcs
    import ida_hexrays
    import ida_idp
    import ida_typeinf
    import idautils
    import llvmlite.binding as llvm

    from idavator.cfg_verify import try_verify

    hx = ida_hexrays
    COND = {hx.m_jcnd, hx.m_jz, hx.m_jnz, hx.m_jae, hx.m_jb, hx.m_ja, hx.m_jbe,
            hx.m_jg, hx.m_jge, hx.m_jl, hx.m_jle, hx.m_jtbl}
    TERM = COND | {hx.m_goto, hx.m_ret}
    OPMAP = {"add": hx.m_add, "sub": hx.m_sub, "mul": hx.m_mul, "and": hx.m_and,
             "or": hx.m_or, "xor": hx.m_xor, "shl": hx.m_shl, "lshr": hx.m_shr,
             "ashr": hx.m_sar}

    mod = llvm.parse_assembly(ir_text)
    fn = next(g for g in mod.functions if g.name == llvm_fn_name and not g.is_declaration)
    args = list(fn.arguments)
    blocks = list(fn.blocks)
    assert len(blocks) == 1, "single-block only for this milestone"

    idapro.open_database(str(examples_dir / "cp"), True)
    try:
        assert hx.init_hexrays_plugin()
        # mregs MUST be resolved AFTER the DB/processor is loaded.
        ABI = [hx.reg2mreg(ida_idp.str2reg(r))
               for r in ("rdi", "rsi", "rdx", "rcx", "r8", "r9")]
        EAX = hx.reg2mreg(ida_idp.str2reg("rax"))
        print(f"mregs: rax={EAX} rdi={ABI[0]}")

        # Linear host (no conditional terminators) with an m_ret at PREOPTIMIZED.
        host = None
        for ea in idautils.Functions():
            f = ida_funcs.get_func(ea)
            if f is None or not (8 <= f.end_ea - f.start_ea <= 200):
                continue
            if ida_hexrays.decompile(ea) is None:
                continue
            hf = hx.hexrays_failure_t()
            mbr = hx.mba_ranges_t()
            mbr.ranges.push_back(f)
            m = hx.gen_microcode(mbr, hf, None, hx.DECOMP_NO_WAIT, hx.MMAT_PREOPTIMIZED)
            if m is None:
                continue
            tails = [int(b.tail.opcode) for i in range(m.qty)
                     if (b := m.get_mblock(i)) is not None and b.tail is not None]
            if hx.m_ret in tails and not (set(tails) & COND):
                host = ea
                break
        assert host is not None, "no linear host with m_ret found"

        # Force the prototype from the LLVM signature: int f(int, int, ...).
        params = ", ".join(f"int a{i}" for i in range(len(args)))
        tif = ida_typeinf.tinfo_t()
        ida_typeinf.parse_decl(tif, None, f"int __fastcall f({params});", 0)
        ida_typeinf.apply_tinfo(host, tif, ida_typeinf.TINFO_DEFINITE)

        box = {"interr": None, "err": None}

        class _Hook(hx.Hexrays_Hooks):
            def preoptimized(self, mba):
                try:
                    ret_blk = None
                    for i in range(mba.qty):
                        b = mba.get_mblock(i)
                        if b is not None and b.tail is not None and int(b.tail.opcode) == hx.m_ret:
                            ret_blk = b
                            break
                    if ret_blk is None:
                        return 0
                    ea = int(ret_blk.tail.ea)

                    # (a) Full-body replace: drop every non-terminator instruction.
                    for i in range(mba.qty):
                        b = mba.get_mblock(i)
                        if b is None:
                            continue
                        ins = b.head
                        while ins is not None:
                            nxt = ins.next
                            if int(ins.opcode) not in TERM:
                                b.remove_from_block(ins)
                            ins = nxt
                        b.mark_lists_dirty()

                    # (b/c) Emit the LLVM computation into the ret block.
                    valmap: dict[str, tuple] = {}
                    for i, a in enumerate(args):
                        valmap[a.name] = ("reg", ABI[i], _type_size(a.type))

                    def desc(operand, default_size):
                        # llvmlite str(operand) of an SSA result is the FULL
                        # defining instruction; use the structured .name instead.
                        nm = operand.name
                        if nm and nm in valmap:
                            return valmap[nm]
                        s = str(operand).strip()
                        num = re.search(r"(-?\d+)\s*$", s)
                        if num:
                            return ("num", int(num.group(1)),
                                    _type_size(s) or default_size)
                        raise ValueError(f"unhandled operand {s!r}")

                    def fill(mop, d):
                        kind, val, size = d
                        if kind == "reg":
                            mop.make_reg(val, size)
                        else:
                            mop.make_number(val & ((1 << (8 * size)) - 1), size)

                    anchor = ret_blk.tail.prev  # insert before the ret
                    for bb in blocks:
                        for ins in bb.instructions:
                            op = ins.opcode
                            ops = list(ins.operands)
                            if op in OPMAP:
                                size = _type_size(ins.type)
                                mi = hx.minsn_t(ea)
                                mi.opcode = OPMAP[op]
                                fill(mi.l, desc(ops[0], size))
                                fill(mi.r, desc(ops[1], size))
                                kreg = mba.alloc_kreg(size)
                                mi.d.make_reg(kreg, size)
                                ret_blk.insert_into_block(mi, anchor)
                                anchor = mi
                                valmap[ins.name] = ("reg", kreg, size)
                            elif op == "ret":
                                mi = hx.minsn_t(ea)
                                mi.opcode = hx.m_mov
                                fill(mi.l, desc(ops[0], 4))
                                mi.d.make_reg(EAX, 4)
                                ret_blk.insert_into_block(mi, anchor)
                                anchor = mi
                    ret_blk.mark_lists_dirty()
                    ok, code = try_verify(mba, "after full-body emit")
                    box["interr"] = code
                    if not ok:
                        for i in range(mba.qty):
                            b = mba.get_mblock(i)
                            if b is None:
                                continue
                            row, x = [], b.head
                            while x is not None:
                                row.append(x.dstr())
                                x = x.next
                            print(f"  POST-EMIT blk[{i}] t={int(b.type)}: {row}")
                except Exception as exc:  # noqa: BLE001
                    box["err"] = repr(exc)
                return 0

        hook = _Hook()
        assert hook.hook()
        try:
            hx.mark_cfunc_dirty(host)
            cf = hx.decompile(host)
        finally:
            hook.unhook()
        return (str(cf) if cf is not None else "<None>"), box, host
    finally:
        idapro.close_database()


@pytest.mark.ida
class TestFullBodyDrop:
    def test_multi_op_drops_clean(self, examples_dir: Path) -> None:
        if not _idalib():
            pytest.skip("idalib unavailable")
        ir = ("define i32 @f(i32 %x) {\nentry:\n"
              "  %a = mul i32 %x, 3\n  %b = add i32 %a, 7\n  ret i32 %b\n}\n")
        text, box, host = _drop(examples_dir, ir, "f")
        print(f"\n=== full-body drop (host={host:#x} interr={box['interr']} "
              f"err={box['err']}) ===\n{text}")
        assert box["err"] is None, f"hook error: {box['err']}"
        assert text != "<None>", "decompile failed"
        # clean: the host's side effects are gone; only our computation remains.
        assert "* 3" in text or "3 *" in text, f"mul missing:\n{text}"
        assert "+ 7" in text, f"add missing:\n{text}"
        assert "gmon" not in text and "MEMORY" not in text, f"host residue:\n{text}"
