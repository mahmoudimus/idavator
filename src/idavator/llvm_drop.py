"""LLVM IR -> Hex-Rays microcode DROP (Model 2).

The drop hooks the decompile at ``hxe_preoptimized``, rebuilds the target
function's microcode from the LLVM IR (full-body replace), and lets the normal
``decompile()`` pipeline wire the CFG, allocate lvars, optimize, and build the
ctree -> clean pseudocode. We build DIRECTLY on LLVM (LLVM IR is already the IR --
no  portable IR); ``cfg_verify`` decodes any INTERR.

Proven slices folded here: full-body replace, the op map, SSA results via
``alloc_kreg`` scratch registers, ABI arg mapping, casts. Multi-block control flow
(the 2-way if/else mechanic) is proven in tests/test_drop_controlflow.py and is the
next layered addition. See memory ``idavator_drop_microcode_hook_architecture``.
"""
from __future__ import annotations

import logging
import re
from contextlib import suppress

import ida_bytes
import ida_frame
import ida_funcs
import ida_hexrays as hx
import ida_ida
import ida_idaapi
import ida_idp
import ida_name
import ida_strlist
import ida_typeinf
import llvmlite.binding as llvm

from idavator.cfg_verify import try_verify

logger = logging.getLogger("idavator.llvm_drop")

# Instruction-tail opcodes (terminators / control flow) -- never cleared.
_TERM = frozenset({
    hx.m_goto, hx.m_ret, hx.m_jcnd, hx.m_jz, hx.m_jnz, hx.m_jtbl,
    hx.m_jae, hx.m_jb, hx.m_ja, hx.m_jbe, hx.m_jg, hx.m_jge, hx.m_jl, hx.m_jle,
})

_BINOP = {
    "add": hx.m_add, "sub": hx.m_sub, "mul": hx.m_mul, "and": hx.m_and,
    "or": hx.m_or, "xor": hx.m_xor, "shl": hx.m_shl, "lshr": hx.m_shr,
    "ashr": hx.m_sar,
    "udiv": hx.m_udiv, "sdiv": hx.m_sdiv, "urem": hx.m_umod, "srem": hx.m_smod,
}
# zext/sext/trunc -> microcode widening/narrowing.
_CAST = {"zext": hx.m_xdu, "sext": hx.m_xds, "trunc": hx.m_low}
# bit-identical reinterpretations -> no microcode, just alias the operand.
_NOOP_CAST = frozenset({"bitcast", "ptrtoint", "inttoptr"})

# icmp predicate -> a 2-way conditional jump that branches to ``d`` when the
# predicate holds (the fall-through, serial+1, takes the FALSE arm).
_ICMP_JMP = {
    "eq": hx.m_jz, "ne": hx.m_jnz,
    "ugt": hx.m_ja, "uge": hx.m_jae, "ult": hx.m_jb, "ule": hx.m_jbe,
    "sgt": hx.m_jg, "sge": hx.m_jge, "slt": hx.m_jl, "sle": hx.m_jle,
}

# IDA rotate intrinsics that survive into FAITHFUL pseudocode (e.g. rotr_sz ->
# `return __ROR8__(a0, a1)`) -> emit as a Hex-Rays helper call. Deliberately NOT
# the stack canary (`__readfsqword`/`__stack_chk_fail`): the decompiler ELIDES
# that boilerplate from final output, so reconstructing it would DIVERGE from the
# round-trip reference (and `make_helper` + reg-arg movs crashes the decompiler;
# the args must ride in the mcallinfo via create_helper_call).
_HELPER_INTRINSICS = frozenset({
    "__ROL1__", "__ROL2__", "__ROL4__", "__ROL8__",
    "__ROR1__", "__ROR2__", "__ROR4__", "__ROR8__",
})

# A private string-constant global used as a VALUE (e.g. an error message handed
# to a call/printf, decayed via getelementptr) stringifies in llvmlite as its
# full definition: ``@name = private [unnamed_addr ]constant [N x i8] c"..."``.
# LLVM truncates the symbol (``aInvalidKindInG``) and IDA auto-names the literal
# from its (longer) content (``aInvalidKindInGenTempname``), so get_name_ea on
# the LLVM name misses. We instead decode the c"..." body and match it against
# the IDB string table by exact content -> the literal's address.
_STRCONST_RE = re.compile(
    r'private\s+(?:unnamed_addr\s+)?constant\s+\[\d+\s+x\s+i8\]\s+c"(.*)"\s*$',
    re.S,
)


def _decode_llvm_cstr(body: str) -> bytes:
    """Decode an LLVM ``c"..."`` string body to raw bytes. LLVM escapes a byte as
    ``\\XX`` (two hex digits, e.g. ``\\22`` = ``"``, ``\\0a`` = ``\\n``, ``\\00`` =
    NUL); every other character is a literal byte. The trailing NUL is preserved
    here and stripped by the caller for matching against IDB strlit contents."""
    out = bytearray()
    i, n = 0, len(body)
    while i < n:
        ch = body[i]
        if ch == "\\" and i + 3 <= n:
            try:
                out.append(int(body[i + 1:i + 3], 16))
                i += 3
                continue
            except ValueError:
                pass
        out.append(ord(ch) & 0xFF)
        i += 1
    return bytes(out)


def _type_size(type_str) -> int:
    # llvmlite uses OPAQUE pointers (LLVM 14+): a pointer stringifies to "ptr",
    # not "i32*", and the pointee type is lost -- so a string "*" probe misses it.
    # Detect the pointer first via the structured ``is_pointer`` flag.
    if getattr(type_str, "is_pointer", False):
        return 8
    s = str(type_str)
    if s == "ptr" or "*" in s:
        return 8
    for tok, sz in (("i64", 8), ("i32", 4), ("i16", 2), ("i8", 1), ("i1", 1),
                    ("double", 8), ("float", 4)):
        if tok in s:
            return sz
    return 4


def _round_up(x: int, a: int) -> int:
    return (x + a - 1) // a * a if a else x


def _split_fields(s: str) -> list[str]:
    """Top-level comma split of a struct body, respecting [] / {} nesting."""
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def _type_sa(tok: str, structs: dict) -> tuple[int, int]:
    """(size, align) of an LLVM type token; nested ``%name`` via ``structs``."""
    tok = tok.strip()
    if tok == "ptr" or tok.endswith("*"):
        return (8, 8)
    m = re.match(r"i(\d+)$", tok)
    if m:
        sz = max(1, (int(m.group(1)) + 7) // 8)
        return (sz, sz if sz in (1, 2, 4, 8) else 1)
    if tok in ("double", "float"):
        return (8, 8) if tok == "double" else (4, 4)
    m = re.match(r"\[\s*(\d+)\s+x\s+(.+)\]$", tok)
    if m:
        esz, eal = _type_sa(m.group(2), structs)
        return (int(m.group(1)) * _round_up(esz, eal), eal)
    # struct name -- llvmlite renders `%timespec`, the IR text declares
    # `%"timespec"`; canonicalize by stripping quotes.
    if tok.replace('"', "") in structs:
        return structs[tok.replace('"', "")]
    raise ValueError(f"unsized type {tok!r}")


def _parse_struct_layouts(ir_text: str) -> dict:
    """name -> (size, align) for every ``%name = type {..}`` whose layout is
    computable (natural C alignment). Un-computable structs are absent (the drop
    raises, the round-trip records a build error)."""
    raw = {m.group(1).strip().replace('"', ""): _split_fields(m.group(2))
           for m in re.finditer(r'(%[\w".:$]+)\s*=\s*type\s*\{(.*)\}', ir_text)}
    structs: dict = {}

    def layout(name, seen):
        name = name.replace('"', "")
        if name in structs:
            return structs[name]
        if name in seen:
            raise ValueError("recursive struct")
        off, maxal = 0, 1
        for f in raw[name]:
            if f.replace('"', "") in raw:
                layout(f, seen | {name})
            sz, al = _type_sa(f, structs)
            off = _round_up(off, al) + sz
            maxal = max(maxal, al)
        structs[name] = (_round_up(off, maxal), maxal)
        return structs[name]

    for nm in raw:
        with suppress(Exception):
            layout(nm, set())
    return structs


class LLVMDropConverter:
    """Drop a (straight-line) LLVM function into a host's decompiled output."""

    def __init__(self, ir_text: str):
        self.module = llvm.parse_assembly(ir_text)
        self._struct_size = _parse_struct_layouts(ir_text)  # name -> (size, align)
        self._allocas: dict = {}  # set per-drop in _build (scalar-slot kregs)
        self._addr_taken: dict = {}  # name -> (stkoff, size) for &local allocas
        self._cur_mba = None         # current mba (make_stkvar needs it)
        self._call_spd_ea = None     # host resting-frame ea for stack-passing calls
        self._canary_kreg = None     # shared kreg for __readfsqword (canary fold)
        self._ret_off = None         # frame off of a promoted return slot, or None
        self._ret_kreg = None        # kreg of a promoted scalar return slot, or None
        self._str_index = None       # IDB string-literal content -> ea (lazy)

    # -- public API ------------------------------------------------------
    def drop(self, host_ea: int, llvm_fn_name: str):
        """Rebuild ``host_ea``'s microcode from the named LLVM function and return
        the resulting ``cfunc_t`` (or None on failure)."""
        fn = next((g for g in self.module.functions
                   if g.name == llvm_fn_name and not g.is_declaration), None)
        if fn is None:
            raise ValueError(f"no definition for @{llvm_fn_name}")
        self._force_prototype(host_ea, fn)
        # A FUNC_NORET host would keep its __noreturn attribute and discard the
        # rebuilt return path; clear it so the dropped body (which returns) shows.
        hf = ida_funcs.get_func(host_ea)
        if hf is not None and (hf.flags & ida_funcs.FUNC_NORET):
            hf.flags &= ~ida_funcs.FUNC_NORET
            ida_funcs.update_func(hf)

        box = {"interr": None, "err": None}
        conv = self

        class _Hook(hx.Hexrays_Hooks):
            def preoptimized(self, mba):  # hxe_preoptimized: m_ret exists
                try:
                    conv._build(mba, fn)
                    ok, code = try_verify(mba, "after llvm drop")
                    box["interr"] = code
                except Exception:  # noqa: BLE001
                    import traceback
                    box["err"] = traceback.format_exc()
                return 0

        hook = _Hook()
        hook.hook()
        try:
            hx.mark_cfunc_dirty(host_ea)
            cf = hx.decompile(host_ea)
        finally:
            hook.unhook()
        if box["err"]:
            logger.error("drop build failed:\n%s", box["err"])
        self.last_interr = box["interr"]
        self.last_error = box["err"]
        return cf

    # -- internals -------------------------------------------------------
    @staticmethod
    def _pointee_size(fn, arg) -> int:
        """Opaque pointers drop the pointee type; recover the access width from
        the first load/store that consumes ``arg`` (default 4)."""
        for bb in fn.blocks:
            for ins in bb.instructions:
                ops = list(ins.operands)
                if ins.opcode == "load" and ops and ops[0].name == arg.name:
                    return _type_size(ins.type)
                if (ins.opcode == "store" and len(ops) >= 2
                        and ops[1].name == arg.name):
                    return _type_size(ops[0].type)
        return 4

    @classmethod
    def _arg_ctype(cls, arg, fn) -> str:
        if getattr(arg.type, "is_pointer", False) or str(arg.type) == "ptr":
            # Match the pointee C type to the access width so an N-byte ldx/stx
            # renders as a clean ``*a`` (no reinterpret cast).
            return {1: "char *", 2: "__int16 *", 4: "_DWORD *",
                    8: "__int64 *"}.get(cls._pointee_size(fn, arg), "_DWORD *")
        s = str(arg.type)
        if "i64" in s:
            return "__int64"
        if "i16" in s:
            return "__int16"
        if "i8" in s:
            return "char"
        return "int"

    @staticmethod
    def _ret_ctype(fn) -> str:
        for bb in fn.blocks:
            for ins in bb.instructions:
                if ins.opcode == "ret":
                    ops = list(ins.operands)
                    if not ops:
                        return "void"
                    t = ops[0].type
                    if getattr(t, "is_pointer", False) or str(t) == "ptr":
                        return "void *"
                    s = str(t)
                    if "i64" in s:
                        return "__int64"
                    if "i16" in s:
                        return "__int16"
                    if "i8" in s:
                        return "char"
                    return "int"
        return "void"

    @classmethod
    def _force_prototype(cls, host_ea: int, fn) -> None:
        args = list(fn.arguments)
        params = ", ".join(f"{cls._arg_ctype(a, fn)} a{i}"
                           for i, a in enumerate(args)) or "void"
        ret = cls._ret_ctype(fn)
        tif = ida_typeinf.tinfo_t()
        ida_typeinf.parse_decl(tif, None, f"{ret} __fastcall f({params});", 0)
        ida_typeinf.apply_tinfo(host_ea, tif, ida_typeinf.TINFO_DEFINITE)

    @staticmethod
    def _abi():
        # Integer arg registers depend on the target ABI: a PE (Windows) target
        # uses the Microsoft x64 convention (rcx/rdx/r8/r9); everything else
        # (ELF/Mach-O) uses System V (rdi/rsi/rdx/rcx/r8/r9). Return reg = rax.
        if ida_ida.inf_get_filetype() == ida_ida.f_PE:
            names = ("rcx", "rdx", "r8", "r9")
        else:
            names = ("rdi", "rsi", "rdx", "rcx", "r8", "r9")
        argregs = [hx.reg2mreg(ida_idp.str2reg(r)) for r in names]
        return (argregs, hx.reg2mreg(ida_idp.str2reg("rax")),
                hx.reg2mreg(ida_idp.str2reg("ds")))

    # -- microcode emit helpers -----------------------------------------
    def _fill(self, mop, d) -> None:
        kind, val, size = d
        if kind == "reg":
            mop.make_reg(val, size)
        elif kind == "gvar":
            mop.make_gvar(val)
            mop.size = size
        elif kind == "stkaddr":
            # &local: mop_a wrapping a stkvar (cf. &global = mop_a wrapping mop_v).
            inner = hx.mop_addr_t()
            inner.make_stkvar(self._cur_mba, val)
            mop.t = hx.mop_a
            mop.a = inner
            mop.size = 8
        else:
            mop.make_number(val & ((1 << (8 * size)) - 1), size)

    def _strconst_ea(self, operand_str: str):
        """Resolve a private string-constant operand (``@x = private constant
        [N x i8] c"..."``) to the address of the matching IDB string literal, or
        None. The LLVM symbol is truncated and IDA names the literal from its
        (longer) content, so name lookup fails; we match by exact content against
        the IDB string table instead (built once, cached on the instance)."""
        m = _STRCONST_RE.search(operand_str)
        if m is None:
            return None
        content = _decode_llvm_cstr(m.group(1)).rstrip(b"\x00")
        if self._str_index is None:
            self._str_index = {}
            ida_strlist.build_strlist()
            for i in range(ida_strlist.get_strlist_qty()):
                si = ida_strlist.string_info_t()
                if not ida_strlist.get_strlist_item(si, i):
                    continue
                raw = ida_bytes.get_strlit_contents(si.ea, si.length, si.type)
                if raw is None:
                    continue
                # First literal wins (lowest index == lowest ea for a content).
                self._str_index.setdefault(raw.rstrip(b"\x00"), si.ea)
        return self._str_index.get(content)

    def _desc(self, operand, vmap, default_size):
        """Resolve an LLVM operand to a value descriptor (reg kreg / numeric /
        gvar / stkaddr). A global used as a VALUE is its address (an array global
        decays; make_gvar carries the symbol); an address-taken alloca is the
        address of its frame slot (&local). A private string-constant value is its
        IDB string-literal address (matched by content)."""
        nm = operand.name
        if nm and nm in vmap:
            return vmap[nm]
        if nm and nm in self._addr_taken:
            return ("stkaddr", self._addr_taken[nm][0], 8)
        s = str(operand).strip()
        if nm and "@" in s:
            ea = ida_name.get_name_ea(ida_idaapi.BADADDR, nm)
            if ea != ida_idaapi.BADADDR:
                return ("gvar", ea, default_size)
            sea = self._strconst_ea(s)
            if sea is not None:
                return ("gvar", sea, default_size)
        num = re.search(r"(-?\d+)\s*$", s)
        if num:
            return ("num", int(num.group(1)), _type_size(s) or default_size)
        raise ValueError(f"unhandled operand {s!r}")

    def _global_ea(self, operand):
        """Resolve an LLVM global operand (``@name``) to its IDB address, or
        None if it is not a (resolvable) global. A private string-constant global
        resolves to its IDB string literal (matched by content)."""
        s = str(operand)
        if operand.name and "@" in s:
            ea = ida_name.get_name_ea(ida_idaapi.BADADDR, operand.name)
            if ea != ida_idaapi.BADADDR:
                return ea
            return self._strconst_ea(s)
        return None

    @staticmethod
    def _wire(blk, succs) -> None:
        """Replace blk's successors with ``succs``, fixing peer predsets."""
        mba = blk.mba
        for old in [int(s) for s in blk.succset]:
            blk.succset._del(old)
            ob = mba.get_mblock(old)
            if ob is not None:
                with suppress(Exception):
                    ob.predset._del(blk.serial)
                ob.mark_lists_dirty()
        for ns in succs:
            blk.succset.push_back(ns)
            nb = mba.get_mblock(ns)
            if nb is not None and blk.serial not in [int(p) for p in nb.predset]:
                nb.predset.push_back(blk.serial)
                nb.mark_lists_dirty()
        blk.mark_lists_dirty()

    @staticmethod
    def _clear(blk) -> None:
        ins = blk.head
        while ins is not None:
            nxt = ins.next
            blk.remove_from_block(ins)
            ins = nxt
        blk.mark_lists_dirty()

    def _emit_value(self, mba, blk, anchor, ea, ins, vmap, ds):
        """Emit one non-terminator LLVM instruction into ``blk`` before
        ``anchor``; record its SSA result in ``vmap``. Returns the new anchor."""
        op = ins.opcode
        ops = list(ins.operands)
        if op in _BINOP:
            size = _type_size(ins.type)
            mi = hx.minsn_t(ea)
            mi.opcode = _BINOP[op]
            self._fill(mi.l, self._desc(ops[0], vmap, size))
            r_desc = self._desc(ops[1], vmap, size)
            if op in ("shl", "lshr", "ashr"):
                # shift-amount operand must be size 1 (INTERR 50835).
                r_desc = (r_desc[0], r_desc[1], 1)
            self._fill(mi.r, r_desc)
            kreg = mba.alloc_kreg(size)
            mi.d.make_reg(kreg, size)
            blk.insert_into_block(mi, anchor)
            vmap[ins.name] = ("reg", kreg, size)
            return mi
        if op in _NOOP_CAST:
            # same-bits cast (ptr<->int, ptr<->ptr): alias the operand, no insn.
            vmap[ins.name] = self._desc(ops[0], vmap, 8)
            return anchor
        if op in _CAST:
            in_sz = _type_size(ops[0].type)
            out_sz = _type_size(ins.type)
            if in_sz == out_sz:
                # Same BYTE width (e.g. `zext i1 to i8` -- both 1 byte): a no-op
                # reinterpretation, not a real widen/narrow. m_xdu/m_xds/m_low all
                # INTERR on equal l/d sizes (50837/50838), so alias the operand.
                vmap[ins.name] = self._desc(ops[0], vmap, out_sz)
                return anchor
            mi = hx.minsn_t(ea)
            mi.opcode = _CAST[op]
            self._fill(mi.l, self._desc(ops[0], vmap, in_sz))
            kreg = mba.alloc_kreg(out_sz)
            mi.d.make_reg(kreg, out_sz)
            blk.insert_into_block(mi, anchor)
            vmap[ins.name] = ("reg", kreg, out_sz)
            return mi
        if op == "alloca":
            # Storage modeled as a kreg (pre-allocated in _scan_allocas); the
            # alloca itself emits nothing -- load/store route to that kreg.
            return anchor
        if op == "load":
            out_sz = _type_size(ins.type)
            if self._is_ret_slot(ops[0], vmap):
                # promoted return slot: the loaded value IS the return register.
                _ar, eax, _d = self._abi()
                vmap[ins.name] = ("reg", eax, out_sz)
                return anchor
            slot = self._allocas.get(ops[0].name)
            if slot is not None:
                # %v = load <ty>, ptr %a  (a is a scalar slot) -> mov slot, v
                mi = hx.minsn_t(ea)
                mi.opcode = hx.m_mov
                mi.l.make_reg(slot[0], out_sz)
                kreg = mba.alloc_kreg(out_sz)
                mi.d.make_reg(kreg, out_sz)
                blk.insert_into_block(mi, anchor)
                vmap[ins.name] = ("reg", kreg, out_sz)
                return mi
            stk = self._stkvar_slot(ops[0], vmap)
            if stk is not None:
                # %v = load (frame-slot alloca or GEP-of-alloca field) -> mov stkvar, v
                mi = hx.minsn_t(ea)
                mi.opcode = hx.m_mov
                mi.l.make_stkvar(mba, stk[0])
                mi.l.size = out_sz
                kreg = mba.alloc_kreg(out_sz)
                mi.d.make_reg(kreg, out_sz)
                blk.insert_into_block(mi, anchor)
                vmap[ins.name] = ("reg", kreg, out_sz)
                return mi
            gea = self._global_ea(ops[0])
            if gea is not None:
                # %v = load <ty>, ptr @g  -> mov g, v  (the gvar IS the location)
                mi = hx.minsn_t(ea)
                mi.opcode = hx.m_mov
                mi.l.make_gvar(gea)
                mi.l.size = out_sz
                kreg = mba.alloc_kreg(out_sz)
                mi.d.make_reg(kreg, out_sz)
                blk.insert_into_block(mi, anchor)
                vmap[ins.name] = ("reg", kreg, out_sz)
                return mi
            # %v = load <ty>, <ty>* %p  ->  ldx ds, p, v
            ad = self._desc(ops[0], vmap, 8)
            mi = hx.minsn_t(ea)
            mi.opcode = hx.m_ldx
            mi.l.make_reg(ds, 2)
            self._fill(mi.r, (ad[0], ad[1], 8))
            kreg = mba.alloc_kreg(out_sz)
            mi.d.make_reg(kreg, out_sz)
            blk.insert_into_block(mi, anchor)
            vmap[ins.name] = ("reg", kreg, out_sz)
            return mi
        if op == "store":
            val_sz = _type_size(ops[0].type)
            vd = self._desc(ops[0], vmap, val_sz)
            if self._is_ret_slot(ops[1], vmap):
                # promoted return slot: write the return register directly (each
                # path writes rax, matching native -- no post-merge read block).
                _ar, eax, _d = self._abi()
                mi = hx.minsn_t(ea)
                mi.opcode = hx.m_mov
                self._fill(mi.l, (vd[0], vd[1], val_sz))
                mi.d.make_reg(eax, val_sz)
                blk.insert_into_block(mi, anchor)
                return mi
            slot = self._allocas.get(ops[1].name)
            if slot is not None:
                # store <ty> %v, ptr %a  (a is a scalar slot) -> mov v, slot
                mi = hx.minsn_t(ea)
                mi.opcode = hx.m_mov
                self._fill(mi.l, (vd[0], vd[1], val_sz))
                mi.d.make_reg(slot[0], val_sz)
                blk.insert_into_block(mi, anchor)
                return mi
            stk = self._stkvar_slot(ops[1], vmap)
            if stk is not None:
                # store (frame-slot alloca or GEP-of-alloca field) -> mov v, stkvar
                mi = hx.minsn_t(ea)
                mi.opcode = hx.m_mov
                self._fill(mi.l, (vd[0], vd[1], val_sz))
                mi.d.make_stkvar(mba, stk[0])
                mi.d.size = val_sz
                blk.insert_into_block(mi, anchor)
                return mi
            gea = self._global_ea(ops[1])
            if gea is not None:
                # store <ty> %v, ptr @g  -> mov v, g  (the gvar IS the location)
                mi = hx.minsn_t(ea)
                mi.opcode = hx.m_mov
                self._fill(mi.l, (vd[0], vd[1], val_sz))
                mi.d.make_gvar(gea)
                mi.d.size = val_sz
                blk.insert_into_block(mi, anchor)
                return mi
            # store <ty> %v, <ty>* %p  ->  stx v, ds, p
            ad = self._desc(ops[1], vmap, 8)
            mi = hx.minsn_t(ea)
            mi.opcode = hx.m_stx
            self._fill(mi.l, (vd[0], vd[1], val_sz))
            mi.r.make_reg(ds, 2)
            self._fill(mi.d, (ad[0], ad[1], 8))
            blk.insert_into_block(mi, anchor)
            return mi
        if op == "getelementptr":
            slot = self._addr_taken.get(ops[0].name)
            if slot is not None:
                # GEP into a frame-slot alloca -> &stkvar(off + field). The result
                # is itself a frame address; record it so a downstream load/store/
                # call resolves to the same stkvar.
                vmap[ins.name] = ("stkaddr",
                                  slot[0] + self._gep_field_offset(ins, ops, vmap),
                                  8)
                return anchor
            # %q = getelementptr <ty>, <ty>* %p, i64 %idx  ->  q = p + idx*sizeof(ty)
            # (single-index array form; the result is an 8-byte address).
            m = re.search(r"getelementptr\s+(?:inbounds\s+)?([\w]+)",
                          str(ins).strip())
            elem_sz = _type_size(m.group(1)) if m else 1
            base = self._desc(ops[0], vmap, 8)
            idx = self._desc(ops[1], vmap, 8)
            if idx[0] == "num":
                off = idx[1] * elem_sz
                if off == 0:
                    vmap[ins.name] = (base[0], base[1], 8)  # alias the base
                    return anchor
                mi = hx.minsn_t(ea)
                mi.opcode = hx.m_add
                self._fill(mi.l, (base[0], base[1], 8))
                mi.r.make_number(off, 8)
            else:
                scaled = idx
                if elem_sz != 1:
                    ml = hx.minsn_t(ea)
                    ml.opcode = hx.m_mul
                    self._fill(ml.l, (idx[0], idx[1], 8))
                    ml.r.make_number(elem_sz, 8)
                    sk = mba.alloc_kreg(8)
                    ml.d.make_reg(sk, 8)
                    blk.insert_into_block(ml, anchor)
                    anchor = ml
                    scaled = ("reg", sk, 8)
                mi = hx.minsn_t(ea)
                mi.opcode = hx.m_add
                self._fill(mi.l, (base[0], base[1], 8))
                self._fill(mi.r, (scaled[0], scaled[1], 8))
            kreg = mba.alloc_kreg(8)
            mi.d.make_reg(kreg, 8)
            blk.insert_into_block(mi, anchor)
            vmap[ins.name] = ("reg", kreg, 8)
            return mi
        if op == "icmp":
            # Folded into the branch (see _build_multiblock); no value emitted.
            return anchor
        if op == "call":
            callee = list(ins.operands)[-1].name
            if callee in ("__readfsqword", "__readgsqword"):
                # Stack canary read -> ONE shared (unwritten) kreg for every read,
                # so `saved_canary == reread_canary` is `K == K` -> folds true ->
                # Hex-Rays prunes the __stack_chk_fail branch (the optimizer elides
                # the canary). See memory idavator_drop_canary_gate.
                sz = _type_size(ins.type) or 8
                if self._canary_kreg is None:
                    self._canary_kreg = mba.alloc_kreg(sz)
                vmap[ins.name] = ("reg", self._canary_kreg, sz)
                return anchor
            if callee == "__stack_chk_fail":
                return anchor  # canary fail path -> elided (branch is dead)
            raise RuntimeError("call must be split into its own block "
                               "(handled by the segment splitter, not _emit_value)")
        logger.warning("unhandled LLVM opcode: %s", op)
        return anchor

    def _emit_call(self, mba, blk, anchor, ea, ins, vmap, argregs):
        """Emit `mov`s into the ABI arg-regs then `m_call l=gvar(callee), d=rax`
        as the block TAIL. The call must terminate its block (it falls through to
        the continuation, BLT_1WAY) -- a call defining rax mid-block is fine for
        50864 but later maturities want calls block-terminal. Returns the call."""
        ops = list(ins.operands)
        callee = ops[-1]
        call_args = ops[:-1]
        _argregs, eax, _ds = self._abi()
        if len(call_args) > len(argregs):
            raise NotImplementedError(
                "stack-passed call argument (more args than ABI registers)")
        # Callee: a direct named function/global -> gvar. An indirect call through
        # an SSA value (function pointer) verifies but won't decompile -- Hex-Rays
        # needs the callee fn-ptr TYPE to rebuild the call signature; deferred.
        if callee.name in vmap:
            # Indirect call (fn pointer). The mcallinfo itself builds (set_type +
            # retregs/return_regs/spoiled + mop_f size clears 50757/50743/50740 and
            # the call renders), but multiple downstream issues remain (a
            # compare-size INTERR 50831 not in the icmp fold, + per-function build
            # errors) -- not converging with targeted fixes. Deferred to a focused
            # pass. Recipe banked in memory idavator_drop_call_construction.
            raise NotImplementedError(
                "indirect call (function pointer): mcallinfo builds but downstream "
                "verify cascade (50831+) is unresolved")
        callee_ea = ida_name.get_name_ea(ida_idaapi.BADADDR, callee.name)
        if callee_ea == ida_idaapi.BADADDR:
            if callee.name in _HELPER_INTRINSICS:
                arg_descs = [self._desc(a, vmap, _type_size(a.type))
                             for a in call_args]
                return self._emit_helper_call(
                    mba, blk, anchor, ea, ins, callee.name, arg_descs, eax)
            raise ValueError(f"unresolved callee @{callee.name}")
        # Resolve args once; a call that materializes a frame address (&local)
        # must carry the host resting-frame ea so Hex-Rays computes a
        # frame-consistent mcallinfo.call_spd -- else WARN_BAD_CALL_SP ("bad sp
        # value at call"). See memory idavator_sp_gate_call_ea_cracked.
        arg_descs = [self._desc(a, vmap, _type_size(a.type)) for a in call_args]
        passes_stkaddr = any(d[0] == "stkaddr" for d in arg_descs)
        for i, (a, d) in enumerate(zip(call_args, arg_descs)):
            asz = _type_size(a.type)
            mv = hx.minsn_t(ea)
            mv.opcode = hx.m_mov
            self._fill(mv.l, d)
            mv.d.make_reg(argregs[i], asz)
            blk.insert_into_block(mv, anchor)
            anchor = mv
        call_ea = (self._call_spd_ea
                   if passes_stkaddr and self._call_spd_ea is not None else ea)
        mc = hx.minsn_t(call_ea)
        mc.opcode = hx.m_call
        mc.l.make_gvar(callee_ea)
        rsz = _type_size(ins.type) if str(ins.type) != "void" else 8
        mc.d.make_reg(eax, rsz)  # call defines rax (the continuation captures it)
        blk.insert_into_block(mc, anchor)
        return mc

    @staticmethod
    def _int_tinfo(size):
        btf = {1: ida_typeinf.BTF_INT8, 2: ida_typeinf.BTF_INT16,
               4: ida_typeinf.BTF_INT32, 8: ida_typeinf.BTF_INT64}.get(
                   size, ida_typeinf.BTF_INT)
        return ida_typeinf.tinfo_t(btf)

    def _emit_helper_call(self, mba, blk, anchor, ea, ins, name, arg_descs, eax):
        """Emit an unresolved rotate intrinsic (`__ROR8__` ...) as a Hex-Rays
        HELPER call. Args ride in the mcallinfo via ``create_helper_call`` (the
        `make_helper` + reg-arg-mov shape crashes the decompiler). ``out`` = rax so
        the segment-split continuation captures the result like any other call."""
        callargs = hx.mcallargs_t()
        for a, d in zip(list(ins.operands)[:-1], arg_descs):
            sz = _type_size(a.type)
            arg = hx.mcallarg_t()
            self._fill(arg, (d[0], d[1], sz))
            arg.size = sz
            arg.type = self._int_tinfo(sz)
            callargs.push_back(arg)
        rsz = _type_size(ins.type) if str(ins.type) != "void" else 8
        out = hx.mop_t()
        out.make_reg(eax, rsz)
        mc = mba.create_helper_call(ea, name, self._int_tinfo(rsz), callargs, out)
        if mc is None:
            raise RuntimeError(f"create_helper_call({name}) returned None")
        blk.insert_into_block(mc, anchor)
        return mc

    def _capture_call_result(self, mba, blk, anchor, ea, call_ins, eax, vmap):
        """At a continuation block's START, copy the call's rax result into a
        scratch kreg so it survives a later call that clobbers rax. Registers
        the kreg in vmap as the call's SSA value."""
        rsz = _type_size(call_ins.type)
        kreg = mba.alloc_kreg(rsz)
        mv = hx.minsn_t(ea)
        mv.opcode = hx.m_mov
        mv.l.make_reg(eax, rsz)
        mv.d.make_reg(kreg, rsz)
        blk.insert_into_block(mv, anchor)
        vmap[call_ins.name] = ("reg", kreg, rsz)
        return mv

    def _emit_ret_value(self, blk, anchor, ea, term, eax, vmap):
        """Emit `mov <retval>, eax` for an LLVM ret (no terminator). A promoted
        return slot is already in the return reg, so no move is emitted."""
        ops = list(term.operands)
        if not ops:
            return anchor
        sz = _type_size(ops[0].type)
        d = self._desc(ops[0], vmap, sz)
        if d[0] == "reg" and d[1] == eax:
            return anchor  # already in the return reg (promoted return slot)
        mi = hx.minsn_t(ea)
        mi.opcode = hx.m_mov
        self._fill(mi.l, d)
        mi.d.make_reg(eax, sz)
        blk.insert_into_block(mi, anchor)
        return mi

    @staticmethod
    def _alloca_elem_size(ins) -> int:
        m = re.search(r"alloca\s+(?:inalloca\s+)?([^,\n]+)", str(ins).strip())
        ty = m.group(1).strip() if m else "i64"
        return 8 if ("*" in ty or ty == "ptr") else _type_size(ty)

    def _stkvar_slot(self, operand, vmap):
        """(stkoff, size) if ``operand`` addresses a frame slot -- an
        address-taken alloca or a GEP-of-alloca field result (stkaddr) -- else
        None."""
        s = self._addr_taken.get(operand.name)
        if s is not None:
            return s
        d = vmap.get(operand.name)
        if d is not None and d[0] == "stkaddr":
            return (d[1], 8)
        return None

    def _array_dims(self, type_str):
        """(count, elem_str, elem_size) for an ``[N x T]`` array. T may be a
        scalar/ptr or a ``%struct`` with a computable layout; else None (e.g. a
        va_list element)."""
        m = re.search(r"\[\s*(\d+)\s+x\s+([^\]]+?)\s*\]", type_str)
        if not m:
            return None
        n, elem = int(m.group(1)), m.group(2).strip()
        if elem == "ptr" or elem in ("i8", "i16", "i32", "i64"):
            return (n, elem, _type_size(elem))
        key = elem.replace('"', "")
        if key not in self._struct_size:
            # llvmlite disambiguates the SAME struct as %foo.NN in some renderings
            # (e.g. the alloca shows [2 x %timespec.15], the GEP/def use %timespec).
            key = re.sub(r"\.\d+$", "", key)
        if key in self._struct_size:
            return (n, elem, self._struct_size[key][0])
        return None

    def _gep_field_offset(self, ins, ops, vmap) -> int:
        """Constant byte offset of ``getelementptr [N x T], ptr %alloca, I0, I1``
        into a frame-slot alloca: ``I0*sizeof([N x T]) + I1*sizeof(T)``."""
        dims = self._array_dims(str(ins))
        if dims is None:
            raise NotImplementedError("GEP-on-stack: struct/va_list element")
        n, _elem, esz = dims
        strides = (n * esz, esz)  # [i0 over whole array, i1 over element]
        total = 0
        for k, o in enumerate(ops[1:]):
            d = self._desc(o, vmap, 8)
            if d[0] != "num":
                raise NotImplementedError("GEP-on-stack: non-constant index")
            total += d[1] * (strides[k] if k < len(strides) else esz)
        return total

    def _scan_allocas(self, mba, fn) -> dict:
        """Classify each alloca; return the SCALAR-SLOT kreg map and populate
        ``self._addr_taken`` (name -> (stkoff, size)) for address-taken ones.

        - scalar slot (used only as a direct load/store pointer) -> kreg;
          Hex-Rays propagates it away.
        - address-taken (its address escapes -- call arg, returned, stored as a
          value) but NOT GEP'd -> an existing host frame slot; &local renders as
          mop_a(stkvar) and the stack-passing call carries the resting-frame ea
          (the SP fix, memory idavator_sp_gate_call_ea_cracked).
        - GEP'd (struct field offsets) -> NotImplementedError (task #3: needs
          struct layout)."""
        names = {ins.name for bb in fn.blocks for ins in bb.instructions
                 if ins.opcode == "alloca"}
        self._addr_taken = {}
        if not names:
            return {}
        gepd: set = set()
        escaped: set = set()
        for bb in fn.blocks:
            for ins in bb.instructions:
                for idx, o in enumerate(ins.operands):
                    if o.name not in names:
                        continue
                    if ins.opcode == "load" and idx == 0:
                        continue
                    if ins.opcode == "store" and idx == 1:
                        continue
                    if ins.opcode == "getelementptr" and idx == 0:
                        gepd.add(o.name)
                    else:
                        escaped.add(o.name)
        allocas = {}
        off = 0
        for bb in fn.blocks:
            for ins in bb.instructions:
                if ins.opcode != "alloca":
                    continue
                nm = ins.name
                sz = self._alloca_elem_size(ins)
                if nm in gepd:
                    # GEP'd alloca -> a frame slot sized to the WHOLE array; each
                    # GEP resolves to &stkvar(off + field). Scalar/ptr element
                    # arrays only; struct/va_list elements need real layout.
                    dims = self._array_dims(str(ins))
                    if dims is None:
                        raise NotImplementedError(
                            f"GEP-on-stack alloca %{nm} (struct/va_list element "
                            f"needs real layout -- not the scalar-array slice)")
                    arr_sz = dims[0] * dims[2]
                    self._addr_taken[nm] = (off, arr_sz)
                    off += max(arr_sz, 8)
                elif nm in escaped:
                    # existing host frame slot -- NO frame extension (the
                    # subframe INTERR chain). Distinct 8-aligned offsets.
                    self._addr_taken[nm] = (off, sz)
                    off += max(sz, 8)
                else:
                    allocas[nm] = (mba.alloc_kreg(sz), sz)
        return allocas

    def _detect_ret_slot(self, fn) -> None:
        """Promote the lifter's RETURN SLOT to the return register (noreturn fns).

        The lift emits ``%funcresult = alloca T``: every returning path does
        ``store v, funcresult`` and the terminal block ``%r = load funcresult;
        ret %r``. Materialising that slot as an intermediate var (a kreg/stkvar
        READ at a post-merge block) is fine normally, but when a NORETURN call
        prunes an edge into that merge, Hex-Rays' value-numbering INTERRs (50342).
        The real compiler writes the retval straight to rax on each path. So,
        GATED on a noreturn call being present, route the slot's load/store to the
        return reg (the load aliases rax; each store writes rax) -- matching native
        and eliminating the post-merge read block. Non-noreturn fns are untouched.
        See memory idavator_drop_noreturn_50342_rootcause.
        """
        has_noret = any(self._callee_is_noreturn(ins)
                        for bb in fn.blocks for ins in bb.instructions
                        if ins.opcode == "call")
        if not has_noret:
            return
        ret_names = set()
        for bb in fn.blocks:
            term = list(bb.instructions)[-1]
            if term.opcode == "ret":
                ops = list(term.operands)
                if ops:
                    ret_names.add(ops[0].name)
        if len(ret_names) != 1:
            return
        rv = next(iter(ret_names))
        src = next((list(ins.operands)[0].name
                    for bb in fn.blocks for ins in bb.instructions
                    if ins.opcode == "load" and ins.name == rv), None)
        if src is None or not self._ret_slot_uses_ok(fn, src):
            return
        if src in self._addr_taken:
            self._ret_off = self._addr_taken[src][0]
        elif src in self._allocas:
            self._ret_kreg = self._allocas[src][0]

    def _ret_slot_uses_ok(self, fn, slot_name) -> bool:
        """Promotable only if the slot's address never escapes: every use of it
        (and its bitcast aliases) is a load/store pointer or a bitcast feeding one.
        A call-arg / value use means the address is needed -> don't promote."""
        alias = {slot_name}
        changed = True
        while changed:
            changed = False
            for bb in fn.blocks:
                for ins in bb.instructions:
                    if (ins.opcode == "bitcast"
                            and list(ins.operands)[0].name in alias
                            and ins.name not in alias):
                        alias.add(ins.name)
                        changed = True
        for bb in fn.blocks:
            for ins in bb.instructions:
                for idx, o in enumerate(ins.operands):
                    if o.name not in alias:
                        continue
                    if ins.opcode == "load" and idx == 0:
                        continue
                    if ins.opcode == "store" and idx == 1:
                        continue
                    if ins.opcode == "bitcast" and idx == 0:
                        continue
                    return False
        return True

    def _is_ret_slot(self, operand, vmap) -> bool:
        """True if ``operand`` addresses the promoted return slot -- a direct
        alloca, a bitcast alias resolving to its frame offset, or its scalar
        kreg. Cheap no-op when nothing was promoted."""
        if self._ret_off is not None:
            stk = self._stkvar_slot(operand, vmap)
            if stk is not None and stk[0] == self._ret_off:
                return True
        if self._ret_kreg is not None:
            slot = self._allocas.get(operand.name)
            if slot is not None and slot[0] == self._ret_kreg:
                return True
        return False

    def _synthesize_ret_block(self, mba):
        """A noreturn host has no m_ret block to use as the return sink. Copy a
        valid code block (for a real start/end -- INTERR 50869 otherwise), clear
        it, give it an operandless m_ret (the value flows via rax), and place it
        just before the special STOP block."""
        src = next((b for i in range(mba.qty)
                    if (b := mba.get_mblock(i)) is not None and int(b.type) == 0
                    and b.start != ida_idaapi.BADADDR and b.start < b.end), None)
        if src is None:
            raise RuntimeError("noreturn host: no valid block to copy for m_ret")
        retb = mba.copy_block(src, mba.qty - 1)  # insert before the STOP block
        self._clear(retb)
        for s in [int(x) for x in retb.succset]:
            retb.succset._del(s)
        for p in [int(x) for x in retb.predset]:
            retb.predset._del(p)
        rt = hx.minsn_t(mba.entry_ea)
        rt.opcode = hx.m_ret
        retb.insert_into_block(rt, None)
        retb.mark_lists_dirty()
        return retb

    @staticmethod
    def _resting_frame_ea(mba) -> int:
        """The host ea with the deepest (most negative) get_spd -- where the
        frame is fully allocated. A call that materializes a frame address
        (&local) carries this ea so Hex-Rays computes a frame-consistent
        call_spd (else WARN_BAD_CALL_SP). Falls back to entry_ea."""
        pfn = ida_funcs.get_func(mba.entry_ea)
        if pfn is None:
            return int(mba.entry_ea)
        best_ea, best_spd = int(mba.entry_ea), 0
        ea = pfn.start_ea
        while ea < pfn.end_ea and ea != ida_idaapi.BADADDR:
            spd = ida_frame.get_spd(pfn, ea)
            if spd < best_spd:
                best_spd, best_ea = spd, int(ea)
            ea = ida_bytes.next_head(ea, pfn.end_ea)
        return best_ea

    # -- build dispatch --------------------------------------------------
    def _build(self, mba, fn) -> None:
        self._cur_mba = mba
        self._call_spd_ea = self._resting_frame_ea(mba)
        self._canary_kreg = None
        self._ret_off = None
        self._ret_kreg = None
        argregs, eax, ds = self._abi()
        retb = next((mba.get_mblock(i) for i in range(mba.qty)
                     if (b := mba.get_mblock(i)) is not None and b.tail is not None
                     and int(b.tail.opcode) == hx.m_ret), None)
        if retb is None:
            retb = self._synthesize_ret_block(mba)

        self._allocas = self._scan_allocas(mba, fn)
        self._detect_ret_slot(fn)
        vmap: dict[str, tuple] = {}
        for i, a in enumerate(fn.arguments):
            if i >= len(argregs):
                raise NotImplementedError(
                    "stack-passed argument (more args than ABI registers)")
            vmap[a.name] = ("reg", argregs[i], _type_size(a.type))

        # A call must terminate its block, so a function with any call needs the
        # multi-block (segment-splitting) path even if it has one LLVM block.
        has_call = any(ins.opcode == "call"
                       for bb in fn.blocks for ins in bb.instructions)
        if len(list(fn.blocks)) == 1 and not has_call:
            self._build_singleblock(mba, fn, retb, eax, ds, vmap)
        else:
            self._build_multiblock(mba, fn, retb, eax, ds, vmap)
        mba.mark_chains_dirty()

    def _build_singleblock(self, mba, fn, retb, eax, ds, vmap) -> None:
        ea = mba.entry_ea
        # full-body replace: drop every non-terminator instruction.
        for i in range(mba.qty):
            b = mba.get_mblock(i)
            if b is None:
                continue
            ins = b.head
            while ins is not None:
                nxt = ins.next
                if int(ins.opcode) not in _TERM:
                    b.remove_from_block(ins)
                ins = nxt
            b.mark_lists_dirty()

        anchor = retb.tail.prev
        for ins in list(fn.blocks)[0].instructions:
            if ins.opcode == "ret":
                anchor = self._emit_ret_value(retb, anchor, ea, ins, eax, vmap)
            else:
                anchor = self._emit_value(mba, retb, anchor, ea, ins, vmap, ds)
        retb.mark_lists_dirty()

    @staticmethod
    def _phi_incomings(ins):
        """[(value_operand, pred_block_name)] for a phi. The incoming labels are
        text-only (not in .operands), so zip the value operands (textual order)
        with the labels parsed from str(ins)."""
        ops = list(ins.operands)
        preds = re.findall(r"\[\s*[^,\[\]]+,\s*%([\w.]+)\s*\]", str(ins))
        return list(zip(ops, preds))

    @staticmethod
    def _is_canary_call(ins) -> bool:
        """A stack-protector call -- __readfsqword/__readgsqword (canary read) or
        __stack_chk_fail (the fail path). These are ELIDED (not split into their
        own block), matching the optimizer, which drops the canary from faithful
        output. See memory idavator_drop_canary_gate."""
        if ins.opcode != "call":
            return False
        return list(ins.operands)[-1].name in (
            "__readfsqword", "__readgsqword", "__stack_chk_fail")

    @staticmethod
    def _callee_is_noreturn(ins) -> bool:
        """True if the call's DIRECT callee is __noreturn (xalloc_die/abort/...).
        Such a call ends its block with NO successor (BLT_0WAY) and everything
        after it is dead. Indirect/unresolved -> False."""
        callee_ea = ida_name.get_name_ea(ida_idaapi.BADADDR,
                                         list(ins.operands)[-1].name)
        if callee_ea == ida_idaapi.BADADDR:
            return False
        f = ida_funcs.get_func(callee_ea)
        return bool(f is not None and (f.flags & ida_funcs.FUNC_NORET))

    @staticmethod
    def _segment_block(bb):
        """Split an LLVM block's instruction stream into SEGMENTS at calls. A
        call must end its microcode block (BLT_1WAY, falls through), so each call
        closes a segment; the next segment captures that call's result at its
        start. Returns [seg,...]; only the LAST seg carries the terminator. A
        noreturn call closes the FINAL segment (BLT_0WAY) -- the rest is dead."""
        term = list(bb.instructions)[-1]
        segs, cur = [], {"values": [], "call": None, "term": None,
                         "prev_call": None}
        for ins in bb.instructions:
            if ins is term:
                break
            if ins.opcode == "phi":
                continue
            if ins.opcode == "call" and not LLVMDropConverter._is_canary_call(ins):
                cur["call"] = ins
                if LLVMDropConverter._callee_is_noreturn(ins):
                    cur["noreturn"] = True
                    segs.append(cur)
                    return segs  # noreturn -> no continuation; drop the dead tail
                segs.append(cur)
                cur = {"values": [], "call": None, "term": None,
                       "prev_call": ins}
            else:
                # canary calls stay in-segment -> _emit_value elides them.
                cur["values"].append(ins)
        cur["term"] = term
        segs.append(cur)
        return segs

    def _build_multiblock(self, mba, fn, retb, eax, ds, vmap) -> None:
        """One microcode block per SEGMENT (an LLVM block split at calls).
        A call-segment is a BLT_1WAY tail that falls through to its continuation;
        the terminal segment carries the LLVM terminator. Conditional ``br``
        lowers to a 2-way jump with a FALSE-arm trampoline at serial+1 (+ a lazy
        TRUE-arm trampoline when a phi needs that edge); phi is destructed out of
        SSA by copying each incoming value into the phi's kreg on its edge block.
        Hex-Rays rebuilds the CFG + lvars from these terminators."""
        ea = mba.entry_ea
        argregs, _eax, _ds = self._abi()
        llvm_blocks = list(fn.blocks)

        # Pre-scan icmp defs so a `br %c` can fold its compare into the jump.
        icmp_map: dict[str, tuple] = {}
        for bb in llvm_blocks:
            for ins in bb.instructions:
                if ins.opcode == "icmp":
                    pred = re.search(r"icmp\s+(\w+)\s", str(ins).strip())
                    icmp_map[ins.name] = (pred.group(1) if pred else "ne",
                                          list(ins.operands))

        # Pre-scan phi nodes -> per-block list + the set of edges needing a copy.
        phis: dict[str, list] = {}
        edges_need_copy: set[tuple] = set()
        for bb in llvm_blocks:
            bps = [(ins.name, ins, self._phi_incomings(ins))
                   for ins in bb.instructions if ins.opcode == "phi"]
            if bps:
                phis[bb.name] = bps
                for _, _, incs in bps:
                    for _val, pred_name in incs:
                        edges_need_copy.add((pred_name, bb.name))

        # Plan the segment layout. name_serial[bb] = its FIRST segment; the
        # terminal segment of a conditional br gets a FALSE trampoline (always)
        # and a TRUE trampoline (only if a phi drops a copy on that edge).
        plan, serial = [], 1
        name_serial: dict[str, int] = {}
        term_entry: dict[str, dict] = {}
        for bb in llvm_blocks:
            segs = self._segment_block(bb)
            for si, seg in enumerate(segs):
                e = {**seg, "bb": bb, "code": serial,
                     "ftramp": None, "ttramp": None}
                if si == 0:
                    name_serial[bb.name] = serial
                serial += 1
                if seg["term"] is not None:
                    term_entry[bb.name] = e
                    tops = list(seg["term"].operands)
                    if seg["term"].opcode == "br" and len(tops) == 3:
                        e["ftramp"] = serial
                        serial += 1
                        if (bb.name, tops[2].name) in edges_need_copy:
                            e["ttramp"] = serial
                            serial += 1
                plan.append(e)
        needed = serial - 1

        # Mint enough code blocks before retb (copy_block inherits a valid
        # start/end; INTERR 50869 otherwise), then clear serials 1..needed.
        avail = retb.serial - 1
        for _ in range(max(0, needed - avail)):
            mba.copy_block(retb, retb.serial)
        for s in range(1, needed + 1):
            self._clear(mba.get_mblock(s))
        for s in range(needed + 1, retb.serial):  # leftover host blocks -> dead
            lb = mba.get_mblock(s)
            self._clear(lb)
            g = hx.minsn_t(ea)
            g.opcode = hx.m_goto
            g.l.make_blkref(retb.serial)
            lb.insert_into_block(g, None)
            lb.type = hx.BLT_1WAY
            self._wire(lb, [retb.serial])

        # phi result kregs must exist before pass A (the phi's own block reads
        # them); register them in vmap up front.
        phi_kreg: dict[str, tuple] = {}
        for bps in phis.values():
            for pname, pins, _ in bps:
                sz = _type_size(pins.type)
                kreg = mba.alloc_kreg(sz)
                phi_kreg[pname] = (kreg, sz)
                vmap[pname] = ("reg", kreg, sz)

        # PASS A: per segment, capture the previous call's result, emit value
        # instructions, then (for a call-segment) the call tail + fall-through.
        for e in plan:
            blk = mba.get_mblock(e["code"])
            anchor = None
            if e["prev_call"] is not None and str(e["prev_call"].type) != "void":
                anchor = self._capture_call_result(
                    mba, blk, anchor, ea, e["prev_call"], eax, vmap)
            for ins in e["values"]:
                anchor = self._emit_value(mba, blk, anchor, ea, ins, vmap, ds)
            if e["call"] is not None:
                self._emit_call(mba, blk, anchor, ea, e["call"], vmap, argregs)
                if e.get("noreturn"):
                    blk.type = hx.BLT_0WAY  # noreturn tail: control never leaves
                else:
                    blk.type = hx.BLT_1WAY  # call tail -> continuation (serial+1)
                    self._wire(blk, [e["code"] + 1])

        # PASS A.5: out-of-SSA phi copies onto each incoming edge block (append
        # after the values; pass B appends the terminator after them, so the copy
        # always precedes the branch on that edge).
        def edge_serial(pred_name, target_name):
            pe = term_entry[pred_name]
            t = pe["term"]
            tops = list(t.operands)
            if t.opcode == "br" and len(tops) == 1:
                return pe["code"]
            if t.opcode == "br":
                if tops[1].name == target_name:
                    return pe["ftramp"]
                if tops[2].name == target_name:
                    return pe["ttramp"]
            raise ValueError(f"no edge {pred_name}->{target_name}")

        for bname, bps in phis.items():
            for pname, _pins, incs in bps:
                kreg, sz = phi_kreg[pname]
                for val_op, pred_name in incs:
                    eb = mba.get_mblock(edge_serial(pred_name, bname))
                    mi = hx.minsn_t(ea)
                    mi.opcode = hx.m_mov
                    self._fill(mi.l, self._desc(val_op, vmap, sz))
                    mi.d.make_reg(kreg, sz)
                    eb.insert_into_block(mi, eb.tail)

        # PASS B: terminators on TERMINAL segments only (call-segments were
        # already wired BLT_1WAY in pass A).
        for e in plan:
            if e["term"] is None:
                continue
            blk = mba.get_mblock(e["code"])
            term, ops = e["term"], list(e["term"].operands)
            anchor = blk.tail
            if term.opcode == "ret":
                anchor = self._emit_ret_value(blk, anchor, ea, term, eax, vmap)
                g = hx.minsn_t(ea)
                g.opcode = hx.m_goto
                g.l.make_blkref(retb.serial)
                blk.insert_into_block(g, anchor)
                blk.type = hx.BLT_1WAY
                self._wire(blk, [retb.serial])
            elif term.opcode == "br" and len(ops) == 1:
                tgt = name_serial[ops[0].name]
                g = hx.minsn_t(ea)
                g.opcode = hx.m_goto
                g.l.make_blkref(tgt)
                blk.insert_into_block(g, anchor)
                blk.type = hx.BLT_1WAY
                self._wire(blk, [tgt])
            elif term.opcode == "br":
                # br %cond, %false, %true  (llvmlite operand order!)
                cond = ops[0]
                false_s = name_serial[ops[1].name]
                true_s = name_serial[ops[2].name]
                mi = hx.minsn_t(ea)
                if cond.name in icmp_map:
                    pred, iops = icmp_map[cond.name]
                    sz = _type_size(iops[0].type)
                    mi.opcode = _ICMP_JMP.get(pred, hx.m_jnz)
                    self._fill(mi.l, self._desc(iops[0], vmap, sz))
                    self._fill(mi.r, self._desc(iops[1], vmap, sz))
                else:
                    sz = _type_size(cond.type)
                    mi.opcode = hx.m_jnz  # jump to TRUE when cond != 0
                    self._fill(mi.l, self._desc(cond, vmap, sz))
                    mi.r.make_number(0, sz)
                # taken arm = true trampoline (if any) else the true block.
                taken = e["ttramp"] if e["ttramp"] else true_s
                mi.d.make_blkref(taken)
                blk.insert_into_block(mi, anchor)
                blk.type = hx.BLT_2WAY
                self._wire(blk, [e["ftramp"], taken])  # [fall-through, taken]
                ft = mba.get_mblock(e["ftramp"])
                gf = hx.minsn_t(ea)
                gf.opcode = hx.m_goto
                gf.l.make_blkref(false_s)
                ft.insert_into_block(gf, ft.tail)
                ft.type = hx.BLT_1WAY
                self._wire(ft, [false_s])
                if e["ttramp"]:
                    tt = mba.get_mblock(e["ttramp"])
                    gt = hx.minsn_t(ea)
                    gt.opcode = hx.m_goto
                    gt.l.make_blkref(true_s)
                    tt.insert_into_block(gt, tt.tail)
                    tt.type = hx.BLT_1WAY
                    self._wire(tt, [true_s])
            elif term.opcode == "unreachable":
                # After a noreturn call or the elided canary fail path: a dead
                # goto to the ret block (the block is unreachable once the canary
                # compare folds, so Hex-Rays prunes it).
                g = hx.minsn_t(ea)
                g.opcode = hx.m_goto
                g.l.make_blkref(retb.serial)
                blk.insert_into_block(g, anchor)
                blk.type = hx.BLT_1WAY
                self._wire(blk, [retb.serial])
            else:
                raise NotImplementedError(
                    f"unhandled terminator {term.opcode!r}")
        retb.mark_lists_dirty()


def drop_llvm_function(ir_text: str, host_ea: int, llvm_fn_name: str):
    """Convenience: drop ``@llvm_fn_name`` from ``ir_text`` into ``host_ea``."""
    return LLVMDropConverter(ir_text).drop(host_ea, llvm_fn_name)
