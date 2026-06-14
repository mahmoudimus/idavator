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
import struct
from contextlib import suppress

import ida_bytes
import ida_frame
import ida_funcs
import ida_hexrays as hx
import ida_ida
import ida_idaapi
import ida_idp
import ida_name
import ida_nalt
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

# IEEE floating-point arithmetic -> Hex-Rays FPU microcode (the +F opcodes,
# m_f2i..m_fdiv). Two-operand ops; both operands and the result are the SAME
# fp width (float=4 / double=8). The insn carries IPROP_FPINSN (set_fpinsn).
_FP_BINOP = {
    "fadd": hx.m_fadd, "fsub": hx.m_fsub, "fmul": hx.m_fmul, "fdiv": hx.m_fdiv,
}
# fp<->int conversions. m_i2f/m_u2f take an integer source -> fp dest;
# m_f2i/m_f2u take an fp source -> integer dest. Source/dest WIDTHS are the
# operand/result type sizes (a `sitofp i32 -> float` is 4->4; `fptoui float ->
# i32` is 4->4). Like _CAST, these alloc a dest kreg of the result width.
_FP_CAST = {
    "sitofp": hx.m_i2f, "uitofp": hx.m_u2f,
    "fptosi": hx.m_f2i, "fptoui": hx.m_f2u,
}
# fp precision change (float<->double): m_f2f. Source/dest are the two fp widths.
_FP_RESIZE = {"fpext": hx.m_f2f, "fptrunc": hx.m_f2f}

# An icmp PREDICATE on two fptoui/fptosi-of-float operands -> the FPU jump that
# native uses for the float compare it lifted from. HexRays renders an ORDERED
# float compare with the UNSIGNED jcc family + the FPU flag (`jbe.fpu` etc., as
# seen in the GLBOPT2 microcode), so the unsigned LLVM predicates map straight
# through; the signed predicates (a < b) use the same unsigned-fpu jcc.
_FPU_JMP = {
    "ugt": hx.m_ja, "uge": hx.m_jae, "ult": hx.m_jb, "ule": hx.m_jbe,
    "sgt": hx.m_ja, "sge": hx.m_jae, "slt": hx.m_jb, "sle": hx.m_jbe,
    "eq": hx.m_jz, "ne": hx.m_jnz,
}
# Same predicate -> the FPU setcc (for an icmp result consumed as a VALUE).
_FPU_SET = {
    "ugt": hx.m_seta, "uge": hx.m_setae, "ult": hx.m_setb, "ule": hx.m_setbe,
    "sgt": hx.m_seta, "sge": hx.m_setae, "slt": hx.m_setb, "sle": hx.m_setbe,
    "eq": hx.m_setz, "ne": hx.m_setnz,
}

# An LLVM ``fcmp`` predicate -> the FPU conditional jump that branches when it
# holds (with set_fpinsn). The ida2llvm lifter now emits ``fcmp`` directly for a
# native FLOAT compare (the int-cast+icmp path destroyed the ordering); this map is
# the EXACT INVERSE of the lifter's opcode->fcmp table, so the FP compare survives
# the IR round-trip back to the original ``jbe.fpu``/``ja.fpu`` microcode. x86
# NaN-aware semantics: an ORDERED predicate (ogt/oge) uses the carry-clear jcc
# (ja/jae), an UNORDERED one (ult/ule/ueq) the carry-set jcc (jb/jbe/jz); ``one``
# (ordered not-equal) is the ZF=0 jump (jnz). Predicates the lifter never emits
# (olt/ole/ugt/uge/oeq/une/ord/uno) are mapped for completeness/robustness using
# the same NaN rules so a re-lifted body is still lowered correctly.
_FCMP_JMP = {
    "ogt": hx.m_ja, "oge": hx.m_jae, "olt": hx.m_jb, "ole": hx.m_jbe,
    "ugt": hx.m_ja, "uge": hx.m_jae, "ult": hx.m_jb, "ule": hx.m_jbe,
    "oeq": hx.m_jz, "ueq": hx.m_jz, "one": hx.m_jnz, "une": hx.m_jnz,
}
# Same predicate -> the FPU setcc (for an fcmp result consumed as a VALUE, e.g. a
# ``select`` condition or short-circuit boolean arm).
_FCMP_SET = {
    "ogt": hx.m_seta, "oge": hx.m_setae, "olt": hx.m_setb, "ole": hx.m_setbe,
    "ugt": hx.m_seta, "uge": hx.m_setae, "ult": hx.m_setb, "ule": hx.m_setbe,
    "oeq": hx.m_setz, "ueq": hx.m_setz, "one": hx.m_setnz, "une": hx.m_setnz,
}

# An LLVM floating-point CONSTANT operand stringifies as ``<fpty> 0xHHHH...``
# where the hex is the IEEE-754 DOUBLE bit-pattern of the value (LLVM stores ALL
# fp literals as doubles in textual IR, even ``float``-typed ones). Capture the
# fp type and the 64-bit hex so _desc can re-materialise it at the operand's
# real width via mop_t.make_fpnum.
_FPCONST_RE = re.compile(r"^(float|double)\s+(0x[0-9A-Fa-f]+|[-+]?[0-9.eE+-]+)$")

# icmp predicate -> a 2-way conditional jump that branches to ``d`` when the
# predicate holds (the fall-through, serial+1, takes the FALSE arm).
_ICMP_JMP = {
    "eq": hx.m_jz, "ne": hx.m_jnz,
    "ugt": hx.m_ja, "uge": hx.m_jae, "ult": hx.m_jb, "ule": hx.m_jbe,
    "sgt": hx.m_jg, "sge": hx.m_jge, "slt": hx.m_jl, "sle": hx.m_jle,
}

# icmp predicate -> a setcc that materialises the i1 result as a 1-byte value
# (``d = (l <pred> r) ? 1 : 0``). Used when an icmp result is CONSUMED as a value
# (a ``select`` condition / short-circuit arm) rather than folded into a branch.
_ICMP_SET = {
    "eq": hx.m_setz, "ne": hx.m_setnz,
    "ugt": hx.m_seta, "uge": hx.m_setae, "ult": hx.m_setb, "ule": hx.m_setbe,
    "sgt": hx.m_setg, "sge": hx.m_setge, "slt": hx.m_setl, "sle": hx.m_setle,
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

# Pure-WRITER libc calls that write THROUGH their destination pointer arg WITHOUT
# reading any caller memory (a constant/scalar fill). A result-DISCARDED call to
# one of these, emitted as a bare ``m_call gvar`` with no mcallinfo, is dead-code-
# eliminated by Hex-Rays' glbopt when a later store in the same block fully covers
# the written bytes (the pointer write is invisible without a callinfo) -- e.g.
# qset_acl's ``memset(&ctx,0,4); ctx.mode = mode`` (the mode store shadows all 4
# bytes). Routed through the explicit-mcallinfo fixed-arg path so HR models the
# clobber and keeps the call.
#
# RESTRICTED to pure writers ONLY (NOT memcpy/memmove). A reader-clobber like
# ``memcpy(dst, &v, n)`` is never at risk of the shadow-store fold (its dst is a
# live output, not a re-overwritten local), so it never needed this path -- and
# forcing an FCI_FINAL mcallinfo onto a DISCARDED ``memcpy(dst, &local, n)`` that
# is the LAST instruction of its block makes HR's glbopt fold away an UNRELATED
# preceding store whose value feeds &local (get_nonce: the
# ``LODWORD(v.tv_sec) = getgid()`` store collapsed to a bare ``getgid()``,
# silently dropping the consumed call's result). Scoped to pure writers keeps the
# qset_acl fix and leaves every reader-clobber (and its neighbours) untouched.
_MEMCLOBBER_FNS = frozenset({
    "memset", "bzero",
    "__memset_chk",
})

# Variadic-prologue intrinsics the *body* emits over the real ``__va_list_tag``
# storage (the SysV va_list machine HexRays renders as the ``va_start(ap, last)``
# / ``va_arg(ap, T)`` / ``va_end(ap)`` macros -- helper-call form, exactly like
# ``__ROR8__``, NOT the ``!va_start`` IR intrinsic). The lifter (ida2llvm) ALSO
# bolts a redundant synth scaffold ON TOP -- ``%ArgList = alloca i8*`` (uninit) +
# ``call @llvm.va_start`` / ``@llvm.va_end`` -- which has NO native counterpart
# (the real machine is the body's ``@va_start``/``@va_arg`` over ``authors``). The
# scaffold (``llvm.va_start``/``llvm.va_end``) is DEAD -> no-op; the body
# (``va_start``/``va_arg``/``va_end``) lowers to a Hex-Rays helper call. Both are
# kept IN-segment (like the canary) so they render INLINE, not block-terminal.
_VA_HELPER = frozenset({"va_start", "va_arg", "va_end"})
_VA_SCAFFOLD = frozenset({"llvm.va_start.p0", "llvm.va_end.p0"})
_VARARG_INTRINSICS = _VA_HELPER | _VA_SCAFFOLD

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


def _fpconst_bytes(operand_str: str):
    """Decode an LLVM floating-point constant operand (``float 0x43F0...`` or
    ``double 1.5``) to (ieee_bytes, size). LLVM stores EVERY fp literal as a
    64-bit IEEE-754 DOUBLE bit-pattern in the textual ``0x...`` form, regardless
    of the operand's declared type -- so ``float 0x43F0000000000000`` is the
    double encoding of the value the 4-byte float represents (1.8446744e19). We
    decode that double, then re-pack at the OPERAND'S width (single for a
    ``float`` operand, double for a ``double``) so ``make_fpnum`` builds the
    right-sized mop_fn -- matching native's ``(float)1.8446744e19``. Returns
    ``None`` if the string is not an fp constant."""
    m = _FPCONST_RE.match(operand_str.strip())
    if m is None:
        return None
    fpty, lit = m.group(1), m.group(2)
    if lit.lower().startswith("0x"):
        # 64-bit IEEE-754 double bit pattern (LLVM's canonical fp-literal form).
        value = struct.unpack("<d", struct.pack("<Q", int(lit, 16) & 0xFFFFFFFFFFFFFFFF))[0]
    else:
        value = float(lit)
    if fpty == "float":
        return struct.pack("<f", value), 4
    return struct.pack("<d", value), 8


def _type_size(type_str) -> int:
    # llvmlite uses OPAQUE pointers (LLVM 14+): a pointer stringifies to "ptr",
    # not "i32*", and the pointee type is lost -- so a string "*" probe misses it.
    # Detect the pointer first via the structured ``is_pointer`` flag.
    if getattr(type_str, "is_pointer", False):
        return 8
    s = str(type_str)
    if s == "ptr" or "*" in s:
        return 8
    # An exact ``iN`` integer type -- size is ceil(N/8). This covers the WIDE
    # integers the frontend uses to lower a whole-struct copy (``load i448`` /
    # ``store i448`` == a 56-byte ``struct quoting_options`` memcpy). The old
    # substring scan matched ``i8``/``i1`` etc. as substrings and otherwise fell
    # through to the 4-byte default, so ``i448`` mis-sized to 4 and the struct
    # copy SCALARIZED to its first field (quotearg_char_mem ``options.style =
    # ...`` instead of ``options = default_quoting_options``).
    m = re.fullmatch(r"i(\d+)", s)
    if m:
        return max(1, (int(m.group(1)) + 7) // 8)
    for tok, sz in (("i64", 8), ("i32", 4), ("i16", 2), ("i8", 1), ("i1", 1),
                    ("double", 8), ("float", 4)):
        if tok in s:
            return sz
    return 4


_LEGAL_KREG_SIZES = (1, 2, 4, 8, 16)


def _legal_kreg_size(nbytes: int) -> int:
    """Round a byte width up to a size ``mba.alloc_kreg(check_size=True)`` will
    accept (a basic-type size: ``{1, 2, 4, 8, 16}``).

    SROA byte-splits a wide scalar into NON-power-of-2 slices -- e.g. it splits
    an ``i32`` into a low byte plus an ``i24`` high slice, or an ``i64`` arg into
    ``i48``/``i56`` tails. ``_type_size`` reports those as 3/6/7 bytes, and
    ``alloc_kreg`` REJECTS 3/5/6/7 (only basic-type sizes are valid), raising
    ``RuntimeError: Unknown exception``. Round the slice up to the next legal
    kreg width so it lives in a real register; the PRODUCING ``trunc`` masks the
    value to the slice's LOGICAL bit-width (``_int_bits``) so the paired widen
    (``zext``/``sext``) re-extends it faithfully -- the high padding bits of the
    rounded kreg carry no meaning."""
    for s in _LEGAL_KREG_SIZES:
        if nbytes <= s:
            return s
    return _LEGAL_KREG_SIZES[-1]


def _int_bits(type_str):
    """Logical bit count of an ``iN`` integer type, else ``None`` (pointer / fp /
    aggregate). Used to mask an illegal-width slice down to its true bit-width
    after rounding its kreg up to a legal size."""
    m = re.fullmatch(r"i(\d+)", str(type_str).strip())
    return int(m.group(1)) if m else None


def _is_ptr_type(type_str) -> bool:
    """True if an LLVM type is a pointer (opaque ``ptr`` or a legacy ``T*``).
    Used to tell a full-pointer slot access from a sub-pointer deref on a
    pointer-typed alloca."""
    if getattr(type_str, "is_pointer", False):
        return True
    s = str(type_str).strip()
    return s == "ptr" or s.endswith("*")


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


def _parse_struct_field_offsets(ir_text: str, sizes: dict) -> dict:
    """``name -> [byte offset of each top-level field]`` for every ``%name = type
    {..}`` whose layout is computable, using the SAME natural-C alignment walk as
    ``_parse_struct_layouts`` (each member rounded up to its own alignment; packed
    structs -- ``<{..}>`` -- get no padding). ``sizes`` is the already-computed
    ``name -> (size, align)`` map so a nested ``%struct`` field is sized by its
    cached layout.

    The offsets HONOR ABI padding: ``%timeval = type {i64, i32}`` yields
    ``[0, 8]`` (the ``i32`` lands at 8, not 4, because the i64 occupies 0..7 and
    the i32's own 4-byte alignment leaves it at 8 in a 16-byte struct). A struct
    whose body references an un-laid-out type is simply absent (the GEP path then
    declines -> native fallback), mirroring ``_parse_struct_layouts``."""
    out: dict = {}
    for m in re.finditer(r'(%[\w".:$]+)\s*=\s*type\s*(<?)\s*\{(.*)\}', ir_text):
        name = m.group(1).strip().replace('"', "")
        packed = m.group(2) == "<"
        try:
            off, offsets = 0, []
            for f in _split_fields(m.group(3)):
                sz, al = _type_sa(f, sizes)
                if not packed:
                    off = _round_up(off, al)
                offsets.append(off)
                off += sz
            out[name] = offsets
        except Exception:  # noqa: BLE001 -- un-laid-out member; skip (native fallback)
            continue
    return out


class LLVMDropConverter:
    """Drop a (straight-line) LLVM function into a host's decompiled output."""

    def __init__(self, ir_text: str):
        self._ir_text = ir_text
        self.module = llvm.parse_assembly(ir_text)
        self._sroa_module = None     # lazily-built SROA-optimized copy (fallback)
        self._kreg_call_results = False  # off by default; the scoped 50342 retry in
        # ``drop`` flips it so ``_capture_call_result`` ALSO kills the ABI return
        # register after kreg-copying a call's result. See ``_capture_call_result``.
        self._struct_size = _parse_struct_layouts(ir_text)  # name -> (size, align)
        # name -> [byte offset of each field] (same natural-C/packed walk as the
        # size pass); feeds the struct-field GEP-on-stack resolution.
        self._struct_field_off = _parse_struct_field_offsets(
            ir_text, self._struct_size)
        self._allocas: dict = {}  # set per-drop in _build (scalar-slot kregs)
        self._addr_taken: dict = {}  # name -> (stkoff, size) for &local allocas
        self._array_alloca_elt: dict = {}  # GEP'd ``[N x T]`` alloca name ->
        # (elem_str, elem_byte_size, count). Drives the post-decompile ARRAY-LVAR
        # retype pass (_save_array_lvar_types): a frame slot that the drop laid out
        # as ONE whole-aggregate region but resolves GEP-by-GEP to bare sp+const
        # scalar refs would otherwise be shattered by Hex-Rays into ~N disjoint
        # scalar stkvars; typing the slot lvar as ``T[N]`` makes it cohere into a
        # single array lvar (native's ``char buf[104]``). Populated in _build.
        self._arg_spill_kill: set = set()  # arg regs to KILL after the entry spill
        # (an arg whose spill slot is read across a clobbering call -- create_hole's
        # ``size``; see _arg_spill_slots_across_call).
        self._ptr_allocas: dict = {}  # ptr-typed addr_taken alloca name -> stkoff
        self._ptr_deref_alias: set = set()  # bitcast aliases rooted at a ptr alloca
        self._ptr_alloca_pointee: dict = {}  # ptr alloca name -> pointee byte width
        self._ptr_alloca_pointee_struct: set = set()  # ptr alloca/alias names whose
        # pointee is a known multi-field STRUCT (cursor slot, e.g. extent_info*).
        # A direct-bitcast pointer-WIDTH store of a NON-pointer arithmetic value
        # into such a slot DEFINES the pointer (cursor advance) rather than writing
        # the struct pointee -- an 8-byte store can never be a full struct-pointee
        # write; native renders it `cur = base + idx`. A SCALAR pointee (i64* ->
        # *total_n_read = 0) stays a deref.
        self._cur_mba = None         # current mba (make_stkvar needs it)
        self._call_spd_ea = None     # host resting-frame ea for stack-passing calls
        self._canary_kreg = None     # shared kreg for __readfsqword (canary fold)
        self._ret_off = None         # frame off of a promoted return slot, or None
        self._ret_kreg = None        # kreg of a promoted scalar return slot, or None
        self._ret_phi = None         # name of a phi whose result feeds `ret`, or None
        self._icmp_defs = {}         # icmp SSA name -> (pred, [operands]) for select
        self._fnptr_bitcast = {}     # bitcast-result SSA name -> underlying fn name
        # (a ``bitcast @fn to <wider variadic>`` callee -- the lifter wraps a
        # variadic call that carries surplus varargs in a function-pointer bitcast;
        # _emit_call sees through it so the call routes as a DIRECT variadic call.)
        self._ptr_origin_vals = set()  # i64 SSA names that are really pointers
        # (a pointer-typed alloca read as i64, or a ptrtoint address) -- typed as
        # ``void *`` for a variadic arg so it renders clean, not a signed cast.
        self._array_elt_addr = {}  # ptrtoint-of-GEP-into-array-alloca SSA name ->
        # its EFFECTIVE byte offset for the variadic-arg decline gate. The value is 0
        # for a faithful whole-buffer pointer: a ZERO-offset address (``&buf[0]``,
        # e.g. ``scanf("%s", &buf)``) into a ``[N x T]`` buffer the ARRAY-LVAR retype
        # pass can cohere into one ``T[N]`` lvar (a nameable scalar/ptr element) -- it
        # then renders like native ``scanf("%s", buf)`` EVEN if the buffer is GEP'd at
        # other offsets too. A NONZERO offset (``&perms[1]``), a non-constant index,
        # OR a zero-offset address into a buffer the retype CANNOT cohere all carry a
        # non-zero effective offset (None for the sentinel), so _emit_call_vararg
        # declines them for a clean native fallback rather than ship a divergent body.
        # Populated in drop() where the offsets + per-buffer cohesibility are known.
        self._str_index = None       # IDB string-literal content -> ea (lazy)

    # -- public API ------------------------------------------------------
    def drop(self, host_ea: int, llvm_fn_name: str):
        """Rebuild ``host_ea``'s microcode from the named LLVM function and return
        the resulting ``cfunc_t`` (or None on failure).

        SROA FALLBACK (scoped, zero-regression): the plain drop runs first. ONLY
        if it returns a LATE failure -- ``cf is None`` with NO build error and NO
        early INTERR caught by ``try_verify`` (the 50342-style value-numbering
        failure that surfaces only after ``preoptimized``) -- do we retry from an
        SROA-optimized copy of the module. SROA collapses the lifter's return-slot
        alloca into a return phi, which the return-phi promotion writes straight to
        the return reg (clearing 50342). Because the retry only runs when the plain
        drop already FAILED, every currently-passing function is untouched."""
        cf, box = self._drop_from_module(self.module, host_ea, llvm_fn_name)
        is_late_failure = (cf is None and box["err"] is None
                           and box["interr"] is None)
        # FAITHFUL 50342 retry (scoped, zero-regression): a late failure where a
        # call result carried in the ABI return register is reused as a body
        # INTERMEDIATE while rax is independently redefined as the return value --
        # the two versions collide at the body-less BLT_STOP (INTERR 50342). Retry
        # the SAME (plain) module with ``_kreg_call_results`` so each captured call
        # result is kicked OUT of rax into its kreg (rax killed), mirroring native's
        # ``%bucket`` stkvar. This stays FAITHFUL (same IR, no SROA reshaping) and
        # runs BEFORE the SROA fallback, so a fix here beats SROA's coarser body.
        # Because it only runs when the plain drop already FAILED, every
        # currently-passing function is untouched.
        if is_late_failure:
            self._kreg_call_results = True
            try:
                cfk, boxk = self._drop_from_module(
                    self.module, host_ea, llvm_fn_name)
            finally:
                self._kreg_call_results = False
            if cfk is not None:
                cf, box = cfk, boxk
                is_late_failure = False
        if is_late_failure:
            opt = self._get_sroa_module()
            if opt is not None and any(
                    g.name == llvm_fn_name and not g.is_declaration
                    for g in opt.functions):
                logger.info("drop @%s: late failure -> SROA fallback retry",
                            llvm_fn_name)
                cf2, box2 = self._drop_from_module(opt, host_ea, llvm_fn_name)
                if cf2 is not None:
                    cf, box = cf2, box2
        if box["err"]:
            logger.error("drop build failed:\n%s", box["err"])
        self.last_interr = box["interr"]
        self.last_error = box["err"]
        return cf

    def _drop_from_module(self, module, host_ea: int, llvm_fn_name: str):
        """Rebuild ``host_ea`` from ``llvm_fn_name`` in ``module``; return
        ``(cfunc_t|None, box)`` where ``box`` carries ``interr``/``err``."""
        fn = next((g for g in module.functions
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
            # Type the struct-pointer cursor slots AND the GEP'd array slots, then
            # re-drop so the type propagation re-runs -- the only thing that beats
            # the decompiler's own inference (param-propagated pointers; a buffer
            # shattered into per-offset scalar stkvars) is a persistent user type at
            # each lvar's ACTUAL (post-decompile) location, applied between two full
            # decompiles. Both retypes share the one re-decompile. Build failures
            # leave cf untyped (unchanged).
            if cf is not None and box["err"] is None:
                retyped = self._save_struct_ptr_lvar_types(host_ea, fn, cf)
                retyped = self._save_array_lvar_types(host_ea, fn, cf) or retyped
                if retyped:
                    hx.mark_cfunc_dirty(host_ea)
                    cf = hx.decompile(host_ea)
        finally:
            hook.unhook()
        return cf, box

    def _get_sroa_module(self):
        """An SROA(+simplifycfg+instnamer)-optimized copy of the module, built
        lazily and cached. SROA promotes the lifter's return-slot alloca to a
        return phi (then promoted to the return reg); instnamer is required because
        SROA emits anonymous ``%0``/``%1`` that the converter keys by NAME. Returns
        None if the new-PM pipeline is unavailable (the plain drop result stands)."""
        if self._sroa_module is not None:
            return self._sroa_module
        try:
            opt = llvm.parse_assembly(self._ir_text)
            llvm.initialize_all_targets()
            llvm.initialize_native_target()
            llvm.initialize_native_asmprinter()
            tm = llvm.Target.from_default_triple().create_target_machine()
            pb = llvm.create_pass_builder(
                tm, llvm.create_pipeline_tuning_options())
            mpm = llvm.create_new_module_pass_manager()
            mpm.add_sroa_pass()
            mpm.add_simplify_cfg_pass()
            mpm.add_instruction_namer_pass()
            mpm.run(opt, pb)
            self._sroa_module = opt
        except Exception:  # noqa: BLE001
            import traceback
            logger.warning("SROA fallback unavailable:\n%s",
                           traceback.format_exc())
            self._sroa_module = None
        return self._sroa_module

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
    def _arg_reg_names():
        # Integer arg registers depend on the target ABI: a PE (Windows) target
        # uses the Microsoft x64 convention (rcx/rdx/r8/r9); everything else
        # (ELF/Mach-O) uses System V (rdi/rsi/rdx/rcx/r8/r9).
        if ida_ida.inf_get_filetype() == ida_ida.f_PE:
            return ("rcx", "rdx", "r8", "r9")
        return ("rdi", "rsi", "rdx", "rcx", "r8", "r9")

    @staticmethod
    def _abi():
        # Return reg = rax (rdi/... for args -- see _arg_reg_names).
        names = LLVMDropConverter._arg_reg_names()
        argregs = [hx.reg2mreg(ida_idp.str2reg(r)) for r in names]
        return (argregs, hx.reg2mreg(ida_idp.str2reg("rax")),
                hx.reg2mreg(ida_idp.str2reg("ds")))

    # -- microcode emit helpers -----------------------------------------
    def _fill(self, mop, d) -> None:
        kind, val, size = d
        if kind == "reg":
            mop.make_reg(val, size)
        elif kind == "fpnum":
            # IEEE floating-point constant (mop_fn). ``val`` is the packed
            # IEEE-754 bytes; make_fpnum infers the size from len(bytes) and
            # builds an fnumber_t of the operand's fp width.
            if not mop.make_fpnum(val):
                raise ValueError(f"make_fpnum failed for {size}-byte fp constant")
        elif kind == "gvar":
            mop.make_gvar(val)
            mop.size = size
        elif kind == "gvaraddr":
            # &global: mop_a wrapping a gvar (cf. &local = mop_a wrapping a
            # stkvar). A global used as a POINTER VALUE is its ADDRESS, not the
            # lvalue read at that address -- e.g. a decayed string constant
            # handed to a call is `gettext("msg")`, not
            # `gettext(*(const char **)"msg")`.
            inner = hx.mop_addr_t()
            inner.make_gvar(val)
            mop.t = hx.mop_a
            mop.a = inner
            mop.size = 8
        elif kind == "stkaddr":
            # &local: mop_a wrapping a stkvar (cf. &global = mop_a wrapping mop_v).
            inner = hx.mop_addr_t()
            inner.make_stkvar(self._cur_mba, val)
            mop.t = hx.mop_a
            mop.a = inner
            mop.size = 8
        elif kind == "stkvar":
            # The VALUE held in a host frame slot (an incoming >6th param spilled
            # to the caller stack -- cf. _fill's stkaddr is the slot's ADDRESS).
            mop.make_stkvar(self._cur_mba, val)
            mop.size = size
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
        gvar / gvaraddr / stkaddr). A global used as a POINTER VALUE is its
        ADDRESS (``gvaraddr`` = &global) -- an array/string global decays to a
        ``ptr`` whose value IS its address, so a decayed string constant handed
        to a call must render ``gettext("msg")``, not the lvalue read
        ``gettext(*(const char **)"msg")``. An address-taken alloca is the
        address of its frame slot (&local). A private string-constant value is
        its IDB string-literal address (matched by content)."""
        nm = operand.name
        if nm and nm in vmap:
            return vmap[nm]
        if nm and nm in self._addr_taken:
            return ("stkaddr", self._addr_taken[nm][0], 8)
        s = str(operand).strip()
        if nm and "@" in s:
            # A global operand of pointer type used as a VALUE is its address:
            # in opaque-pointer IR a global symbol's operand type is ``ptr`` and
            # its value IS its address (the array decayed). Materialise &global
            # (gvaraddr) rather than the lvalue read at that address (gvar).
            ptr_value = str(operand.type) == "ptr"
            ea = ida_name.get_name_ea(ida_idaapi.BADADDR, nm)
            if ea != ida_idaapi.BADADDR:
                return ("gvaraddr", ea, 8) if ptr_value \
                    else ("gvar", ea, default_size)
            sea = self._strconst_ea(s)
            if sea is not None:
                return ("gvaraddr", sea, 8) if ptr_value \
                    else ("gvar", sea, default_size)
        if s.split()[-1:] == ["undef"]:
            # an ``undef`` value is a don't-care (SROA leaves it on dead phi
            # incomings / masked-insert leftovers / a `ret <ty> undef` whose path
            # is pruned). Resolve it to zero of the operand's width -- any concrete
            # value is correct, and 0 keeps the value-numbering trivial.
            return ("num", 0, _type_size(s) or default_size)
        fpc = _fpconst_bytes(s)
        if fpc is not None:
            # IEEE fp literal (``float 0x43F0...``) -> mop_fn. Must precede the
            # integer-tail regex below, which would otherwise capture the hex
            # bit-pattern's trailing digits as a bogus integer.
            ieee, sz = fpc
            return ("fpnum", ieee, sz)
        num = re.search(r"(-?\d+)\s*$", s)
        if num:
            return ("num", int(num.group(1)), _type_size(s) or default_size)
        if s.split()[-1:] == ["null"]:
            # a `null` pointer constant (e.g. a phi incoming after SROA folds the
            # return slot: ``[ null, %B ]``) -> the zero address.
            return ("num", 0, _type_size(s) or default_size)
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

    def _fp_compare_operands(self, iops):
        """If both operands of an icmp are fptoui/fptosi conversions of float
        SSA values (the lifter's lowering of a native float compare), return the
        two ORIGINAL float operand descriptors and their width as
        ``(l_op, r_op, fp_size)``; else ``None``. Both sides must convert from the
        SAME fp width (an asymmetric pair is not a single float compare)."""
        a = self._fp_cvt_src.get(iops[0].name)
        b = self._fp_cvt_src.get(iops[1].name)
        if a is None or b is None or a[1] != b[1]:
            return None
        return a[0], b[0], a[1]

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
        # A reused/minted host block may carry the host's stale MBL_NORET ("dead
        # end: doesn't return execution control") -- set on blocks that, in the
        # ORIGINAL function, sat after a noreturn call (xalloc_die / the
        # __stack_chk_fail path). The drop repurposes such a block as a normal
        # call-continuation or branch-arm that DOES fall through to the return.
        # If the stale flag survives, Hex-Rays (from MMAT_CALLS on) severs the
        # block's successor edge and DCEs its body -- silently dropping the
        # COMPUTED return value it produces (e.g. `return hash_lookup(...) != 0`),
        # leaving only the constant arm (`return 0`). The block's noreturn-ness
        # must be re-derived from its NEW content, so clear the flag on wipe; a
        # genuinely-noreturn tail relies on BLT_0WAY (set explicitly in PASS A),
        # not on this inherited bit. See memory idavator_drop_retslot_mbl_noret.
        blk.flags &= ~hx.MBL_NORET
        blk.mark_lists_dirty()

    def _emit_i1(self, mba, blk, anchor, ea, operand, vmap):
        """Materialise an i1 ``operand`` as a 1-byte value descriptor, returning
        ``(desc, anchor)``. An icmp result (otherwise folded into a branch, never
        in ``vmap``) is emitted here as a ``setcc`` (``d = (l <pred> r) ? 1 : 0``);
        a constant ``true``/``false`` -> 1/0; anything already resolvable (a prior
        select result, a value arg) -> ``_desc``. Needed because ``select`` (and a
        short-circuit ``select`` arm) consumes an i1 as a VALUE."""
        nm = operand.name
        if nm and nm in self._icmp_defs and nm not in vmap:
            pred, iops = self._icmp_defs[nm]
            isz = _type_size(iops[0].type)
            mi = hx.minsn_t(ea)
            mi.opcode = _ICMP_SET.get(pred, hx.m_setnz)
            self._fill(mi.l, self._desc(iops[0], vmap, isz))
            self._fill(mi.r, self._desc(iops[1], vmap, isz))
            kreg = mba.alloc_kreg(1)
            mi.d.make_reg(kreg, 1)
            blk.insert_into_block(mi, anchor)
            vmap[nm] = ("reg", kreg, 1)
            return ("reg", kreg, 1), mi
        if nm and nm in self._fcmp_defs and nm not in vmap:
            # An fcmp result consumed as a VALUE (select cond / short-circuit arm) ->
            # the FPU setcc on the float operands (set_fpinsn), matching native's
            # ``v = (a <pred> b)`` rather than a truncated integer compare.
            fpred, fops = self._fcmp_defs[nm]
            fsz = _type_size(fops[0].type)
            mi = hx.minsn_t(ea)
            mi.opcode = _FCMP_SET.get(fpred, hx.m_setnz)
            self._fill(mi.l, self._desc(fops[0], vmap, fsz))
            self._fill(mi.r, self._desc(fops[1], vmap, fsz))
            mi.set_fpinsn()
            kreg = mba.alloc_kreg(1)
            mi.d.make_reg(kreg, 1)
            blk.insert_into_block(mi, anchor)
            vmap[nm] = ("reg", kreg, 1)
            return ("reg", kreg, 1), mi
        s = str(operand).strip()
        if s.split()[-1:] == ["true"]:
            return ("num", 1, 1), anchor
        if s.split()[-1:] == ["false"]:
            return ("num", 0, 1), anchor
        d = self._desc(operand, vmap, 1)
        return (d[0], d[1], 1), anchor

    def _direct_lvalue(self, operand, vmap, size):
        """Resolve ``operand`` (a load source / store dest pointer) to a DIRECT
        memory lvalue descriptor of ``size`` bytes -- ``("gvar", ea, size)`` for a
        global or ``("stkvar", off, size)`` for an escaped-alloca frame slot --
        or ``None`` when it is a runtime pointer that needs a deref. Used to lower
        a WHOLE-STRUCT copy (``load i448`` -> ``store i448``, the
        ``options = default_quoting_options`` 56-byte memcpy) as one mem-to-mem
        ``m_mov`` lvalue->lvalue: the value never lives in a register, so it sizes
        past the kreg cap (``alloc_kreg`` rejects >16) that scalarized it before.

        The pointer typically arrives already resolved in ``vmap`` -- the lifter
        prefixes the load/store with a no-op ``bitcast``/zero-offset ``getelementptr``
        that aliases the base to an ADDRESS descriptor (``gvaraddr`` &global /
        ``stkaddr`` &local). The LVALUE at that address is the same location read at
        ``size`` bytes (``gvar``/``stkvar``)."""
        d = vmap.get(operand.name)
        if d is not None:
            if d[0] in ("gvar", "gvaraddr"):
                return ("gvar", d[1], size)
            if d[0] in ("stkaddr", "stkvar"):
                return ("stkvar", d[1], size)
        stk = self._stkvar_slot(operand, vmap)
        if stk is not None:
            return ("stkvar", stk[0], size)
        gea = self._global_ea(operand)
        if gea is not None:
            return ("gvar", gea, size)
        return None

    def _emit_narrow(self, mba, blk, anchor, ea, ins, vmap, src_op,
                     in_sz, out_sz, out_bits, out_raw):
        """Lower a NARROWING cast (``trunc``, or a width-shrinking ptrtoint /
        inttoptr) into a legal-width kreg, recording the result in ``vmap``.

        ``in_sz``/``out_sz`` are the LEGAL (rounded) kreg widths; ``out_bits`` is
        the logical bit count of the result type (``None`` for a non-integer
        result, e.g. a pointer); ``out_raw`` is its raw ``ceil(N/8)`` byte size.

        When the result type is a non-power-of-2 slice (``out_raw`` not a basic
        size, e.g. ``i24`` -> 3B rounded to 4B) the value must be MASKED to its
        ``out_bits`` logical bits -- a single ``m_and`` both truncates the high
        source bytes AND clears the rounded kreg's padding bits, so the paired
        ``zext`` re-extends it faithfully (zero high bits == zero-extension). A
        truncation to a LEGAL width keeps the plain ``m_low``."""
        illegal = (out_bits is not None
                   and (out_bits % 8 != 0 or out_raw not in _LEGAL_KREG_SIZES))
        if illegal and out_bits < out_sz * 8:
            # Mask to the logical bit-width directly from the (legal-width)
            # source; m_and l/r/d all share out_sz. This subsumes the narrow.
            mi = hx.minsn_t(ea)
            mi.opcode = hx.m_and
            self._fill(mi.l, self._desc(src_op, vmap, out_sz))
            mi.r.make_number((1 << out_bits) - 1, out_sz)
            kreg = mba.alloc_kreg(out_sz)
            mi.d.make_reg(kreg, out_sz)
            blk.insert_into_block(mi, anchor)
            vmap[ins.name] = ("reg", kreg, out_sz)
            return mi
        if in_sz == out_sz:
            # Rounded to the same legal width and no masking needed -- alias.
            vmap[ins.name] = self._desc(src_op, vmap, in_sz)
            return anchor
        mi = hx.minsn_t(ea)
        mi.opcode = hx.m_low
        self._fill(mi.l, self._desc(src_op, vmap, in_sz))
        kreg = mba.alloc_kreg(out_sz)
        mi.d.make_reg(kreg, out_sz)
        blk.insert_into_block(mi, anchor)
        vmap[ins.name] = ("reg", kreg, out_sz)
        return mi

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
        if op in _FP_BINOP:
            # IEEE fp arithmetic: l <op> r -> d, all the SAME fp width (the result
            # type, == both operand types for a well-typed fadd/fmul/...). Mirrors
            # the integer _BINOP path but emits an FPU microinsn (set_fpinsn) so
            # the verifier/optimizer treats l/r/d as floats, not raw bit-patterns.
            # An fp CONSTANT operand resolves to a mop_fn via _desc (`fpnum`).
            size = _type_size(ins.type)
            mi = hx.minsn_t(ea)
            mi.opcode = _FP_BINOP[op]
            self._fill(mi.l, self._desc(ops[0], vmap, size))
            self._fill(mi.r, self._desc(ops[1], vmap, size))
            kreg = mba.alloc_kreg(size)
            mi.d.make_reg(kreg, size)
            mi.set_fpinsn()
            blk.insert_into_block(mi, anchor)
            vmap[ins.name] = ("reg", kreg, size)
            return mi
        if op in _FP_CAST or op in _FP_RESIZE:
            # fp<->int conversion (m_i2f/m_u2f/m_f2i/m_f2u) or fp precision change
            # (m_f2f). The source width is the OPERAND type's size, the dest width
            # the RESULT type's size -- a `sitofp i32 -> float` reads a 4-byte int
            # and writes a 4-byte float; a `fptoui float -> i32` the reverse. Like
            # _CAST these alloc a dest kreg of the result width. set_fpinsn marks
            # it an FPU op (the source OR dest is fp).
            in_sz = _type_size(ops[0].type)
            out_sz = _type_size(ins.type)
            mi = hx.minsn_t(ea)
            mi.opcode = (_FP_CAST.get(op) or _FP_RESIZE.get(op))
            self._fill(mi.l, self._desc(ops[0], vmap, in_sz))
            kreg = mba.alloc_kreg(out_sz)
            mi.d.make_reg(kreg, out_sz)
            mi.set_fpinsn()
            blk.insert_into_block(mi, anchor)
            vmap[ins.name] = ("reg", kreg, out_sz)
            return mi
        if op == "bitcast" and ops and ops[0].name == "IDA_QWORD":
            # ``bitcast i8* ()* @IDA_QWORD to i8*`` is the lifter's TYPE MARKER for
            # the ``_QWORD`` argument of ``va_arg(ap, _QWORD)`` (ida2llvm models the
            # type as a declared marker fn). It is DEAD -- the ``@va_arg`` call takes
            # only the va_list ptr; this bitcast feeds nothing. ``@IDA_QWORD`` is a
            # function declare, not a value, so ``_desc`` cannot resolve it. No-op.
            return anchor
        if op in _NOOP_CAST:
            # ptrtoint/inttoptr/bitcast are usually bit-identical reinterpretations
            # (ptr<->int, ptr<->ptr) of the SAME width -- alias the operand, no insn.
            # But ptrtoint/inttoptr can also CHANGE width (e.g. `ptrtoint i8* %p to
            # i8` truncates 8->1, `inttoptr i32 %x to ptr` widens 4->8); a width
            # change is a REAL narrow/widen, not an alias, and must lower like
            # trunc/zext (m_low / m_xdu into a fresh kreg). Otherwise the consumer
            # -- typically an icmp folded to a conditional jump -- sees an operand
            # whose size mismatches its comparand (`m_jz l=kr.8 r=#0.1`) and the
            # verifier rejects it (INTERR 50831, verify.cpp conditional-branch
            # operand-size check requires l.size == r.size).
            in_raw = _type_size(ops[0].type)
            out_raw = _type_size(ins.type)
            if in_raw == out_raw or out_raw == 0 or in_raw == 0:
                vmap[ins.name] = self._desc(ops[0], vmap, 8)
                return anchor
            # SROA can emit a width-changing ptrtoint/inttoptr through a
            # non-power-of-2 slice; round both ends to a legal kreg width
            # (alloc_kreg rejects 3/5/6/7) and mask a narrowing result to its
            # logical bits so a later widen re-extends it faithfully.
            in_sz = _legal_kreg_size(in_raw)
            out_sz = _legal_kreg_size(out_raw)
            if out_raw < in_raw:
                return self._emit_narrow(
                    mba, blk, anchor, ea, ins, vmap, ops[0],
                    in_sz, out_sz, _int_bits(ins.type), out_raw)
            if in_sz == out_sz:
                vmap[ins.name] = self._desc(ops[0], vmap, in_sz)
                return anchor
            mi = hx.minsn_t(ea)
            mi.opcode = hx.m_xdu
            self._fill(mi.l, self._desc(ops[0], vmap, in_sz))
            kreg = mba.alloc_kreg(out_sz)
            mi.d.make_reg(kreg, out_sz)
            blk.insert_into_block(mi, anchor)
            vmap[ins.name] = ("reg", kreg, out_sz)
            return mi
        if op in _CAST:
            in_raw = _type_size(ops[0].type)
            out_raw = _type_size(ins.type)
            # SROA byte-splits a wide scalar into NON-power-of-2 slices (i24/i48/
            # i56 -> 3/6/7B); alloc_kreg(check_size) REJECTS those. Round each end
            # to a legal kreg width and, for a NARROWING trunc to such a slice,
            # MASK the value to its logical bits so the paired widen re-extends it
            # faithfully (the rounded kreg's high padding bits carry no meaning).
            in_sz = _legal_kreg_size(in_raw)
            out_sz = _legal_kreg_size(out_raw)
            if op == "trunc":
                return self._emit_narrow(
                    mba, blk, anchor, ea, ins, vmap, ops[0],
                    in_sz, out_sz, _int_bits(ins.type), out_raw)
            # zext / sext (a WIDEN): the source already holds a value masked /
            # sign-correct in its logical bits within ``in_sz`` bytes.
            in_bits = _int_bits(ops[0].type)
            src_illegal = in_bits is not None and (in_bits % 8 != 0
                                                   or in_raw not in _LEGAL_KREG_SIZES)
            if in_sz == out_sz and not (op == "sext" and src_illegal):
                # Same legal BYTE width (e.g. `zext i1 to i8`, or `zext i24 to i32`
                # where both round to 4): a no-op reinterpretation. m_xdu/m_xds/
                # m_low INTERR on equal l/d sizes (50837/50838) -> alias. For zext
                # this is exact (high bits already zero from the producer's mask);
                # sext from an ILLEGAL slice still needs sign-replication, handled
                # below.
                vmap[ins.name] = self._desc(ops[0], vmap, in_sz)
                return anchor
            if op == "sext" and src_illegal:
                # Sign-extend from logical bit ``in_bits-1`` of a value that lives
                # masked in a wider kreg: arithmetic ``(x << k) >>s k`` with
                # k = in_sz*8 - in_bits replicates the sign bit, then widen to
                # out_sz. (No such site in cp.ll today; kept for faithfulness.)
                k = in_sz * 8 - in_bits
                shl = hx.minsn_t(ea)
                shl.opcode = hx.m_shl
                self._fill(shl.l, self._desc(ops[0], vmap, in_sz))
                shl.r.make_number(k, 1)
                sk = mba.alloc_kreg(in_sz)
                shl.d.make_reg(sk, in_sz)
                blk.insert_into_block(shl, anchor)
                anchor = shl
                sar = hx.minsn_t(ea)
                sar.opcode = hx.m_sar
                sar.l.make_reg(sk, in_sz)
                sar.r.make_number(k, 1)
                rk = mba.alloc_kreg(in_sz)
                sar.d.make_reg(rk, in_sz)
                blk.insert_into_block(sar, anchor)
                anchor = sar
                if in_sz == out_sz:
                    vmap[ins.name] = ("reg", rk, out_sz)
                    return sar
                wi = hx.minsn_t(ea)
                wi.opcode = hx.m_xds
                wi.l.make_reg(rk, in_sz)
                wk = mba.alloc_kreg(out_sz)
                wi.d.make_reg(wk, out_sz)
                blk.insert_into_block(wi, anchor)
                vmap[ins.name] = ("reg", wk, out_sz)
                return wi
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
            if out_sz > 16:
                # A WHOLE-STRUCT load (``load i448`` == read all 56 bytes of a
                # ``struct quoting_options``). The value is too wide for a kreg
                # (alloc_kreg rejects >16) and there is no register to hold it --
                # it only ever feeds the paired ``store`` of a struct COPY. Defer:
                # record the SOURCE lvalue and emit nothing; the store lowers the
                # pair as one mem-to-mem ``m_mov`` (see ``store``/``_direct_lvalue``).
                src_lv = self._direct_lvalue(ops[0], vmap, out_sz)
                if src_lv is None:
                    raise NotImplementedError(
                        f"aggregate load %{ins.name} ({out_sz}B) from a non-direct "
                        f"lvalue -- needs a memcpy through a runtime pointer")
                vmap[ins.name] = ("memref", src_lv, out_sz)
                return anchor
            if self._is_ret_slot(ops[0], vmap):
                # promoted return slot: the loaded value IS the return register.
                _ar, eax, _d = self._abi()
                vmap[ins.name] = ("reg", eax, out_sz)
                return anchor
            if (ops[0].name in self._ptr_deref_alias
                    and not _is_ptr_type(ins.type)
                    and not self._is_punned_ptr_value(ops[0], out_sz)):
                # *X (deref) of a pointer-alloca slot: the lifter reaches it via a
                # no-op bitcast and a load of the POINTEE type (e.g. `*name` as i8,
                # or a pointer-width `*total_n_read` where total_n_read is a
                # `size_t*`). Read the slot's POINTER value, then ldx through it --
                # native's `mov %X, r; ldx ds, r`. The distinguisher from a slot
                # read is the result's TYPE *and* width: the lifter type-puns BOTH
                # a full pointer-VALUE read (`load ptr, bitcast %X`, ins.type is a
                # pointer -> stays a slot read below; OR `load i64, bitcast %X`
                # where the slot's pointee is SUB-pointer-width -- the
                # ``_is_punned_ptr_value`` case: passing the pointer BY VALUE to a
                # callback, e.g. ``safe_hasher``'s ``table->hasher(key, ...)``) AND
                # ``*X`` as a non-pointer load (`load i64, bitcast %X` for `*p`
                # where p is a pointer to an 8-byte value). A non-pointer result
                # whose width MATCHES the pointee is the unambiguous deref; a
                # pointer-width read of a sub-pointer pointee is a punned VALUE read
                # and falls through to the slot read below.
                poff = self._ptr_deref_off(ops[0], vmap)
                pr = mba.alloc_kreg(8)
                mv = hx.minsn_t(ea)
                mv.opcode = hx.m_mov
                mv.l.make_stkvar(mba, poff)
                mv.l.size = 8
                mv.d.make_reg(pr, 8)
                blk.insert_into_block(mv, anchor)
                mi = hx.minsn_t(ea)
                mi.opcode = hx.m_ldx
                mi.l.make_reg(ds, 2)
                mi.r.make_reg(pr, 8)
                kreg = mba.alloc_kreg(out_sz)
                mi.d.make_reg(kreg, out_sz)
                blk.insert_into_block(mi, mv)
                vmap[ins.name] = ("reg", kreg, out_sz)
                return mi
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
            if vd[0] == "memref":
                # WHOLE-STRUCT copy: the stored value is a deferred aggregate load
                # (see the ``load`` ``out_sz > 16`` branch). Lower the load/store
                # pair as ONE mem-to-mem ``m_mov`` SRC_lvalue -> DST_lvalue, both
                # sized to the struct (HexRays supports UDT-sized operands). This
                # is native's ``options = default_quoting_options`` -- no register
                # round-trip, so it sidesteps the kreg size cap that scalarized the
                # copy to its first field.
                src_lv = vd[1]
                dst_lv = self._direct_lvalue(ops[1], vmap, val_sz)
                if dst_lv is None:
                    raise NotImplementedError(
                        f"aggregate store ({val_sz}B) to a non-direct lvalue -- "
                        f"needs a memcpy through a runtime pointer")
                mi = hx.minsn_t(ea)
                mi.opcode = hx.m_mov
                self._fill(mi.l, src_lv)
                self._fill(mi.d, dst_lv)
                # An UNTYPED operand of a non-basic size (56B) trips the verifier's
                # ``is_valid_size`` check (INTERR 50757, verify.cpp); the check is
                # SKIPPED for a UDT (``is_udt()``). Flag both operands so the
                # struct-sized mov is accepted -- this is the microcode model of a
                # whole-struct assignment (UDT-to-UDT mov).
                mi.l.set_udt()
                mi.d.set_udt()
                blk.insert_into_block(mi, anchor)
                return mi
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
            if (ops[1].name in self._ptr_deref_alias
                    and not _is_ptr_type(ops[0].type)
                    and not (val_sz == 8
                             and ops[1].name in self._ptr_alloca_pointee_struct)):
                # *X = v (deref write) of a pointer-alloca slot: a store through
                # the no-op bitcast writes the POINTEE *X (e.g. `oa->style = 10`,
                # or a pointer-width `*total_n_read = 0` where total_n_read is a
                # `size_t*`). Read the slot's POINTER value, then stx through it --
                # native's `mov %X, r; stx v, ds, r`. The distinguisher from a
                # slot DEFINE is the stored value's TYPE, not its width: a store of
                # a POINTER value (`_is_ptr_type(ops[0].type)`) instead DEFINES the
                # pointer and falls through to the slot-write path below (e.g.
                # `bucket = *table`, `oa = &default`). A non-pointer value of ANY
                # width (i8 field OR a full i64 `*p = 0`) is a deref -- EXCEPT a
                # pointer-WIDTH (8B) store into a STRUCT-pointee slot
                # (`_ptr_alloca_pointee_struct`): the lifter lowers cursor
                # arithmetic `last_ei = scan->ext_info + idx` to `store i64
                # <ptrtoint+add>, bitcast(%last_ei to i64*)` -- a non-pointer i64
                # value, but it DEFINES the cursor (an 8-byte store can never be a
                # full struct-pointee write; a genuine first-field deref
                # `last_ei->ext_logical = v` LOADs the pointer first so its bitcast
                # is rooted at the load, not the alloca -> not in
                # `_ptr_deref_alias`). A scalar i64 pointee (`*total_n_read`) is
                # NOT struct -> stays a deref.
                poff = self._ptr_deref_off(ops[1], vmap)
                pr = mba.alloc_kreg(8)
                mv = hx.minsn_t(ea)
                mv.opcode = hx.m_mov
                mv.l.make_stkvar(mba, poff)
                mv.l.size = 8
                mv.d.make_reg(pr, 8)
                blk.insert_into_block(mv, anchor)
                mi = hx.minsn_t(ea)
                mi.opcode = hx.m_stx
                self._fill(mi.l, (vd[0], vd[1], val_sz))
                mi.r.make_reg(ds, 2)
                mi.d.make_reg(pr, 8)
                blk.insert_into_block(mi, mv)
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
                    # The off==0 decay of a global base yields its ADDRESS, not
                    # the lvalue read at it: a gvar base decays to gvaraddr
                    # (&global). _desc already returns gvaraddr for a ptr-typed
                    # global operand; normalise a gvar base here for safety.
                    if base[0] == "gvar":
                        vmap[ins.name] = ("gvaraddr", base[1], 8)
                    else:
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
        if op in ("icmp", "fcmp"):
            # Folded into the branch (see _build_multiblock); no value emitted. An
            # fcmp consumed as a value (select/short-circuit) is materialised on
            # demand by _emit_i1 (FPU setcc), not here.
            return anchor
        if op == "select":
            return self._emit_select(mba, blk, anchor, ea, ins, vmap)
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
            if callee in _VA_SCAFFOLD:
                # The lifter's redundant synth prologue -- ``call @llvm.va_start``
                # / ``@llvm.va_end`` on an uninit ``%ArgList`` -- is DEAD (the real
                # va_list machine is the body's ``@va_start``/``@va_arg`` over the
                # ``__va_list_tag`` storage). Native has NO counterpart; drop it.
                return anchor
            if callee in _VA_HELPER:
                # The body's va_list machine: ``va_start(ap, last)`` /
                # ``va_arg(ap, T)`` / ``va_end(ap)``. HexRays renders these as
                # helper-call MACROS over ``__va_list_tag`` (like ``__ROR8__``),
                # NOT the ``!va_start`` IR intrinsic -- so a helper call matches
                # the native rendering. ``va_arg`` yields the next argument; copy
                # rax into a stable kreg and record it so the consuming store
                # (``store i64 %va_arg_result, i64* %vN``) reads a live value
                # across any later rax-clobbering call.
                _ar, eax, _ds = self._abi()
                arg_descs = [self._desc(a, vmap, _type_size(a.type))
                             for a in list(ins.operands)[:-1]]
                anchor = self._emit_helper_call(
                    mba, blk, anchor, ea, ins, callee, arg_descs, eax)
                if str(ins.type) != "void":
                    rsz = _type_size(ins.type)
                    kreg = mba.alloc_kreg(rsz)
                    mv = hx.minsn_t(ea)
                    mv.opcode = hx.m_mov
                    mv.l.make_reg(eax, rsz)
                    mv.d.make_reg(kreg, rsz)
                    blk.insert_into_block(mv, anchor)
                    anchor = mv
                    vmap[ins.name] = ("reg", kreg, rsz)
                return anchor
            raise RuntimeError("call must be split into its own block "
                               "(handled by the segment splitter, not _emit_value)")
        logger.warning("unhandled LLVM opcode: %s", op)
        return anchor

    def _emit_select(self, mba, blk, anchor, ea, ins, vmap):
        """Lower ``%r = select i1 %c, T %a, T %b`` (a branchless ternary).

        Three forms, all branchless (a 2-way merge can re-trip the noreturn-merge
        INTERR family -- avoid emitting one):

        - SHORT-CIRCUIT boolean (one arm a constant i1): instcombine emits
          ``select i1 %c, i1 true,  i1 %b`` == ``%c | %b`` (or.cond) and
          ``select i1 %c, i1 %a,    i1 false`` == ``%c & %a`` (and.cond). Lower to
          ``m_or``/``m_and`` on the two 1-byte i1 operands (each materialised via
          ``_emit_i1`` -- an icmp arm becomes a setcc). Exact and cheap.
        - BOOLEAN MATERIALISE ``select i1 %c, <N> 1, <N> 0`` == ``zext c to N``:
          the (1-byte) condition widened to the result width.
        - GENERAL ``select i1 %c, T %a, T %b``: a branchless blend
          ``r = b + ((a - b) & mask)`` where ``mask = 0 - (T)c`` is all-ones when
          ``c`` else 0 (so ``r == a`` when ``c``, else ``b``). T is the arm width."""
        ops = list(ins.operands)
        cond, tval, fval = ops[0], ops[1], ops[2]
        out_sz = _type_size(ins.type)
        ts, fs = str(tval).strip(), str(fval).strip()
        t_is_i1 = _type_size(tval.type) == 1 and "i1" in str(tval.type)
        f_is_i1 = _type_size(fval.type) == 1 and "i1" in str(fval.type)

        # SHORT-CIRCUIT: a constant i1 arm collapses to a bitwise op on the i1s.
        if t_is_i1 and ts.split()[-1:] == ["true"]:
            # select c, true, b  ==  c | b
            cd, anchor = self._emit_i1(mba, blk, anchor, ea, cond, vmap)
            bd, anchor = self._emit_i1(mba, blk, anchor, ea, fval, vmap)
            return self._emit_bool_binop(mba, blk, anchor, ea, ins, hx.m_or, cd,
                                         bd, vmap, out_sz)
        if f_is_i1 and fs.split()[-1:] == ["false"]:
            # select c, a, false  ==  c & a
            cd, anchor = self._emit_i1(mba, blk, anchor, ea, cond, vmap)
            ad, anchor = self._emit_i1(mba, blk, anchor, ea, tval, vmap)
            return self._emit_bool_binop(mba, blk, anchor, ea, ins, hx.m_and, cd,
                                         ad, vmap, out_sz)
        if f_is_i1 and fs.split()[-1:] == ["true"]:
            # select c, a, true  ==  !c | a  ==  (c==0) | a
            cd, anchor = self._emit_i1(mba, blk, anchor, ea, cond, vmap)
            ad, anchor = self._emit_i1(mba, blk, anchor, ea, tval, vmap)
            nc = hx.minsn_t(ea)
            nc.opcode = hx.m_setz
            self._fill(nc.l, cd)
            nc.r.make_number(0, 1)
            nk = mba.alloc_kreg(1)
            nc.d.make_reg(nk, 1)
            blk.insert_into_block(nc, anchor)
            anchor = nc
            return self._emit_bool_binop(mba, blk, anchor, ea, ins, hx.m_or,
                                         ("reg", nk, 1), ad, vmap, out_sz)
        if t_is_i1 and ts.split()[-1:] == ["false"]:
            # select c, false, b  ==  !c & b
            cd, anchor = self._emit_i1(mba, blk, anchor, ea, cond, vmap)
            bd, anchor = self._emit_i1(mba, blk, anchor, ea, fval, vmap)
            nc = hx.minsn_t(ea)
            nc.opcode = hx.m_setz
            self._fill(nc.l, cd)
            nc.r.make_number(0, 1)
            nk = mba.alloc_kreg(1)
            nc.d.make_reg(nk, 1)
            blk.insert_into_block(nc, anchor)
            anchor = nc
            return self._emit_bool_binop(mba, blk, anchor, ea, ins, hx.m_and,
                                         ("reg", nk, 1), bd, vmap, out_sz)

        # BOOLEAN MATERIALISE: select c, 1, 0  ==  zext c.
        td = self._desc(tval, vmap, out_sz) \
            if tval.name not in self._icmp_defs and tval.name not in self._fcmp_defs \
            else None
        fd = self._desc(fval, vmap, out_sz) \
            if fval.name not in self._icmp_defs and fval.name not in self._fcmp_defs \
            else None
        if (td and td[0] == "num" and td[1] == 1
                and fd and fd[0] == "num" and fd[1] == 0):
            cd, anchor = self._emit_i1(mba, blk, anchor, ea, cond, vmap)
            if out_sz == 1:
                vmap[ins.name] = (cd[0], cd[1], 1)
                return anchor
            mi = hx.minsn_t(ea)
            mi.opcode = hx.m_xdu
            self._fill(mi.l, cd)
            kreg = mba.alloc_kreg(out_sz)
            mi.d.make_reg(kreg, out_sz)
            blk.insert_into_block(mi, anchor)
            vmap[ins.name] = ("reg", kreg, out_sz)
            return mi

        # GENERAL branchless blend: r = b + ((a - b) & mask), mask = 0 - (T)c.
        ad = self._desc(tval, vmap, out_sz)
        bd = self._desc(fval, vmap, out_sz)
        cd, anchor = self._emit_i1(mba, blk, anchor, ea, cond, vmap)
        # mask = 0 - zext(c, out_sz)  (all-ones when c, else 0).
        if out_sz != 1:
            zc = hx.minsn_t(ea)
            zc.opcode = hx.m_xdu
            self._fill(zc.l, cd)
            zk = mba.alloc_kreg(out_sz)
            zc.d.make_reg(zk, out_sz)
            blk.insert_into_block(zc, anchor)
            anchor = zc
            cwide = ("reg", zk, out_sz)
        else:
            cwide = cd
        mneg = hx.minsn_t(ea)
        mneg.opcode = hx.m_neg
        self._fill(mneg.l, (cwide[0], cwide[1], out_sz))
        mk = mba.alloc_kreg(out_sz)
        mneg.d.make_reg(mk, out_sz)
        blk.insert_into_block(mneg, anchor)
        anchor = mneg
        # diff = a - b
        sub = hx.minsn_t(ea)
        sub.opcode = hx.m_sub
        self._fill(sub.l, (ad[0], ad[1], out_sz))
        self._fill(sub.r, (bd[0], bd[1], out_sz))
        sk = mba.alloc_kreg(out_sz)
        sub.d.make_reg(sk, out_sz)
        blk.insert_into_block(sub, anchor)
        anchor = sub
        # masked = diff & mask
        msk = hx.minsn_t(ea)
        msk.opcode = hx.m_and
        msk.l.make_reg(sk, out_sz)
        msk.r.make_reg(mk, out_sz)
        ck = mba.alloc_kreg(out_sz)
        msk.d.make_reg(ck, out_sz)
        blk.insert_into_block(msk, anchor)
        anchor = msk
        # result = b + masked
        add = hx.minsn_t(ea)
        add.opcode = hx.m_add
        self._fill(add.l, (bd[0], bd[1], out_sz))
        add.r.make_reg(ck, out_sz)
        rk = mba.alloc_kreg(out_sz)
        add.d.make_reg(rk, out_sz)
        blk.insert_into_block(add, anchor)
        vmap[ins.name] = ("reg", rk, out_sz)
        return add

    def _emit_bool_binop(self, mba, blk, anchor, ea, ins, opcode, ld, rd, vmap,
                         out_sz):
        """Emit ``d = ld <opcode> rd`` (m_or/m_and on two 1-byte i1 values) for a
        short-circuit select; record the 1-byte result in ``vmap`` (widened to the
        select's result width if it is wider than i1)."""
        mi = hx.minsn_t(ea)
        mi.opcode = opcode
        self._fill(mi.l, (ld[0], ld[1], 1))
        self._fill(mi.r, (rd[0], rd[1], 1))
        bk = mba.alloc_kreg(1)
        mi.d.make_reg(bk, 1)
        blk.insert_into_block(mi, anchor)
        anchor = mi
        if out_sz == 1:
            vmap[ins.name] = ("reg", bk, 1)
            return mi
        wi = hx.minsn_t(ea)
        wi.opcode = hx.m_xdu
        wi.l.make_reg(bk, 1)
        wk = mba.alloc_kreg(out_sz)
        wi.d.make_reg(wk, out_sz)
        blk.insert_into_block(wi, anchor)
        vmap[ins.name] = ("reg", wk, out_sz)
        return wi

    def _formal_arg_sizes(self, callee_ea, nargs):
        """``[size_0, ..., size_{nargs-1}]`` of a resolved DIRECT callee's formal
        parameters from its IDB prototype, or ``None`` when no usable widening
        applies -- an unresolved ea, a non-function/vararg type, or a prototype
        whose arity differs from the call (defer rather than mis-pair widths).

        Used by ``_emit_call`` to widen each in-register arg to its real ABI width
        when the lifted IR operand type is narrower than the callee declares (the
        ``memset``/``size_t`` case). A vararg callee is excluded: its surplus-arg
        widths are carried by the explicit mcallinfo path, not here."""
        if callee_ea == ida_idaapi.BADADDR:
            return None
        ctif = ida_typeinf.tinfo_t()
        if not ida_nalt.get_tinfo(ctif, callee_ea) or not ctif.is_func():
            return None
        if ctif.is_vararg_cc() or ctif.get_nargs() != nargs:
            return None
        out = []
        for i in range(nargs):
            sz = ctif.get_nth_arg(i).get_size()
            out.append(sz if sz not in (0, ida_idaapi.BADADDR) else 0)
        return out

    def _emit_call(self, mba, blk, anchor, ea, ins, vmap, argregs):
        """Emit `mov`s into the ABI arg-regs then `m_call l=gvar(callee), d=rax`
        as the block TAIL. The call must terminate its block (it falls through to
        the continuation, BLT_1WAY) -- a call defining rax mid-block is fine for
        50864 but later maturities want calls block-terminal. Returns the call."""
        ops = list(ins.operands)
        callee = ops[-1]
        call_args = ops[:-1]
        _argregs, eax, _ds = self._abi()
        # See through a function-pointer bitcast callee: the lifter wraps a
        # variadic call that carries surplus varargs as ``%c = bitcast @fn to
        # <wider variadic>; call %c(...)``, so the callee operand is the SSA
        # bitcast result. Resolve it back to the underlying NAMED function so the
        # routing below treats it as a DIRECT call (else `callee.name in vmap` is
        # false but get_name_ea(".11")==BADADDR and it mis-routes / drops the call
        # -> INTERR 50406). The wider call type is irrelevant: arg count + the real
        # @fn prototype drive the variadic dispatch.
        #
        # ONLY for a result-DISCARDED call (the error/fprintf surplus-vararg form
        # _emit_call_vararg models). A CONSUMED-result variadic call (e.g.
        # ``v = openat(dirfd, path, oflag, mode)`` where mode rides as a vararg)
        # has no result-discarded recipe in _emit_call_vararg -- keep the old
        # behaviour (the bitcast callee falls through to the indirect-call path,
        # which builds) rather than route it to a decline that loses the build.
        is_fnptr_cast = (callee.name in self._fnptr_bitcast
                         and not self._value_used(ins))
        callee_name = (self._fnptr_bitcast[callee.name]
                       if is_fnptr_cast else callee.name)
        # More integer args than ABI registers: the 7th+ ride on the stack. A
        # direct (named) callee whose prototype is known lets us build an
        # explicit mcallinfo (set_type does the SysV reg/stack classification),
        # so the stack args travel IN the call -- no SP-modeled pushes. Indirect
        # / unresolved callees fall through to their existing handling below.
        if (len(call_args) > len(argregs) and callee_name not in vmap
                and ida_name.get_name_ea(ida_idaapi.BADADDR, callee_name)
                != ida_idaapi.BADADDR):
            return self._emit_call_stackargs(
                mba, blk, anchor, ea, ins, vmap, callee_name=callee_name)
        if len(call_args) > len(argregs):
            raise NotImplementedError(
                "stack-passed call argument (more args than ABI registers)")
        # Callee: a direct named function/global -> gvar. An indirect call through
        # an SSA value (a function pointer, e.g. a loaded struct field) is lowered
        # to Hex-Rays' native m_icall form -- see _emit_call_indirect. A
        # function-pointer bitcast is NOT indirect (it has a resolved name).
        if callee_name in vmap and not is_fnptr_cast:
            return self._emit_call_indirect(mba, blk, anchor, ea, ins, vmap)
        callee_ea = ida_name.get_name_ea(ida_idaapi.BADADDR, callee_name)
        if callee_ea == ida_idaapi.BADADDR:
            if callee.name in _HELPER_INTRINSICS:
                arg_descs = [self._desc(a, vmap, _type_size(a.type))
                             for a in call_args]
                return self._emit_helper_call(
                    mba, blk, anchor, ea, ins, callee.name, arg_descs, eax)
            raise ValueError(f"unresolved callee @{callee.name}")
        # Variadic callee (e.g. printf/error) carrying trailing varargs: a bare
        # `m_call gvar` with no mcallinfo leaves Hex-Rays to re-discover the
        # vararg count from its own analysis, and it drops them (rendering only a
        # mis-resolved fmt). Build an EXPLICIT mcallinfo declaring every passed
        # arg (fixed prefix + varargs) so the call carries them deterministically.
        # Only when surplus args are actually present beyond the fixed prototype.
        ctif = ida_typeinf.tinfo_t()
        is_vararg = (ida_nalt.get_tinfo(ctif, callee_ea) and ctif.is_func()
                     and ctif.is_vararg_cc())
        if is_vararg and len(call_args) > ctif.get_nargs():
            return self._emit_call_vararg(
                mba, blk, anchor, ea, ins, vmap, callee_ea, ctif)
        # Variadic callee invoked with EXACTLY its fixed args (no surplus vararg).
        # A bare `m_call gvar` to a variadic prototype still lets Hex-Rays' own
        # vararg recovery invent trailing args -- e.g. `open("/dev/urandom", 0)`
        # mis-renders as `open(a0, a1, a2)`, reading stale incoming-param regs as
        # phantom varargs. Pin the call with an EXPLICIT fixed-arg variadic
        # mcallinfo (FCI_FINAL) so no re-derivation occurs. set_type keeps the
        # ellipsis cc; we fill only the fixed args (no surplus tail).
        if is_vararg and len(call_args) <= ctif.get_nargs():
            return self._emit_call_vararg_fixed(
                mba, blk, anchor, ea, ins, vmap, callee_ea, ctif)
        # A RESULT-DISCARDED pure-WRITER libc call (memset/bzero -- see
        # _MEMCLOBBER_FNS) writes THROUGH its destination pointer arg. A bare
        # ``m_call gvar`` carries no mcallinfo, so Hex-Rays' glbopt cannot see the
        # memory write: when a later store in the same block covers the written
        # bytes (e.g. qset_acl's ``memset(&ctx,0,4); ctx.mode = mode`` -- the mode
        # store overwrites all 4 bytes of the {i32} struct) the WHOLE call is dead-
        # code-eliminated and the zero-init is silently dropped. The NATIVE decompile
        # keeps it because its call carries a typed mcallinfo (``<fast:"void *s"
        # &ctx,...>``) that models the pointer write. Emit such a call with an
        # EXPLICIT mcallinfo (set_type + FCI_FINAL, via the fixed-arg path) so HR
        # models the clobber and preserves the call. Scoped to resolved, non-vararg,
        # DISCARDED-result pure-writer callees (the only ones HR can wrongly fold);
        # every other call -- including reader-clobbers like memcpy -- is unchanged.
        if (not is_vararg and not self._value_used(ins)
                and callee_name in _MEMCLOBBER_FNS):
            mctif = ida_typeinf.tinfo_t()
            if (ida_nalt.get_tinfo(mctif, callee_ea) and mctif.is_func()
                    and not mctif.is_vararg_cc()
                    and mctif.get_nargs() == len(call_args)):
                return self._emit_call_vararg_fixed(
                    mba, blk, anchor, ea, ins, vmap, callee_ea, mctif)
        # Resolve args once; a call that materializes a frame address (&local)
        # must carry the host resting-frame ea so Hex-Rays computes a
        # frame-consistent mcallinfo.call_spd -- else WARN_BAD_CALL_SP ("bad sp
        # value at call"). See memory idavator_sp_gate_call_ea_cracked.
        arg_descs = [self._desc(a, vmap, _type_size(a.type)) for a in call_args]
        passes_stkaddr = any(d[0] == "stkaddr" for d in arg_descs)
        # Argument register WIDTH comes from the callee's REAL prototype, not the
        # lifted IR operand type. ida2llvm may declare an extern narrower than its
        # IDB prototype -- e.g. ``memset`` lifts as ``(i8*, i32, i32)`` so the size
        # arg is an i32 (4 bytes), but IDA's real ``memset`` takes ``size_t n``
        # (8 bytes). A 4-byte ``mov #4, edx`` feeding an 8-byte ``size_t`` use
        # leaves the high dword of rdx undefined: Hex-Rays then can't fold the
        # constant (renders ``LODWORD(v)=4; memset(&ctx,0,v)`` instead of
        # ``memset(&ctx,0,sizeof(ctx))``) and the partial-def trips "local variable
        # allocation has failed". Widen each in-register arg to its formal param
        # size (zero-extend a narrower reg via m_xdu; a number just takes the wider
        # size) -- exactly the formal-size widening _emit_call_stackargs already
        # applies to the 7th+ args. Only for a resolved, non-vararg callee whose
        # prototype arity matches the call (vararg widths ride in the mcallinfo).
        formal_szs = self._formal_arg_sizes(callee_ea, len(call_args))
        for i, (a, d) in enumerate(zip(call_args, arg_descs)):
            asz = _type_size(a.type)
            fsz = formal_szs[i] if formal_szs is not None else asz
            if fsz > asz:
                if d[0] == "num":
                    d, asz = ("num", d[1], fsz), fsz
                elif d[0] == "reg":
                    mi = hx.minsn_t(ea)
                    mi.opcode = hx.m_xdu
                    self._fill(mi.l, (d[0], d[1], asz))
                    kreg = mba.alloc_kreg(fsz)
                    mi.d.make_reg(kreg, fsz)
                    blk.insert_into_block(mi, anchor)
                    anchor = mi
                    d, asz = ("reg", kreg, fsz), fsz
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

    def _emit_call_vararg(self, mba, blk, anchor, ea, ins, vmap, callee_ea, ctif):
        """Emit a DIRECT call to a VARIADIC callee (printf/error/...) carrying
        trailing varargs, with an EXPLICIT variadic ``mcallinfo_t`` so Hex-Rays
        renders every argument (the fixed prefix AND the varargs).

        A bare ``m_call gvar`` with no callinfo leaves the vararg count to
        Hex-Rays' own analysis, which drops the trailing args (it renders only a
        mis-resolved fmt). We MIRROR Hex-Rays' own post-CALLS variadic form
        (native ``mov call $".printf"<...:"const char *format" &fmt, _QWORD v1,
        _QWORD v2> => int .4, eax.4``):

        * ``set_type`` with the callee's REAL variadic tinfo -- this KEEPS the
          ellipsis cc (``is_vararg()`` stays true) and creates the FIXED args with
          their SysV arglocs (fmt in rdi, ...). Do NOT rebuild a concrete
          fastcall prototype: that normalizes ellipsis away and the final
          decompile then drops the call (INTERR 50406/50743).
        * APPEND each surplus vararg to ``fi.args`` with the next SysV integer
          register argloc (rsi, rdx, rcx, r8, r9), exactly as the variadic tail
          appears natively. ``solid_args`` stays the fixed count; the varargs ride
          beyond it.

        The result is DISCARDED (every cp variadic call site discards it), so the
        call is emitted as a bare result-discarded ``m_call`` (``d.size`` 0, no
        retregs) -- a pre-built return register conflicts with Hex-Rays' own
        callinfo re-derivation at MMAT_CALLS (INTERR 50743/50406)."""
        # The result-discarded modeling below has no recipe for a CONSUMED vararg
        # result -- defer (native fallback) rather than silently corrupt. Checked
        # first so no partial microcode is emitted on the unsupported path.
        ops = list(ins.operands)
        call_args = ops[:-1]
        if self._value_used(ins):
            raise NotImplementedError(
                f"vararg call result consumed for @{ops[-1].name}")
        # Decline a vararg that is a non-faithful stack-array element address (see
        # the population site for the full taxonomy + the B5 proof). The EFFECTIVE
        # offset in ``_array_elt_addr`` folds in buffer cohesibility: it is 0 for a
        # zero-offset whole-buffer pointer into a buffer the ARRAY-LVAR retype can
        # cohere into one ``T[N]`` lvar (a clean ``scanf("%s", buf)`` target that
        # renders faithfully even when the buffer is also GEP'd at other offsets). A
        # nonzero offset (``&perms[1]``), a non-constant index, OR a zero offset into
        # a buffer the retype CANNOT cohere all carry a non-zero effective offset and
        # DECLINE for a clean native fallback -- never a building-but-divergent body.
        # (Scoped to the variadic path -- non-vararg array-address uses unaffected.)
        def _is_mid_array_addr(name):
            # True iff NAME is an array-element address that is NOT a faithful
            # whole-buffer pointer (effective offset != 0 -> decline).
            if name not in self._array_elt_addr:
                return False
            return self._array_elt_addr[name] != 0
        if any(_is_mid_array_addr(a.name) for a in call_args):
            raise NotImplementedError(
                f"vararg stack-array element address for @{ops[-1].name}")
        _argregs, eax, _ds = self._abi()
        arg_reg_names = self._arg_reg_names()
        nfixed = ctif.get_nargs()
        # Return width from the callee PROTOTYPE, not the LLVM call type (the lift
        # may type the result narrower than the ABI return reg); void -> 8.
        crettype = ctif.get_rettype()
        rsz = crettype.get_size()
        if rsz in (0, ida_idaapi.BADADDR) or crettype.is_void():
            rsz = 8
        # set_type with the REAL variadic prototype keeps cc=ellipsis + fixed
        # arglocs (is_vararg() stays true).
        fi = hx.mcallinfo_t(callee_ea, 0)
        if not fi.set_type(ctif):
            raise NotImplementedError(
                f"vararg call: set_type failed for @{ins.operands[-1].name}")

        def _fill_arg(arg, a):
            """Fill one mcallarg's mop VALUE from LLVM operand ``a``, widening a
            narrower value into the formal slot (verifier wants mop.size == slot)."""
            nonlocal anchor
            fsz = arg.type.get_size()
            if fsz in (0, ida_idaapi.BADADDR):
                fsz = _type_size(a.type)
            asz = _type_size(a.type)
            d = self._desc(a, vmap, asz)
            if asz < fsz and d[0] in ("reg", "num"):
                if d[0] == "num":
                    d = ("num", d[1], fsz)
                else:
                    mi = hx.minsn_t(ea)
                    mi.opcode = hx.m_xdu
                    self._fill(mi.l, (d[0], d[1], asz))
                    kreg = mba.alloc_kreg(fsz)
                    mi.d.make_reg(kreg, fsz)
                    blk.insert_into_block(mi, anchor)
                    anchor = mi
                    d = ("reg", kreg, fsz)
            tmp = hx.mop_t()
            self._fill(tmp, (d[0], d[1], fsz))
            arg.copy_mop(tmp)
            arg.size = fsz

        # Fixed args: set_type already created their slots + arglocs; fill values.
        for i in range(min(nfixed, fi.args.size())):
            _fill_arg(fi.args[i], call_args[i])
        # Variadic tail: APPEND each surplus arg with the next SysV integer
        # register argloc, mirroring the native variadic tail.
        for i in range(nfixed, len(call_args)):
            a = call_args[i]
            arg = hx.mcallarg_t()
            asz = _type_size(a.type)
            fsz = asz if asz in (1, 2, 4, 8) else 8
            # A pointer operand renders cleanly as a ``void *`` vararg (else the
            # value shows as a ``(signed __int64)`` reinterpret cast). The lifter
            # also reads a pointer-typed alloca as an i64 (and materialises an
            # address via ptrtoint) -- ``_ptr_origin_vals`` flags those so an 8-byte
            # pointer-origin value is typed as a pointer, not int (matching native).
            is_ptr = (getattr(a.type, "is_pointer", False)
                      or str(a.type) == "ptr"
                      or (a.name in self._ptr_origin_vals and asz == 8))
            if is_ptr:
                pt = ida_typeinf.tinfo_t()
                pt.create_ptr(ida_typeinf.tinfo_t(ida_typeinf.BTF_VOID))
                arg.type = pt
                fsz = 8
            else:
                arg.type = self._int_tinfo(fsz)
            arg.size = fsz
            if i < len(arg_reg_names):
                arg.argloc._set_reg1(ida_idp.str2reg(arg_reg_names[i]))
            _fill_arg(arg, a)
            fi.args.push_back(arg)
        # Result scaffolding. The variadic callees we recover (printf/error/...)
        # DISCARD their result in every cp call site; model the call as a
        # result-discarded one: d.size = 0, NO retregs/return_regs. A pre-built
        # mcallinfo that DECLARES a return register conflicts with Hex-Rays' own
        # re-derivation of the callinfo at MMAT_CALLS (used_retvals vs retval size
        # -> INTERR 50743); a result-discarded call sidesteps that and lets
        # Hex-Rays' printf-format machinery recover the varargs. spoiled keeps rax
        # clobbered (an ABI call destroys it).
        fi.spoiled.add(eax, rsz)
        fi.return_type = self._int_tinfo(rsz)
        fi.flags |= hx.FCI_HASFMT
        call_ea = (self._call_spd_ea
                   if self._call_spd_ea is not None else ea)
        fi.call_spd = ida_frame.get_spd(ida_funcs.get_func(mba.entry_ea),
                                        call_ea) if ida_funcs.get_func(
                                            mba.entry_ea) else 0
        mc = hx.minsn_t(call_ea)
        mc.opcode = hx.m_call
        mc.l.make_gvar(callee_ea)
        mc.d._make_callinfo(fi)
        mc.d.size = 0  # result discarded
        blk.insert_into_block(mc, anchor)
        return mc

    def _emit_call_vararg_fixed(self, mba, blk, anchor, ea, ins, vmap,
                                callee_ea, ctif):
        """Emit a DIRECT call to a VARIADIC callee invoked with EXACTLY its fixed
        args (NO surplus vararg), pinning the arg list so Hex-Rays does not invent
        phantom trailing varargs.

        A bare ``m_call gvar`` to a variadic prototype (e.g. ``open(const char *,
        int, ...)``) lets Hex-Rays' own vararg recovery read stale incoming-param
        registers as phantom varargs -- ``open("/dev/urandom", 0)`` mis-renders as
        ``open(a0, a1, a2)``. We build an EXPLICIT mcallinfo: ``set_type`` with the
        REAL variadic tinfo (KEEPS the ellipsis cc + the fixed-arg SysV arglocs),
        fill ONLY the fixed args, and set ``FCI_FINAL`` so the call list is taken
        as authoritative (no re-derivation).

        Unlike ``_emit_call_vararg`` (which models a result-DISCARDED variadic tail
        call), this handles both the discarded and the CONSUMED result: when the
        result is used we seed retregs/return_regs (mirroring
        ``_emit_call_stackargs``) and wrap the call in a ``mov ... => rax`` so the
        continuation captures it; when discarded we emit a bare result-0 call."""
        _argregs, eax, _ds = self._abi()
        ops = list(ins.operands)
        call_args = ops[:-1]
        crettype = ctif.get_rettype()
        rsz = crettype.get_size()
        if rsz in (0, ida_idaapi.BADADDR) or crettype.is_void():
            rsz = 8
        fi = hx.mcallinfo_t(callee_ea, 0)
        if not fi.set_type(ctif):
            raise NotImplementedError(
                f"fixed-arg vararg call: set_type failed for @{ops[-1].name}")
        # Fill each FIXED formal arg's mop VALUE (set_type fixed its argloc/type);
        # widen a narrower LLVM value into the formal slot (verifier wants
        # mop.size == formal slot size).
        for i in range(min(len(call_args), fi.args.size())):
            arg = fi.args[i]
            a = call_args[i]
            fsz = arg.type.get_size()
            asz = _type_size(a.type)
            if fsz in (0, ida_idaapi.BADADDR):
                fsz = asz
            d = self._desc(a, vmap, asz)
            if asz < fsz and d[0] in ("reg", "num"):
                if d[0] == "num":
                    d = ("num", d[1], fsz)
                else:
                    mi = hx.minsn_t(ea)
                    mi.opcode = hx.m_xdu
                    self._fill(mi.l, (d[0], d[1], asz))
                    kreg = mba.alloc_kreg(fsz)
                    mi.d.make_reg(kreg, fsz)
                    blk.insert_into_block(mi, anchor)
                    anchor = mi
                    d = ("reg", kreg, fsz)
            tmp = hx.mop_t()
            self._fill(tmp, (d[0], d[1], fsz))
            arg.copy_mop(tmp)
            arg.size = fsz
        # FCI_FINAL: take this arg list as authoritative (suppress phantom-vararg
        # re-derivation). spoiled keeps rax clobbered (an ABI call destroys it).
        fi.flags |= hx.FCI_FINAL
        fi.spoiled.add(eax, rsz)
        fi.return_type = (crettype if crettype.get_size() not in
                          (0, ida_idaapi.BADADDR) and not crettype.is_void()
                          else self._int_tinfo(rsz))
        call_ea = (self._call_spd_ea
                   if self._call_spd_ea is not None else ea)
        fi.call_spd = ida_frame.get_spd(ida_funcs.get_func(mba.entry_ea),
                                        call_ea) if ida_funcs.get_func(
                                            mba.entry_ea) else 0
        if not self._value_used(ins):
            # Result discarded -> bare result-0 call (no retregs).
            mc = hx.minsn_t(call_ea)
            mc.opcode = hx.m_call
            mc.l.make_gvar(callee_ea)
            mc.d._make_callinfo(fi)
            mc.d.size = 0
            blk.insert_into_block(mc, anchor)
            return mc
        # Result consumed -> seed retregs/return_regs and wrap in mov => rax so the
        # continuation captures the result (cf. _emit_call_stackargs).
        ret_mop = hx.mop_t()
        ret_mop.make_reg(eax, rsz)
        fi.retregs.push_back(ret_mop)
        fi.return_regs.add(eax, rsz)
        inner = hx.minsn_t(call_ea)
        inner.opcode = hx.m_call
        inner.l.make_gvar(callee_ea)
        inner.d._make_callinfo(fi)
        inner.d.size = rsz
        mov = hx.minsn_t(call_ea)
        mov.opcode = hx.m_mov
        mov.l.make_insn(inner)
        mov.l.size = rsz
        mov.d.make_reg(eax, rsz)
        blk.insert_into_block(mov, anchor)
        return mov

    def _emit_call_stackargs(self, mba, blk, anchor, ea, ins, vmap,
                             callee_name=None):
        """Emit a direct call with MORE integer args than ABI registers (the
        7th+ travel on the stack). Build an explicit ``mcallinfo_t`` from the
        callee's known prototype: ``set_type`` does the SysV reg/stack
        classification (regs for 0..5, ALOC_STACK for 6+), so each stack arg
        rides IN the call -- no SP-modeled ``push`` sequence, no WARN_BAD_CALL_SP.

        ``callee_name`` overrides the callee operand's name (a function-pointer
        bitcast resolves to the underlying @fn -- see ``_emit_call``).

        The emitted shape mirrors Hex-Rays' own post-CALLS form:
        ``mov (call gvar<...> => mop_f(callinfo)) => rax``. We carry the
        host resting-frame ea on the call (frame allocated) and seed
        retregs/return_regs/spoiled so the value-numbering of the result is
        consistent (cf. the indirect-call recipe, memory
        idavator_drop_call_construction)."""
        ops = list(ins.operands)
        callee = ops[-1]
        call_args = ops[:-1]
        _argregs, eax, _ds = self._abi()
        if callee_name is None:
            callee_name = callee.name
        callee_ea = ida_name.get_name_ea(ida_idaapi.BADADDR, callee_name)
        tif = ida_typeinf.tinfo_t()
        if not ida_nalt.get_tinfo(tif, callee_ea) or not tif.is_func():
            raise NotImplementedError(
                f"stack-passed args: no prototype for @{callee_name}")
        nargs = tif.get_nargs()
        if nargs != len(call_args):
            # The lift's arg count must match the prototype for set_type's
            # arglocs to line up; otherwise defer rather than mis-place args.
            raise NotImplementedError(
                f"stack-passed args: prototype arity {nargs} != "
                f"call arity {len(call_args)} for @{callee_name}")

        fi = hx.mcallinfo_t(callee_ea, 0)
        if not fi.set_type(tif):
            raise NotImplementedError(
                f"stack-passed args: set_type failed for @{callee_name}")
        # Return width comes from the PROTOTYPE, not the LLVM call type: the lift
        # may type the result narrower (e.g. `i1` for an `int`-returning fn), but
        # the call's retreg/return_type must match the real ABI return register
        # (else INTERR 50743 -- retreg count vs retval size). void -> 8 (rax def).
        rettype = tif.get_rettype()
        rsz = rettype.get_size()
        if rsz in (0, ida_idaapi.BADADDR) or rettype.is_void():
            rsz = 8
        # Fill each formal arg's mop VALUE (set_type already fixed its argloc +
        # type); copy_mop overwrites only the mop_t base, preserving argloc/type.
        # The verifier requires mop.size == formal type size (INTERR 50735), so
        # a narrower LLVM value (e.g. i1 into an `int` slot) is widened first --
        # exactly what Hex-Rays renders natively (xdu.4(%v.1)).
        for i, a in enumerate(call_args):
            arg = fi.args[i]
            fsz = arg.type.get_size()
            asz = _type_size(a.type)
            d = self._desc(a, vmap, asz)
            if asz < fsz and d[0] in ("reg", "num"):
                if d[0] == "num":
                    d = ("num", d[1], fsz)  # numbers just take the wider size
                else:
                    mi = hx.minsn_t(ea)
                    mi.opcode = hx.m_xdu
                    self._fill(mi.l, (d[0], d[1], asz))
                    kreg = mba.alloc_kreg(fsz)
                    mi.d.make_reg(kreg, fsz)
                    blk.insert_into_block(mi, anchor)
                    anchor = mi
                    d = ("reg", kreg, fsz)
            tmp = hx.mop_t()
            self._fill(tmp, (d[0], d[1], fsz))
            arg.copy_mop(tmp)
            arg.size = fsz
        # Result scaffolding: retregs/return_regs/spoiled keep the rax def
        # consistent under value-numbering (50743/50740 class).
        ret_mop = hx.mop_t()
        ret_mop.make_reg(eax, rsz)
        fi.retregs.push_back(ret_mop)
        fi.return_regs.add(eax, rsz)
        fi.spoiled.add(eax, rsz)
        fi.return_type = (rettype if rettype.get_size() not in
                          (0, ida_idaapi.BADADDR) and not rettype.is_void()
                          else self._int_tinfo(rsz))
        # Frame must be allocated at the call ea (else WARN_BAD_CALL_SP for the
        # stack args / any &local), and record call_spd to match.
        call_ea = (self._call_spd_ea
                   if self._call_spd_ea is not None else ea)
        fi.call_spd = ida_frame.get_spd(ida_funcs.get_func(mba.entry_ea),
                                        call_ea) if ida_funcs.get_func(
                                            mba.entry_ea) else 0
        # Inner m_call: l = callee gvar, d = mop_f(callinfo). Wrap in a mov to
        # rax so the continuation captures the result like any other call.
        inner = hx.minsn_t(call_ea)
        inner.opcode = hx.m_call
        inner.l.make_gvar(callee_ea)
        inner.d._make_callinfo(fi)
        inner.d.size = rsz
        mov = hx.minsn_t(call_ea)
        mov.opcode = hx.m_mov
        mov.l.make_insn(inner)
        mov.l.size = rsz
        mov.d.make_reg(eax, rsz)
        blk.insert_into_block(mov, anchor)
        return mov

    def _emit_call_indirect(self, mba, blk, anchor, ea, ins, vmap):
        """Emit an INDIRECT call -- the callee is an SSA function-pointer VALUE
        (e.g. an ``inttoptr`` of a loaded struct field), not a named symbol -- in
        Hex-Rays' native ``m_icall`` form::

            mov  (icall cs.2, <callee-value>.8 <mcallinfo>) => rax

        This mirrors Hex-Rays' own post-CALLS lowering. Reference (``hash_lookup``
        @ MMAT_CALLS): ``icall cs.2,[ds:(table+0x38)].8<fast:_QWORD a0,_QWORD a1>
        => __int64 .8, rax.8`` with an mcallinfo of ``cc=0x70 callee=BADADDR
        solid_args=2 return_type=__int64``, ``retregs`` EMPTY, ``return_regs=rax``.

        We synthesize a ``__fastcall`` prototype for the callee so ``set_type``
        does the SysV reg/cc classification (regs rdi/rsi/...; cc=0x70) exactly
        like a direct call. The native call carries flags
        ``FCI_PROP|FCI_DEAD|FCI_SPLOK``: ``FCI_PROP`` is REQUIRED -- it lets the
        verifier accept an empty ``retregs`` list (without it the retreg/return
        cross-check fires INTERR 50745 first)."""
        ops = list(ins.operands)
        callee = ops[-1]
        call_args = ops[:-1]
        _argregs, eax, _ds = self._abi()
        rsz = _type_size(ins.type) if str(ins.type) != "void" else 8
        # Synthesize a __fastcall fn-ptr prototype so set_type fixes the SysV
        # arglocs/cc exactly like a direct call (regs rdi/rsi/... ; cc=0x70).
        ft = ida_typeinf.func_type_data_t()
        ft.cc = ida_typeinf.CM_CC_FASTCALL
        ft.rettype = self._int_tinfo(rsz)
        for a in call_args:
            fa = ida_typeinf.funcarg_t()
            fa.type = self._int_tinfo(_type_size(a.type))
            ft.push_back(fa)
        tif = ida_typeinf.tinfo_t()
        tif.create_func(ft)
        fi = hx.mcallinfo_t(ida_idaapi.BADADDR, 0)
        if not fi.set_type(tif):
            raise NotImplementedError("indirect call: set_type failed")
        # Fill each formal arg's mop VALUE (set_type already fixed its argloc +
        # type + size); the verifier requires mop.size == formal type size.
        for i, a in enumerate(call_args):
            arg = fi.args[i]
            fsz = arg.type.get_size()
            d = self._desc(a, vmap, _type_size(a.type))
            tmp = hx.mop_t()
            self._fill(tmp, (d[0], d[1], fsz))
            arg.copy_mop(tmp)
            arg.size = fsz
        # Result scaffolding -- MIRROR NATIVE: retregs EMPTY, return_regs=rax. The
        # empty retregs list is only legal because FCI_PROP is set below.
        fi.return_regs.add(eax, rsz)
        fi.spoiled.add(eax, rsz)
        fi.return_type = self._int_tinfo(rsz)
        fi.solid_args = len(call_args)
        # NATIVE flags (hash_lookup): FCI_PROP|FCI_DEAD|FCI_SPLOK. FCI_PROP makes
        # the verifier SKIP the retreg/return-size cross-check (else 50745/50743).
        fi.flags |= hx.FCI_PROP | hx.FCI_DEAD | hx.FCI_SPLOK
        call_ea = (self._call_spd_ea
                   if self._call_spd_ea is not None else ea)
        fi.call_spd = ida_frame.get_spd(ida_funcs.get_func(mba.entry_ea),
                                        call_ea) if ida_funcs.get_func(
                                            mba.entry_ea) else 0
        # Inner m_icall: l = cs.2 (code segment), r = callee fn-ptr VALUE.8,
        # d = mop_f(callinfo). Wrap in a mov to rax so the continuation captures
        # the result like any other call.
        cs = hx.reg2mreg(ida_idp.str2reg("cs"))
        inner = hx.minsn_t(call_ea)
        inner.opcode = hx.m_icall
        inner.l.make_reg(cs, 2)
        cdesc = self._desc(callee, vmap, 8)
        self._fill(inner.r, (cdesc[0], cdesc[1], 8))
        inner.d._make_callinfo(fi)
        inner.d.size = rsz
        mov = hx.minsn_t(call_ea)
        mov.opcode = hx.m_mov
        mov.l.make_insn(inner)
        mov.l.size = rsz
        mov.d.make_reg(eax, rsz)
        blk.insert_into_block(mov, anchor)
        return mov

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
        the kreg in vmap as the call's SSA value.

        When ``self._kreg_call_results`` is set (the scoped 50342 retry in
        ``drop``), ALSO kill the ABI return register (``mov #0, rax``) right after
        the kreg copy. The plain copy alone is INERT when nothing clobbers rax
        between here and a later rax-redefinition: Hex-Rays proves ``kreg == rax``
        and copy-propagates the kreg back into rax, so the call result (used as a
        loop/body INTERMEDIATE) ends up sharing the physical return register with
        the function's RETURN value. Both versions then converge at the body-less
        ``BLT_STOP`` as one register holding two distinct SSA values -> an
        un-numberable phi (INTERR 50342 at MMAT_GLBOPT2). Killing rax severs the
        copy chain so the intermediate must live in the kreg (Hex-Rays parks it in
        a stack slot, exactly as native carries it in ``%bucket``), leaving rax
        solely for the return phi-sink. Mirrors the ``_arg_spill_kill`` discipline
        for clobbered arg registers."""
        rsz = _type_size(call_ins.type)
        kreg = mba.alloc_kreg(rsz)
        mv = hx.minsn_t(ea)
        mv.opcode = hx.m_mov
        mv.l.make_reg(eax, rsz)
        mv.d.make_reg(kreg, rsz)
        blk.insert_into_block(mv, anchor)
        vmap[call_ins.name] = ("reg", kreg, rsz)
        anchor = mv
        if self._kreg_call_results:
            kill = hx.minsn_t(ea)
            kill.opcode = hx.m_mov
            kill.l.make_number(0, rsz)
            kill.d.make_reg(eax, rsz)
            blk.insert_into_block(kill, anchor)
            anchor = kill
        return anchor

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

    def _alloca_struct_key(self, ins):
        """The ``self._struct_size`` key for a ``%struct`` alloca (quotes stripped,
        llvmlite ``.NN`` disambiguation suffix tolerated), or None for a scalar/ptr
        alloca. Mirrors _array_dims's element-key handling."""
        m = re.search(r"alloca\s+(?:inalloca\s+)?([^,\n]+)", str(ins).strip())
        ty = m.group(1).strip() if m else "i64"
        if "*" in ty or ty == "ptr":
            return None
        key = ty.replace('"', "")
        if key not in self._struct_size:
            key = re.sub(r"\.\d+$", "", key)
        return key if key in self._struct_size else None

    @staticmethod
    def _alloca_is_ptr(ins) -> bool:
        """True if ``ins`` is ``alloca ptr`` (or ``alloca T*``) -- a slot that
        holds a single pointer value. Such a slot is type-punned by the lifter:
        ``bitcast %slot`` reaches both the pointer itself and ``*slot``."""
        m = re.search(r"alloca\s+(?:inalloca\s+)?([^,\n]+)", str(ins).strip())
        ty = m.group(1).strip() if m else ""
        return ty == "ptr" or ty.endswith("*")

    @staticmethod
    def _pointee_size_of_decl(decl_ty) -> int:
        """Byte width of the POINTEE of a pointer-typed alloca's DECLARED element
        type string (from the original IR text, ``_alloca_decl_types``).
        ``i8*`` -> 1 (pointee ``i8``); ``i64*`` -> 8; a pointer-to-pointer
        (``i8**``) or an opaque / missing ``ptr`` -> 8 (the pointer width, the
        conservative DEFAULT that keeps the legacy 8-byte deref behaviour). Used
        to tell a pointer-VALUE read punned to a non-pointer integer (load width
        == pointer width but pointee sub-pointer-width) from a real ``*slot``
        deref (load width == pointee width)."""
        ty = (decl_ty or "").strip()
        if ty.endswith("*") and not ty.endswith("**"):
            inner = ty[:-1].strip()
            if inner == "ptr":
                return 8
            return _type_size(inner)
        # pointer-to-pointer / opaque ``ptr`` / unknown -- pointee is itself
        # pointer-width (or erased); default to 8 so the existing deref
        # classification is unchanged.
        return 8

    def _alloca_decl_types(self, fn) -> dict:
        """``alloca-name -> declared slot type string`` parsed from the ORIGINAL IR
        text for ``fn``.

        ``llvm.parse_assembly`` normalizes to OPAQUE pointers, so ``str(ins)`` for
        a pointer alloca is the type-erased ``alloca ptr`` -- the pointee struct
        (``%"hash_entry"*``) survives only in the source text. The cursor-typing
        pass needs the pointee, so it reads the declared types straight from
        ``self._ir_text`` (``%"bucket" = alloca %"hash_entry"*``)."""
        body = re.search(
            r'(?ms)^define[^\n]*@"?' + re.escape(fn.name) + r'"?\(.*?\n\}',
            self._ir_text)
        if body is None:
            return {}
        out: dict = {}
        for m in re.finditer(
                r'%"?([\w.$]+)"?\s*=\s*alloca\s+(?:inalloca\s+)?([^,\n]+)',
                body.group(0)):
            out[m.group(1)] = m.group(2).strip()
        return out

    def _pointee_struct_of_type(self, ty: str):
        """The known-struct name of a single-level pointer type string ``T*`` whose
        pointee ``T`` has a computed layout (present in ``self._struct_size``), else
        None. Only ``T*`` qualifies -- ``T**`` is a pointer-to-pointer, not a
        struct cursor; a scalar/opaque pointee (``i8*``, ``ptr``) has no struct."""
        ty = ty.strip()
        if not ty.endswith("*") or ty.endswith("**"):
            return None
        key = ty[:-1].strip()
        if not key.startswith("%"):
            return None
        key = key.replace('"', "")
        if key not in self._struct_size:
            key = re.sub(r"\.\d+$", "", key)
        return key if key in self._struct_size else None

    def _cursor_struct_ptrs(self, fn, names, gepd, escaped, decl_types) -> set:
        """Names of CLEAN (load/store-only -> would otherwise kreg) single-level
        struct-pointer allocas that are genuine CURSORS deserving a typed frame
        slot, not a propagated temporary.

        After the m_ldx pointer-slot-define lift fix, a through-pointer pointer
        member load (``bucket = table->bucket`` at off 0, ``cursor = cursor->next``
        at +8) lowers to a pointer-typed ``store T* v, T** %slot`` -- correct
        SEMANTICS (no spurious deref-write), but the slot is now accessed cleanly
        so ``_scan_allocas`` kregs it and the decompiler renders an UNTYPED
        ``void**`` walk (``i = *(void***)a0; *i; i += 2``) instead of native's typed
        ``for(bucket=table->bucket;;++bucket){if(bucket->data)..}``. Re-anchoring
        such a slot to a real frame slot (like native) lets ``_save_struct_ptr_lvar_
        types`` type it ``T*`` and recover the field-access render.

        A CURSOR must satisfy ALL of:
          * a resolvable single-level struct-pointer pointee (``%hash_entry`` etc.);
          * written at least once from a DERIVED pointer -- a ``load`` from memory
            or a ``getelementptr`` result (the cursor advance / member read). This is
            the pointer-slot-DEFINE the lift fix now emits;
          * its LOADED value is USED beyond a bare ``ret`` -- it is dereferenced,
            advanced, compared, or passed. A walked cursor reads back; a value that
            is only defined-then-returned does not.

        EXCLUSIONS (would otherwise regress self-consistency):
          * the return slot ``funcresult`` -- stored from a ``load`` (``return *p``)
            and loaded only to feed ``ret``; typing it as a frame cursor breaks the
            return-slot promotion (clone_quoting_options, quoting_options_from_style).
            The loaded-beyond-ret test already rejects it; the name guard is cheap
            belt-and-braces.
          * a pure PARAM-COPY slot (its only store value is an incoming argument,
            ``store %".1", %table``): typing it pins a propagatable parameter into a
            stack local and adds ``table = a0`` noise. Its store value is an argument
            (not a load/gep), so it is not in ``derived_into``."""
        clean = (names - gepd - escaped) - {"funcresult"}
        if not clean:
            return set()
        # An operand ValueRef does NOT expose its defining instruction's opcode
        # (llvmlite reports is_instruction=False for an operand), so map each
        # named instruction's result -> opcode and look operands up there.
        defop: dict = {}
        for bb in fn.blocks:
            for ins in bb.instructions:
                if ins.name:
                    defop[ins.name] = ins.opcode
        # name of each load's pointer-source alloca: ``%v = load <ty>, ptr %slot``.
        load_src: dict = {}
        for bb in fn.blocks:
            for ins in bb.instructions:
                if ins.opcode == "load" and ins.name:
                    sops = list(ins.operands)
                    if sops:
                        load_src[ins.name] = sops[0].name
        derived_into: set = set()
        used_loaded: set = set()  # cursor whose loaded value is used beyond `ret`
        for bb in fn.blocks:
            for ins in bb.instructions:
                if ins.opcode == "store":
                    sops = list(ins.operands)
                    val, dst = sops[0], sops[1]
                    if (dst.name in clean
                            and defop.get(val.name) in ("load", "getelementptr")):
                        derived_into.add(dst.name)
                if ins.opcode == "ret":
                    continue
                # any non-ret instruction consuming a load-of-cursor result marks
                # that cursor as genuinely walked (deref / advance / compare / arg).
                for o in ins.operands:
                    base = load_src.get(o.name)
                    if base in clean:
                        used_loaded.add(base)
        out = set()
        for nm in clean:
            if nm not in derived_into or nm not in used_loaded:
                continue
            if self._pointee_struct_of_type(decl_types.get(nm, "")) is not None:
                out.add(nm)
        return out

    def _struct_ptr_alloca_slots(self, fn) -> dict:
        """``name -> (stkoff, struct_name)`` for every ESCAPING single-level
        struct-pointer alloca (and every clean struct-pointer CURSOR -- see
        ``_cursor_struct_ptrs``), using the SAME synthetic frame-offset packing as
        ``_scan_allocas`` so the offsets match the ``make_stkvar(mba, off)`` the
        drop emits -- i.e. the offset by which ``_save_struct_ptr_lvar_types``
        locates the decompiled cursor stkvar.

        The offset bookkeeping here is a faithful replica of ``_scan_allocas``'s
        ``off`` accounting for gepd/escaped/cursor allocas; it needs no ``mba``
        because struct-pointer slots are never re-anchored to a host offset (only
        escaping aggregate structs are) and scalar kreg allocas do not consume
        ``off``. Mirrors the classification exactly to avoid drift."""
        names = {ins.name for bb in fn.blocks for ins in bb.instructions
                 if ins.opcode == "alloca"}
        if not names:
            return {}
        decl_types = self._alloca_decl_types(fn)
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
        cursors = self._cursor_struct_ptrs(fn, names, gepd, escaped, decl_types)
        out: dict = {}
        off = 0
        for bb in fn.blocks:
            for ins in bb.instructions:
                if ins.opcode != "alloca":
                    continue
                nm = ins.name
                sz = self._alloca_elem_size(ins)
                if nm in gepd:
                    # Mirror _scan_allocas' gepd sizing EXACTLY so the synthetic
                    # ``off`` accounting (used to locate cursor stkvars) stays in
                    # lockstep: an ``[N x T]`` array uses _array_dims; a bare laid-out
                    # ``%struct`` uses the whole-struct size; a scalar byte-pun /
                    # va_list / anonymous gepd alloca is where _scan_allocas raises ->
                    # nothing to type, bail.
                    dims = self._array_dims(str(ins))
                    if dims is not None:
                        off += max(dims[0] * dims[2], 8)
                    elif nm and self._alloca_struct_key(ins) is not None:
                        off += max(sz, 8)
                    else:
                        return out
                elif nm in escaped or nm in cursors:
                    # Opaque-pointer normalization erases the pointee from
                    # ``str(ins)``; resolve the struct from the source-text decl.
                    struct = self._pointee_struct_of_type(decl_types.get(nm, ""))
                    if struct is not None:
                        out[nm] = (off, struct)
                    off += max(sz, 8)
                # scalar kreg allocas do not consume a frame offset
        return out

    @staticmethod
    def _struct_ptr_tif(struct: str):
        """A ``T*`` ``tinfo_t`` for the IR struct key ``struct`` (e.g.
        ``%hash_entry``), resolved as the named IDB type ``hash_entry`` (present
        from DWARF on a non-stripped target), or None if it cannot be resolved."""
        type_name = struct.lstrip("%")
        base = ida_typeinf.tinfo_t()
        if not base.get_named_type(None, type_name):
            return None
        ptr = ida_typeinf.tinfo_t()
        ptr.create_ptr(base)
        return ptr if ptr.is_ptr() else None

    def _save_struct_ptr_lvar_types(self, host_ea: int, fn, cf) -> bool:
        """Type each escaping struct-pointer cursor slot as ``T*`` so the decompiler
        renders the faithful cursor walk (``for(bucket=table->bucket;;++bucket){...
        if (bucket->data) break;}``) instead of an untyped ``*(_QWORD*)`` blob walk.

        A pre-decompile ``save_user_lvar_settings`` keyed by a synthetic
        ``vdloc_t().set_stkoff(N)`` does NOT match the real lvar's richer location
        encoding, so it is silently ignored. The mechanism that sticks (and beats
        the decompiler's own ``Hash_table*``/param-propagated inference) is, GIVEN a
        first ``cf``: locate the cursor stkvar by the frame offset the drop assigned
        it, then persist a user type at that lvar's ACTUAL location via
        ``modify_user_lvar_info(MLI_TYPE)``. The caller re-decompiles so the
        structural/type analysis re-runs (a mere ``refresh_func_ctext`` keeps the
        old shape).

        Returns True if any type was applied (caller must re-decompile). The
        pointee struct must resolve as a named IDB type; otherwise the slot is left
        as-is -- identical to prior behaviour, a strict zero-regression
        enhancement."""
        if cf is None:
            return False
        slots = self._struct_ptr_alloca_slots(fn)
        if not slots:
            return False
        # offset -> struct-tif (only resolvable structs)
        want: dict = {}
        for _nm, (stkoff, struct) in slots.items():
            tif = self._struct_ptr_tif(struct)
            if tif is not None:
                want[stkoff] = tif
        if not want:
            return False
        applied = False
        for v in cf.get_lvars():
            if not v.is_stk_var():
                continue
            tif = want.get(v.get_stkoff())
            if tif is None:
                continue
            if v.type() is not None and v.type().dstr() == tif.dstr():
                continue  # already this type (idempotent re-drop)
            info = hx.lvar_saved_info_t()
            info.ll.location = v.location
            info.ll.defea = v.defea
            info.type = tif
            info.size = 8
            if hx.modify_user_lvar_info(host_ea, hx.MLI_TYPE, info):
                applied = True
        return applied

    @staticmethod
    def _array_elt_tinfo(elem_str: str, elem_size: int):
        """A ``tinfo_t`` for one cohesion-eligible GEP'd-array element, else None.

        SCOPED TO BYTE BUFFERS (``i8`` element -> ``char``): a ``[N x i8]`` stack
        buffer is the case Hex-Rays shatters into ~N disjoint scalar stkvars when the
        drop resolves each GEP to a bare ``sp+const`` ref (native's ``char buf[104]``
        fed to ``scanf("%s", buf)`` becomes ``scanf("%s", &fragment)``). Typing the
        slot ``char[N]`` is what cohers it.

        WIDER integer / ``ptr`` element arrays are deliberately EXCLUDED: a small
        ``[2 x ptr]`` / ``[10 x ptr]`` aggregate local already drops faithfully, and
        forcing a ``void *[N]`` lvar over a slot the body ALSO uses as a single
        returned pointer (``hash_insert``'s ``&matched_ent`` / ``return
        matched_ent[0]``) trips Hex-Rays' "local variable allocation has failed"
        banner -- a regression. Returning None there leaves the slot exactly as the
        drop emitted it (strict zero-regression)."""
        if elem_str == "i8":
            return ida_typeinf.tinfo_t(ida_typeinf.BTF_CHAR)
        return None

    def _save_array_lvar_types(self, host_ea: int, fn, cf) -> bool:
        """Type each GEP'd ``[N x T]`` frame slot as a single ``T[N]`` array lvar so
        the decompiler renders the buffer WHOLE (native's ``char buf[104]`` fed to
        ``scanf("%s", buf)``) instead of shattering the GEP-by-GEP ``sp+const``
        scalar refs into ~N disjoint scalar stkvars.

        Exactly mirrors ``_save_struct_ptr_lvar_types``: GIVEN a first ``cf``, locate
        each array slot's stkvar by the frame offset the drop assigned it
        (``self._addr_taken[name][0]``), then persist a user type at that lvar's
        ACTUAL (post-decompile) location via ``modify_user_lvar_info(MLI_TYPE)``. The
        caller re-decompiles so type/structural analysis re-runs over the now-cohesive
        slot. Returns True iff any type was applied (caller must re-decompile)."""
        if cf is None or not self._array_alloca_elt:
            return False
        # frame offset -> array tinfo (only faithfully-typeable element kinds).
        want: dict = {}
        for nm, (elem_str, elem_size, count) in self._array_alloca_elt.items():
            slot = self._addr_taken.get(nm)
            if slot is None or count <= 0:
                continue
            elt = self._array_elt_tinfo(elem_str, elem_size)
            if elt is None:
                continue
            arr = ida_typeinf.tinfo_t()
            if not arr.create_array(elt, count) or not arr.is_array():
                continue
            want[slot[0]] = arr
        if not want:
            return False
        applied = False
        for v in cf.get_lvars():
            if not v.is_stk_var():
                continue
            tif = want.get(v.get_stkoff())
            if tif is None:
                continue
            if v.type() is not None and v.type().dstr() == tif.dstr():
                continue  # already this array type (idempotent re-drop)
            info = hx.lvar_saved_info_t()
            info.ll.location = v.location
            info.ll.defea = v.defea
            info.type = tif
            info.size = tif.get_size()
            if hx.modify_user_lvar_info(host_ea, hx.MLI_TYPE, info):
                applied = True
        return applied

    def _scan_ptr_deref_aliases(self, fn) -> None:
        """SSA names that are a no-op ``bitcast`` chain rooted DIRECTLY at a
        pointer-typed addr-taken alloca (``self._ptr_allocas``), with no ``load``
        or ``getelementptr`` in between.

        The lifter emits ``*X`` (deref) as ``bitcast %X to ptr; load/store <ty>``
        -- the bitcast reinterprets the SLOT but the intent is the pointee. A
        DIRECT slot access uses ``load/store ... ptr %X`` (no bitcast) or
        ``load ptr, bitcast %X`` (the full pointer value). Recording the
        direct-bitcast aliases lets the load/store emit apply the deref-vs-slot
        rule by valtype (see ``_emit_value``); a deref of a pointer LOADED from
        the slot (``load ptr, %X; bitcast; gep``) already works via the generic
        ldx/stx path and is intentionally NOT in this set.

        Each alias inherits its ROOT alloca's pointee byte width (recorded in
        ``self._ptr_alloca_pointee``) so the load/store emit can tell a punned
        pointer-VALUE read (load width == pointer width on a sub-pointer pointee)
        from a real ``*slot`` deref (load width == pointee width)."""
        self._ptr_deref_alias = set()
        if not self._ptr_allocas:
            return
        changed = True
        while changed:
            changed = False
            for bb in fn.blocks:
                for ins in bb.instructions:
                    if ins.opcode != "bitcast" or ins.name in self._ptr_deref_alias:
                        continue
                    src = list(ins.operands)[0].name
                    if src in self._ptr_allocas or src in self._ptr_deref_alias:
                        self._ptr_deref_alias.add(ins.name)
                        # Inherit the root alloca's pointee width (default 8 = the
                        # legacy behaviour for an opaque / pointer-pointee slot).
                        self._ptr_alloca_pointee[ins.name] = \
                            self._ptr_alloca_pointee.get(src, 8)
                        # Inherit the struct-pointee flag so a direct-bitcast
                        # cursor DEFINE through the alias is recognised.
                        if src in self._ptr_alloca_pointee_struct:
                            self._ptr_alloca_pointee_struct.add(ins.name)
                        changed = True

    def _alloca_elem_size(self, ins) -> int:
        m = re.search(r"alloca\s+(?:inalloca\s+)?([^,\n]+)", str(ins).strip())
        ty = m.group(1).strip() if m else "i64"
        if "*" in ty or ty == "ptr":
            return 8
        # A ``%struct`` alloca: prefer the already-parsed layout size; _type_size
        # has no struct case and returns 4 (the opaque-name fallback), which would
        # under-size an escaping struct slot and collide with the next alloca.
        key = self._alloca_struct_key(ins)
        if key is not None:
            return self._struct_size[key][0]
        return _type_size(ty)

    def _is_punned_ptr_value(self, operand, access_sz: int) -> bool:
        """True if a ``_ptr_deref_alias`` access of ``access_sz`` bytes is a
        POINTER-VALUE read punned to a non-pointer integer rather than a real
        ``*slot`` deref.

        The lifter type-puns a pointer-alloca slot two ways: a deref ``*slot``
        loads exactly ``sizeof(pointee)`` bytes, while passing the pointer BY
        VALUE through a non-pointer integer (``load i64, bitcast i8** %slot to
        i64*`` -- ``safe_hasher``'s ``table->hasher(key, ...)``) loads the POINTER
        width (8). When the access is pointer-width but the slot's pointee is
        SUB-pointer-width, only the value read fits -- a real deref would have
        loaded the (narrower) pointee. An 8-byte pointee is genuinely ambiguous
        (both a value read and ``*p`` move 8 bytes) and stays a deref (the prior
        behaviour, correct for ``*total_n_read`` etc.)."""
        if access_sz != 8:
            return False
        pointee = self._ptr_alloca_pointee.get(operand.name, 8)
        return pointee < 8

    def _ptr_deref_off(self, operand, vmap) -> int:
        """Frame offset of the pointer-alloca slot a ``_ptr_deref_alias`` operand
        is rooted at. The alias is a no-op ``bitcast`` chain over the alloca, so
        its resolved stkaddr offset IS the slot offset."""
        d = vmap.get(operand.name)
        if d is not None and d[0] == "stkaddr":
            return d[1]
        s = self._addr_taken.get(operand.name)
        if s is not None:
            return s[0]
        raise ValueError(f"ptr-deref alias {operand.name!r} did not resolve")

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

    def _struct_key(self, tok: str):
        """Canonical ``self._struct_size``/``self._struct_field_off`` key for a
        ``%name`` type token (quotes stripped, llvmlite ``.NN`` suffix tolerated),
        or None if it is not a laid-out struct."""
        tok = tok.strip()
        if not tok.startswith("%"):
            return None
        key = tok.replace('"', "")
        if key not in self._struct_size:
            key = re.sub(r"\.\d+$", "", key)
        return key if key in self._struct_size else None

    def _type_byte_size(self, tok: str) -> int:
        """Byte size of an LLVM type token honouring struct layout (``_type_size``
        has no struct case and returns 4 for an opaque ``%name``)."""
        tok = tok.strip()
        m = re.match(r"\[\s*(\d+)\s+x\s+(.+)\]$", tok)
        if m:
            return int(m.group(1)) * self._type_byte_size(m.group(2))
        key = self._struct_key(tok)
        if key is not None:
            return self._struct_size[key][0]
        return _type_size(tok)

    def _gep_walk_offset(self, lead_ty: str, ops, vmap) -> int:
        """Constant byte offset of a ``getelementptr LEAD_TY, ptr %base, I0, I1..``
        walking the index list through the leading aggregate type. Supports a
        SCALAR lead (single pointer-arithmetic index ``i32 0`` on ``i32 %mode``),
        an ``[N x T]`` array (first index strides whole-elements, the rest descend
        into T), a ``%struct`` (an index SELECTS field F -> its real byte offset),
        and nested array-of-struct / struct-in-struct. Raises NotImplementedError
        on a non-constant index or an un-laid-out element (clean native fallback).

        The FIRST index is C pointer arithmetic over ``LEAD_TY`` (``ptr[i]``); each
        subsequent index descends one aggregate level. This mirrors LLVM GEP
        semantics and de-generalises to the scalar identity GEP the lifter emits
        for a reload (``getelementptr i32, ptr %mode, i32 0`` -> 0)."""
        cur = lead_ty.strip()
        total = 0
        for k, o in enumerate(ops[1:]):
            d = self._desc(o, vmap, 8)
            if d[0] != "num":
                raise NotImplementedError("GEP-on-stack: non-constant index")
            idx = d[1]
            if k == 0:
                # pointer arithmetic over the whole leading type, then descend it.
                total += idx * self._type_byte_size(cur)
                continue
            m = re.match(r"\[\s*(\d+)\s+x\s+(.+)\]$", cur)
            if m:
                cur = m.group(2).strip()
                total += idx * self._type_byte_size(cur)
                continue
            key = self._struct_key(cur)
            if key is not None and key in self._struct_field_off:
                offsets = self._struct_field_off[key]
                if not 0 <= idx < len(offsets):
                    raise NotImplementedError("GEP-on-stack: field index OOB")
                total += offsets[idx]
                # descend into the selected field's type for any deeper index.
                raw = re.search(
                    r'%"?' + re.escape(key) + r'"?\s*=\s*type\s*<?\s*\{(.*)\}',
                    self._ir_text)
                if raw is not None:
                    fields = _split_fields(raw.group(1))
                    if idx < len(fields):
                        cur = fields[idx].strip()
                continue
            raise NotImplementedError(
                f"GEP-on-stack: un-laid-out element {cur!r}")
        return total

    def _gep_lead_type(self, ins) -> str:
        """The leading source element type of a ``getelementptr`` from its IR text
        (``getelementptr inbounds [2 x %timeval], ptr %a, ..`` -> ``[2 x %timeval]``;
        ``getelementptr i32, ptr %mode, ..`` -> ``i32``)."""
        s = str(ins).strip()
        m = re.search(
            r"getelementptr\s+(?:inbounds\s+)?(?:nuw\s+)?(.+?)\s*,\s*ptr\b", s)
        if m:
            return m.group(1).strip()
        # legacy typed-pointer form: ``getelementptr i32, i32* %p`` -- take the
        # token before the first comma.
        m = re.search(
            r"getelementptr\s+(?:inbounds\s+)?(?:nuw\s+)?([^,]+),", s)
        return m.group(1).strip() if m else "i8"

    def _gep_field_offset(self, ins, ops, vmap) -> int:
        """Constant byte offset of ``getelementptr [N x T], ptr %alloca, I0, I1``
        into a frame-slot alloca: ``I0*sizeof([N x T]) + I1*sizeof(T)``.

        The scalar-array fast path stays for the (overwhelmingly common) ``[N x T]``
        scalar-element form; a struct / array-of-struct / scalar-identity GEP
        (``_array_dims`` None) routes through the general type-walking resolver."""
        dims = self._array_dims(str(ins))
        if dims is None:
            return self._gep_walk_offset(self._gep_lead_type(ins), ops, vmap)
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
        self._array_alloca_elt = {}
        self._arg_spill_kill = set()
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
        # A clean (load/store-only) single-level struct-pointer CURSOR (e.g.
        # ``bucket``/``cursor`` after the m_ldx pointer-slot-define lift fix) would
        # otherwise be a kreg and render as an untyped ``void**`` walk. Re-anchor it
        # to a real frame slot (the escaped path below) so ``_save_struct_ptr_lvar_
        # types`` can type it ``T*`` -- recovering native's typed ``bucket->data`` /
        # ``++bucket``. _cursor_struct_ptrs excludes pure param-copy slots. The set
        # MUST match _struct_ptr_alloca_slots' so the typed stkoff lines up.
        decl_types = self._alloca_decl_types(fn)
        cursors = self._cursor_struct_ptrs(fn, names, gepd, escaped, decl_types)
        escaped |= cursors
        # Distinct CONSTANT byte offsets each GEP'd alloca is reached at. The
        # ARRAY-LVAR retype fires ONLY for a buffer touched at >= 2 distinct offsets
        # -- the genuinely FRAGMENTING case (OLLVM's ``[104 x i8] v56`` GEP'd at 21
        # offsets that Hex-Rays shatters into scalar stkvars). A buffer reached only
        # at offset 0 (every cp.ll byte buffer: ``samedir_template(.., buf)``) is
        # already coherent on its own; forcing a ``char[N]`` lvar over it gains
        # nothing and trips Hex-Rays' "local variable allocation has failed" banner.
        gep_offsets: dict[str, set] = {n: set() for n in gepd}
        for bb in fn.blocks:
            for ins in bb.instructions:
                if ins.opcode != "getelementptr":
                    continue
                sops = list(ins.operands)
                if not sops or sops[0].name not in gepd:
                    continue
                try:
                    gep_offsets[sops[0].name].add(
                        self._gep_field_offset(ins, sops, {}))
                except Exception:  # noqa: BLE001 - unknown offset counts as distinct
                    gep_offsets[sops[0].name].add(None)
        # An incoming reg arg spilled to a slot that is then read ACROSS a call
        # (and consumed by that call) must rest in a real frame slot, NOT a scalar
        # kreg: Hex-Rays forwards a kreg copy of the raw incoming register into the
        # clobbering call site, losing the post-clobber re-load (create_hole's
        # ``size`` -> uninit ``v5``). Native keeps it on the stack. Force the slot
        # ``escaped`` (the real-frame-slot path) and record its source register so
        # _build_multiblock can KILL the register after the entry spill (else
        # Hex-Rays back-substitutes the raw register past the spill). See
        # _arg_spill_slots_across_call.
        argregs, _eax, _ds = self._abi()
        reg_arg_names = {a.name for i, a in enumerate(fn.arguments)
                         if i < len(argregs)}
        spill_slots = self._arg_spill_slots_across_call(fn, reg_arg_names)
        if spill_slots:
            arg_index = {a.name: i for i, a in enumerate(fn.arguments)}
            for slot_name, arg_name in spill_slots.items():
                # Only a pure scalar (load/store-only) spill slot -- a GEP'd or
                # already-escaped slot has its own handling.
                if slot_name in gepd or slot_name in escaped:
                    continue
                escaped.add(slot_name)
                i = arg_index.get(arg_name)
                if i is not None and i < len(argregs):
                    self._arg_spill_kill.add(argregs[i])
        # Host-frame member offsets, keyed by the source name the lifter preserved.
        # An escaping STRUCT alloca must rest at its REAL host offset: the synthetic
        # sequential packing below can land a struct's base on top of a DIFFERENT,
        # independently-materialised host scalar (e.g. ``%storage`` -> synthetic +16
        # == host ``new_size`` in hash_rehash). Hex-Rays then reads/writes the wrong
        # slot -> garbage decompile + the post-noreturn-merge INTERR 50342. Native
        # uses the true frame offsets; matching them by name de-collides the slot.
        host_off = self._host_frame_offsets(mba)
        self._ptr_allocas = {}
        self._ptr_alloca_pointee = {}
        self._ptr_alloca_pointee_struct = set()
        allocas = {}
        # A frame-slot alloca that the HOST FRAME ALSO NAMES must rest at its TRUE
        # host offset, not at a synthetic sequential offset. The old packing (0, 8,
        # 16, ...) ignored the host layout, so a host-named escaped slot routinely
        # got an offset belonging to a DIFFERENT host variable and Hex-Rays then
        # ALIASED the two distinct values onto one stkvar:
        #   create_hole   -- ``punch_holes`` (escaped via ``bitcast i1*``) packed at
        #                    synthetic +8 == host ``size`` (+8): the body read
        #                    ``size = a2`` and gated on ``if (!size)`` (native gates
        #                    on ``punch_holes``);
        #   quotearg_buffer-- ``p`` packed at +8 == host ``o`` (+8);
        #   sparse_copy   -- ``total_n_read`` (+8) / ``last_write_made_hole`` (+16)
        #                    SWAPPED, so ``*total_n_read += n_read`` corrupted
        #                    ``last_write_made_hole``.
        # ANONYMOUS allocas (no anchorable host member -- a deref temp, an SROA
        # leftover, or a host bool the frame byte-packs sub-qword) keep the original
        # LOW synthetic packing from 0. Overlaying a host LOCAL slot is BENIGN
        # (Hex-Rays treats an unnamed scratch slot sharing bytes with a host local as
        # scratch, exactly as the old ``off=0`` packing did) and crucially stays in
        # the LOCAL frame -- it must NOT be pushed UP into the ``__saved_registers`` /
        # ``__return_address`` / outgoing-args region (that corrupts the frame:
        # create_hard_link -> "local variable allocation has failed"; a synthetic slot
        # landing on ``__saved_registers`` renders as ``savedregs``). The ONE thing
        # the low packing must avoid is sharing a slot with a host-NAMED alloca
        # RE-ANCHORED below, so it skips those placed ranges.
        #
        # Re-anchor ONLY to an 8-ALIGNED host offset. The destructive collisions are
        # 8-byte slots at 8-aligned offsets (``punch_holes``@24 vs ``size``@8,
        # ``p``@8 vs ``o``@8, ``total_n_read``@8 / ``last_write_made_hole``@16). A
        # host frame's SUB-8 byte-packed bools (create_hard_link ``dereference``@4,
        # last_component ``saw_slash``@23) are NOT a destructive alias -- the original
        # synthetic packing handled them fine -- and re-anchoring an escaped slot to a
        # mid-qword offset (then sizing it to 8 bytes) OVERLAPS the neighbouring packed
        # member and itself trips the allocation failure. (The earlier struct-only
        # re-anchor under-covered the named 8-aligned collision; scalar/ptr escaped
        # slots need the SAME host-offset de-collision.)
        def _anchorable(name: str) -> bool:
            return name in host_off and host_off[name][0] % 8 == 0
        placed = [
            (host_off[ins.name][0],
             host_off[ins.name][0] + max(self._alloca_elem_size(ins), 8))
            for bb in fn.blocks for ins in bb.instructions
            if ins.opcode == "alloca" and _anchorable(ins.name)
            and (ins.name in gepd or ins.name in escaped)
        ]

        def _synthetic(start: int, size: int) -> int:
            want = max(size, 8)
            o, moved = start, True
            while moved:
                moved = False
                for lo, hi in placed:
                    if o < hi and lo < o + want:
                        o, moved = (hi + 7) & ~7, True
            return o
        off = 0
        for bb in fn.blocks:
            for ins in bb.instructions:
                if ins.opcode != "alloca":
                    continue
                nm = ins.name
                sz = self._alloca_elem_size(ins)
                named = _anchorable(nm)
                if nm in gepd:
                    # GEP'd alloca -> a frame slot sized to the WHOLE aggregate; each
                    # GEP resolves to &stkvar(slot + field). A host-named slot rests
                    # at its real offset; an anonymous one keeps a low synthetic slot.
                    #   * ``[N x T]`` array (scalar OR struct element) -> ``_array_dims``
                    #     gives the whole-array size and ``_gep_field_offset`` strides
                    #     it (the long-standing scalar-array path; AoS already worked);
                    #   * a bare ``%struct`` alloca GEP'd at its real FIELD offsets
                    #     (``getelementptr %struct, ptr %a, 0, i32 F``) -> ``_array_dims``
                    #     is None but ``_alloca_struct_key`` resolves a laid-out struct;
                    #     size the slot to the whole struct and let
                    #     ``_gep_field_offset``'s field-offset walker resolve each GEP.
                    #     This is the ADDITIVE struct-layout case (previously RAISED).
                    #
                    # DECLINE (keep raising -> clean native fallback, EXACTLY as
                    # ebe211f) for:
                    #   * a SCALAR / pointer / va_list alloca GEP'd at a constant
                    #     offset -- this is the lifter's byte-pun reload
                    #     (``getelementptr i16, ptr %v7, i32 0`` + ``bitcast .. to
                    #     i8*`` + masked sub-byte store). Routing the scalar through a
                    #     frame slot drops the sub-register store, AND the relifted
                    #     byte offsets diverge from native's ``BYTE1``/``HIBYTE`` form
                    #     (rpl_mknod/fdutimens/set_owner): not faithfully droppable, so
                    #     it MUST fall back rather than ship a divergent body;
                    #   * an ANONYMOUS (numeric-IR) gepd alloca -- llvmlite reports
                    #     ``name == ''`` for ALL numeric SSA values, collapsing the
                    #     whole-function name map onto one '' bucket (the OLLVM
                    #     obfuscated IR); not faithfully droppable through the
                    #     name-keyed path.
                    dims = self._array_dims(str(ins))
                    if dims is not None:
                        arr_sz = dims[0] * dims[2]
                        # Record (elem_str, elem_byte_size, count) so the
                        # post-decompile retype pass can give this slot a single
                        # ``T[N]`` lvar instead of letting Hex-Rays shatter the
                        # GEP-by-GEP scalar refs into N disjoint stkvars. Scoped to
                        # cohesion-eligible byte buffers (_array_elt_tinfo) that are
                        # ACTUALLY fragmenting -- reached at >= 2 distinct offsets.
                        # A single-offset (offset-0-only) buffer is already coherent;
                        # retyping it gains nothing and trips the allocation-failed
                        # banner (force_linkat). Both gates -> strict zero-regression.
                        if (self._array_elt_tinfo(dims[1], dims[2]) is not None
                                and len(gep_offsets.get(nm, ())) >= 2):
                            self._array_alloca_elt[nm] = (dims[1], dims[2], dims[0])
                    elif nm and self._alloca_struct_key(ins) is not None:
                        arr_sz = sz  # whole-struct size (via _struct_size)
                    else:
                        raise NotImplementedError(
                            f"GEP-on-stack alloca %{nm} (scalar byte-pun / va_list / "
                            f"anonymous numeric IR -- not a laid-out struct field; "
                            f"native fallback)")
                    if named:
                        slot_off = host_off[nm][0]
                    else:
                        slot_off = _synthetic(off, arr_sz)
                        off = slot_off + max(arr_sz, 8)
                    self._addr_taken[nm] = (slot_off, arr_sz)
                elif nm in escaped:
                    # An escaped slot (address used as a value / via a deref bitcast).
                    # Host-named & 8-aligned -> its TRUE host offset (the de-collision
                    # above); else a low synthetic slot skipping the re-anchored ones.
                    # NO frame extension into host territory (the subframe INTERR chain).
                    if named:
                        slot_off = host_off[nm][0]
                    else:
                        slot_off = _synthetic(off, sz)
                        off = slot_off + max(sz, 8)
                    self._addr_taken[nm] = (slot_off, sz)
                    if self._alloca_is_ptr(ins):
                        # A pointer-typed slot: the lifter type-puns it via a no-op
                        # ``bitcast %slot`` for BOTH the pointer value (slot access)
                        # and a deref ``*slot`` (sub-pointer load/store). Record it
                        # so the load/store emit can tell the two apart (the deref
                        # alias set + valtype rule, mirroring native's
                        # ``mov %slot, r; ldx ds, r``).
                        self._ptr_allocas[nm] = slot_off
                        # Record the slot's POINTEE byte width (``i8*`` -> 1).
                        # A pointer-VALUE read punned to a non-pointer integer
                        # (``load i64, bitcast i8** %slot to i64*`` -- pass the
                        # key BY VALUE to a callback) loads the POINTER width (8)
                        # from a slot whose pointee is sub-pointer-width, whereas a
                        # genuine ``*slot`` deref loads exactly ``sizeof(pointee)``.
                        # The width lets the load/store emit tell the punned
                        # pointer-value read from a real deref (see ``_emit_value``).
                        # ``str(ins)`` is the OPAQUE ``alloca ptr`` (llvmlite erased
                        # the pointee), so take the typed declaration from the
                        # ORIGINAL IR text (``decl_types`` = ``i8*``).
                        self._ptr_alloca_pointee[nm] = \
                            self._pointee_size_of_decl(decl_types.get(nm))
                        # A single-level struct-pointer slot (``%extent_info*``):
                        # remember the pointee is a multi-field struct so a
                        # direct-bitcast pointer-width store of arithmetic is
                        # recognised as a cursor DEFINE, not a struct-pointee write.
                        if self._pointee_struct_of_type(
                                decl_types.get(nm, "")) is not None:
                            self._ptr_alloca_pointee_struct.add(nm)
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
        and eliminating the post-merge read block.

        Originally GATED on a noreturn call (the only case that INTERRs 50342). But
        the redundant funcresult routing also makes NON-noreturn + void fns drop
        silent garbage -- an uninit ``return v1`` (the slot kreg is never written),
        or a duplicate post-merge read. Writing the return reg straight on each path
        is what native does universally, so the promotion now fires for ANY fn whose
        return matches the funcresult-slot SHAPE (a single ret-value name loaded from
        a non-escaping alloca). The shape gate below (single ret name + load-of-slot
        + ``_ret_slot_uses_ok``) is the real guard; noreturn is no longer required.
        See memory idavator_drop_noreturn_50342_rootcause.
        """
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

    def _detect_ret_phi(self, fn) -> None:
        """Promote a RETURN PHI to the return register (the SROA-of-the-slot form).

        SROA collapses the ``%funcresult`` alloca/load/store return slot into a
        ``phi`` at the ret block: ``%fr.0 = phi [%v,@A],[null,@B]; ret %fr.0``.
        Destructuring that phi the normal way (a kreg READ at the post-merge ret
        block) re-fires the SAME value-numbering INTERR 50342 that
        ``_detect_ret_slot`` clears for the alloca form -- because a noreturn call
        still prunes an edge into the merge. The fix is identical in spirit: write
        the return reg straight on each incoming edge instead of materialising a
        merge var. We record the phi NAME; ``_build_multiblock`` then uses eax as
        that phi's "kreg" (PASS A.5 writes each incoming to eax on its edge, PASS
        B's ret skips the now-redundant mov). Originally GATED on a noreturn call;
        generalized alongside ``_detect_ret_slot`` to fire for ANY fn whose ret is a
        single phi result -- writing the return reg per-edge is what native does and
        avoids the materialised post-merge read regardless of noreturn."""
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
        if any(ins.opcode == "phi" and ins.name == rv
               for bb in fn.blocks for ins in bb.instructions):
            self._ret_phi = rv

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
    def _host_frame_offsets(mba) -> dict:
        """``{member_name: (frame_offset, size)}`` for the host function's frame
        (the lifter preserves source names, so a ``%storage`` alloca matches the
        host ``storage`` frame member). Empty on any failure / no frame."""
        out: dict = {}
        pfn = ida_funcs.get_func(mba.entry_ea)
        if pfn is None:
            return out
        tif = ida_typeinf.tinfo_t()
        if not ida_frame.get_func_frame(tif, pfn):
            return out
        udt = ida_typeinf.udt_type_data_t()
        if not tif.get_udt_details(udt):
            return out
        for i in range(udt.size()):
            m = udt[i]
            out[m.name] = (m.offset // 8, m.size // 8)
        return out

    @classmethod
    def _incoming_stack_offsets(cls, mba, fn, nregs) -> dict:
        """``{arg_index: host_frame_offset}`` for each incoming param that the
        SysV ABI passed on the CALLER's stack (index >= ``nregs``).

        ``_force_prototype`` ran before the frame was rebuilt, declaring every
        param as the ``a{i}`` member; ``set_type``/the frame builder classified
        params 0..5 into registers and spilled 6+ to the incoming-args region.
        We resolve each stack arg to its real host frame offset so a stkvar mop
        reads it. PRIMARY: match the ``a{i}`` frame member by name. FALLBACK:
        translate the prototype argloc's ``stkoff`` against the frame's
        incoming-args base (first member above ``__return_address``). The
        inverse of ``_emit_call_stackargs`` (the caller-side reg/stack split)."""
        out: dict = {}
        nargs = len(list(fn.arguments))
        if nargs <= nregs:
            return out
        host_off = cls._host_frame_offsets(mba)
        for i in range(nregs, nargs):
            slot = host_off.get(f"a{i}")
            if slot is not None:
                out[i] = slot[0]
        if len(out) == nargs - nregs:
            return out
        # Fallback: derive from the applied prototype's stack arglocs. The argloc
        # stkoff is relative to the incoming-args region; add the frame offset of
        # that region (right after the return address) to get the member offset.
        pfn = ida_funcs.get_func(mba.entry_ea)
        if pfn is None:
            return out
        tif = ida_typeinf.tinfo_t()
        if not ida_nalt.get_tinfo(tif, pfn.start_ea):
            return out
        ftd = ida_typeinf.func_type_data_t()
        if not tif.get_func_details(ftd) or ftd.size() < nargs:
            return out
        ret_member = host_off.get("__return_address")
        args_base = (ret_member[0] + ret_member[1]) if ret_member else None
        for i in range(nregs, nargs):
            if i in out:
                continue
            al = ftd[i].argloc
            if al.atype() == ida_typeinf.ALOC_STACK and args_base is not None:
                out[i] = args_base + al.stkoff()
        return out

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
        self._ret_phi = None
        argregs, eax, ds = self._abi()
        retb = next((mba.get_mblock(i) for i in range(mba.qty)
                     if (b := mba.get_mblock(i)) is not None and b.tail is not None
                     and int(b.tail.opcode) == hx.m_ret), None)
        if retb is None:
            retb = self._synthesize_ret_block(mba)

        # icmp defs (name -> (pred, operands)): a `select` consuming an icmp result
        # as a value materialises it via setcc (icmp itself is otherwise folded into
        # branches and emits nothing). _build_multiblock reuses this for the br fold.
        self._icmp_defs = {}
        # fp->int conversion results (fptoui/fptosi name -> the FP SOURCE operand).
        # The ida2llvm lifter has NO fcmp: it lowers a native FLOAT comparison
        # (`jbe.fpu xmm0, xmm1`) into `fptoui/fptosi float %x to i32` on BOTH sides
        # followed by an integer `icmp` on the converted ints. That integer compare
        # is NOT equivalent to the float compare it replaced (truncation loses the
        # fraction), and renders the spurious `(unsigned int)` casts native never
        # shows. When an icmp's BOTH operands are such conversions of same-typed
        # floats, the br-fold recovers the original FPU compare (jcc + set_fpinsn)
        # on the pre-conversion floats -- restoring native's `v5 > (float)(...)`.
        self._fp_cvt_src = {}
        # fcmp defs (name -> (pred, operands)): the lifter emits a real ``fcmp`` for a
        # native FLOAT compare; the br-fold lowers it straight to the FPU jcc
        # (``jbe.fpu`` etc., set_fpinsn) on the float operands, and a select/short-
        # circuit consumer materialises it via the FPU setcc. This is the direct
        # successor of the legacy fptoui+icmp recovery (_fp_compare_operands).
        self._fcmp_defs = {}
        # bitcast-of-function callees: the lifter carries a variadic call's surplus
        # varargs by widening the callee TYPE -- ``%c = bitcast @fn to <wider
        # variadic>; call %c(...)``. The call's callee operand is then the SSA
        # bitcast result (an indirect-looking value), not @fn. Map each such
        # bitcast result name -> the underlying named function so _emit_call can
        # see through it and route the call as a DIRECT variadic call to @fn.
        self._fnptr_bitcast = {}
        for bb in fn.blocks:
            for ins in bb.instructions:
                if ins.opcode == "icmp":
                    pred = re.search(r"icmp\s+(\w+)\s", str(ins).strip())
                    self._icmp_defs[ins.name] = (
                        pred.group(1) if pred else "ne", list(ins.operands))
                elif ins.opcode == "fcmp":
                    pred = re.search(r"fcmp\s+(\w+)\s", str(ins).strip())
                    self._fcmp_defs[ins.name] = (
                        pred.group(1) if pred else "une", list(ins.operands))
                elif ins.opcode in ("fptoui", "fptosi"):
                    fsrc = list(ins.operands)[0]
                    self._fp_cvt_src[ins.name] = (fsrc, _type_size(fsrc.type))
                elif ins.opcode == "bitcast" and ins.name:
                    sops = list(ins.operands)
                    if sops:
                        src = sops[0]
                        # A function source renders as `declare`/`define ... @name`
                        # AND resolves to a real function ea. A struct/data bitcast
                        # (load/gep/alloca source) is excluded.
                        sstr = str(src).strip()
                        if (src.name and (sstr.startswith("declare ")
                                          or sstr.startswith("define "))
                                and ida_name.get_name_ea(
                                    ida_idaapi.BADADDR, src.name)
                                != ida_idaapi.BADADDR):
                            self._fnptr_bitcast[ins.name] = src.name

        # Pointer-origin i64 values: the lifter reads a POINTER-typed alloca as an
        # i64 via the ``%c = bitcast ptr %slot to ptr; %v = load i64, ptr %c``
        # idiom (and materialises an address via ``ptrtoint``). Such a value feeding
        # a variadic ``%s`` arg is a pointer, but its LLVM type is i64, so
        # _emit_call_vararg would render it as a ``(signed __int64)`` reinterpret
        # cast instead of a clean ``void *``. Track each so the vararg path types it
        # as a pointer (matching native's clean pointer arg). Conservative: only the
        # load-of-ptr-alloca and ptrtoint forms; everything else stays int.
        self._ptr_origin_vals = set()
        _alloca_is_ptr = {}
        _bc_src = {}
        # Opaque pointers make every alloca's TYPE just ``ptr``; the held element
        # type is only in the instruction text (``%slot = alloca ptr, align 8``).
        _alloca_ptr_re = re.compile(r"=\s*alloca\s+ptr\b")
        for bb in fn.blocks:
            for ins in bb.instructions:
                if not ins.name:
                    continue
                if ins.opcode == "alloca":
                    _alloca_is_ptr[ins.name] = bool(
                        _alloca_ptr_re.search(str(ins).strip()))
                elif ins.opcode == "bitcast":
                    sops = list(ins.operands)
                    if sops:
                        _bc_src[ins.name] = sops[0].name
        for bb in fn.blocks:
            for ins in bb.instructions:
                if not ins.name:
                    continue
                if ins.opcode == "ptrtoint":
                    self._ptr_origin_vals.add(ins.name)
                elif ins.opcode == "load":
                    sops = list(ins.operands)
                    if not sops:
                        continue
                    src = sops[0].name
                    # peel a no-op bitcast off the load's pointer source.
                    src = _bc_src.get(src, src)
                    if _alloca_is_ptr.get(src):
                        self._ptr_origin_vals.add(ins.name)

        # Address of a STACK-ARRAY ELEMENT (``ptrtoint`` of a ``getelementptr`` into
        # a ``[N x T]`` alloca) passed as a VARIADIC arg. A MID-array element
        # (``&perms[1]`` -- a NONZERO GEP offset) is divergent (the forced prototype
        # fragments ``[12 x i8] perms`` and Hex-Rays tail-duplicates the call), so
        # _emit_call_vararg DECLINES it for a clean native fallback (NEVER a
        # building-but-divergent body). A WHOLE-buffer pointer (``&buf[0]``, offset 0)
        # USED to also fragment when the buffer was touched at other offsets (OLLVM's
        # ``[104 x i8] v56``, GEP'd at 25+ offsets -> ``scanf("%s", &buf)`` rendered
        # over a 4-byte fragment). The ARRAY-LVAR retype pass now cohers such a buffer
        # into ONE ``T[N]`` lvar, so the offset-0 pointer renders native's
        # ``scanf("%s", buf)`` and rides the variadic path (see below).
        #
        # The faithful shape is a ZERO-offset whole-buffer address (``&buf[0]``, a
        # ``scanf``/``fgets`` target). This stays coherent -- and is rendered as ONE
        # ``T[N]`` lvar -- WHEN the post-decompile ARRAY-LVAR retype pass
        # (_save_array_lvar_types) can type the slot: a ``[N x T]`` alloca whose
        # element T is a nameable scalar/ptr. Such a buffer renders like native's
        # ``scanf("%s", buf)`` EVEN when it is also GEP'd at nonzero offsets
        # elsewhere (a multi-purpose ``char buf[104]`` that scanf fills then strtok
        # walks -- exactly native C). A NONZERO offset (``&perms[1]``) is a
        # mid-element address and still declines. A buffer the retype CANNOT cohere
        # (struct/exotic element, or an alloca the layout cannot lay out) keeps the
        # old conservative rule: a zero-offset address into it is treated as
        # fragmenting (None) and declines. ``self._array_elt_addr`` maps each
        # array-element ptrtoint name to its EFFECTIVE offset for the gate. Scoped to
        # the vararg path so non-variadic array-address uses are untouched.
        _arr_alloca = set()
        _arr_nameable: set = set()  # [N x T] with a retype-nameable element T
        _arr_re = re.compile(r"=\s*alloca\s+\[\d+ x ")
        for bb in fn.blocks:
            for ins in bb.instructions:
                if ins.name and ins.opcode == "alloca" and _arr_re.search(
                        str(ins).strip()):
                    _arr_alloca.add(ins.name)
                    dims = self._array_dims(str(ins))
                    if dims is not None and self._array_elt_tinfo(
                            dims[1], dims[2]) is not None:
                        _arr_nameable.add(ins.name)
        # For each GEP-into-array result: (source alloca name, CONSTANT byte offset).
        # offset is None for a non-constant index. Track distinct offsets per array.
        _gep_into_arr: dict[str, tuple[str, int | None]] = {}
        _arr_distinct: dict[str, set] = {n: set() for n in _arr_alloca}
        for bb in fn.blocks:
            for ins in bb.instructions:
                if ins.name and ins.opcode == "getelementptr":
                    sops = list(ins.operands)
                    if sops and sops[0].name in _arr_alloca:
                        base = sops[0].name
                        try:
                            # vmap is not built yet, but a CONSTANT-index GEP needs
                            # none (operands resolve to ("num", k) directly). ANY
                            # failure (non-constant index -> NotImplementedError, an
                            # exotic operand -> ValueError, an un-laid-out element)
                            # means the offset is unknown: record None so the address
                            # is treated as nonzero and DECLINES. Never let offset
                            # analysis crash an otherwise-working drop.
                            off = self._gep_field_offset(ins, sops, {})
                        except Exception:  # noqa: BLE001 - conservative: unknown -> decline
                            off = None
                        _gep_into_arr[ins.name] = (base, off)
                        _arr_distinct[base].add(off)
        # A buffer is COHESIBLE iff the retype will actually fire on it: a nameable
        # byte buffer reached at >= 2 distinct offsets (the fragmenting case the
        # retype repairs). EXACTLY mirrors the _scan_allocas gate, so the offset-0
        # vararg gate proceeds iff the slot really cohers into one ``T[N]`` lvar.
        _arr_cohesible = {n for n in _arr_nameable
                          if len(_arr_distinct.get(n, ())) >= 2}
        self._array_elt_addr = {}
        for bb in fn.blocks:
            for ins in bb.instructions:
                if ins.name and ins.opcode == "ptrtoint":
                    sops = list(ins.operands)
                    if sops and sops[0].name in _gep_into_arr:
                        base, off = _gep_into_arr[sops[0].name]
                        # A zero-offset address into a buffer the retype pass cannot
                        # cohere is not a faithful whole-buffer pointer -> sentinel
                        # None forces decline. A cohesible buffer keeps off==0 and
                        # rides the variadic path (renders ``scanf("%s", buf)``).
                        if off == 0 and base not in _arr_cohesible:
                            off = None
                        self._array_elt_addr[ins.name] = off

        self._allocas = self._scan_allocas(mba, fn)
        self._scan_ptr_deref_aliases(fn)
        self._detect_ret_slot(fn)
        self._detect_ret_phi(fn)
        vmap: dict[str, tuple] = {}
        recv_stkoffs = self._incoming_stack_offsets(mba, fn, len(argregs))
        # Reg args whose value is read across a call need a stable kreg home (the
        # raw caller-saved arg register is clobbered by the intervening call's
        # arg-setup). Mirrors the lifter's plain-IR spill-to-alloca that the SROA
        # fallback removed. The entry block materialises ``mov argreg, kreg`` for
        # each (see _build_multiblock); args used only before any call keep the
        # raw register (inert -- no copy emitted).
        reg_arg_names = {a.name for i, a in enumerate(fn.arguments)
                         if i < len(argregs)}
        preserve = self._names_used_across_call(fn, reg_arg_names)
        arg_preserve: list[tuple] = []  # (kreg, argreg, size) entry copies
        for i, a in enumerate(fn.arguments):
            if i < len(argregs):
                sz = _type_size(a.type)
                if a.name in preserve:
                    kreg = mba.alloc_kreg(max(sz, 8))
                    arg_preserve.append((kreg, argregs[i], max(sz, 8)))
                    vmap[a.name] = ("reg", kreg, sz)
                else:
                    vmap[a.name] = ("reg", argregs[i], sz)
                continue
            # Incoming param 7+ (SysV): the caller spilled it to its stack; after
            # the standard prologue it rests in the host frame's incoming-args
            # region. _force_prototype declared it as the `a{i}` frame member, so
            # read its VALUE straight from that slot (a stkvar mop) -- the inverse
            # of _emit_call_stackargs' caller-side reg/stack split.
            off = recv_stkoffs.get(i)
            if off is None:
                raise NotImplementedError(
                    f"incoming stack arg #{i}: no host frame slot "
                    "(set_type/frame layout did not materialise it)")
            vmap[a.name] = ("stkvar", off, _type_size(a.type))

        # A call must terminate its block, so a function with any call needs the
        # multi-block (segment-splitting) path even if it has one LLVM block.
        has_call = any(ins.opcode == "call"
                       for bb in fn.blocks for ins in bb.instructions)
        if len(list(fn.blocks)) == 1 and not has_call:
            self._build_singleblock(mba, fn, retb, eax, ds, vmap)
        else:
            self._build_multiblock(mba, fn, retb, eax, ds, vmap,
                                   arg_preserve)
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
    def _switch_arms(ins):
        """(value_operand, default_name, [(case_int, target_name), ...]) for an
        LLVM ``switch``. llvmlite exposes the operands as
        ``[cond, default_label, caseval0, label0, caseval1, label1, ...]`` -- the
        case integers are unnamed constants (parse the literal from str), and the
        labels are text-only (parse the block names from str, textual order)."""
        ops = list(ins.operands)
        cond = ops[0]
        # Block names in textual order: the default first, then each case target.
        names = re.findall(r'label %"?([\w.@]+)"?', str(ins))
        default_name = names[0]
        target_names = names[1:]
        case_vals = []
        for o in ops[2::2]:
            m = re.search(r"(-?\d+)\s*$", str(o).strip())
            case_vals.append(int(m.group(1)) if m else 0)
        return cond, default_name, list(zip(case_vals, target_names))

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
    def _is_vararg_intrinsic_call(ins) -> bool:
        """A va_list-machine call -- the body's ``va_start``/``va_arg``/``va_end``
        (HexRays helper macros over ``__va_list_tag``) or the lifter's redundant
        ``llvm.va_start``/``llvm.va_end`` synth scaffold (dead, on an uninit
        ``%ArgList``). Kept IN-segment (NOT split into its own block) -- native
        renders them inline; ``_emit_value`` no-ops the scaffold and lowers the
        body macros via a helper call. See ``_VARARG_INTRINSICS``."""
        if ins.opcode != "call":
            return False
        return list(ins.operands)[-1].name in _VARARG_INTRINSICS

    @staticmethod
    def _value_used(ins) -> bool:
        """True if the SSA result of ``ins`` is referenced as an operand by any
        instruction in its function (llvmlite ValueRef exposes no use list, so we
        scan). A void/unnamed result is never used."""
        name = getattr(ins, "name", "")
        if not name:
            return False
        for blk in ins.function.blocks:
            for other in blk.instructions:
                for op in other.operands:
                    if op.name == name:
                        return True
        return False

    @staticmethod
    def _names_used_across_call(fn, names) -> set:
        """Subset of ``names`` (incoming-arg SSA names) whose VALUE is consumed
        at-or-after a ``call`` in the function's linearised instruction order.

        An incoming integer argument lives in a SysV caller-saved register
        (rdi/rsi/...); a ``call`` clobbers every such register (its own argument
        setup overwrites them). When a function reads an arg's value AFTER a call
        -- as the SROA-promoted ``remember_copied`` reads ``name`` (rdi) for
        ``xstrdup`` only AFTER ``xmalloc`` whose ``mov 0x18, rdi`` arg-setup
        already clobbered rdi -- the raw arg register is stale. Such args need a
        STABLE home (a kreg the decompiler can place in a callee-saved register /
        stack), mirroring the lifter's plain-IR spill-to-alloca that the SROA
        fallback optimised away. Args used only BEFORE any call keep the raw
        register (no clobber, no copy -- inert)."""
        if not names:
            return set()
        seen_call = False
        out: set = set()
        for bb in fn.blocks:
            for ins in bb.instructions:
                for op in ins.operands:
                    if seen_call and op.name in names:
                        out.add(op.name)
                if ins.opcode == "call":
                    seen_call = True
        return out

    @staticmethod
    def _arg_spill_slots_across_call(fn, reg_arg_names) -> dict:
        """``{spill_alloca_name: arg_register_name}`` for each incoming NON-POINTER
        scalar reg arg that is (1) passed BY VALUE to a call and (2) re-read AFTER
        that call, both THROUGH its spill slot.

        The lifter spills each param to an alloca (``store %arg, %slot``) and
        re-loads it. ``_names_used_across_call`` catches an arg whose SSA name is
        read past a call directly, but it MISSES the value that flows through the
        spill slot: ``create_hole`` stores ``size`` (arg3=rcx, an ``i64``) to
        ``%size`` at entry, then ``lseek(fd, load %size, 1)`` passes it BY VALUE
        (clobbering rcx), and ``punch_hole(fd, file_end - load %size, load %size)``
        re-loads it AFTER. ``%size``'s loads are the across-call reads, not ``%.4``.

        c3f71ce's preserve-kreg does NOT rescue this: the value is consumed by the
        clobbering call as its argument, so Hex-Rays copy-propagates the raw
        incoming register forward INTO that call site (the register is still the
        live representative there) and the post-clobber load resolves to a fresh,
        undefined register version (``rcx1`` -> uninit ``v5``). Native keeps the
        value on the STACK (``mov [rbp+size], rcx`` then re-loads ``[rbp+size]`` on
        each use); the stack slot survives the clobber.

        So a slot in this set must (a) rest in a real FRAME SLOT (not a scalar kreg
        Hex-Rays freely forwards) and (b) have its source register KILLED after the
        entry spill, so Hex-Rays cannot back-substitute the raw register past the
        spill and must anchor on the stable slot -- mirroring native, where the arg
        register is dead after the spill store.

        NARROW on purpose -- the trigger that distinguishes the LOST value:
        - NON-POINTER arg only. A POINTER arg spilled-and-reread (rpl_fflush's
          ``stream``, transfer_entries/hash_insert_if_absent/safe_hasher's table/
          entry pointers) is NOT lost: Hex-Rays preserves the pointer naturally,
          and demoting+killing it only churns variable naming. A scalar value
          (``size``) is the one the clobbering call's reg-setup destroys.
        - The slot must be passed BY VALUE to a call (a direct load operand of a
          ``call``), then loaded AGAIN after that SAME call. An arg merely live
          across unrelated calls (no by-value consumption by the clobbering call)
          keeps the raw register fine.
        - Single-store (pure param spill) slot only; a re-stored slot is a normal
          local, not a param spill."""
        if not reg_arg_names:
            return {}
        # Non-pointer reg-arg SSA names (pointer args are preserved naturally).
        scalar_args = {a.name for i, a in enumerate(fn.arguments)
                       if a.name in reg_arg_names and str(a.type) != "ptr"}
        if not scalar_args:
            return {}
        # Map each spill alloca to the scalar arg it is the (sole, entry) spill
        # target of, and count stores (a re-stored slot is not a pure param spill).
        spill: dict = {}        # slot_name -> arg_name
        store_count: dict = {}  # slot_name -> number of stores
        for bb in fn.blocks:
            for ins in bb.instructions:
                if ins.opcode != "store":
                    continue
                ops = list(ins.operands)
                if len(ops) < 2:
                    continue
                store_count[ops[1].name] = store_count.get(ops[1].name, 0) + 1
                if ops[0].name in scalar_args and ops[1].name not in spill:
                    spill[ops[1].name] = ops[0].name
        if not spill:
            return {}
        # A load of a spill slot used directly as a CALL argument marks that slot
        # as "consumed by value" at that call; a later load of the same slot is the
        # post-clobber re-read. Walk in order: a value-arg load arms the slot; a
        # subsequent load confirms the lost-across-clobbering-call pattern.
        out: dict = {}
        armed: set = set()         # slot loaded as a call arg, awaiting a re-read
        load_def: dict = {}        # load-result SSA name -> slot it loaded
        for bb in fn.blocks:
            for ins in bb.instructions:
                if ins.opcode == "load":
                    ld = list(ins.operands)
                    if ld and ld[0].name in spill:
                        if (store_count.get(ld[0].name, 0) == 1
                                and ld[0].name in armed):
                            out[ld[0].name] = spill[ld[0].name]
                        load_def[ins.name] = ld[0].name
                elif ins.opcode == "call":
                    # Arm each spill slot whose load feeds this call BY VALUE
                    # (a direct call operand). The callee operand is the last.
                    call_ops = list(ins.operands)[:-1]
                    for op in call_ops:
                        slot = load_def.get(op.name)
                        if slot is not None:
                            armed.add(slot)
        return out

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
        insns = list(bb.instructions)
        term = insns[-1]
        segs, cur = [], {"values": [], "call": None, "term": None,
                         "prev_call": None}
        # NB: llvmlite returns a FRESH ValueRef wrapper per iteration, so an
        # identity (`is`) check against `term` never matches -- stop by INDEX.
        for idx, ins in enumerate(insns):
            if idx == len(insns) - 1:
                break
            if ins.opcode == "phi":
                continue
            if (ins.opcode == "call"
                    and not LLVMDropConverter._is_canary_call(ins)
                    and not LLVMDropConverter._is_vararg_intrinsic_call(ins)):
                cur["call"] = ins
                if LLVMDropConverter._callee_is_noreturn(ins):
                    cur["noreturn"] = True
                    segs.append(cur)
                    return segs  # noreturn -> no continuation; drop the dead tail
                segs.append(cur)
                cur = {"values": [], "call": None, "term": None,
                       "prev_call": ins}
            else:
                # canary + va_list-machine calls stay in-segment -> _emit_value
                # elides the canary/scaffold and lowers va_start/va_arg inline.
                cur["values"].append(ins)
        cur["term"] = term
        segs.append(cur)
        return segs

    def _build_multiblock(self, mba, fn, retb, eax, ds, vmap,
                          arg_preserve=None) -> None:
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

        # icmp defs (for a `br %c` to fold its compare into the jump) were scanned
        # in _build (shared with the select->setcc materialisation).
        icmp_map = self._icmp_defs

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
                     "ftramp": None, "ttramp": None,
                     "cmps": None, "dtramp": None}
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
                    elif seg["term"].opcode == "switch":
                        # Lower to a chain of equality tests: the switch block is
                        # the FIRST compare (case 0); reserve one extra comparison
                        # block per remaining case. Each compare is a 2-way whose
                        # FALSE arm falls through (serial+1) to the next compare;
                        # the last compare's FALSE arm falls through to a DEFAULT
                        # trampoline (a 1-way `goto default`), mirroring the br
                        # ftramp -- a 2-way false arm must be the next serial.
                        _, _, arms = self._switch_arms(seg["term"])
                        e["cmps"] = [serial + k
                                     for k in range(max(0, len(arms) - 1))]
                        serial += len(e["cmps"])
                        e["dtramp"] = serial
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
        # The found host m_ret block keeps its ORIGINAL body, but the drop re-emits
        # every computation in the minted LLVM blocks and only needs retb as the
        # bare return sink. A side-effecting leftover (e.g. a leaf fn whose whole
        # body -- ``*__errno_location()=0x5F; return -1`` -- lives in the ret block)
        # is NOT dead-code-eliminated and survives as a DUPLICATE store + stale rax
        # write. Strip retb to its m_ret tail (singleblock does its own full wipe;
        # only the multiblock sink retb is missed by the 1..needed clear above).
        while retb.head is not None and retb.head is not retb.tail:
            retb.remove_from_block(retb.head)
        retb.mark_lists_dirty()
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
        # them); register them in vmap up front. The RETURN PHI (its result feeds
        # `ret`, downstream of a noreturn) is promoted: its "kreg" is the return
        # reg (eax), so PASS A.5 writes each incoming straight to eax and PASS B's
        # ret elides the redundant mov -- matching native, no post-merge read
        # block (clears INTERR 50342). See _detect_ret_phi.
        phi_kreg: dict[str, tuple] = {}
        for bps in phis.values():
            for pname, pins, _ in bps:
                sz = _type_size(pins.type)
                if pname == self._ret_phi:
                    phi_kreg[pname] = (eax, sz)
                    vmap[pname] = ("reg", eax, sz)
                else:
                    kreg = mba.alloc_kreg(sz)
                    phi_kreg[pname] = (kreg, sz)
                    vmap[pname] = ("reg", kreg, sz)

        # Preserve clobbered reg args: copy each incoming arg register that is
        # read across a call into its stable kreg at the ENTRY block head, BEFORE
        # any value emission (so the first call's arg-setup overwrites the raw
        # register, not the saved kreg). The decompiler propagates the kreg away
        # when nothing clobbers the source register (inert for non-clobbered
        # args). vmap already points each preserved arg at its kreg.
        if arg_preserve:
            entry_blk = mba.get_mblock(plan[0]["code"])
            anchor = None
            for kreg, argreg, sz in arg_preserve:
                mv = hx.minsn_t(ea)
                mv.opcode = hx.m_mov
                mv.l.make_reg(argreg, sz)
                mv.d.make_reg(kreg, sz)
                entry_blk.insert_into_block(mv, anchor)
                anchor = mv

        # PASS A: per segment, capture the previous call's result, emit value
        # instructions, then (for a call-segment) the call tail + fall-through.
        for e in plan:
            blk = mba.get_mblock(e["code"])
            anchor = None
            if e is plan[0] and arg_preserve:
                # The entry block already holds the arg-preservation copies; emit
                # this segment's values AFTER them (not at the block head).
                anchor = entry_blk.tail
            if (e["prev_call"] is not None
                    and str(e["prev_call"].type) != "void"
                    and self._value_used(e["prev_call"])):
                # Skip capturing a result no instruction consumes: the kreg copy
                # would be dead, and for a result-discarded call (e.g. a variadic
                # printf modeled with d.size 0) it would read an undefined rax.
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

        # KILL each arg-spill source register right after its entry spill store.
        # The lifter spills the arg (``mov argreg, stkvar``); the value then lives
        # in the frame slot (read by both the clobbering call and the post-clobber
        # re-load). Without this, Hex-Rays copy-propagates the raw incoming
        # register forward into the clobbering call site and the post-clobber load
        # resolves to a fresh, undefined register version (create_hole's ``size``
        # -> uninit ``v5``). Writing the dead register (``mov #0, reg``) severs the
        # copy chain so Hex-Rays must anchor on the stable slot -- native has the
        # register dead after its ``mov [rbp+size], rcx`` spill. The kill goes AFTER
        # the spill store (so the slot is written first) but in the ENTRY block,
        # before any branch leaves it. See _arg_spill_slots_across_call.
        if self._arg_spill_kill:
            entry_blk = mba.get_mblock(plan[0]["code"])
            killset = set(self._arg_spill_kill)
            ins = entry_blk.head
            while ins is not None and killset:
                # The spill store of a killed register: ``mov <killreg>, <stkvar>``.
                if (ins.opcode == hx.m_mov and ins.l.t == hx.mop_r
                        and ins.l.r in killset and ins.d.t == hx.mop_S):
                    reg = ins.l.r
                    killset.discard(reg)
                    kill = hx.minsn_t(ea)
                    kill.opcode = hx.m_mov
                    kill.l.make_number(0, 8)
                    kill.d.make_reg(reg, 8)
                    entry_blk.insert_into_block(kill, ins)
                    ins = kill  # continue past the inserted kill
                ins = ins.next

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
                if cond.name in self._fcmp_defs:
                    # Native FLOAT compare lifted as a real ``fcmp`` -> the FPU jcc
                    # (jbe.fpu/ja.fpu/...) on the float operands, set_fpinsn. The
                    # fcmp predicate already encodes ordered/unordered, so it maps
                    # straight to the carry/zero jcc family (_FCMP_JMP).
                    fpred, fops = self._fcmp_defs[cond.name]
                    fsz = _type_size(fops[0].type)
                    mi.opcode = _FCMP_JMP.get(fpred, hx.m_jnz)
                    self._fill(mi.l, self._desc(fops[0], vmap, fsz))
                    self._fill(mi.r, self._desc(fops[1], vmap, fsz))
                    mi.set_fpinsn()
                elif cond.name in icmp_map:
                    pred, iops = icmp_map[cond.name]
                    fpcmp = self._fp_compare_operands(iops)
                    if fpcmp is not None:
                        # Recovered FLOAT compare: the icmp's fptoui/fptosi operands
                        # were the lifter's lowering of a native float comparison.
                        # Emit the FPU jump (jbe.fpu/...) on the ORIGINAL floats so
                        # the decompiler renders `v5 > (float)(...)`, not the lossy
                        # `(unsigned int)v5 > (unsigned int)(...)` the int compare
                        # produces. set_fpinsn makes it the ``.fpu`` jcc native uses.
                        l_op, r_op, fsz = fpcmp
                        mi.opcode = _FPU_JMP.get(pred, hx.m_jnz)
                        self._fill(mi.l, self._desc(l_op, vmap, fsz))
                        self._fill(mi.r, self._desc(r_op, vmap, fsz))
                        mi.set_fpinsn()
                    else:
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
            elif term.opcode == "switch":
                # switch %v, default [ vi -> Bi ]  ->  a chain of equality tests.
                # Each comparison block does `jz %v, vi -> Bi` (2-way) and falls
                # through (serial+1) to the next compare; the last compare falls
                # through to a DEFAULT trampoline (1-way goto default). Hex-Rays
                # re-folds the chain back into a switch/if-ladder.
                cond, default_name, arms = self._switch_arms(term)
                default_s = name_serial[default_name]
                csz = _type_size(cond.type)
                cdesc = self._desc(cond, vmap, csz)
                # A switch-edge phi copy needs a per-case trampoline (not built);
                # defer rather than silently drop the out-of-SSA copy.
                for _v, tgt in arms:
                    if (e["bb"].name, tgt) in edges_need_copy:
                        raise NotImplementedError(
                            "switch edge needs a phi copy (per-case trampoline)")
                if not arms:
                    # degenerate switch (default only) -> unconditional goto.
                    g = hx.minsn_t(ea)
                    g.opcode = hx.m_goto
                    g.l.make_blkref(default_s)
                    blk.insert_into_block(g, anchor)
                    blk.type = hx.BLT_1WAY
                    self._wire(blk, [default_s])
                    continue
                cmp_blocks = [blk] + [mba.get_mblock(s) for s in e["cmps"]]
                for k, (cval, tgt) in enumerate(arms):
                    cblk = cmp_blocks[k]
                    target_s = name_serial[tgt]
                    # FALSE/fall-through arm: the next compare, or the default
                    # trampoline after the last compare (must be serial+1).
                    fall_s = (cmp_blocks[k + 1].serial
                              if k + 1 < len(cmp_blocks) else e["dtramp"])
                    mi = hx.minsn_t(ea)
                    mi.opcode = hx.m_jz
                    self._fill(mi.l, (cdesc[0], cdesc[1], csz))
                    mi.r.make_number(cval & ((1 << (8 * csz)) - 1), csz)
                    mi.d.make_blkref(target_s)
                    cblk.insert_into_block(mi, cblk.tail)
                    cblk.type = hx.BLT_2WAY
                    self._wire(cblk, [fall_s, target_s])  # [fall-through, taken]
                dt = mba.get_mblock(e["dtramp"])
                gd = hx.minsn_t(ea)
                gd.opcode = hx.m_goto
                gd.l.make_blkref(default_s)
                dt.insert_into_block(gd, dt.tail)
                dt.type = hx.BLT_1WAY
                self._wire(dt, [default_s])
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
