"""Round-trip fidelity harness for the LLVM->microcode drop.

The round trip is: a function's lifted LLVM IR -> DROP (rebuild its microcode,
let Hex-Rays decompile) -> compare the dropped pseudocode against the original
via the libclang :mod:`idavator.oracle`. If lift-then-drop is faithful, the
dropped function is semantically identical to the original.

A real lifted module (``examples/cp.ll``) is full of constructs the drop does not
yet handle (switch, struct/multi-index GEP, float/vector, vararg, intrinsics).
Feeding those to the drop can crash a later Hex-Rays maturity, so the harness
PRE-FILTERS to the supported subset (parse-only -- never drops an unsupported
function) and reports coverage. Only supported functions are actually dropped.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import llvmlite.binding as llvm

from idavator.oracle import clang_available, fidelity_ledger

# NB: the coverage analyzer (is_supported / module_coverage) is deliberately
# IDA-free (llvmlite only) so it runs standalone. Only round_trip() drops, and it
# imports LLVMDropConverter at the IDA boundary, inside the function.

# Opcodes the drop converter lowers (see idavator.llvm_drop).
SUPPORTED_OPS = frozenset({
    "add", "sub", "mul", "and", "or", "xor", "shl", "lshr", "ashr",
    "udiv", "sdiv", "urem", "srem",
    "zext", "sext", "trunc", "bitcast", "ptrtoint", "inttoptr",
    "load", "store", "call", "icmp", "br", "ret", "phi", "getelementptr",
    "alloca",
})

# Scalar value types the drop can size (integer + opaque pointer).
_OK_TYPE = re.compile(r"^(i\d+|ptr)$")


def _type_ok(t) -> bool:
    return bool(_OK_TYPE.match(str(t).strip()))


def is_supported(fn) -> tuple[bool, str]:
    """(supported, reason). A function is droppable only if every instruction
    uses a lowered opcode over integer/pointer scalars, GEP is single-index, and
    the signature is non-vararg int/pointer."""
    if fn.is_declaration:
        return False, "declaration"
    for arg in fn.arguments:
        if not _type_ok(arg.type):
            return False, f"argtype:{arg.type}"
    # alloca is droppable as a scalar slot (kreg), address-taken into an existing
    # host frame slot (&local -> mop_a(stkvar)), OR GEP'd as a scalar/ptr-element
    # ARRAY ([N x ptr/iX]) -> &stkvar(off + field). A GEP over a struct/va_list
    # element still needs real struct layout -- unsupported.
    alloca_names = {ins.name for bb in fn.blocks for ins in bb.instructions
                    if ins.opcode == "alloca"}
    if alloca_names:
        for bb in fn.blocks:
            for ins in bb.instructions:
                if ins.opcode != "getelementptr":
                    continue
                ops = list(ins.operands)
                if not (ops and ops[0].name in alloca_names):
                    continue
                txt = str(ins)
                if "va_list" in txt:
                    return False, "alloca:gep-valist"
                # scalar/ptr OR a %struct element array; the drop computes the
                # struct size from the parsed layout (raises if un-computable).
                if not re.search(r"\[\s*\d+\s+x\s+(?:ptr|i\d+|%[\w\".:$]+)\s*\]",
                                 txt):
                    return False, "alloca:gep-struct"
    for bb in fn.blocks:
        for ins in bb.instructions:
            op = ins.opcode
            if op not in SUPPORTED_OPS:
                return False, f"opcode:{op}"
            if op not in ("store", "br", "ret") and not _type_ok(ins.type):
                return False, f"restype:{ins.type}"
            ops = list(ins.operands)
            if op == "getelementptr" and len(ops) != 2:
                return False, "gep:multi-index"
            # operand value types (skip the callee/label operands)
            checkable = ops[:-1] if op in ("call", "br") else ops
            for o in checkable:
                ot = str(o.type).strip()
                if ot in ("label", "void"):
                    continue
                if not _type_ok(o.type) and "*" not in ot:
                    return False, f"optype:{ot}"
    return True, ""


@dataclass
class Coverage:
    supported: list = field(default_factory=list)
    unsupported: dict = field(default_factory=dict)  # name -> reason

    @property
    def total(self) -> int:
        return len(self.supported) + len(self.unsupported)

    def reason_histogram(self) -> dict:
        hist: dict = {}
        for reason in self.unsupported.values():
            key = reason.split(":", 1)[0]
            hist[key] = hist.get(key, 0) + 1
        return dict(sorted(hist.items(), key=lambda kv: -kv[1]))


def module_coverage(ir_text: str) -> Coverage:
    """Parse a module and classify every defined function as drop-supported or
    not (parse-only; nothing is dropped)."""
    module = llvm.parse_assembly(ir_text)
    cov = Coverage()
    for fn in module.functions:
        if fn.is_declaration:
            continue
        ok, reason = is_supported(fn)
        if ok:
            cov.supported.append(fn.name)
        else:
            cov.unsupported[fn.name] = reason
    return cov


@dataclass
class RoundTripResult:
    name: str
    ok: bool
    error: str | None = None
    interr: int | None = None
    ledger: dict = field(default_factory=dict)
    dropped_c: str = ""


def round_trip(ir_text: str, fn_name: str, host_ea: int,
               original_c: str) -> RoundTripResult:
    """Drop ``fn_name``'s IR into ``host_ea`` and compare the result to
    ``original_c`` via the oracle. ``original_c`` is the host's pseudocode BEFORE
    the drop (the round-trip reference)."""
    from idavator.llvm_drop import LLVMDropConverter  # IDA boundary

    conv = LLVMDropConverter(ir_text)
    cf = conv.drop(host_ea, fn_name)
    if conv.last_error is not None:
        return RoundTripResult(fn_name, False, error=conv.last_error,
                               interr=conv.last_interr)
    if cf is None:
        return RoundTripResult(fn_name, False, error="decompile returned None",
                               interr=conv.last_interr)
    dropped = str(cf)
    if not clang_available():
        # No oracle -> report the drop succeeded but fidelity unverified.
        return RoundTripResult(fn_name, True, dropped_c=dropped,
                               ledger={"oracle": "unavailable"})
    ledger = fidelity_ledger(original_c, dropped)
    return RoundTripResult(fn_name, ledger == {}, ledger=ledger,
                           dropped_c=dropped)
