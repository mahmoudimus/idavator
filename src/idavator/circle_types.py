"""Consume CiRCLE struct-recovery output and expose it as LLVM types.

CiRCLE (vul337/CiRCLE, MIT) recovers struct layouts from IDA Hex-Rays microcode
and writes a ``circle_result.json``. This module turns that JSON into byte-accurate
LLVM identified ``%struct`` types plus a ``(func_ea, lvar_name) -> struct`` index, so
the lifter can type recovered struct-pointer locals instead of falling back to i8*.

IDA-free by design (operates purely on the JSON + llvmlite), so it unit-tests offline.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from llvmlite import ir

# CiRCLE scalar C type names -> LLVM scalar types. Anything not listed (embedded
# structs/unions/unknown) falls back to a byte array of the field's declared size.
_SCALARS: dict[str, ir.Type] = {
    "bool": ir.IntType(1),
    "char": ir.IntType(8),
    "signed char": ir.IntType(8),
    "unsigned char": ir.IntType(8),
    "short": ir.IntType(16),
    "unsigned short": ir.IntType(16),
    "int": ir.IntType(32),
    "unsigned int": ir.IntType(32),
    "long": ir.IntType(64),
    "unsigned long": ir.IntType(64),
    "long long": ir.IntType(64),
    "unsigned long long": ir.IntType(64),
    "float": ir.FloatType(),
    "double": ir.DoubleType(),
}


class CircleTypes:
    """Recovered CiRCLE structs as LLVM types, queryable by (func_ea, lvar name)."""

    def __init__(self, result: dict[str, Any], module: ir.Module) -> None:
        self._context = module.context
        self.struct_types: dict[str, ir.IdentifiedStructType] = build_struct_types(
            result, module
        )
        self.index: dict[tuple[int, str], str] = build_lvar_index(result)

    @classmethod
    def from_json(cls, path: str | Path, module: ir.Module) -> "CircleTypes":
        result = json.loads(Path(path).read_text())
        return cls(result, module)

    def lvar_type(self, func_ea: int, lvar_name: str) -> Optional[ir.Type]:
        """LLVM type for a CiRCLE-bound local: a pointer to its recovered struct.

        Returns ``None`` when the (func_ea, lvar) pair was not recovered, so callers
        fall back to the normal lift.
        """
        struct_name = self.index.get((func_ea, lvar_name))
        if struct_name is None:
            return None
        return self.struct_types[struct_name].as_pointer()


def _field_llvm_type(
    field: dict[str, Any], struct_types: dict[str, ir.IdentifiedStructType]
) -> ir.Type:
    ptr_level = int(field.get("ptr_level", 0) or 0)
    if ptr_level > 0:
        relation = field.get("relation") or {}
        target = relation.get("target")
        base: ir.Type
        if relation.get("type") == "PointerToStruct" and target in struct_types:
            base = struct_types[target]
        else:
            base = ir.IntType(8)  # opaque pointer base
        typ = base
        for _ in range(ptr_level):
            typ = typ.as_pointer()
        return typ

    type_name = (field.get("type") or "").strip()
    scalar = _SCALARS.get(type_name)
    if scalar is not None:
        return scalar
    # Embedded struct / union / unknown: byte-accurate filler of the declared size.
    size = int(field.get("size", 0) or 0)
    return ir.ArrayType(ir.IntType(8), max(size, 1))


def _layout_fields(
    fields: dict[str, Any], struct_types: dict[str, ir.IdentifiedStructType]
) -> list[ir.Type]:
    """Ordered LLVM element list with i8 padding inserted for inter-field gaps."""
    items = sorted(
        ((int(off, 16), info) for off, info in fields.items()), key=lambda x: x[0]
    )
    elements: list[ir.Type] = []
    cursor = 0
    for offset, info in items:
        if offset > cursor:
            elements.append(ir.ArrayType(ir.IntType(8), offset - cursor))
            cursor = offset
        elif offset < cursor:
            # Overlapping/out-of-order field (e.g. union-like): skip to keep layout
            # monotonic rather than emit a negative-size pad.
            continue
        el = _field_llvm_type(info, struct_types)
        elements.append(el)
        cursor = offset + int(info.get("size", 0) or 0)
    return elements


def build_struct_types(
    result: dict[str, Any], module: ir.Module
) -> dict[str, ir.IdentifiedStructType]:
    """Create a packed identified ``%struct`` per recovered struct.

    Two passes so struct-pointer fields can reference structs declared later:
    pass 1 creates empty identified types, pass 2 fills bodies.
    """
    context = module.context
    struct_types: dict[str, ir.IdentifiedStructType] = {}
    for name in result:
        struct_types[name] = context.get_identified_type(name)

    for name, struct in result.items():
        t = struct_types[name]
        # Identified types are cached per-context by name; if a prior build already
        # defined this one (same long-lived context), reuse it instead of re-setting.
        if not t.is_opaque:
            continue
        fields = struct.get("fields") or {}
        elements = _layout_fields(fields, struct_types)
        if not elements:
            elements = [ir.ArrayType(ir.IntType(8), 1)]
        t.set_body(*elements)
        t.packed = True
    return struct_types


def build_lvar_index(result: dict[str, Any]) -> dict[tuple[int, str], str]:
    """Map (func_ea:int, lvar_name) -> struct name from every struct's bind_lvars."""
    index: dict[tuple[int, str], str] = {}
    for name, struct in result.items():
        for binding in struct.get("bind_lvars") or []:
            func_ea = int(binding["func_ea"], 16)
            for lvar in binding.get("lvars") or []:
                index[(func_ea, lvar["name"])] = name
    return index
