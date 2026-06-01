from __future__ import annotations

import json
from pathlib import Path

from llvmlite import ir

from idavator.circle_types import CircleTypes

# A miniature circle_result.json: a leaf struct, a struct with a struct** field
# (carrying a PointerToStruct relation), and lvar bindings in two functions.
_RESULT = {
    "Hash_tuning": {
        "name": "Hash_tuning",
        "fields": {
            "0x0": {"type": "float", "size": 4, "ptr_level": 0},
            "0x4": {"type": "float", "size": 4, "ptr_level": 0},
            "0x10": {"type": "char", "size": 1, "ptr_level": 0},
        },
        "bind_lvars": [
            {"func_ea": "0x1094A", "lvars": [{"name": "tuning", "kind": "arg", "index": 0}]},
        ],
    },
    "cp_options": {
        "name": "cp_options",
        "fields": {
            "0x0": {
                "type": "struct struct_x * *",
                "size": 8,
                "ptr_level": 2,
                "relation": {"type": "PointerToStruct", "target": "struct_x"},
            },
            "0x4": {"type": "int", "size": 4, "ptr_level": 0},
        },
        "bind_lvars": [
            {"func_ea": "0x5D83", "lvars": [{"name": "x", "kind": "arg", "index": 1}]},
        ],
    },
    "struct_x": {
        "name": "struct_x",
        "fields": {"0x0": {"type": "int", "size": 4, "ptr_level": 0}},
        "bind_lvars": [],
    },
}


def test_builds_identified_struct_types_for_each_recovered_struct() -> None:
    module = ir.Module(context=ir.Context())
    ct = CircleTypes(_RESULT, module)
    assert set(ct.struct_types) == {"Hash_tuning", "cp_options", "struct_x"}
    for name, t in ct.struct_types.items():
        assert isinstance(t, ir.IdentifiedStructType)
        assert t.elements is not None and len(t.elements) > 0, name


def test_scalar_fields_map_to_llvm_scalars_with_gap_padding() -> None:
    module = ir.Module(context=ir.Context())
    ct = CircleTypes(_RESULT, module)
    ht = ct.struct_types["Hash_tuning"]
    # 0x0:f32, 0x4:f32, gap 0x8..0x10 (8 bytes), 0x10:i8
    assert ht.elements[0] == ir.FloatType()
    assert ht.elements[1] == ir.FloatType()
    assert ht.elements[-1] == ir.IntType(8)
    # total byte size must equal the highest field end (0x10 + 1 = 17)
    assert _packed_size(ht) == 0x11


def test_struct_pointer_field_uses_relation_target() -> None:
    module = ir.Module(context=ir.Context())
    ct = CircleTypes(_RESULT, module)
    cp = ct.struct_types["cp_options"]
    f0 = cp.elements[0]
    # struct_x ** : pointer to pointer to the identified struct_x
    assert isinstance(f0, ir.PointerType)
    assert isinstance(f0.pointee, ir.PointerType)
    assert f0.pointee.pointee is ct.struct_types["struct_x"]


def test_lvar_index_maps_func_ea_and_name_to_struct_pointer_type() -> None:
    module = ir.Module(context=ir.Context())
    ct = CircleTypes(_RESULT, module)
    t = ct.lvar_type(0x1094A, "tuning")
    assert isinstance(t, ir.PointerType)
    assert t.pointee is ct.struct_types["Hash_tuning"]
    assert ct.lvar_type(0x5D83, "x").pointee is ct.struct_types["cp_options"]
    assert ct.lvar_type(0xDEAD, "nope") is None
    assert ct.lvar_type(0x1094A, "other") is None


def test_from_json_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "circle_result.json"
    p.write_text(json.dumps(_RESULT))
    module = ir.Module(context=ir.Context())
    ct = CircleTypes.from_json(p, module)
    assert ct.lvar_type(0x1094A, "tuning").pointee is ct.struct_types["Hash_tuning"]


def _packed_size(t: ir.IdentifiedStructType) -> int:
    total = 0
    for el in t.elements:
        total += _elem_size(el)
    return total


def _elem_size(el: ir.Type) -> int:
    if isinstance(el, ir.PointerType):
        return 8
    if isinstance(el, ir.IntType):
        return el.width // 8
    if isinstance(el, ir.FloatType):
        return 4
    if isinstance(el, ir.DoubleType):
        return 8
    if isinstance(el, ir.ArrayType):
        return el.count * _elem_size(el.element)
    raise AssertionError(f"unexpected element type {el}")
