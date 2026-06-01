from __future__ import annotations

from llvmlite import ir

from idavator.circle_types import CircleTypes
from idavator.type_providers import TypeProvider, resolve_lvar_type


class _FakeProvider:
    def __init__(self, mapping: dict[tuple[int, str], ir.Type]) -> None:
        self.mapping = mapping

    def lvar_type(self, func_ea: int, lvar_name: str):
        return self.mapping.get((func_ea, lvar_name))


def test_resolve_returns_first_provider_hit() -> None:
    i32, i8 = ir.IntType(32), ir.IntType(8)
    p1 = _FakeProvider({(0x10, "a"): i32})
    p2 = _FakeProvider({(0x10, "a"): i8, (0x10, "b"): i8})
    # first provider wins for "a"; "b" falls through to the second
    assert resolve_lvar_type([p1, p2], 0x10, "a") is i32
    assert resolve_lvar_type([p1, p2], 0x10, "b") is i8


def test_resolve_returns_none_when_no_provider_matches() -> None:
    p = _FakeProvider({(0x10, "a"): ir.IntType(32)})
    assert resolve_lvar_type([p], 0x10, "missing") is None
    assert resolve_lvar_type([], 0x10, "a") is None


def test_circle_types_satisfies_type_provider_protocol() -> None:
    module = ir.Module(context=ir.Context())
    ct = CircleTypes(
        {
            "S": {
                "name": "S",
                "fields": {"0x0": {"type": "int", "size": 4, "ptr_level": 0}},
                "bind_lvars": [
                    {"func_ea": "0x10", "lvars": [{"name": "a", "kind": "arg"}]}
                ],
            }
        },
        module,
    )
    assert isinstance(ct, TypeProvider)
    # and it works as a provider through resolve_lvar_type
    assert resolve_lvar_type([ct], 0x10, "a").pointee is ct.struct_types["S"]
