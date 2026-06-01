"""Lift-time type providers.

A ``TypeProvider`` supplies an LLVM type for a function-local during lifting. The
lifter consults a list of providers when allocating each local; the first provider
to return a type wins, and ``None`` defers to the normal IDA-derived type. Providers
are opt-in and composable, which keeps the lifter agnostic of any particular
type-recovery source (CiRCLE today, others later).
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Optional, Protocol, runtime_checkable

from llvmlite import ir


@runtime_checkable
class TypeProvider(Protocol):
    """Supplies an LLVM type for ``(func_ea, lvar_name)`` or ``None`` to defer."""

    def lvar_type(self, func_ea: int, lvar_name: str) -> Optional[ir.Type]:
        ...


def resolve_lvar_type(
    providers: Iterable[TypeProvider], func_ea: int, lvar_name: str
) -> Optional[ir.Type]:
    """Return the first provider's type for this local, or ``None`` if none match."""
    for provider in providers:
        typ = provider.lvar_type(func_ea, lvar_name)
        if typ is not None:
            return typ
    return None
