from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class IRPassContext:
    source_binary: str | None = None
    target_mode: str = "host"
    metadata: dict[str, object] = field(default_factory=dict)


class IRTextPass(Protocol):
    name: str

    def run(self, ir_text: str, ctx: IRPassContext) -> str:
        ...


def run_ir_passes(
    ir_text: str, pass_names: Sequence[str], ctx: IRPassContext | None = None
) -> str:
    if not pass_names:
        return ir_text

    ctx = ctx or IRPassContext()
    registry = _pass_registry()
    for pass_name in pass_names:
        normalized = pass_name.strip()
        if not normalized:
            continue
        try:
            ir_pass = registry[normalized]
        except KeyError as exc:
            available = ", ".join(sorted(registry))
            raise ValueError(
                f"unknown IR pass {normalized!r}; available passes: {available}"
            ) from exc
        ir_text = ir_pass.run(ir_text, ctx)
    return ir_text


def parse_ir_passes(pass_spec: str | None) -> tuple[str, ...]:
    if not pass_spec:
        return ()
    return tuple(part.strip() for part in pass_spec.split(",") if part.strip())


def _pass_registry() -> dict[str, IRTextPass]:
    from .concurrency import ConcurrencyAnnotationPass
    from .verify import VerifyIRPass

    passes: tuple[IRTextPass, ...] = (ConcurrencyAnnotationPass(), VerifyIRPass())
    return {ir_pass.name: ir_pass for ir_pass in passes}
