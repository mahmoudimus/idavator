from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from tests.baseline_kit import assert_metrics_match_case, compute_metrics


def _idalib_available() -> bool:
    try:
        import idapro  # noqa: F401

        return True
    except ImportError:
        return False


_PASS_TO_CASE: dict[tuple[str, ...], str] = {
    (): "cp_lift",
    ("concurrency",): "cp_lift_concurrency",
    ("concurrency", "verify"): "cp_lift_pipeline",
}


@pytest.mark.ida
@pytest.mark.parametrize("passes", [(), ("concurrency",), ("concurrency", "verify")])
def test_ida_lift_matches_baseline(
    passes: tuple[str, ...],
    examples_dir: Path,
    artifacts_dir: Path,
    baseline_update: bool,
) -> None:
    if not _idalib_available():
        pytest.skip("idalib is not available in this environment")

    binary = examples_dir / "cp"
    if not binary.exists():
        pytest.skip(f"missing example binary: {binary}")

    from idavator.cli import lift_binary_to_llvm

    case = _PASS_TO_CASE[passes]
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "lifted.ll"
        ok = lift_binary_to_llvm(
            input_binary=str(binary),
            output_llvm_ir=str(out),
            target_mode="host",
            verbose=False,
            ir_passes=passes,
        )
        assert ok, "lift_binary_to_llvm failed"
        ir_text = out.read_text()

    assert_metrics_match_case(
        case=case,
        ir_text=ir_text,
        artifacts_dir=artifacts_dir,
        baseline_update=baseline_update,
        source=f"examples/cp via ida2llvm passes={','.join(passes) or 'none'}",
        passes=passes,
    )

    metrics = compute_metrics(ir_text)
    assert metrics.defines > 0
    assert metrics.calls > 0
