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


@pytest.mark.xfail(
    reason="The committed metrics baselines were generated on the dev macOS IDA, "
    "which lifts examples/cp to ~1995 calls / ~1.27 MB of IR. IDA 9.3 Linux "
    "(idalib CI) recovers fewer functions/calls from the same binary (~1455 "
    "calls / ~1.01 MB) -- a real cross-IDA-build decompiler-coverage difference, "
    "not a lift regression (the lift succeeds; metrics.defines/calls > 0). A "
    "single committed baseline cannot satisfy both IDA builds; regenerating for "
    "Linux would simply flip the failure onto macOS dev. Left xfail (not "
    "baseline-flipped) to keep the dev baseline authoritative. Run "
    "'pytest --baseline-update' in the idapro-9.3 container to regenerate IF the "
    "Linux IDA is adopted as the canonical baseline env.",
    strict=False,
)
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
