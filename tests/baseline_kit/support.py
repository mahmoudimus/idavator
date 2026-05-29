from __future__ import annotations

from pathlib import Path

import pytest

from .metrics import (
    IRMetrics,
    compare_metrics,
    compute_metrics,
    load_baseline,
    save_baseline,
)


def baseline_path(artifacts_dir: Path, case: str) -> Path:
    return artifacts_dir / case / "metrics.json"


def assert_metrics_match_case(
    *,
    case: str,
    ir_text: str,
    artifacts_dir: Path,
    baseline_update: bool,
    source: str,
    passes: tuple[str, ...] = (),
) -> IRMetrics:
    actual = compute_metrics(ir_text)
    path = baseline_path(artifacts_dir, case)

    if baseline_update:
        save_baseline(
            path,
            case=case,
            metrics=actual,
            source=source,
            passes=passes,
        )
        return actual

    if not path.exists():
        pytest.fail(
            f"missing baseline {path}; run pytest --baseline-update to create it"
        )

    expected_payload = load_baseline(path)
    expected = IRMetrics(**expected_payload["metrics"])
    comparison = compare_metrics(expected, actual, case=case)
    comparison.raise_if_regressed()
    return actual
