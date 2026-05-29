from __future__ import annotations

from pathlib import Path

import pytest

from idavator.ir_passes import IRPassContext, run_ir_passes
from tests.baseline_kit import (
    assert_metrics_match_case,
    compute_metrics,
    verify_ir,
    write_metrics_report,
)

_BASELINE_REPORT: dict[str, dict] = {}


@pytest.mark.parametrize(
    ("case", "ir_name", "passes"),
    [
        ("cp_lift", "cp_lift.ll", ()),
        ("cp_lift_concurrency", "cp_lift.ll", ("concurrency",)),
        ("cp_lift_pipeline", "cp_lift.ll", ("concurrency", "verify")),
        ("fixture_tls_syscall", "tls_syscall_sample.ll", ("concurrency",)),
    ],
)
def test_baseline_metrics(
    case: str,
    ir_name: str,
    passes: tuple[str, ...],
    fixtures_dir: Path,
    artifacts_dir: Path,
    baseline_update: bool,
) -> None:
    ir_text = (fixtures_dir / ir_name).read_text()
    if passes:
        ir_text = run_ir_passes(ir_text, passes, IRPassContext())

    source = f"tests/fixtures/{ir_name}" + (
        f" passes={','.join(passes)}" if passes else ""
    )
    actual = assert_metrics_match_case(
        case=case,
        ir_text=ir_text,
        artifacts_dir=artifacts_dir,
        baseline_update=baseline_update,
        source=source,
        passes=passes,
    )

    if not baseline_update and "verify" in passes:
        pytest.importorskip("llvmlite")
        verify_ir(ir_text)

    _BASELINE_REPORT[case] = actual.to_dict()


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if not _BASELINE_REPORT:
        return
    report = _BASELINE_REPORT
    out = Path(session.config.rootpath) / "tests" / "reports" / "last-metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_metrics_report(out, report)
