from __future__ import annotations

from pathlib import Path

import pytest

from tests.baseline_kit import compute_metrics, verify_ir
from idavator.ir_passes import IRPassContext, run_ir_passes


def test_concurrency_pass_appends_metadata(fixtures_dir: Path) -> None:
    ir_text = (fixtures_dir / "tls_syscall_sample.ll").read_text()
    out = run_ir_passes(ir_text, ("concurrency",), IRPassContext())

    assert "!idavator.concurrency" in out
    metrics = compute_metrics(out)
    assert metrics.concurrency_metadata is True
    assert metrics.tls_read_metadata >= 1
    assert metrics.tls_read_calls == 1
    assert metrics.virtual_fs is True


def test_verify_pass_accepts_valid_ir(fixtures_dir: Path) -> None:
    pytest.importorskip("llvmlite")
    ir_text = (fixtures_dir / "tls_syscall_sample.ll").read_text()
    out = run_ir_passes(ir_text, ("verify",), IRPassContext())
    verify_ir(out)
    assert out == ir_text


def test_pipeline_runs_in_order(fixtures_dir: Path) -> None:
    pytest.importorskip("llvmlite")
    ir_text = (fixtures_dir / "tls_syscall_sample.ll").read_text()
    out = run_ir_passes(ir_text, ("concurrency", "verify"), IRPassContext())
    assert "!idavator.concurrency" in out
    verify_ir(out)


def test_unknown_pass_raises() -> None:
    with pytest.raises(ValueError, match="unknown IR pass"):
        run_ir_passes("define void @f() { ret void }", ("nope",), IRPassContext())
