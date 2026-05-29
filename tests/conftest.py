from __future__ import annotations

from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
EXAMPLES_DIR = ROOT / "examples"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--baseline-update",
        action="store_true",
        default=False,
        help="Refresh committed baseline metrics JSON files",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "ida: requires IDA Pro idalib (skipped when unavailable)"
    )


@pytest.fixture
def baseline_update(request: pytest.FixtureRequest) -> bool:
    return bool(request.config.getoption("--baseline-update"))


@pytest.fixture
def artifacts_dir() -> Path:
    return ARTIFACTS_DIR


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture
def examples_dir() -> Path:
    return EXAMPLES_DIR


@pytest.fixture
def metrics_report(tmp_path: Path) -> Path:
    return tmp_path / "metrics-report.json"
