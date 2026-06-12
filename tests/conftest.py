from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
EXAMPLES_DIR = ROOT / "examples"


def _setup_oracle_env() -> None:
    """Best-effort: make 's clang_loader importable and point at IDA's
    libclang so the AST oracle (idavator.oracle) works without manual env. All
    paths are env-overridable; missing paths are a silent no-op (oracle skips)."""
    _src = os.environ.get("_SRC")
    if not _src:
        # idavator and  conventionally share an .../idapro/ parent.
        candidate = Path(__file__).resolve().parents[3] / "" / "src"
        if candidate.exists():
            _src = str(candidate)
    if _src and Path(_src).exists() and _src not in sys.path:
        sys.path.insert(0, _src)
    if not os.environ.get("IDA_INSTALL_DIR"):
        for app in sorted(Path("/Applications").glob("IDA*.app"), reverse=True):
            macos = app / "Contents" / "MacOS"
            if macos.exists():
                os.environ["IDA_INSTALL_DIR"] = str(macos)
                break


_setup_oracle_env()


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
