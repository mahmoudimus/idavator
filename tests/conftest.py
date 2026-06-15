from __future__ import annotations

import os
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
EXAMPLES_DIR = ROOT / "examples"


def _setup_oracle_env() -> None:
    """Best-effort: point at IDA's bundled libclang so the AST oracle
    (idavator.oracle) works without manual env. Env-overridable; a missing IDA
    install is a silent no-op (the oracle reports clang_available() == False and
    its tests skip). The clang loader + bindings are vendored under
    idavator._vendor -- no sibling-repo path injection needed."""
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
