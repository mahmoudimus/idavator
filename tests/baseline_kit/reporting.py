from __future__ import annotations

import json
from pathlib import Path

from .metrics import BASELINE_VERSION


def write_metrics_report(path: Path, results: dict[str, dict]) -> None:
    path.write_text(
        json.dumps(
            {"version": BASELINE_VERSION, "results": results},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
