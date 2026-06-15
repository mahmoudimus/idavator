"""Cross-IDA-build-robust lift assertions for examples/cp.

The committed exact-metrics baselines (tests/artifacts/cp_lift*/metrics.json) were
generated on the dev macOS IDA, which lifts examples/cp to ~1995 calls / ~1.27 MB.
IDA 9.3 Linux (idalib CI) recovers fewer call sites from the SAME binary (~1455
calls / ~1.01 MB) -- a real cross-IDA-build decompiler-coverage difference, not a
lift regression (the lift itself succeeds and the function-recovery count is
IDENTICAL: defines == 575 on both builds).

A single exact baseline cannot satisfy both builds, so this test asserts the
build-STABLE invariants plus a conservative structural FLOOR plus the pass-specific
delta the test actually cares about -- enough to catch a real lift regression (the
module collapsing, functions vanishing, the concurrency metadata not being emitted)
while tolerating the benign per-build call/byte coverage delta.

``--baseline-update`` still refreshes the committed exact-metrics JSON (for the
record / cross-build drift tracking); it is no longer the pass/fail gate.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from tests.baseline_kit import (
    assert_metrics_match_case,
    compute_metrics,
    load_baseline,
)


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

# Conservative structural floors that BOTH the dev macOS (~1995 calls) and IDA 9.3
# Linux (~1455 calls) builds clear comfortably -- a real lift regression (recovery
# collapsing) drops well below these; the benign cross-build coverage delta does
# not. Kept ~20% under the lower (Linux) build so normal drift never trips them.
_FLOORS = {
    "defines": 500,    # 575 on both builds (function recovery is build-stable)
    "calls": 1200,     # 1455 (Linux) / 1995 (macOS)
    "loads": 5000,     # 5972 / 6638
    "stores": 2800,    # 3369 / 4038
    "globals": 400,    # 468 / 478
    "declares": 100,   # 128 / 132
    "bytes": 900_000,  # 1.01 MB / 1.27 MB
    "lines": 25_000,   # 31527 / 37562
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

    if baseline_update:
        # Refresh the committed exact-metrics JSON for the record (not the gate).
        assert_metrics_match_case(
            case=case,
            ir_text=ir_text,
            artifacts_dir=artifacts_dir,
            baseline_update=True,
            source=f"examples/cp via ida2llvm passes={','.join(passes) or 'none'}",
            passes=passes,
        )

    metrics = compute_metrics(ir_text)
    md = metrics.to_dict()

    # 1. Structural FLOOR: the lift recovered a substantial module on this build.
    for key, floor in _FLOORS.items():
        assert md[key] >= floor, (
            f"{case}: {key}={md[key]} below cross-build floor {floor} "
            f"(lift regression -- recovery collapsed)")

    # 2. Build-STABLE invariants (identical on both IDA builds).
    assert md["defines"] == 575, (
        f"{case}: defines={md['defines']} (expected the build-stable 575 -- a "
        "change here is a real function-recovery regression, not a render delta)")
    assert md["virtual_fs"] is True, f"{case}: virtual_fs flag lost"
    assert md["syscall_calls"] == 0, (
        f"{case}: unexpected raw syscall calls: {md['syscall_calls']}")
    assert md["tls_read_calls"] > 0, (
        f"{case}: no TLS reads recovered (lift regression)")

    # 3. Pass-specific DELTA -- the thing each pass set actually changes.
    concurrency_on = "concurrency" in passes
    assert md["concurrency_metadata"] is concurrency_on, (
        f"{case}: concurrency_metadata={md['concurrency_metadata']} but "
        f"concurrency pass {'on' if concurrency_on else 'off'}")
    if concurrency_on:
        # the concurrency pass annotates every recovered TLS read.
        assert md["tls_read_metadata"] == md["tls_read_calls"], (
            f"{case}: tls.read metadata ({md['tls_read_metadata']}) does not "
            f"annotate every TLS read ({md['tls_read_calls']})")
    else:
        assert md["tls_read_metadata"] == 0, (
            f"{case}: tls.read metadata present without the concurrency pass")

    # 4. The committed baseline (where present) stays a valid, loadable record and
    #    its build-stable anchor agrees -- guards the JSON against silent rot.
    baseline_file = artifacts_dir / case / "metrics.json"
    if baseline_file.exists():
        payload = load_baseline(baseline_file)
        assert payload["metrics"]["defines"] == md["defines"], (
            f"{case}: committed baseline defines "
            f"{payload['metrics']['defines']} != current {md['defines']}")
