from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from idavator.events import EventEmitter
from idavator.persistence import (
    FIDELITY_EVENT,
    FidelityEvent,
    FidelityStore,
    fidelity_summary,
    worst_offenders,
)


def _idalib_available() -> bool:
    try:
        import idapro  # noqa: F401

        return True
    except ImportError:
        return False


def _emit(emitter: EventEmitter, **kwargs) -> None:
    emitter.emit(FIDELITY_EVENT, FidelityEvent(**kwargs))


def test_store_persists_events_and_summarizes_by_kind(tmp_path: Path) -> None:
    db = tmp_path / "fidelity.db"
    emitter = EventEmitter()
    store = FidelityStore()
    store.subscribe(emitter)
    store.start(db, binary="cp", output="cp.ll")

    _emit(emitter, kind="type_fallback_ptrsize", severity="imprecision", function="main", ea=0x1000)
    _emit(emitter, kind="type_fallback_ptrsize", severity="imprecision", function="foo", ea=0x2000)
    _emit(emitter, kind="value_zero_substituted", severity="corruption", function="main", ea=0x1004, detail="bitcast")

    store.stop()

    assert fidelity_summary(db) == {
        "type_fallback_ptrsize": 2,
        "value_zero_substituted": 1,
    }


def test_worst_offenders_orders_functions_by_event_count(tmp_path: Path) -> None:
    db = tmp_path / "fidelity.db"
    emitter = EventEmitter()
    store = FidelityStore()
    store.subscribe(emitter)
    store.start(db, binary="cp", output="cp.ll")

    for _ in range(3):
        _emit(emitter, kind="type_guessed", severity="imprecision", function="busy")
    _emit(emitter, kind="type_guessed", severity="imprecision", function="quiet")

    store.stop()

    assert worst_offenders(db) == [("busy", 3), ("quiet", 1)]


def test_runs_accumulate_and_are_isolated_by_run_id(tmp_path: Path) -> None:
    db = tmp_path / "fidelity.db"

    emitter1 = EventEmitter()
    store1 = FidelityStore()
    store1.subscribe(emitter1)
    store1.start(db, binary="cp", output="cp.ll")
    _emit(emitter1, kind="type_guessed", severity="imprecision", function="a")
    _emit(emitter1, kind="type_guessed", severity="imprecision", function="b")
    run1 = store1.run_id
    store1.stop()

    emitter2 = EventEmitter()
    store2 = FidelityStore()
    store2.subscribe(emitter2)
    store2.start(db, binary="cp", output="cp.ll")
    _emit(emitter2, kind="value_zero_substituted", severity="corruption", function="a")
    run2 = store2.run_id
    store2.stop()

    # Per-run isolation
    assert fidelity_summary(db, run1) == {"type_guessed": 2}
    assert fidelity_summary(db, run2) == {"value_zero_substituted": 1}
    # Accumulated across both runs
    assert fidelity_summary(db) == {"type_guessed": 2, "value_zero_substituted": 1}
    assert run1 != run2


@pytest.mark.ida
def test_ida_lift_writes_fidelity_db(examples_dir: Path, tmp_path: Path) -> None:
    if not _idalib_available():
        pytest.skip("idalib is not available in this environment")
    binary = examples_dir / "cp"
    if not binary.exists():
        pytest.skip(f"missing example binary: {binary}")

    from idavator.cli import lift_binary_to_llvm

    db = tmp_path / "fidelity.db"
    out = tmp_path / "lifted.ll"
    ok = lift_binary_to_llvm(
        input_binary=str(binary),
        output_llvm_ir=str(out),
        target_mode="host",
        fidelity_db=str(db),
    )
    assert ok, "lift_binary_to_llvm failed"

    assert db.exists()
    # start() always writes exactly one lift_run row per lift.
    conn = sqlite3.connect(db)
    try:
        runs = conn.execute("SELECT COUNT(*) FROM lift_run").fetchone()[0]
    finally:
        conn.close()
    assert runs == 1
    assert isinstance(fidelity_summary(db), dict)
