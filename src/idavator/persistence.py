"""Fidelity-ledger persistence.

Owns all sqlite3 storage for the lifter's fidelity-loss events. A FidelityStore
runs a single background thread that owns the sqlite connection and drains a queue
of FidelityEvents; ``emit -> enqueue`` is non-blocking, and ``stop()`` drains and
joins so the database is durable before any read.
"""
from __future__ import annotations

import queue
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from idavator.events import EventEmitter

# Event key used on the emitter for fidelity-loss events.
FIDELITY_EVENT = "fidelity_event"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS lift_run (
    id INTEGER PRIMARY KEY,
    binary TEXT,
    output TEXT,
    created_at TEXT,
    total_functions INTEGER,
    total_events INTEGER
);
CREATE TABLE IF NOT EXISTS fidelity_event (
    id INTEGER PRIMARY KEY,
    run_id INTEGER REFERENCES lift_run(id),
    kind TEXT NOT NULL,
    severity TEXT NOT NULL,
    function TEXT,
    ea INTEGER,
    detail TEXT
);
CREATE INDEX IF NOT EXISTS ix_event_run_kind ON fidelity_event(run_id, kind);
"""

# Severity buckets for fidelity-loss events.
SEVERITY_CORRUPTION = "corruption"
SEVERITY_IMPRECISION = "imprecision"
SEVERITY_HARD_FAIL = "hard_fail"


@dataclass
class FidelityEvent:
    kind: str
    severity: str
    function: str | None = None
    ea: int | None = None
    detail: str | None = None


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)
    conn.commit()


_STOP = object()  # sentinel pushed onto the queue to stop the worker


class FidelityStore:
    """Async sqlite3 sink for fidelity events."""

    def __init__(self) -> None:
        self._queue: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._db_path: Path | None = None
        self._run_id: int | None = None
        self._binary: str | None = None
        self._output: str | None = None

    def subscribe(self, emitter: EventEmitter) -> None:
        emitter.on(FIDELITY_EVENT, self.enqueue)

    def enqueue(self, event: FidelityEvent) -> None:
        self._queue.put(event)

    @property
    def run_id(self) -> int | None:
        return self._run_id

    def start(self, db_path, *, binary: str, output: str) -> None:
        self._db_path = Path(db_path)
        self._binary = binary
        self._output = output

        # Establish the run row synchronously so run_id is available immediately;
        # only event draining happens on the worker thread.
        conn = sqlite3.connect(self._db_path)
        try:
            init_schema(conn)
            cur = conn.execute(
                "INSERT INTO lift_run"
                " (binary, output, created_at, total_functions, total_events)"
                " VALUES (?, ?, ?, 0, 0)",
                (binary, output, datetime.now(timezone.utc).isoformat()),
            )
            self._run_id = cur.lastrowid
            conn.commit()
        finally:
            conn.close()

        self._thread = threading.Thread(
            target=self._run, name="fidelity-store", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        conn = sqlite3.connect(self._db_path)
        try:
            total = 0
            functions: set[str] = set()
            while True:
                item = self._queue.get()
                if item is _STOP:
                    break
                ev: FidelityEvent = item
                conn.execute(
                    "INSERT INTO fidelity_event"
                    " (run_id, kind, severity, function, ea, detail)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    (self._run_id, ev.kind, ev.severity, ev.function, ev.ea, ev.detail),
                )
                total += 1
                if ev.function:
                    functions.add(ev.function)

            conn.execute(
                "UPDATE lift_run SET total_functions = ?, total_events = ? WHERE id = ?",
                (len(functions), total, self._run_id),
            )
            conn.commit()
        finally:
            conn.close()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._queue.put(_STOP)
        self._thread.join()
        self._thread = None


def _connect(db_path) -> sqlite3.Connection:
    return sqlite3.connect(Path(db_path))


def fidelity_summary(db_path, run_id: int | None = None) -> dict[str, int]:
    conn = _connect(db_path)
    try:
        if run_id is None:
            rows = conn.execute(
                "SELECT kind, COUNT(*) FROM fidelity_event GROUP BY kind"
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT kind, COUNT(*) FROM fidelity_event WHERE run_id = ? GROUP BY kind",
                (run_id,),
            ).fetchall()
        return {kind: count for kind, count in rows}
    finally:
        conn.close()


def worst_offenders(db_path, run_id: int | None = None, limit: int = 20):
    conn = _connect(db_path)
    try:
        params: tuple = ()
        sql = (
            "SELECT function, COUNT(*) AS n FROM fidelity_event"
            " WHERE function IS NOT NULL"
        )
        if run_id is not None:
            sql += " AND run_id = ?"
            params = (run_id,)
        sql += " GROUP BY function ORDER BY n DESC LIMIT ?"
        params = (*params, limit)
        rows = conn.execute(sql, params).fetchall()
        return [(fn, n) for fn, n in rows]
    finally:
        conn.close()
