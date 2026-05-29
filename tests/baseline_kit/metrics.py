"""LLVM IR baseline metrics and comparison for regression tracking."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


BASELINE_VERSION = 1

_TLS_READ_CALL_RE = re.compile(
    r'call\s+\S+\s+@"(__readfs(?:byte|word|dword|qword))"',
    re.MULTILINE,
)
_TLS_WRITE_CALL_RE = re.compile(
    r'call\s+void\s+@"(__writefs(?:byte|word|dword|qword))"',
    re.MULTILINE,
)
_SYSCALL_CALL_RE = re.compile(
    r'call\s+\S+\s+(?:\([^)]*\)\s+)?@"?syscall"?\(',
    re.MULTILINE,
)
_DEFINE_RE = re.compile(r"^define\s+[^@]*@([^(]+)\(", re.MULTILINE)
_METADATA_DEF_RE = re.compile(r"^!\d+\s*=", re.MULTILINE)
_CONCURRENCY_BLOCK_RE = re.compile(
    r"\n!idavator\.concurrency = !\{[^}]*\}(?:\n!\d+ = !\{[^}]*\})*",
    re.MULTILINE,
)


@dataclass(frozen=True)
class IRMetrics:
    lines: int
    bytes: int
    defines: int
    declares: int
    calls: int
    loads: int
    stores: int
    globals: int
    tls_read_calls: int
    tls_write_calls: int
    syscall_calls: int
    tls_read_metadata: int
    futex_metadata: int
    virtual_fs: bool
    concurrency_metadata: bool
    define_names_hash: str
    fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MetricsComparison:
    case: str
    matched: bool
    deltas: dict[str, dict[str, int | bool | str]] = field(default_factory=dict)

    def raise_if_regressed(self) -> None:
        if self.matched:
            return
        lines = [f"baseline regression in case {self.case!r}:"]
        for key, change in sorted(self.deltas.items()):
            lines.append(
                f"  {key}: expected={change['expected']!r} actual={change['actual']!r}"
            )
        raise AssertionError("\n".join(lines))


def compute_metrics(ir_text: str) -> IRMetrics:
    define_names = _DEFINE_RE.findall(ir_text)
    return IRMetrics(
        lines=ir_text.count("\n"),
        bytes=len(ir_text.encode()),
        defines=len(define_names),
        declares=len(re.findall(r"^declare\s+", ir_text, re.MULTILINE)),
        calls=len(re.findall(r"\bcall\b", ir_text)),
        loads=len(re.findall(r"\bload\b", ir_text)),
        stores=len(re.findall(r"\bstore\b", ir_text)),
        globals=len(re.findall(r"^@", ir_text, re.MULTILINE)),
        tls_read_calls=len(_TLS_READ_CALL_RE.findall(ir_text)),
        tls_write_calls=len(_TLS_WRITE_CALL_RE.findall(ir_text)),
        syscall_calls=len(_SYSCALL_CALL_RE.findall(ir_text)),
        tls_read_metadata=ir_text.count('!"tls.read"'),
        futex_metadata=ir_text.count('!"futex.syscall"'),
        virtual_fs='@"virtual_fs"' in ir_text or "@virtual_fs" in ir_text,
        concurrency_metadata="!idavator.concurrency" in ir_text,
        define_names_hash=_hash_sorted(define_names),
        fingerprint=fingerprint_ir(ir_text),
    )


def fingerprint_ir(ir_text: str) -> str:
    normalized = normalize_ir_for_fingerprint(ir_text)
    return hashlib.sha256(normalized.encode()).hexdigest()


def normalize_ir_for_fingerprint(ir_text: str) -> str:
    text = ir_text.replace("\r\n", "\n")
    text = _CONCURRENCY_BLOCK_RE.sub("", text)
    text = _METADATA_DEF_RE.sub("!NUM =", text)
    text = re.sub(r"!\d+", "!NUM", text)
    lines = []
    for line in text.splitlines():
        if line.startswith(";"):
            continue
        lines.append(line)
    return "\n".join(lines).strip() + "\n"


def load_baseline(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if data.get("version") != BASELINE_VERSION:
        raise ValueError(f"unsupported baseline version in {path}")
    return data


def save_baseline(
    path: Path,
    *,
    case: str,
    metrics: IRMetrics,
    source: str,
    passes: tuple[str, ...] = (),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": BASELINE_VERSION,
        "case": case,
        "source": source,
        "passes": list(passes),
        "metrics": metrics.to_dict(),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def compare_metrics(
    expected: IRMetrics, actual: IRMetrics, *, case: str
) -> MetricsComparison:
    deltas: dict[str, dict[str, int | bool | str]] = {}
    for key, expected_value in expected.to_dict().items():
        actual_value = actual.to_dict()[key]
        if expected_value != actual_value:
            deltas[key] = {"expected": expected_value, "actual": actual_value}
    return MetricsComparison(case=case, matched=not deltas, deltas=deltas)


def verify_ir(ir_text: str) -> None:
    import llvmlite.binding as llvm

    module = llvm.parse_assembly(ir_text)
    module.verify()


def _hash_sorted(values: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode()).hexdigest()
