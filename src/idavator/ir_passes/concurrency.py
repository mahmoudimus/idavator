import re
from dataclasses import dataclass

from .pipeline import IRPassContext


_TLS_READ_RE = re.compile(
    r'^\s*(?:%[^=]+ = )?call\s+\S+\s+@"(__readfs(?:byte|word|dword|qword))"\(i32\s+([^)]+)\)',
    re.MULTILINE,
)
_TLS_WRITE_RE = re.compile(
    r'^\s*call\s+void\s+@"(__writefs(?:byte|word|dword|qword))"\(i32\s+([^,]+),\s+[^)]+\)',
    re.MULTILINE,
)
_SYSCALL_RE = re.compile(
    r'^\s*(?:%[^=]+ = )?call\s+\S+\s+(?:\([^)]*\)\s+)?@"?syscall"?\(([^)]*)\)',
    re.MULTILINE,
)
_EXISTING_METADATA_RE = re.compile(r"^!(\d+)\s*=", re.MULTILINE)

_FUTEX_SYSCALL_X86_64 = 202
_FUTEX_OPS = {
    0: "FUTEX_WAIT",
    1: "FUTEX_WAKE",
    2: "FUTEX_FD",
    3: "FUTEX_REQUEUE",
    4: "FUTEX_CMP_REQUEUE",
    5: "FUTEX_WAKE_OP",
    6: "FUTEX_LOCK_PI",
    7: "FUTEX_UNLOCK_PI",
    8: "FUTEX_TRYLOCK_PI",
    9: "FUTEX_WAIT_BITSET",
    10: "FUTEX_WAKE_BITSET",
    11: "FUTEX_WAIT_REQUEUE_PI",
    12: "FUTEX_CMP_REQUEUE_PI",
}
_FUTEX_PRIVATE_FLAG = 128
_FUTEX_CLOCK_REALTIME = 256
_FUTEX_CMD_MASK = ~(_FUTEX_PRIVATE_FLAG | _FUTEX_CLOCK_REALTIME)


@dataclass(frozen=True)
class ConcurrencyAnnotation:
    kind: str
    fields: tuple[tuple[str, str], ...]


class ConcurrencyAnnotationPass:
    name = "concurrency"

    def run(self, ir_text: str, ctx: IRPassContext) -> str:
        _ = ctx
        return annotate_concurrency(ir_text)


def annotate_concurrency(ir_text: str) -> str:
    annotations = list(_find_concurrency_annotations(ir_text))
    if not annotations:
        return ir_text

    first_metadata_id = _next_metadata_id(ir_text)
    metadata_lines = ["", "!idavator.concurrency = !{"]
    metadata_lines.extend(
        f"  !{first_metadata_id + idx}{',' if idx + 1 < len(annotations) else ''}"
        for idx in range(len(annotations))
    )
    metadata_lines.append("}")
    metadata_lines.extend(
        _format_metadata(first_metadata_id + idx, annotation)
        for idx, annotation in enumerate(annotations)
    )

    return f"{ir_text.rstrip()}\n" + "\n".join(metadata_lines) + "\n"


def _find_concurrency_annotations(ir_text: str):
    if '@"virtual_fs"' in ir_text or "@virtual_fs" in ir_text:
        yield ConcurrencyAnnotation(
            "tls.storage",
            (("symbol", "virtual_fs"), ("semantics", "virtual thread-local storage")),
        )
    yield from _find_tls_annotations(ir_text)
    yield from _find_syscall_annotations(ir_text)


def _find_tls_annotations(ir_text: str):
    for match in _TLS_READ_RE.finditer(ir_text):
        helper, offset = match.groups()
        yield ConcurrencyAnnotation(
            "tls.read",
            (
                ("helper", helper),
                ("offset", offset.strip()),
                ("line", str(_line_no(ir_text, match.start()))),
            ),
        )

    for match in _TLS_WRITE_RE.finditer(ir_text):
        helper, offset = match.groups()
        yield ConcurrencyAnnotation(
            "tls.write",
            (
                ("helper", helper),
                ("offset", offset.strip()),
                ("line", str(_line_no(ir_text, match.start()))),
            ),
        )


def _find_syscall_annotations(ir_text: str):
    for match in _SYSCALL_RE.finditer(ir_text):
        args = _split_call_args(match.group(1))
        if not args:
            continue
        syscall_no = _parse_int_arg(args[0])
        line = str(_line_no(ir_text, match.start()))
        if syscall_no is None:
            yield ConcurrencyAnnotation(
                "syscall",
                (("number", args[0].strip()), ("line", line)),
            )
            continue

        fields = [("number", str(syscall_no)), ("line", line)]
        if syscall_no == _FUTEX_SYSCALL_X86_64:
            yield _format_futex_annotation(args, fields)
        else:
            yield ConcurrencyAnnotation("syscall", tuple(fields))


def _format_futex_annotation(
    args: list[str], fields: list[tuple[str, str]]
) -> ConcurrencyAnnotation:
    fields[0] = ("number", "202")
    fields.append(("name", "futex"))
    if len(args) > 1:
        fields.append(("uaddr", args[1].strip()))
    if len(args) > 2:
        fields.extend(_format_futex_op(args[2]))
    return ConcurrencyAnnotation("futex.syscall", tuple(fields))


def _format_futex_op(op_arg: str) -> list[tuple[str, str]]:
    op_value = _parse_int_arg(op_arg)
    if op_value is None:
        return [("op", op_arg.strip())]

    base_op = op_value & _FUTEX_CMD_MASK
    fields = [("op", str(op_value)), ("op_name", _FUTEX_OPS.get(base_op, "UNKNOWN"))]
    if op_value & _FUTEX_PRIVATE_FLAG:
        fields.append(("private", "true"))
    if op_value & _FUTEX_CLOCK_REALTIME:
        fields.append(("clock_realtime", "true"))
    return fields


def _split_call_args(args_text: str) -> list[str]:
    args: list[str] = []
    depth = 0
    start = 0
    for idx, char in enumerate(args_text):
        if char in "(<[":
            depth += 1
        elif char in ")>]":
            depth = max(0, depth - 1)
        elif char == "," and depth == 0:
            args.append(args_text[start:idx].strip())
            start = idx + 1
    tail = args_text[start:].strip()
    if tail:
        args.append(tail)
    return args


def _parse_int_arg(arg: str) -> int | None:
    match = re.search(r"\bi(?:8|16|32|64)\s+(-?(?:0x[0-9A-Fa-f]+|\d+))\b", arg)
    if match is None:
        return None
    return int(match.group(1), 0)


def _format_metadata(metadata_id: int, annotation: ConcurrencyAnnotation) -> str:
    values = [f'!"{_escape(annotation.kind)}"']
    for key, value in annotation.fields:
        values.append(f'!"{_escape(key)}"')
        if re.fullmatch(r"-?\d+", value):
            values.append(f"i64 {value}")
        else:
            values.append(f'!"{_escape(value)}"')
    return f"!{metadata_id} = !{{{', '.join(values)}}}"


def _next_metadata_id(ir_text: str) -> int:
    existing = [int(match.group(1)) for match in _EXISTING_METADATA_RE.finditer(ir_text)]
    return max(existing, default=-1) + 1


def _line_no(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _escape(value: str) -> str:
    return value.replace("\\", "\\5C").replace('"', "\\22")
