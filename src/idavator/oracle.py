"""Semantic-equivalence oracle for the LLVM->microcode drop, via libclang.

Hex-Rays renders semantically-equivalent code differently: it renames locals
(``a0``/``v3``), changes the constant base (``100`` vs ``0x64``), folds
``x << 2`` into ``4 * x``, adds cosmetic casts (``(const char *)a0``), and inverts
a test while swapping the arms of an ``if``. A textual diff therefore produces
false mismatches.

This oracle parses both C functions with **IDA's libclang** (discovered through
the vendored ``clang_loader``) and compares a CANONICAL form of each function body --
the AST after the normalizations that undo those cosmetic choices:

* locals/params are alpha-renamed to positional ids (``#0``, ``#1`` ...);
* integer literals collapse to their value (``100`` == ``0x64``);
* ``a << k`` / ``a >> k`` (k constant) fold to ``a * 2**k`` / ``a / 2**k``;
* commutative operators (``+ * & | ^ == != && ||``) sort their operands;
* comparisons canonicalize direction (``a > b`` -> ``b < a``);
* an ``if`` is stored as the smaller of ``(cond, then, else)`` and
  ``(!cond, else, then)`` so an inverted test with swapped arms matches;
* casts / parentheses / implicit-conversion nodes are transparent.

It is a structural-semantic oracle robust to Hex-Rays' COSMETIC rendering, not a
full equivalence prover: it does not inline temporaries or reassociate across
statements, so a body Hex-Rays restructured into extra locals may still differ
(the :func:`fidelity_ledger` reports where). See the memory note
``idavator_libclang_ast_oracle`` for the plumbing + scope rationale.

Inputs are COMPLETE C function definitions (``int f(int x){ ... }``); the drop's
``str(cfunc)`` is one already, and the Hex-Rays pseudotypes are typedef'd by a
small prelude so libclang can parse them.
"""
from __future__ import annotations

import re

try:
    from idavator._clang_loader import load_clang_index

    _CLANG_IMPORT_ERROR: Exception | None = None
except Exception as _e:  # noqa: BLE001 - optional dependency (IDA's libclang)
    load_clang_index = None  # type: ignore[assignment]
    _CLANG_IMPORT_ERROR = _e

# Hex-Rays pseudotypes so a raw ``str(cfunc)`` parses standalone.
_PRELUDE = (
    "typedef unsigned char _BYTE;typedef unsigned short _WORD;"
    "typedef unsigned int _DWORD;typedef unsigned long long _QWORD;"
    "typedef long long __int64;typedef int __int32;typedef short __int16;"
    "typedef signed char __int8;\n"
)
_PARSE_ARGS = ["-x", "c", "-fms-extensions", "-w"]

# Transparent wrappers: descend to the meaningful child.
_TRANSPARENT = {
    "UNEXPOSED_EXPR", "PAREN_EXPR", "CSTYLE_CAST_EXPR", "FIRST_EXPR",
    "CXX_FUNCTIONAL_CAST_EXPR",
}
_COMMUTATIVE = {"+", "*", "&", "|", "^", "==", "!=", "&&", "||"}

_index = None
_index_loaded = False


class OracleParseError(RuntimeError):
    """The active libclang could not fully parse a function body.

    Raised when canonicalization yields an EMPTY body for a source that clearly
    has statements -- i.e. the parser silently dropped the body on a syntax it
    could not handle. This happens with the pip ``libclang`` *fallback* (an older
    clang used on Linux when IDA's own libclang cannot parse a TU): it rejects
    some Hex-Rays pseudocode constructs that IDA's clang-21 accepts (function-
    pointer casts like ``(*((T (__fastcall **)(...))p + 7))(...)``). The compare
    is then INCONCLUSIVE, not a divergence; callers that must avoid a false
    negative (e.g. the drop decline gate) should treat it as "cannot verify"."""


def _ensure_index():
    """Load IDA's libclang index once; cache the (possibly None) result. The
    vendored loader always imports, so the real availability signal is whether
    the native libclang library could actually be discovered and loaded."""
    global _index, _index_loaded
    if not _index_loaded:
        _index_loaded = True
        if load_clang_index is not None:
            try:
                _index, _, _ = load_clang_index()
            except Exception:  # noqa: BLE001 - any load failure => unavailable
                _index = None
    return _index


def clang_available() -> bool:
    """True only if IDA's libclang actually LOADS (oracle usable). Importing the
    vendored loader is not enough -- without the native libclang library the
    oracle tests must skip (e.g. CI without IDA)."""
    return _ensure_index() is not None


def _get_index():
    idx = _ensure_index()
    if idx is None:
        raise RuntimeError(
            "libclang unavailable: IDA's native libclang could not be loaded "
            f"(loader import error: {_CLANG_IMPORT_ERROR})")
    return idx


def _litval(cursor) -> int:
    tokens = [t.spelling for t in cursor.get_tokens()]
    if not tokens:
        return 0
    return int(re.sub(r"[uUlL]+$", "", tokens[0]), 0)


def _children(cursor):
    return list(cursor.get_children())


def _binop_spelling(cursor) -> str:
    """Operator of a BINARY_OPERATOR cursor, version-independently.

    ``cursor.spelling`` only carries the operator on libclang >= 19; older
    libclang (e.g. the pip ``libclang`` fallback used on Linux) returns ``""``.
    Recover it from the token stream instead: the operator is the first token at
    or after the end of the left operand's extent. This agrees with the
    ``spelling`` value on clang 21 (IDA's own libclang) and works on clang 18."""
    spelled = cursor.spelling
    if spelled:
        return spelled
    kids = _children(cursor)
    if len(kids) == 2:
        left_end = kids[0].extent.end.offset
        for tok in cursor.get_tokens():
            if tok.extent.start.offset >= left_end:
                return tok.spelling
    return spelled


class _Canon:
    """Canonicalize a function body to a comparable nested tuple."""

    def __init__(self):
        self._names: dict[str, int] = {}

    def alpha(self, name: str) -> int:
        return self._names.setdefault(name, len(self._names))

    def expr(self, c):
        kind = c.kind.name
        if kind in _TRANSPARENT:
            kids = _children(c)
            return self.expr(kids[0]) if kids else ("nil",)
        if kind == "INTEGER_LITERAL":
            return ("int", _litval(c))
        if kind in ("DECL_REF_EXPR", "MEMBER_REF_EXPR"):
            return ("var", self.alpha(c.spelling))
        if kind == "UNARY_OPERATOR":
            toks = [t.spelling for t in c.get_tokens()]
            op = toks[0] if toks else "?"
            kids = _children(c)
            inner = self.expr(kids[0]) if kids else ("nil",)
            if op == "-" and inner[:1] == ("int",):
                return ("int", -inner[1])
            return ("un", op, inner)
        if kind == "BINARY_OPERATOR":
            kids = _children(c)
            return self._binop(_binop_spelling(c), self.expr(kids[0]),
                               self.expr(kids[1]))
        if kind == "CALL_EXPR":
            callee = c.spelling or "?"
            kids = _children(c)
            # first child is the (unexposed) callee ref; remainder are args.
            args = tuple(self.expr(k) for k in kids[1:])
            return ("call", callee, args)
        # Fallback: opaque node with its children.
        return (kind, tuple(self.expr(k) for k in _children(c)))

    def _binop(self, op, l, r):
        # shift-by-constant folds to the equivalent mul/div.
        if op == "<<" and r[:1] == ("int",):
            return self._binop("*", l, ("int", 1 << r[1]))
        if op == ">>" and r[:1] == ("int",):
            return ("bin", "/", l, ("int", 1 << r[1]))
        if op in (">", ">="):
            return ("bin", {">": "<", ">=": "<="}[op], r, l)  # swap operands
        if op in _COMMUTATIVE:
            a, b = sorted((l, r), key=repr)
            return ("bin", op, a, b)
        return ("bin", op, l, r)

    def stmt(self, c):
        kind = c.kind.name
        if kind == "COMPOUND_STMT":
            return ("block", tuple(self.stmt(k) for k in _children(c)))
        if kind == "DECL_STMT":
            return ("decls", tuple(self.stmt(k) for k in _children(c)))
        if kind == "VAR_DECL":
            kids = _children(c)
            init = self.expr(kids[-1]) if kids else ("nil",)
            return ("assign", self.alpha(c.spelling), init)
        if kind == "RETURN_STMT":
            kids = _children(c)
            return ("return", self.expr(kids[0]) if kids else None)
        if kind == "IF_STMT":
            kids = _children(c)
            cond = self.expr(kids[0])
            then = self.stmt(kids[1])
            els = self.stmt(kids[2]) if len(kids) > 2 else None
            a = ("if", cond, then, els)
            if els is not None:
                b = ("if", _negate(cond), els, then)
                return min((a, b), key=repr)
            return a
        if kind in ("FOR_STMT", "WHILE_STMT", "DO_STMT"):
            # Loop kind is normalized away (Hex-Rays freely picks for/while/do).
            return ("loop", tuple(self.stmt(k) if k.kind.name.endswith("STMT")
                                  else self.expr(k) for k in _children(c)))
        if kind in ("BREAK_STMT", "CONTINUE_STMT"):
            return (kind.lower(),)
        # Expression-statement / other: canonicalize as an expression list.
        return ("stmt", tuple(self.expr(k) for k in _children(c)))


def _negate(cond):
    if cond[:1] == ("bin",):
        op = cond[1]
        flip = {"<": ("<=", True), "<=": ("<", True),
                "==": ("!=", False), "!=": ("==", False)}
        if op in flip:
            new_op, swap = flip[op]
            a, b = (cond[3], cond[2]) if swap else (cond[2], cond[3])
            return ("bin", new_op, a, b)
    return ("un", "!", cond)


def _function_cursor(tu):
    fn = None
    for cur in tu.cursor.get_children():
        if cur.kind.name == "FUNCTION_DECL" and cur.is_definition():
            fn = cur
    return fn


_HAS_STMT = re.compile(r"[;{]")


def canonical_form(c_function: str):
    """Canonical comparable form of a complete C function definition's body.

    Raises :class:`OracleParseError` if the body canonicalizes to EMPTY while the
    source plainly has statements -- the signal that the active (fallback)
    libclang silently dropped the body on an unsupported syntax (see that
    exception's docstring). A genuinely empty body (``int f(){}``) does not have
    statement punctuation between its braces, so it is not misflagged."""
    tu = _get_index().parse(
        "o.c", args=_PARSE_ARGS,
        unsaved_files=[("o.c", _PRELUDE + c_function)])
    fn = _function_cursor(tu)
    if fn is None:
        raise ValueError("no function definition found in oracle input")
    body = next((k for k in fn.get_children()
                 if k.kind.name == "COMPOUND_STMT"), None)
    if body is None:
        raise ValueError("function has no body")
    canon = _Canon().stmt(body)
    if canon == ("block", ()):
        # Empty canonical body. Distinguish a truly-empty body from a parse that
        # silently dropped statements: look at the source between the OUTERMOST
        # braces for statement punctuation.
        inner = c_function[c_function.find("{") + 1: c_function.rfind("}")]
        stripped = re.sub(r"//[^\n]*|/\*.*?\*/", "", inner, flags=re.DOTALL)
        if _HAS_STMT.search(stripped):
            raise OracleParseError(
                "libclang produced an empty body for a non-empty function "
                "(unsupported pseudocode syntax for this clang version)")
    return canon


def _norm_text(c: str) -> str:
    """Comment-free, whitespace-collapsed body text (the fast-path key)."""
    c = re.sub(r"//[^\n]*|/\*.*?\*/", "", c, flags=re.DOTALL)
    return re.sub(r"\s+", " ", c).strip()


def matches(expected_c: str, actual_c: str) -> bool:
    """True iff two C function definitions share a canonical body form. A verbatim
    drop (common -- the drop often reproduces the original exactly) short-circuits
    BEFORE AST canonicalization, which is incomplete for goto/label/switch bodies."""
    if _norm_text(expected_c) == _norm_text(actual_c):
        return True
    return canonical_form(expected_c) == canonical_form(actual_c)


def fidelity_ledger(expected_c: str, actual_c: str) -> dict:
    """Empty dict iff faithful; else the first divergent canonical subtrees."""
    if _norm_text(expected_c) == _norm_text(actual_c):
        return {}
    exp, act = canonical_form(expected_c), canonical_form(actual_c)
    if exp == act:
        return {}
    return {"expected": _first_diff(exp, act)[0],
            "actual": _first_diff(exp, act)[1]}


def _first_diff(a, b):
    """Return the (a, b) of the smallest differing subtrees."""
    if isinstance(a, tuple) and isinstance(b, tuple) and len(a) == len(b):
        for x, y in zip(a, b):
            if x != y:
                return _first_diff(x, y)
    return a, b
