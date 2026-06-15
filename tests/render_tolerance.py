"""Render-base tolerance for drop-vs-native assertions.

Hex-Rays renders the SAME integer constant differently across IDA builds: IDA 9.3
Linux (idalib CI) prints small constants in DECIMAL (``27``, ``95``, ``-1``) while
the dev macOS IDA prints HEX (``0x1B``, ``0x5F``, ``0xFFFFFFFF``). The dropped body
is byte-faithful either way -- only the literal BASE on IDA's own pseudocode
differs. A raw-text needle pinned to one base therefore false-fails on the other
build.

These helpers compare integer literals by VALUE, not by rendering, so a test still
CATCHES a real divergence (a wrong/absent constant, a missing call) while tolerating
the benign base axis. They normalize ONLY the integer-literal rendering -- nothing
else about the body text is relaxed.
"""
from __future__ import annotations

import re
import zlib


def int_renderings(value: int, *, width: int = 32) -> list[str]:
    """Every way Hex-Rays might render the integer ``value`` at ``width`` bits.

    Covers signed decimal (``-1``), unsigned decimal (``27``), and uppercase hex
    (``0x1B``) for BOTH the signed value and its unsigned 2's-complement at the
    given width (so ``-1`` at 32-bit also matches ``0xFFFFFFFF`` and at 64-bit
    ``0xFFFFFFFFFFFFFFFF``). Suffix-free; callers anchor with word boundaries /
    optional ``[uUlL]*`` so a suffixed literal (``0x48u``, ``24LL``) still matches.
    """
    mask = (1 << width) - 1
    unsigned = value & mask
    # The signed interpretation of the unsigned bit pattern (so 0xFFFFFFFF -> -1).
    signed = unsigned - (1 << width) if unsigned >> (width - 1) else unsigned
    out: set[str] = set()
    for v in (value, unsigned, signed):
        out.add(str(v))
        if v >= 0:
            out.add(f"0x{v:X}")
        else:
            # negative hex is not how Hex-Rays prints; the unsigned form covers it.
            pass
    return sorted(out)


def _alt(value: int, width: int) -> str:
    """Regex alternation matching any base-rendering of ``value`` as a whole token,
    tolerating an optional integer-suffix (``u``/``L``/``LL``)."""
    bodies = sorted({re.escape(r) for r in int_renderings(value, width=width)},
                    key=len, reverse=True)
    return r"(?:" + "|".join(bodies) + r")[uUlL]*"


def contains_int(text: str, value: int, *, width: int = 32) -> bool:
    """True iff ``text`` contains ``value`` rendered in ANY base as a standalone
    token (not a substring of a larger number/identifier)."""
    pat = re.compile(r"(?<![0-9A-Za-z_])" + _alt(value, width) + r"(?![0-9A-Za-z_])")
    return pat.search(text) is not None


def count_int(text: str, value: int, *, width: int = 32) -> int:
    """Number of standalone-token occurrences of ``value`` in ANY base."""
    pat = re.compile(r"(?<![0-9A-Za-z_])" + _alt(value, width) + r"(?![0-9A-Za-z_])")
    return len(pat.findall(text))


def int_pattern(value: int, *, width: int = 32) -> str:
    """A regex fragment (un-anchored) matching ``value`` in any base, for splicing
    into a larger pattern where a constant sits in a specific syntactic position."""
    return _alt(value, width)


def search_with_ints(template: str, text: str, ints: dict[str, int],
                     *, width: int = 32) -> "re.Match[str] | None":
    """Search ``text`` for ``template`` after replacing each ``{name}`` placeholder
    with a base-tolerant alternation for ``ints[name]``.

    The template is a raw regex EXCEPT the ``{name}`` placeholders (so escape any
    regex metacharacters in the literal parts yourself, as with a normal regex).
    """
    pat = template.format(**{k: int_pattern(v, width=width)
                             for k, v in ints.items()})
    return re.search(pat, text)


# ---------------------------------------------------------------------------
# Structural normalization for drop-vs-native pseudocode comparison.
#
# The amd64 idalib build is a RICHER/STRICTER native oracle than arm64: its
# native decompile of cp functions carries DWARF param names + types AND a
# ``__readfsqword(0x28u)`` stack-canary that an LLVM-IR-built drop (cp.ll is
# weakly typed) fundamentally cannot reproduce. So the SAME faithful drop that
# is byte-identical to arm64-native diverges from amd64-native on a fixed set of
# BENIGN, type-driven rendering axes:
#
#   * declared types / ``const`` / ``__cdecl`` vs ``__fastcall`` / param names
#     (DWARF on amd64 native, weak ``_DWORD *``/``aN`` on the IR drop);
#   * cosmetic casts Hex-Rays inserts to satisfy a type (``(const char *)a0``);
#   * the ``__readfsqword(...)`` canary read + the BYREF local ``= 0;`` init it
#     guards (stack protector present on amd64 native, absent on the drop);
#   * leading-underscore count on libc symbols (``__assert_fail`` vs
#     ``_assert_fail``, ``__errno_location`` vs ``_errno_location``);
#   * width-keyword return materialization the weak ``int`` return forces
#     (``LOBYTE(result) = f(...); return result`` for native's ``return f(...)``)
#     and a single-use temp COPY Hex-Rays keeps under one type but folds under
#     another (``p = v6; ... p->x`` vs ``v8 = ...; ... v8->x``).
#
# ``structural_norm`` collapses EXACTLY these axes and nothing else: it strips
# types/casts/canary/byref-init, equalizes leading underscores, undoes the
# width-materialized return and single-use temp-copy, then alpha-renames every
# identifier to a positional id. What survives is the STATEMENT/CALL/CONSTANT/
# CONTROL-FLOW skeleton -- so a real divergence (a missing/extra/reordered
# statement, a wrong callee or constant, a struct-field-vs-raw-offset access, a
# restructured loop or nested-vs-combined branch) STILL fails. It is deliberately
# NOT a full equivalence prover: it does not reassociate arithmetic or inline
# multi-use temporaries, so a body Hex-Rays genuinely restructured stays
# divergent (and such cases are left as documented xfails, never force-passed).
# ---------------------------------------------------------------------------

# Hex-Rays width-extraction keywords: ``LOBYTE(x)``/``LODWORD(x)``/... select a
# sub-register of ``x``. Under a weak (``int``) type they wrap a value that a
# correctly-typed (``bool``/``size_t``) render would not. As a *target* of an
# assignment whose value is then returned, they are pure type artifacts.
_WIDTH_KW = (
    "LOBYTE", "LOWORD", "LODWORD", "HIBYTE", "HIWORD", "HIDWORD",
    "BYTE1", "BYTE2", "BYTE3", "BYTE4", "BYTE5", "BYTE6", "BYTE7",
    "WORD1", "WORD2", "SLOBYTE", "SLOWORD", "SLODWORD",
)
_WIDTH_WRAP = re.compile(
    r"\b(?:" + "|".join(_WIDTH_KW) + r")\(\s*([A-Za-z_]\w*)\s*\)")

# A cast: a parenthesized TYPE immediately preceding a value. Conservative -- the
# inner text must look like a C type (a known type keyword / ``_DWORD`` family /
# a ``struct``/``enum`` tag / a bare identifier) optionally with ``const``,
# ``unsigned``/``signed``, ``struct``/``enum``, and trailing ``*``s -- and be
# followed by something a cast applies to (ident, ``(``, ``*``, ``&``). This
# never matches a grouping paren like ``(a + b)`` (has an operator) or a call.
_TYPE_WORD = (
    r"(?:const\s+|volatile\s+|unsigned\s+|signed\s+|struct\s+|enum\s+)*"
    r"(?:_BYTE|_WORD|_DWORD|_QWORD|__int8|__int16|__int32|__int64|"
    r"char|short|int|long|bool|void|size_t|ssize_t|off_t|__off_t|"
    r"float|double|[A-Za-z_]\w*)"
    r"(?:\s*\*)*"
)
_CAST = re.compile(r"\(\s*" + _TYPE_WORD + r"\s*\)\s*(?=[A-Za-z_(&*])")

_LEADING_US = re.compile(r"\b_{2,}(?=[A-Za-z])")
_IDENT = re.compile(r"\b[A-Za-z_]\w*\b")

# Identifiers that are NOT renamable: language/type keywords and the structural
# vocabulary. Everything else (locals, params, GLOBALS, callee names) collapses
# to a positional id -- callee/global names ARE compared, but via the alpha map
# their RELATIVE identity is what matters (a wrong callee maps to a different id
# than the matching call on the other side, so it still diverges).
_KEYWORDS = frozenset({
    "if", "else", "for", "while", "do", "return", "break", "continue",
    "goto", "switch", "case", "default", "sizeof",
})

# A Hex-Rays SYNTHETIC local name (never a global): ``vN`` / ``aN`` (param) /
# ``result`` / a ``krNN_M`` carry register / ``sN`` saved register. These are
# safe to copy-propagate / collapse a materialized return on even when the decl
# block is COLLAPSED (the ``// [COLLAPSED LOCAL DECLARATIONS...]`` banner), where
# the declared-local set is otherwise unavailable. A named stack local
# (``first_dir_created...``) is recovered from the decls when expanded; when
# collapsed it cannot be the target of a fold here, which is fine -- the fold is
# only NEEDED for the synthetic ``result`` return materialization.
_TEMP_NAME = re.compile(r"(?:v\d+|a\d+|result|kr[0-9A-Fa-f]+_\d+|s\d+)\Z")

# The COPY-PROPAGATION source pattern: a Hex-Rays INTERMEDIATE register temp ONLY
# -- ``vN`` / ``krNN_M`` / ``sN`` / ``result``. Deliberately EXCLUDES ``aN``: an
# incoming PARAM copied/stored into another name (``some_global = a0``) is a real
# store whose drop would lose an observable side effect, so a param RHS must not
# trigger propagation (cf. the missing-statement / global-store unit tests).
_REG_TEMP = re.compile(r"(?:v\d+|result|kr[0-9A-Fa-f]+_\d+|s\d+)\Z")


def _is_local(name: str, locals_: "set[str]") -> bool:
    """A declared local/param OR a Hex-Rays synthetic temp (safe vs a global)."""
    return name in locals_ or bool(_TEMP_NAME.match(name))


def _strip_decls_and_signature(text: str) -> "tuple[list[str], set[str]]":
    """Drop the function signature (return type + params, possibly multi-line)
    and the local-declaration block; return ``(body_lines, local_names)`` where
    ``local_names`` is the set of declared LOCALS + PARAMS.

    Hex-Rays local declarations are exactly the lines carrying a trailing
    ``// <reg-or-frame-offset>`` comment in the decl block; the body proper has
    no such comments. The signature ends at the first line whose stripped text is
    ``{`` (Hex-Rays always puts the opening brace on its own line). ``local_names``
    lets copy-propagation distinguish a removable LOCAL temp copy (``p = v6``)
    from an observable GLOBAL store (``program_name = v3``), which it must never
    drop."""
    lines = text.splitlines()
    # Find the body open brace (first line that is exactly "{").
    start = next((i for i, ln in enumerate(lines) if ln.strip() == "{"), None)
    if start is None:
        return [ln.rstrip() for ln in lines], set()
    # Params: identifiers inside the signature's parentheses (the last identifier
    # of each comma-separated declarator, e.g. ``const char *src`` -> ``src``).
    sig = " ".join(lines[:start])
    locals_: set[str] = set()
    pstart, pend = sig.find("("), sig.rfind(")")
    if 0 <= pstart < pend:
        for decl in sig[pstart + 1:pend].split(","):
            ids = _IDENT.findall(decl)
            if ids:
                locals_.add(ids[-1])
    body = lines[start + 1:]
    # Drop the trailing closing brace.
    while body and body[-1].strip() == "}":
        body = body[:-1]
    out = []
    for ln in body:
        # The collapsed-declarations banner Hex-Rays emits when the local decls are
        # folded for display (``// [COLLAPSED LOCAL DECLARATIONS. ...]``) -- one
        # build may collapse while the other expands; both reduce to NO decls.
        if "COLLAPSED LOCAL DECLARATIONS" in ln:
            continue
        # A declaration line: ``<type> <name>; // <reg-or-frame comment>``. The
        # CODE part (before the ``//``) ends with ``;`` and has no call/assignment
        # -- so a plain ``x = 0;`` or ``f();`` statement (no decl ``//`` comment)
        # is never mistaken for a declaration.
        code = ln.split("//", 1)[0].rstrip() if "//" in ln else ""
        if (code.endswith(";") and "=" not in code and "(" not in code
                and "return" not in code):
            ids = _IDENT.findall(code)
            if ids:
                locals_.add(ids[-1])  # the declarator name (last id on the line)
            continue
        out.append(ln)
    return out, locals_


def _strip_canary_and_byref_init(lines: list[str]) -> list[str]:
    """Drop the ``__readfsqword(...)`` canary read and a BYREF local's ``= 0;``
    init that the stack protector guards.

    Only fires when a canary is actually present (a stack-protected function);
    then the lone ``<ident> = 0;`` statements (the BYREF locals the protector
    zero-inits, which the IR drop does not emit) are removed too. A function with
    no canary keeps all its ``= 0;`` statements, so a genuine ``x = 0;`` in
    unprotected code is never dropped."""
    has_canary = any("__readfsqword" in ln for ln in lines)
    out = []
    for ln in lines:
        if "__readfsqword" in ln:
            continue
        if has_canary and re.fullmatch(r"\s*[A-Za-z_]\w*\s*=\s*0;\s*", ln):
            continue
        out.append(ln)
    return out


def _fold_materialized_return(text: str, locals_: "set[str]") -> str:
    """Collapse ``X = EXPR; return X;`` (and the width-wrapped form, already
    unwrapped to ``X = EXPR;`` by :func:`_normalize_body`) to ``return EXPR;``,
    for a LOCAL ``X`` only.

    Hex-Rays materializes a return value into a temp when the function's declared
    return type forces it (a weak ``int`` return of a ``bool``/``size_t`` call),
    where a correctly-typed render returns the expression directly. Narrow: the
    assigned and returned name must be IDENTICAL, adjacent, and a declared LOCAL
    (so a ``global = EXPR; return global`` -- which keeps an observable store --
    is never collapsed)."""
    def _sub(m: "re.Match[str]") -> str:
        return (f"return {m.group(2)};" if _is_local(m.group(1), locals_)
                else m.group(0))

    return re.sub(
        r"\b([A-Za-z_]\w*)\s*=\s*([^;]+);\s*return\s+\1\s*;", _sub, text)


def _inline_single_use_copy(text: str, locals_: "set[str]") -> str:
    """Copy-propagate a temp COPY ``A = B;`` where ``B`` is a Hex-Rays REGISTER
    TEMP (``vN``/``krN``/``sN``/``result``/``aN``), into the statements that
    follow, then drop the copy.

    Hex-Rays keeps an extra copy of a value in a distinct local under one type
    assignment but folds it away under another (``v6 = o; p = v6; ... p->x ...``
    on native vs ``v8 = ...; ... v8->x ...`` on the drop). The safe, build-robust
    signal is the RHS: a copy whose source is a register-temp is an intermediate
    value Hex-Rays may or may not keep; propagating it is value-preserving. Narrow:

    * the RHS ``B`` must be a register-temp -- so a real store of a PARAM/value
      into a global (``top_level_src_name = src_name``; the missing-statement
      unit test's ``g = a0``) has a non-temp RHS and is NEVER dropped (the
      observable store still diverges if removed). This RHS test needs no decl
      info, so it works whether the decls are expanded OR collapsed (where the
      LHS -- a named local like ``p`` -- is otherwise unknown);
    * propagation is GUARDED -- it fires only when neither ``A`` NOR ``B`` is
      reassigned (``=``/``++``/``--``/``&``/compound ``op=``) after the copy, so
      every later read of ``A`` keeps the same value.

    A copy failing any guard is left untouched."""
    body = text.replace("\n", " ")
    # Process statement-by-statement so "after the copy" is well-defined.
    stmts = [s for s in body.split(";")]
    out: list[str] = []
    i = 0
    while i < len(stmts):
        s = stmts[i]
        m = re.fullmatch(r"\s*([A-Za-z_]\w*)\s*=\s*([A-Za-z_]\w*)\s*", s)
        if (m and m.group(1) != m.group(2)
                and _REG_TEMP.match(m.group(2))):
            lhs, rhs = m.group(1), m.group(2)
            tail = ";".join(stmts[i + 1:])

            def _reassigned(name: str) -> bool:
                w = re.escape(name)
                return bool(
                    re.search(r"\b" + w + r"\s*(?:=(?!=)|\+\+|--|[-+*/&|^]=)",
                              tail)
                    or re.search(r"(?:\+\+|--|&)\s*\b" + w + r"\b", tail))

            if not _reassigned(lhs) and not _reassigned(rhs):
                # Propagate B for every later read of A and drop the copy.
                stmts[i + 1:] = [
                    re.sub(r"\b" + re.escape(lhs) + r"\b", rhs, t)
                    for t in stmts[i + 1:]]
                i += 1
                continue
        out.append(s)
        i += 1
    return ";".join(out)


def _normalize_body(text: str) -> "tuple[str, dict[str, str]]":
    """Apply the benign type-driven normalizations and return the pre-alpha body
    text plus the string/char-literal stash map.

    Collapses ONLY the benign, type-driven rendering axes (types, casts, canary,
    BYREF init, leading underscores, width-materialized return, single-use temp
    copy); preserves statements, calls, constants, control flow, and (stashed)
    string literals. Shared by :func:`structural_norm` (which then alpha-renames)
    and :func:`structural_equiv` (which tokenizes for a homomorphism check)."""
    lines, locals_ = _strip_decls_and_signature(text)
    lines = _strip_canary_and_byref_init(lines)
    body = "\n".join(lines)
    # Protect string/char literals from cast-strip, underscore-collapse, and
    # alpha-rename: their CONTENT is part of the structure (a wrong/absent string
    # IS a divergence) but the words inside must not be tokenized as identifiers
    # (a string word colliding with a local name would skew the alpha map). Map
    # each distinct literal to a stable opaque token by content, so identical
    # strings on both sides compare equal and a changed string diverges.
    literals: dict[str, str] = {}

    def _stash(m: "re.Match[str]") -> str:
        # A NUL-delimited, CONTENT-keyed token: the identifier regex requires a
        # leading [A-Za-z_], so it never tokenizes this placeholder (and the
        # cast/width/fold passes skip it too). Keyed by content (a stable hash) so
        # the SAME string on both bodies collides to one token while a DIFFERENT
        # string gets a different token -- a changed/absent literal still diverges.
        lit = m.group(0)
        key = zlib.crc32(lit.encode()) & 0xFFFFFFFF
        return literals.setdefault(lit, f"\x00{key}\x00")

    body = re.sub(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'', _stash, body)
    # Strip cosmetic casts (repeat: nested ``(const char *)(_DWORD *)x``).
    prev = None
    while prev != body:
        prev = body
        body = _CAST.sub("", body)
    # Unwrap width-extraction keywords applied to a bare identifier: the target
    # of a materialized return / sub-register store is a pure type artifact.
    body = _WIDTH_WRAP.sub(r"\1", body)
    # Equalize libc-symbol leading-underscore count (2+ -> 1).
    body = _LEADING_US.sub("_", body)
    # Undo type-driven temp materialization, then collapse whitespace per stmt.
    body = _fold_materialized_return(body, locals_)
    body = _inline_single_use_copy(body, locals_)
    # Join everything to one whitespace-normalized stream so multi-line call arg
    # lists (native wraps long calls; the drop may not) compare equal.
    body = re.sub(r"\s+", " ", body).strip()
    # Re-fold materialized return AFTER whitespace collapse (the pattern may have
    # spanned the wrapped lines).
    body = _fold_materialized_return(body, locals_)
    return body, literals


def structural_norm(text: str) -> str:
    """Alpha-renamed structural skeleton of a pseudocode body (see module note).

    Every non-keyword identifier collapses to a positional id ``#k`` (assigned in
    first-appearance order) and string/char literals to ``$Lk$``, so two bodies
    compare equal iff they share the same statement/call/constant/control-flow
    structure under a BIJECTIVE name renaming. Use this for exact structural
    equality (e.g. unit tests); use :func:`structural_equiv` when the drop may
    carry a benign weak-typing value-SPLIT (one native value rendered as two
    register-typed names) that a bijection would reject."""
    body, literals = _normalize_body(text)
    names: dict[str, int] = {}

    def _ren(m: "re.Match[str]") -> str:
        tok = m.group(0)
        if tok in _KEYWORDS:
            return tok
        return f"#{names.setdefault(tok, len(names))}"

    body = _IDENT.sub(_ren, body)
    # Restore each stashed literal to a stable ``$Lk$`` (k = first-appearance
    # order in this body), so identical strings render identically and a changed
    # string renders differently.
    for k, (_lit, tok) in enumerate(literals.items()):
        body = body.replace(tok.strip(), f"$L{k}$")
    return re.sub(r"\s+", " ", body).strip()


# A structural token: a string/char-literal stash (NUL-delimited), an integer
# literal (any base/suffix), an identifier, or a single punctuation char.
_TOKEN = re.compile(
    r"\x00\d+\x00"
    r"|0[xX][0-9A-Fa-f]+[uUlL]*"
    r"|\d+[uUlL]*"
    r"|[A-Za-z_]\w*"
    r"|::|->|\+\+|--|<<|>>|<=|>=|==|!=|&&|\|\||[-+*/%&|^]="
    r"|[^\s]")


def _tokenize(body: str) -> list[str]:
    return _TOKEN.findall(body)


def structural_equiv(drop_text: str, native_text: str) -> bool:
    """True iff the dropped body is STRUCTURALLY faithful to the native body,
    tolerating the benign weak-typing axes (see module note) INCLUDING a value
    SPLIT: a single native value the drop renders as two distinct register-typed
    names (``options`` used in both calls on native; ``a3`` in one and its typed
    copy ``v8`` in the other on the drop -- the drop has no visible ``v8 = a3``
    to copy-propagate).

    The check is a one-directional structural homomorphism drop -> native over
    the normalized token streams: every NON-identifier token (keyword, operator,
    bracket, integer constant, string literal) must match position-for-position,
    and every drop identifier must map CONSISTENTLY to a single native identifier.
    This is the safe direction: weak typing only ever SPLITS one source value into
    several typed names, so allowing the drop to MERGE names (two drop names -> one
    native name is fine) matches reality, while the converse -- the drop MERGING
    two genuinely distinct native values into one name -- makes a native name map
    from two different drop names, which is rejected (a real divergence still
    fails). A missing/extra/reordered statement, a wrong callee or constant, a
    struct-field-vs-raw-offset access, or a restructured loop all change the
    non-identifier token stream and so still fail."""
    dbody, _ = _normalize_body(drop_text)
    nbody, _ = _normalize_body(native_text)
    dtoks, ntoks = _tokenize(dbody), _tokenize(nbody)
    if len(dtoks) != len(ntoks):
        return False

    def _is_fixed(toks: list[str], i: int) -> bool:
        """A FIXED identifier (compared literally, never renamed): a CALLEE (next
        token ``(``) or an ENUM/namespace scope member (adjacent ``::``). These
        are emitted by name identically on both builds, so a mismatch here -- a
        WRONG callee or enum constant -- is a real divergence, not a rename."""
        nxt = toks[i + 1] if i + 1 < len(toks) else ""
        prv = toks[i - 1] if i > 0 else ""
        return nxt == "(" or nxt == "::" or prv == "::"

    fwd: dict[str, str] = {}
    for i, (dt, nt) in enumerate(zip(dtoks, ntoks)):
        d_id = bool(_IDENT.fullmatch(dt)) and dt not in _KEYWORDS
        n_id = bool(_IDENT.fullmatch(nt)) and nt not in _KEYWORDS
        if d_id != n_id:
            return False  # an identifier vs a keyword/operator => structural diff
        if not d_id:
            if dt != nt:
                return False  # keyword / operator / constant / literal mismatch
            continue
        # A callee / enum-scope identifier must match LITERALLY (after the shared
        # leading-underscore normalization already applied in _normalize_body).
        if _is_fixed(dtoks, i) or _is_fixed(ntoks, i):
            if dt != nt:
                return False
            continue
        # A value identifier (local / param / global): require a consistent
        # drop -> native mapping. One-directional, so the drop may SPLIT one
        # native value into two typed names (a3 and v8 both -> options); but the
        # converse -- the drop MERGING two distinct native values into one name --
        # makes a native pair require one drop name to map to two natives, which
        # is rejected (a real divergence still fails).
        if dt in fwd:
            if fwd[dt] != nt:
                return False
        else:
            fwd[dt] = nt
    return True

