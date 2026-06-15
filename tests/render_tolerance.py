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
