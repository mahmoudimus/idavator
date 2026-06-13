"""``hash_initialize`` in the committed ``cp.ll`` must carry the CORRECT
store-through-pointer / pointer-copy shapes for ``table->bucket = calloc(...)``
and ``tuninga = tuning`` -- not the STALE-cp.ll lifter collapse.

Background (ticket ida-w829 / the RE-DIAGNOSED 50342). The ``hash_initialize``
and ``renameatu`` bodies in ``cp.ll`` were lifted by an OLD ida2llvm that
predated the m_stx store-through-pointer fix (``b65a9a3``) and the
pointer-copy-not-whole-struct-memcpy fix (``d634a21``). Two consequences in the
STALE body:

1. ``table->bucket = calloc(...)`` was lifted as
   ``store i8* calloc, bitcast(%"Hash_table"** %table to i8**)`` -- a bitcast of
   the ALLOCA SLOT (type ``Hash_table**``), i.e. a slot REDEFINE
   (``table = calloc``), byte-identical in IR to a legit ``table = malloc`` slot
   define. Native does a DEREF (``table->bucket = calloc``). The fixed lifter
   LOADS ``%table`` first and bitcasts the LOADED VALUE (``Hash_table*``), so the
   store is THROUGH the pointer.
2. ``tuninga = tuning`` (a pointer copy) was mis-lifted as
   ``memcpy(%tuninga, %tuning, i64 20)`` (copy the whole 20-byte struct). The
   fixed lifter emits a ``load``+``store`` of the pointer value.

The fix re-lifted both bodies through the CURRENT ida2llvm and re-spliced them
into ``cp.ll`` (CRLF-preserved, 0 collateral). With the corrected
store-through-pointer IR, ``hash_initialize``'s drop clears the late INTERR 50342
on its SROA-fallback path (``real_drop`` becomes True; previously ``cf=None``).

This guard pins the committed ``cp.ll`` artifact (the spliced body), which is
deterministic. It asserts:
  * the ``calloc`` store writes THROUGH a LOADED ``%table`` pointer value
    (a ``bitcast`` of a ``load`` of the alloca), NOT a bitcast of the alloca
    slot itself (``%"Hash_table"** %table``);
  * there is NO ``memcpy(..., i64 20)`` (the mis-lifted pointer-copy is gone).

Fail-without-fix: the pre-splice (HEAD) body has the calloc store as
``store i8* <calloc>, <bitcast %"Hash_table"** %table to i8**>`` (the slot-define
shape) and TWO ``memcpy(..., i64 20)`` calls -- proven by reverting the splice
(``git show HEAD:examples/cp.ll``).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


def _cpll_body(examples_dir: Path, name: str) -> str:
    cpll = examples_dir / "cp.ll"
    if not cpll.exists():
        pytest.skip("missing cp.ll")
    text = cpll.read_text(encoding="utf-8").replace("\r\n", "\n")
    m = re.search(
        r'(?ms)^define[^\n]*@"' + re.escape(name) + r'"\(.*?\n\}', text)
    assert m is not None, f"{name} not found in cp.ll"
    return m.group(0)


def _calloc_store_is_deref(body: str):
    """Classify the store of the ``calloc`` result in ``hash_initialize``.

    Returns one of ``"DEREF"`` (store dest is a bitcast of a LOADED pointer
    value), ``"SLOT-DEFINE"`` (store dest is a bitcast of the ``%table``
    alloca, type ``Hash_table**``), or ``"NOT-FOUND"``.
    """
    lines = body.split("\n")
    for i, line in enumerate(lines):
        if 'call i8* @"calloc"' not in line:
            continue
        mres = re.search(r'(%"?\.?\d+"?)\s*=', line)
        if not mres:
            continue
        res = mres.group(1)
        for j in range(i + 1, min(i + 8, len(lines))):
            ms = re.match(
                r'\s*store i8\* ' + re.escape(res) + r', i8\*\* (%"?\.?\d+"?)',
                lines[j])
            if not ms:
                continue
            dst = ms.group(1)
            for k in range(i, j):
                bc = re.match(
                    r'\s*' + re.escape(dst)
                    + r' = bitcast (%"[^"]+"\*\*?) (%"[^"]+")',
                    lines[k])
                if bc:
                    srctype = bc.group(1)
                    return ("SLOT-DEFINE" if srctype.endswith("**")
                            else "DEREF")
    return "NOT-FOUND"


class TestCallocStoreCollapse:
    def test_calloc_store_is_a_deref(self, examples_dir: Path) -> None:
        """``table->bucket = calloc(...)`` must be a store THROUGH the loaded
        ``%table`` pointer, not a redefine of the ``%table`` alloca slot.

        Fail-without-fix: the HEAD (pre-splice) body classifies as
        ``SLOT-DEFINE`` (``store ..., bitcast %"Hash_table"** %table``)."""
        body = _cpll_body(examples_dir, "hash_initialize")
        shape = _calloc_store_is_deref(body)
        assert shape == "DEREF", (
            "hash_initialize calloc store is not a deref-store "
            f"(classified {shape!r}); the STALE store-collapse "
            "`table = calloc` (slot redefine) was not re-spliced:\n" + body)

    def test_no_pointer_copy_memcpy20(self, examples_dir: Path) -> None:
        """The ``tuninga = tuning`` pointer copy must NOT lift to
        ``memcpy(..., i64 20)`` (whole-struct copy).

        Fail-without-fix: the HEAD body has TWO ``memcpy(..., i64 20)`` calls
        (the ``tuninga`` and ``table->tuning`` pointer stores)."""
        body = _cpll_body(examples_dir, "hash_initialize")
        bad = re.findall(r'call i8\* @"memcpy"\([^)]*i64 20\)', body)
        assert not bad, (
            "hash_initialize still has memcpy(..., 20) pointer-copy(s) -- the "
            f"mis-lifted pointer copy was not re-spliced:\n{bad}")
