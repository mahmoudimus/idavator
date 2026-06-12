"""A through-member deref of a struct-POINTER parameter must survive the
lift+drop -- the first member must be DEREFERENCED, not the struct pointer
passed twice.

The lifter lowered ``ldx ds, x, off`` (a memory load -- ``*(T*)x``) by lifting
the address operand ``x`` with ``dest=True``, which returned the SLOT ADDRESS
(``&x``) for an lvalue operand. ``m_ldx``'s single ``load`` then recovered only
the SLOT VALUE (``x``) -- one indirection short -- emitting
``bitcast %x to i64*; load i64`` (the SAME shape the lifter uses for a plain
pointer-value slot read). The loss was invisible at sub-pointer width (the
drop's ``_ptr_deref_alias`` width rule patched ``*name`` as a byte deref) but
corrupted POINTER-width derefs: ``triple_free`` dropped ``free(a0); free(a0)``
instead of native ``free(*(void**)x); free(x)`` (free the member, then the
struct). ``triple_hash`` / ``triple_compare`` / ``randint_all_free`` lost the
same off-0 ``*(T**)x`` member load.

Ground truth (clang ``-O2`` on the gnulib ``hash-triple.c`` ``triple_free`` and
IDA's own PRISTINE native): ``%2 = load ptr, ptr %0; free(%2); free(%0)`` --
the first free dereferences ``x`` (loads ``*x``), the second frees ``x``.

Fix (``ida2llvm.lift_insn`` m_ldx): lift the address operand as a VALUE
(``dest=False``) so the load DEREFERENCES it (``load(load(%slot))``), distinct
in the IR from a plain slot read (``bitcast %slot; load``).

Fail-without-fix: against the pre-fix lifted IR (address operand lifted with
``dest=True``) the drop emits ``free(a0); free(a0)`` -- the member deref is
ABSENT and the first arg equals the second (proven by reverting the m_ldx
address-operand lift to ``dest=True`` and re-lifting these bodies).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


def _idalib() -> bool:
    try:
        import idapro  # noqa: F401

        return True
    except ImportError:
        return False


def _paths(examples_dir: Path):
    binary = examples_dir / "cp"
    ir_path = examples_dir / "cp.ll"
    if not (binary.exists() and ir_path.exists()):
        pytest.skip("missing cp / cp.ll")
    return binary, ir_path


def _drop_only(examples_dir: Path, name: str) -> str:
    """Drop ``name`` from cp.ll into its own ea in a FRESH session and return the
    dropped pseudocode. Nothing decompiles the ea first -- a prior decompile of
    the same ea perturbs the lvar cache and reshapes the drop (idalib
    non-determinism), so the through-member deref check must run clean. A native
    fallback (build error) is rejected: this asserts a REAL drop."""
    import idapro
    import ida_hexrays
    import ida_idaapi
    import ida_name

    binary, ir_path = _paths(examples_dir)
    from idavator.llvm_drop import LLVMDropConverter

    idapro.open_database(str(binary), True)
    try:
        assert ida_hexrays.init_hexrays_plugin()
        ea = ida_name.get_name_ea(ida_idaapi.BADADDR, name)
        if ea == ida_idaapi.BADADDR:
            pytest.skip(f"{name} not in this binary")
        conv = LLVMDropConverter(ir_path.read_text())
        cf = conv.drop(ea, name)
        # A real drop (not a native fallback): build succeeded with no late error.
        assert conv.last_error is None, conv.last_error
        assert cf is not None, "decompile returned None"
        return str(cf)
    finally:
        idapro.close_database()


@pytest.mark.ida
class TestThroughMemberDerefPreserved:
    def test_triple_free_dereferences_first_member(
            self, examples_dir: Path) -> None:
        """``triple_free`` frees the first member (``*(void**)x``) THEN the struct
        (``x``); the pointer-width through-member deref must not collapse to
        ``free(x); free(x)``.

        Fail-without-fix: the m_ldx address operand lifted as ``dest=True``
        recovers only the slot value, so the drop emits ``free(a0); free(a0)`` --
        no deref of the first member."""
        if not _idalib():
            pytest.skip("idalib unavailable")
        dropped = _drop_only(examples_dir, "triple_free")

        # The first free dereferences the struct's first member: `*(void **)`.
        assert "*(void **)" in dropped, (
            f"first member deref lost -- `free(*(void**)x)` collapsed to "
            f"`free(x)`:\n{dropped}")
        # The first free()'s argument is the dereferenced member; the bug renders
        # BOTH frees with the bare struct pointer (`free(a0); free(a0)`). Drop the
        # function-signature line (`triple_free(...)` itself ends in `free(`) and
        # require exactly one bare-pointer free plus one through-member free.
        body = "\n".join(
            ln for ln in dropped.splitlines() if not ln.lstrip().startswith("void"))
        assert "free(*(void **)" in body, (
            f"first free does not dereference the member:\n{dropped}")
        # The two frees are NOT identical -- the second frees the bare pointer.
        free_args = re.findall(r"\bfree\(([^;]*)\);", body)
        assert len(free_args) == 2, (
            f"expected two free() calls in the body, got {free_args}:\n{dropped}")
        assert free_args[0] != free_args[1], (
            f"both frees pass the same operand (member deref absent):\n{dropped}")

    def test_triple_hash_loads_first_member(self, examples_dir: Path) -> None:
        """``triple_hash`` hashes the first member ``*(const void**)x``, not the
        struct pointer itself.

        Fail-without-fix: drops ``hash_pjw(a0, ...)`` (the raw pointer) instead
        of ``hash_pjw(*(const void **)a0, ...)``."""
        if not _idalib():
            pytest.skip("idalib unavailable")
        dropped = _drop_only(examples_dir, "triple_hash")

        assert "hash_pjw(" in dropped, f"hash_pjw call missing:\n{dropped}"
        # hash_pjw's first argument is the DEREFERENCED member (a pointer load).
        assert "*(const void **)" in dropped or "*(void **)" in dropped, (
            f"first member deref lost -- hash_pjw given the raw struct pointer:\n"
            f"{dropped}")

    def test_triple_compare_dereferences_name_members(
            self, examples_dir: Path) -> None:
        """``triple_compare`` passes the dereferenced name members
        (``*(const char**)x`` / ``*(const char**)y``) to ``same_name``.

        Fail-without-fix: drops ``same_name((const char *)a0, (const char *)a1)``
        -- the raw struct pointers, not the name members."""
        if not _idalib():
            pytest.skip("idalib unavailable")
        dropped = _drop_only(examples_dir, "triple_compare")

        assert "same_name(" in dropped, f"same_name call missing:\n{dropped}"
        # same_name receives the through-member name pointers, not the structs.
        assert dropped.count("*(const char **)") >= 2, (
            f"name-member derefs lost -- same_name given raw struct pointers:\n"
            f"{dropped}")

    def test_randint_all_free_loads_source_member(
            self, examples_dir: Path) -> None:
        """``randint_all_free`` frees ``s->source`` (the first member,
        ``*(randread_source**)s``), not the struct ``s`` itself.

        Fail-without-fix: drops ``randread_free((randread_source *)a0)`` -- the
        raw struct pointer, not the source member."""
        if not _idalib():
            pytest.skip("idalib unavailable")
        dropped = _drop_only(examples_dir, "randint_all_free")

        assert "randread_free(" in dropped, f"randread_free call missing:\n{dropped}"
        # randread_free receives the dereferenced source member.
        assert "*(randread_source **)" in dropped, (
            f"source-member deref lost -- randread_free given the raw struct "
            f"pointer:\n{dropped}")
