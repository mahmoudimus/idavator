"""Non-escaping pointer-alloca deref-vs-slot lowering.

The lifter spills a ``const char *p`` (and ``X->field`` accesses on a struct
pointer) as an ``alloca ptr`` slot and reaches ``*p`` via a NO-OP
``bitcast %p to ptr`` followed by a SUB-pointer ``load i8`` / ``store i32``.
The address-taken pass gives the slot a frame slot (matching native's stkvar),
but the load/store emit must DEREF through the slot's pointer value
(``mov %p, r; ldx ds, r`` -- native), not read the slot's low byte. The pre-fix
emit read the slot, scalarising the pointer to ``name = (char)a0`` and
collapsing ``++name`` to ``name = 0x30`` (the ADDR-TAKEN scalarisation bucket,
ticket ida-j9cm).

A pointer-WIDTH (i64/ptr) bitcast access is left as a slot access: the lifter
type-puns BOTH a full pointer value (``bucket = *table``) and ``*X`` as
``load/store i64, bitcast %X``, and only a sub-pointer width is the unambiguous
deref -- so this fix touches only the i8/i16/i32 case and must NOT perturb the
pointer-width (``transfer_entries`` Hash_table list-walk) shape.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tests.render_tolerance import structural_equiv


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
    """Drop ``name`` from cp.ll into its own ea in a FRESH session and return
    the dropped pseudocode. NB: nothing else decompiles the ea first -- a prior
    ``decompile`` of the SAME ea perturbs the lvar cache and reshapes the drop
    (idalib non-determinism), so the deref-semantics check must run clean."""
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
        assert conv.last_error is None, conv.last_error
        assert cf is not None, "decompile returned None"
        return str(cf)
    finally:
        idapro.close_database()


def _native_only(examples_dir: Path, name: str) -> str:
    """Native decompilation of ``name`` in its OWN fresh session (the oracle).
    Kept in a separate open/close from the drop so neither contaminates the
    other's lvar cache."""
    import idapro
    import ida_hexrays
    import ida_idaapi
    import ida_name

    binary, _ir = _paths(examples_dir)
    idapro.open_database(str(binary), True)
    try:
        assert ida_hexrays.init_hexrays_plugin()
        ea = ida_name.get_name_ea(ida_idaapi.BADADDR, name)
        if ea == ida_idaapi.BADADDR:
            pytest.skip(f"{name} not in this binary")
        return str(ida_hexrays.decompile(ea))
    finally:
        idapro.close_database()


@pytest.mark.ida
class TestAddrTakenPointerAllocaDeref:
    def test_last_component_pointer_is_dereferenced_not_truncated(
            self, examples_dir: Path) -> None:
        """``*name`` / ``*p`` (sub-pointer i8 loads through a bitcast of an
        ``alloca ptr`` slot) must lower to a real DEREF, not a slot read.

        Fail-without-fix: the pre-fix slot read scalarises the pointer to
        ``name = (char)a0`` and collapses ``++name`` to ``name = 0x30`` (proven
        by stashing the fix -> these byte-truncation forms reappear)."""
        if not _idalib():
            pytest.skip("idalib unavailable")
        dropped = _drop_only(examples_dir, "last_component")

        # the pointer is walked by DEREF: a byte load through the pointer value.
        assert ("*(_BYTE *)" in dropped or "*(unsigned __int8 *)" in dropped), (
            f"pointer never dereferenced (scalarised?):\n{dropped}")
        # the pointer advances as a POINTER (+ 1 byte), never assigned a constant.
        assert "+ 1)" in dropped, f"pointer increment lost:\n{dropped}"
        # the pre-fix byte-truncation / collapsed-increment must be ABSENT.
        assert "(char)a0" not in dropped, f"pointer truncated to char:\n{dropped}"
        assert "= 0x30" not in dropped, (
            f"`++name` collapsed to a constant store:\n{dropped}")

    def test_last_component_matches_native(self, examples_dir: Path) -> None:
        """The dropped body is byte-faithful to native's own decompilation
        (same-session capture: native is taken before the drop)."""
        if not _idalib():
            pytest.skip("idalib unavailable")
        native = _native_only(examples_dir, "last_component")
        dropped = _drop_only(examples_dir, "last_component")
        assert dropped == native, (
            f"dropped diverges from native\n--- native ---\n{native}\n"
            f"--- dropped ---\n{dropped}")

    def test_transfer_entries_pointer_width_walk_preserved(
            self, examples_dir: Path) -> None:
        """The pointer-WIDTH (i64) Hash_table list-walk must stay byte-faithful
        to native: the sub-pointer deref fix must NOT add spurious derefs to a
        full-pointer ``load/store i64, bitcast %slot`` (``bucket = *table``).

        This is the regression guard for the load/store asymmetry -- a naive
        "bitcast-load always derefs" rule double-derefs this walk.

        On macOS-arm64 native this is a byte-exact match (native has no DWARF
        struct types, so it renders the same weakly-typed body as the drop). On
        the amd64 idalib build native carries DWARF ``Hash_table``/``hash_entry``
        types and so renders the walk as TYPED struct-field access (``src->bucket``,
        ``dst->n_buckets_used``, ``while`` over ``src->bucket_limit``), whereas the
        weakly-typed IR drop renders RAW pointer-offset arithmetic
        (``*(hash_entry **)a1``, ``*((_QWORD *)a0 + 3)``). That is a GENUINE per-
        build structural divergence (field-access vs raw-offset is not a cosmetic
        type/canary axis), so on that build the case is xfail -- weakening the
        guard to absorb it would defeat its purpose (B5)."""
        if not _idalib():
            pytest.skip("idalib unavailable")
        native = _native_only(examples_dir, "transfer_entries")
        dropped = _drop_only(examples_dir, "transfer_entries")
        if dropped == native or structural_equiv(dropped, native):
            return  # byte-exact or benign-axis match (arm64 / no-DWARF native)
        # Build-specific DWARF-struct divergence: native renders typed field
        # access (``->`` on a named struct ptr) that the weakly-typed drop renders
        # as raw ``*((_QWORD *)aN + k)`` offset arithmetic. Confirm THAT is the
        # residual (not some other regression) before declaring it an xfail -- any
        # OTHER divergence falls through to a real failure (B5).
        raw_offset = "(_QWORD *)a" in dropped or "(hash_entry **)a" in dropped
        typed_fields = "->n_buckets_used" in native and "Hash_table *" in native
        if raw_offset and typed_fields:
            pytest.xfail(
                "amd64 native renders the Hash_table walk via DWARF struct-field "
                "access (src->bucket, dst->n_buckets_used) while the weakly-typed "
                "IR drop renders raw pointer-offset arithmetic -- a genuine per-"
                "build structural divergence, not a cosmetic axis. Byte-exact on "
                "macOS-arm64 native, where the body ships.")
        pytest.fail(
            f"transfer_entries diverges from native\n--- native ---\n{native}\n"
            f"--- dropped ---\n{dropped}")
