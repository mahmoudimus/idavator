"""A STORE-through a struct-POINTER (a field deref write, ``p->f = v``) must
LIFT to a dereference of the pointer VALUE, not a write of the pointer SLOT.

This is the STORE-side mirror of the m_ldx ``r``-as-VALUE fix (``57e7c90``,
guarded by ``test_drop_missing_deref``). ``ida2llvm.lift_insn`` lowered
``stx v, ds, p`` (store ``v`` THROUGH the pointer ``p``) by lifting the address
operand ``p`` with ``dest=True``, which returned the SLOT ADDRESS (``&p``) for a
bare local pointer slot at offset 0. The store then wrote ``v`` INTO the slot
(``new_bucket = v``, a pointer-slot DEFINE) instead of through it
(``new_bucket->data = v``, a store THROUGH the pointer). The two are
byte-identical in the IR (``bitcast %slot; store``) -- but the microcode carries
the distinction (an m_stx *address* operand vs an m_ldx/m_mov *destination*).
Whenever the stored value was itself a POINTER, the drop's ``_ptr_deref_alias``
value-type rule then mistook the deref-store for a slot define and SILENTLY
DROPPED the side effect (``transfer_entries`` lost the rehash ``new_bucket->data
= data``; ``record_file`` collapsed ``ent->name = xstrdup(name)`` to ``ent =
xstrdup(name)`` -- discarding the ``xmalloc`` result).

Fix: when the m_stx address operand is a bare mop_l/mop_S local pointer slot,
LOAD the slot to recover the pointer VALUE and store ``l`` THROUGH it
(``store l, load(slot)``) -- distinct in the IR from a slot define, and (done
inline, not via ``_store_as``) not undone by ``dedereference``.

This guard asserts at the LIFT level (deterministic; idalib pseudocode rendering
is non-deterministic). For ``transfer_entries`` it pins that the off-0
store-through emits ``store <val>, <load of the pointer slot>`` -- a store whose
destination is a LOAD of the alloca, NOT a ``store ..., bitcast %slot``.

Fail-without-fix: with the address operand lifted ``dest=True`` the off-0
store-through emits ``store <val>, bitcast(%slot)`` (the slot-define shape) and
there is NO ``store ..., <load %slot>`` deref-store (proven by reverting the
m_stx treatment and re-lifting).
"""
from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path

import pytest


def _idalib() -> bool:
    try:
        import idapro  # noqa: F401

        return True
    except ImportError:
        return False


def _binary(examples_dir: Path) -> Path:
    binary = examples_dir / "cp"
    if not binary.exists():
        pytest.skip("missing cp")
    return binary


def _lift_fn_ir(examples_dir: Path, name: str) -> str:
    """Lift ``name`` from the cp binary through the CURRENT ida2llvm lifter and
    return its LLVM IR text. Deterministic (the lifter does not depend on the
    idalib lvar cache the way decompilation rendering does)."""
    import ida_idaapi
    import ida_name
    import idapro

    from idavator import ida2llvm
    from idavator.ida2llvm import BIN2LLVMController, ptext, refreshed_funcs

    binary = _binary(examples_dir)
    tmp = Path(tempfile.mkdtemp(prefix="storelift_"))
    dst = tmp / "cp"
    shutil.copy(binary, dst)
    idapro.open_database(str(dst), True)
    try:
        ptext.clear()
        refreshed_funcs.clear()
        ida2llvm.type_providers = []
        c = BIN2LLVMController(target_mode="host")
        c.initialize()
        ea = ida_name.get_name_ea(ida_idaapi.BADADDR, name)
        if ea == ida_idaapi.BADADDR:
            pytest.skip(f"{name} not in this binary")
        c.insertFunctionAtEa(ea)
        text = str(c.m)
    finally:
        idapro.close_database()
        shutil.rmtree(tmp, ignore_errors=True)
    m = re.search(
        r'(?ms)^define[^\n]*@"' + re.escape(name) + r'"\(.*?\n\}', text)
    assert m is not None, f"{name} not found in lifted module"
    return m.group(0)


def _store_dests(ir_body: str):
    """Return the SSA operand NAME each ``store`` writes to (the ``%".N"`` in the
    destination operand), one per store instruction. The destination is the last
    comma-separated operand; strip its leading type so only the value name
    remains (``store i8* %".77", i8** %".79"`` -> ``%".79"``)."""
    dests = []
    for line in ir_body.splitlines():
        s = line.strip()
        m = re.match(r"store\s+.+,\s*[^,\n]*?(%\"?[.\w]+\"?)\s*$", s)
        if m:
            dests.append(m.group(1).strip())
    return dests


@pytest.mark.ida
class TestStoreThroughPointerLift:
    def test_transfer_entries_offset0_store_is_a_deref(
            self, examples_dir: Path) -> None:
        """``transfer_entries`` rehash inserts (``new_bucket->data = data``) lift
        to a deref-store THROUGH the pointer VALUE: ``store i8* %data, i8**
        <load of %new_bucket>`` -- the store destination is a LOAD of the
        pointer alloca, distinct from a ``store ..., bitcast %new_bucket`` slot
        define.

        Fail-without-fix: the off-0 store-through emits ``store ..., bitcast
        %new_bucket`` (slot-define shape) and NO store targets a load of the
        ``new_bucket``/``new_bucketa`` alloca."""
        if not _idalib():
            pytest.skip("idalib unavailable")
        body = _lift_fn_ir(examples_dir, "transfer_entries")

        # SSA names of the loads of the new_bucket / new_bucketa pointer slots:
        # `%".N" = load %"hash_entry"*, %"hash_entry"** %"new_bucket[a]"`.
        load_names = set(
            re.findall(
                r'(%"\.\d+") = load %"hash_entry"\*, %"hash_entry"\*\* '
                r'%"new_bucketa?"',
                body))
        assert load_names, (
            "no `load ... %new_bucket` -- the pointer value is never recovered "
            f"(off-0 store still reads the slot address):\n{body}")

        # at least one store must write THROUGH one of those loaded pointer
        # values (its destination is a cast of / equal to the load), i.e. a
        # deref-store, NOT a `store ..., bitcast %new_bucket` slot define.
        dests = _store_dests(body)
        # resolve casts: a store dest may be `%".M"` where `%".M" = bitcast
        # <load> to i8**`. Build cast -> source map.
        cast_src = dict(
            re.findall(r'(%"\.\d+") = bitcast .*? (%"\.\d+") to ', body))

        def roots_to_load(dest: str) -> bool:
            seen = set()
            cur = dest
            while cur in cast_src and cur not in seen:
                seen.add(cur)
                cur = cast_src[cur]
            return cur in load_names

        deref_stores = [d for d in dests if roots_to_load(d)]
        assert deref_stores, (
            "rehash store-through lost -- NO store writes through a LOADED "
            "new_bucket pointer value; the off-0 `new_bucket->data = data` was "
            f"lowered as a `bitcast %slot; store` slot define:\n{body}")

        # and the slot-define shape (`store ..., bitcast %new_bucket`) must NOT
        # be how the data write is emitted: no store targets a bitcast taken
        # DIRECTLY off the new_bucket alloca.
        bad = re.search(
            r'(%"\.\d+") = bitcast %"hash_entry"\*\* %"new_bucketa?" to', body)
        if bad is not None:
            bad_name = bad.group(1)
            assert bad_name not in dests, (
                "off-0 store-through still emitted as a slot define "
                f"(`store ..., bitcast %new_bucket`):\n{body}")

    def test_record_file_name_store_is_a_deref(
            self, examples_dir: Path) -> None:
        """``record_file`` ``ent->name = xstrdup(name)`` lifts to a store THROUGH
        the ``ent`` pointer value (a deref-store), not a redefinition of the
        ``ent`` slot.

        ``ent`` is a struct (``F_triple``) pointer alloca; the off-0 ``->name``
        store-through must target a LOAD of the slot, not ``bitcast %ent``."""
        if not _idalib():
            pytest.skip("idalib unavailable")
        body = _lift_fn_ir(examples_dir, "record_file")

        # the entry pointer slot's loads (`load %"F_triple"*, %"F_triple"** %ent`
        # -- the alloca name is whatever the lifter assigned; match any pointer
        # alloca whose loaded value is later stored through at off 0).
        # A deref-store off-0 is `store <val>, <cast of a load of a ptr alloca>`.
        # Assert at least one store writes through a LOADED pointer value
        # (not a bitcast-of-alloca), which is the `ent->name = ...` write.
        loads = dict(
            re.findall(r'(%"\.\d+") = load (%"[^"]+"\*), %"[^"]+"\*\* ', body))
        assert loads, f"no pointer-slot loads in record_file:\n{body}"
        cast_src = dict(
            re.findall(r'(%"\.\d+") = bitcast .*? (%"\.\d+") to ', body))
        dests = _store_dests(body)

        def roots_to_load(dest: str) -> bool:
            seen = set()
            cur = dest
            while cur in cast_src and cur not in seen:
                seen.add(cur)
                cur = cast_src[cur]
            return cur in loads

        assert any(roots_to_load(d) for d in dests), (
            "record_file has NO deref-store through a loaded pointer value -- "
            "the `ent->name = xstrdup(..)` store-through was lowered as a slot "
            f"define (xmalloc result discarded):\n{body}")
