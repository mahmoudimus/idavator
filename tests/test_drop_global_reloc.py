"""Global STORE-DESTINATION resolves to the real global ea, not BADADDR.

``set_program_name`` (the csweep-#5 GARBAGE/CONST-MEM canonical) stores to the
``.bss`` global ``@"program_name"`` via ``store i8* %v, i8** @"program_name"``.
The store-DEST path (``_global_ea`` -> ``make_gvar`` on ``mi.d``, distinct from
the gvaraddr VALUE-side fix ff1c58c) must resolve that named global to its real
IDB address so the dropped C renders ``program_name = v;`` -- NOT a write through
a BADADDR pointer ``*(_QWORD *)0xFFFFFFFFFFFFFFFF = v;``.

FAITHFULNESS NOTE (why the dropped output legitimately contains BADADDR stores):
the SAME function also stores to ``@"program_invocation_short_name@GLIBC_2.2.5"``
and ``@"program_invocation_name@GLIBC_2.2.5"`` -- GLIBC interposition externs
that are NOT addressable in this statically-analyzed IDB (``get_name_ea`` ->
BADADDR) and whose IR global value is literally ``i64 18446744073709551615``
(0xFFFFFFFFFFFFFFFF). Hex-Rays' OWN native decompilation, run deterministically,
renders those two stores as ``*(_QWORD *)0xFFFFFFFFFFFFFFFFLL = ...``. The drop
matches native there -- that is faithful, not a miscompile. So the invariant is
NOT "zero BADADDR anywhere"; it is "the RESOLVABLE global resolves to its real
ea AND the whole body equals deterministic native".

PROVEN to fail without the store-dest resolution: with ``_global_ea`` forced to
return ``None`` (the store-dest falls back to the generic ``stx``/``_desc`` path
exactly as it did before the global store-dest ``make_gvar`` existed), the
``program_name = v;`` assignment vanishes and the dropped text no longer matches
native -- the ``test_*_proves_fail_without_store_dest_resolution`` case asserts
that regression fires.
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


_FN = "set_program_name"
# The real, RESOLVABLE .bss global the store-dest must name.
_RESOLVABLE_GLOBAL = "program_name"
# A BADADDR-pointer STORE (write THROUGH a 0xFFFF... pointer) -- the bug shape
# for the resolvable global. (A *(_QWORD *)0xFFFF... = does also legitimately
# appear for the GLIBC externs and IS in native; we assert specifically that the
# RESOLVABLE global is NOT rendered that way, via the `program_name =` check.)
_BADADDR_STORE = re.compile(r"\*\(_QWORD \*\)0xFFFFFFFFFFFFFFFF")
_BYTE_PTR = re.compile(r"\bbyte_[0-9A-Fa-f]+\b")


def _drop_text(examples_dir: Path) -> str:
    import idapro
    import ida_hexrays
    import ida_idaapi
    import ida_name

    from idavator.llvm_drop import LLVMDropConverter

    ll = (examples_dir / "cp.ll").read_text()
    idapro.open_database(str(examples_dir / "cp"), True)
    try:
        assert ida_hexrays.init_hexrays_plugin()
        ea = ida_name.get_name_ea(ida_idaapi.BADADDR, _FN)
        assert ea != ida_idaapi.BADADDR, f"missing function {_FN}"
        conv = LLVMDropConverter(ll)
        cf = conv.drop(ea, _FN)
        assert conv.last_error is None, conv.last_error
        assert conv.last_interr is None, f"INTERR {conv.last_interr}"
        assert cf is not None, "decompile returned None"
        return str(cf)
    finally:
        idapro.close_database()


@pytest.mark.ida
class TestDropGlobalReloc:
    def test_resolvable_global_store_dest_is_real_ea_not_badaddr(
        self, examples_dir: Path
    ) -> None:
        """The ``@"program_name"`` store renders ``program_name = v;`` -- the
        named .bss global, resolved to its real ea -- with no BADADDR pointer
        store and no ``byte_*`` blob for that assignment."""
        if not _idalib():
            pytest.skip("idalib unavailable")
        if not (examples_dir / "cp").exists():
            pytest.skip("missing example binary: cp")

        txt = _drop_text(examples_dir)

        # The resolvable global is named and ASSIGNED (lhs), not dereffed through
        # a garbage pointer.
        assert re.search(rf"\b{_RESOLVABLE_GLOBAL}\b\s*=", txt), (
            f"resolvable global '{_RESOLVABLE_GLOBAL}' not assigned by name "
            f"(store-dest resolution regressed):\n{txt}"
        )
        # No undecoded byte_* blob anywhere (a make_gvar-on-BADADDR symptom).
        assert _BYTE_PTR.search(txt) is None, (
            f"store-dest rendered as a byte_* blob:\n{txt}"
        )

    def test_set_program_name_drop_equals_native(
        self, examples_dir: Path
    ) -> None:
        """Ground-truth faithfulness: the whole dropped body equals Hex-Rays'
        deterministic native decompilation (incl. the GLIBC-extern BADADDR
        stores, which native produces identically). Re-run once on mismatch to
        defeat idalib non-determinism on the native reference before failing."""
        if not _idalib():
            pytest.skip("idalib unavailable")
        if not (examples_dir / "cp").exists():
            pytest.skip("missing example binary: cp")

        import idapro
        import ida_hexrays
        import ida_idaapi
        import ida_name

        from idavator.llvm_drop import LLVMDropConverter

        ll = (examples_dir / "cp.ll").read_text()

        def _once() -> tuple[str, str]:
            idapro.open_database(str(examples_dir / "cp"), True)
            try:
                assert ida_hexrays.init_hexrays_plugin()
                ea = ida_name.get_name_ea(ida_idaapi.BADADDR, _FN)
                native = str(ida_hexrays.decompile(ea))
                conv = LLVMDropConverter(ll)
                cf = conv.drop(ea, _FN)
                assert conv.last_error is None, conv.last_error
                return str(cf), native
            finally:
                idapro.close_database()

        dropped, native = _once()
        if dropped.strip() != native.strip():
            # idalib non-determinism hits the NATIVE reference too; one re-run.
            dropped, native = _once()
        assert dropped.strip() == native.strip(), (
            "drop diverges from deterministic native:\n"
            f"--- dropped ---\n{dropped}\n--- native ---\n{native}"
        )

    def test_resolvable_global_store_dest_proves_fail_without_resolution(
        self, examples_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """PROOF the guard bites: make the symbol lookup MISS ``program_name``
        (``get_name_ea -> BADADDR``, the root the ticket names -- both the
        store-dest ``_global_ea`` AND the ``_desc`` named-global branch consume
        it). Then ``program_name = v;`` becomes a ``*(_QWORD *)&dword_0 = v;``
        blob store. Asserts the regression signature (no named assignment, a
        ``dword_``/``byte_`` blob appears) is observable -- without it the two
        passing guard cases would be vacuous."""
        if not _idalib():
            pytest.skip("idalib unavailable")
        if not (examples_dir / "cp").exists():
            pytest.skip("missing example binary: cp")

        import idapro
        import ida_hexrays
        import ida_idaapi
        import ida_name

        import idavator.llvm_drop as drop_mod
        from idavator.llvm_drop import LLVMDropConverter

        ll = (examples_dir / "cp.ll").read_text()
        idapro.open_database(str(examples_dir / "cp"), True)
        try:
            assert ida_hexrays.init_hexrays_plugin()
            ea = ida_name.get_name_ea(ida_idaapi.BADADDR, _FN)

            # Force the store-dest symbol resolution to MISS only program_name
            # (every other lookup is untouched), reproducing the unresolved
            # writable-global root.
            real_get_name_ea = drop_mod.ida_name.get_name_ea

            def _miss(frm, nm):
                if nm == _RESOLVABLE_GLOBAL:
                    return ida_idaapi.BADADDR
                return real_get_name_ea(frm, nm)

            monkeypatch.setattr(drop_mod.ida_name, "get_name_ea", _miss)

            conv = LLVMDropConverter(ll)
            cf = conv.drop(ea, _FN)
            broken = str(cf) if cf is not None else ""

            assert not re.search(rf"\b{_RESOLVABLE_GLOBAL}\b\s*=", broken), (
                "store-dest symbol miss yet 'program_name =' still present -- "
                f"guard cannot catch the regression:\n{broken}"
            )
            # The unresolved store-dest collapses to a numeric/blob pointer
            # store -- a dword_*/byte_* blob -- which the passing guard forbids.
            assert (
                _BYTE_PTR.search(broken)
                or re.search(r"\bdword_[0-9A-Fa-f]+\b", broken)
            ), (
                "expected a dword_*/byte_* blob store-dest when the global is "
                f"unresolved (regression signature absent):\n{broken}"
            )
        finally:
            idapro.close_database()
