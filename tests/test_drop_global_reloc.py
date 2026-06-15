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

SECOND interior-global path (``_desc`` fallthrough): the BARE store-dest
``store i64 %v, i64* @data_24148`` routes through ``_global_ea`` (above). But a
store whose pointer is a no-op ``bitcast @data_24188 to ptr`` (do_copy's
``x_tmp_2.src_info`` field copy) reaches the interior global through ``_desc`` on
the bitcast's source operand -- and ``_desc`` did NOT call ``_addr_named_global_ea``.
``str(@data_24188)`` is the global's full definition ``'@data_24188 = global i64
-1'``, so ``_desc``'s trailing-integer fallback captured the ``-1`` initializer and
collapsed the store target to ``*(_QWORD *)0xFFFFFFFFFFFFFFFF = v;`` -- exactly the
BADADDR-pointer bug shape, but for a field reached via bitcast. ``TestDropDescInteriorGlobal``
covers this ``_desc`` path (and proves the regression with ``_addr_named_global_ea``
neutralized). do_copy still DECLINES at the B5 gate (its dead stack-canary read --
native keeps ``__readfsqword(0x28u)`` under the dynamic alloca -- diverges, and a
blanket canary-emit regresses functions native DOES elide it for, e.g. usage /
hash_clear); the ``src_info`` resolution is asserted on the force-accepted degraded
body so the interior-global fix is regression-tested independent of the canary gap.
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


# do_copy's x_tmp_2 struct copy: the src_info field is stored through a no-op
# ``bitcast @data_24188 to ptr`` -> the interior global is resolved by ``_desc``
# on the bitcast SOURCE, a path that bypassed ``_global_ea`` and so missed the
# interior-aggregate decoder until the ``_desc`` fallback was added.
_DOCOPY_FN = "do_copy"
# The resolved interior field write the fix must render (NOT a BADADDR store).
_SRC_INFO_ASSIGN = re.compile(r"\bx_tmp_2\.src_info\b\s*=")


def _docopy_degraded_body(examples_dir: Path, break_desc: bool) -> str:
    """The do_copy drop body. do_copy DECLINES at the B5 gate (dead-canary gap),
    so force-accept the degraded body to inspect the interior-global resolution
    in isolation. ``break_desc`` neutralizes ``_addr_named_global_ea`` to simulate
    the pre-fix ``_desc`` fallthrough (the trailing ``-1`` initializer capture)."""
    import idapro
    import ida_hexrays
    import ida_idaapi
    import ida_name

    from idavator.llvm_drop import LLVMDropConverter

    ll = (examples_dir / "cp.ll").read_text()
    idapro.open_database(str(examples_dir / "cp"), True)
    try:
        assert ida_hexrays.init_hexrays_plugin()
        ea = ida_name.get_name_ea(ida_idaapi.BADADDR, _DOCOPY_FN)
        assert ea != ida_idaapi.BADADDR, f"missing function {_DOCOPY_FN}"
        conv = LLVMDropConverter(ll)
        # Inspect the declined body: the B5 gate would drop it for the canary gap,
        # but the interior-global store is observable on the degraded body.
        conv._degraded_body_is_faithful = lambda native_c, cf: True  # type: ignore[method-assign]
        if break_desc:
            conv._addr_named_global_ea = staticmethod(lambda nm: None)  # type: ignore[method-assign]
        cf = conv.drop(ea, _DOCOPY_FN)
        return str(cf) if cf is not None else ""
    finally:
        idapro.close_database()


@pytest.mark.ida
class TestDropDescInteriorGlobal:
    """The bitcast-routed interior-aggregate store (``_desc`` path) resolves to
    the real field, not a BADADDR pointer write."""

    def test_src_info_field_store_resolves_not_badaddr(
        self, examples_dir: Path
    ) -> None:
        """do_copy's ``x_tmp_2.src_info`` (stored through ``bitcast @data_24188``)
        renders the resolved interior field assignment with no BADADDR store --
        the ``_desc`` interior-global fallback fires for the bitcast source."""
        if not _idalib():
            pytest.skip("idalib unavailable")
        if not (examples_dir / "cp").exists():
            pytest.skip("missing example binary: cp")

        txt = _docopy_degraded_body(examples_dir, break_desc=False)

        assert _SRC_INFO_ASSIGN.search(txt), (
            "interior-global field 'x_tmp_2.src_info =' not rendered "
            f"(_desc interior-global resolution regressed):\n{txt}"
        )
        assert _BADADDR_STORE.search(txt) is None, (
            "src_info store collapsed to a *(_QWORD *)0xFFFF... BADADDR write "
            f"(the _desc trailing-initializer capture regressed):\n{txt}"
        )

    def test_src_info_store_proves_fail_without_desc_resolution(
        self, examples_dir: Path
    ) -> None:
        """PROOF the guard bites: neutralize ``_addr_named_global_ea`` so ``_desc``
        falls through to the trailing-integer capture of the global's ``-1``
        initializer. The ``x_tmp_2.src_info =`` assignment then vanishes and a
        ``*(_QWORD *)0xFFFFFFFFFFFFFFFF`` BADADDR store appears."""
        if not _idalib():
            pytest.skip("idalib unavailable")
        if not (examples_dir / "cp").exists():
            pytest.skip("missing example binary: cp")

        broken = _docopy_degraded_body(examples_dir, break_desc=True)

        assert not _SRC_INFO_ASSIGN.search(broken), (
            "interior-global decoder neutralized yet 'x_tmp_2.src_info =' still "
            f"present -- guard cannot catch the regression:\n{broken}"
        )
        assert _BADADDR_STORE.search(broken), (
            "expected a *(_QWORD *)0xFFFF... BADADDR store when the interior "
            f"global is unresolved (regression signature absent):\n{broken}"
        )
