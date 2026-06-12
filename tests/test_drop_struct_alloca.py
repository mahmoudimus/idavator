"""Escaping struct-alloca sizing + host-frame anchor.

A struct alloca (e.g. `%stat`-typed locals) was sized via `_type_size` -> 4 (no
struct case) and packed at a synthetic frame offset, so its `bitcast;ptr-arith`
field accesses collided with an adjacent slot -> wrong-but-green output
(`stat((stat *)&src_st, ...)` instead of `stat(..., &dst_back_sb)`). The fix sizes
escaping struct allocas from `self._struct_size` and re-anchors them at their real
host-frame offset. Regression for memory idavator_drop_noreturn_50342_rootcause
(the struct half / ticket ida-1geh).
"""
from __future__ import annotations

from pathlib import Path

import pytest


def _idalib() -> bool:
    try:
        import idapro  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.ida
class TestStructAllocaSizing:
    def test_source_is_dst_backup_struct_not_collided(
            self, examples_dir: Path) -> None:
        if not _idalib():
            pytest.skip("idalib unavailable")
        import idapro
        import ida_hexrays
        import ida_idaapi
        import ida_name

        binary = examples_dir / "cp"
        ir_path = examples_dir / "cp.ll"
        if not (binary.exists() and ir_path.exists()):
            pytest.skip("missing cp / cp.ll")

        from idavator.llvm_drop import LLVMDropConverter

        idapro.open_database(str(binary), True)
        try:
            assert ida_hexrays.init_hexrays_plugin()
            ea = ida_name.get_name_ea(
                ida_idaapi.BADADDR, "source_is_dst_backup")
            if ea == ida_idaapi.BADADDR:
                pytest.skip("source_is_dst_backup not in this binary")

            conv = LLVMDropConverter(ir_path.read_text())
            cf = conv.drop(ea, "source_is_dst_backup")
            assert conv.last_error is None, conv.last_error
            assert cf is not None, "decompile returned None"
            txt = str(cf)
            # the stat buffer must resolve to its OWN frame slot, not a collided
            # cast of another local (the pre-fix `stat((stat *)&src_st, ...)`).
            assert "dst_back_sb" in txt, f"struct slot collided:\n{txt}"
            assert "write access to const memory" not in txt, txt
        finally:
            idapro.close_database()
