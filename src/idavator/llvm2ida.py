"""
Drop path: LLVM IR -> Hex-Rays microcode inside an already-open IDA database.

This module is intended for the IDA plugin / GUI workflow (interactive drop, patch,
export). Headless CLI support is not provided here; use ``idavator ida2llvm`` for
batch lifting via idalib.

The LLVM IR -> microcode conversion lives in :mod:`idavator.llvm_drop`
(``LLVMDropConverter``); this module is the thin database-facing entry that resolves
each definition to its host and drops it.
"""

import logging
from contextlib import suppress

import ida_ida
import ida_idaapi
import ida_name

from idavator.llvm_drop import LLVMDropConverter


def get_target_triple_from_ida() -> str:
    """Derive an LLVM target triple from the open IDA database."""
    proc = ida_ida.inf_get_procname()
    is_64 = ida_ida.inf_is_64bit()
    arch = "unknown"

    if proc == "metapc":
        arch = "x86_64" if is_64 else "i386"
    elif proc in ("arm", "ARM"):
        arch = "aarch64" if is_64 else "arm"
    elif proc in ("aarch64", "arm64"):
        arch = "aarch64"
    elif proc == "mips":
        arch = "mips64" if is_64 else "mips"
    elif proc in ("ppc", "ppc64"):
        arch = "powerpc64" if is_64 else "powerpc"
    elif proc == "riscv":
        arch = "riscv64" if is_64 else "riscv32"

    os_name = "unknown"
    with suppress(Exception):
        ostype = ida_ida.inf_get_ostype()
        if hasattr(ida_ida, "OSTYPE_WIN") and ostype == ida_ida.OSTYPE_WIN:
            os_name = "windows"
        elif hasattr(ida_ida, "OSTYPE_LINUX") and ostype == ida_ida.OSTYPE_LINUX:
            os_name = "linux"
        elif hasattr(ida_ida, "OSTYPE_MACOS") and ostype == ida_ida.OSTYPE_MACOS:
            os_name = "darwin"

    return f"{arch}-unknown-{os_name}"


def apply_llvm_ir(
    ir_text: str, *, verbose: bool = False, target_ea: int | None = None
) -> bool:
    """
    Apply LLVM IR to the current IDA database (must already be open).

    Each defined LLVM function is DROPPED into the IDB function of the same name
    (the Model-2 microcode-hook rebuild in :class:`LLVMDropConverter` -- the
    converter rebuilds the target's microcode from the IR and lets ``decompile()``
    run the full pipeline). ``target_ea`` forces a single host (used when the IR
    holds exactly one definition and the caller already knows the destination).

    :param ir_text: LLVM IR source
    :param verbose: Enable DEBUG logging on the root logger
    :param target_ea: optional explicit host EA for a single-function IR
    :return: True if at least one function was applied
    """
    logging.getLogger().setLevel(logging.DEBUG if verbose else logging.INFO)

    try:
        converter = LLVMDropConverter(ir_text)
    except Exception as e:  # noqa: BLE001
        logging.error("Failed to parse LLVM IR: %s", e, exc_info=verbose)
        return False

    applied = 0
    for fn in converter.module.functions:
        if fn.is_declaration:
            continue
        host = (target_ea if target_ea is not None
                else ida_name.get_name_ea(ida_idaapi.BADADDR, fn.name))
        if host == ida_idaapi.BADADDR:
            logging.warning("no IDB function named %s to drop into; skipping",
                            fn.name)
            continue
        cf = converter.drop(host, fn.name)
        if cf is None or converter.last_error:
            logging.error("drop @%s failed (interr=%s err=%s)", fn.name,
                          converter.last_interr, converter.last_error)
            continue
        logging.info("Applied @%s -> %#x", fn.name, host)
        applied += 1

    if applied == 0:
        logging.error("No LLVM functions were applied")
    return applied > 0


def apply_llvm_ir_file(path: str, *, verbose: bool = False) -> bool:
    """Load LLVM IR from a file and apply it via :func:`apply_llvm_ir`."""
    with open(path, encoding="utf-8") as f:
        return apply_llvm_ir(f.read(), verbose=verbose)
