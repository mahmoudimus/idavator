import logging
import sys
import traceback
from collections.abc import Generator
from contextlib import contextmanager, suppress
import typer

import idapro

app = typer.Typer(
    add_completion=False,
    help="IDAvator - Headless lift of Hex-Rays microcode to LLVM IR (idalib)",
)

_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
_LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"


_DEFAULT_LOG_FILE = "idavator.log"


def _validate_log_type(log_type: str) -> str:
    if log_type not in ("file", "console"):
        raise typer.BadParameter("must be one of: file, console", param_hint="--log-type")
    return log_type


def apply_subcommand_logging(
    ctx: typer.Context,
    *,
    log_type: str | None,
    log_file: str | None,
    verbose: bool | None,
) -> bool:
    """Honor logging flags after the subcommand; else keep the root callback setup."""
    if log_type is not None:
        _validate_log_type(log_type)
    if log_type is not None or log_file is not None or verbose is not None:
        configure_logging(
            log_type or "file",
            log_file=log_file or _DEFAULT_LOG_FILE,
            level=logging.DEBUG if verbose else logging.INFO,
        )
        return bool(verbose)
    return bool(ctx.obj.get("verbose", False))


def configure_logging(
    log_type: str,
    *,
    log_file: str = _DEFAULT_LOG_FILE,
    level: int = logging.INFO,
) -> None:
    """Configure the root logger (safe to call more than once)."""
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT)
    if log_type == "file":
        handler: logging.Handler = logging.FileHandler(log_file, mode="a")
    else:
        handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)
    handler.setFormatter(formatter)
    root.addHandler(handler)


@app.callback()
def _setup_logging(
    ctx: typer.Context,
    log_type: str = typer.Option(
        "file",
        "--log-type",
        help="Log destination: file or console (stderr)",
    ),
    log_file: str = typer.Option(
        _DEFAULT_LOG_FILE,
        "--log-file",
        help="Log file path when --log-type=file",
    ),
    verbose: bool = typer.Option(
        False,
        "-v",
        "--verbose",
        help="Enable verbose (DEBUG) logging",
    ),
) -> None:
    _validate_log_type(log_type)
    level = logging.DEBUG if verbose else logging.INFO
    configure_logging(log_type, log_file=log_file, level=level)
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose


@contextmanager
def idapro_database(
    input_binary: str, *, run_auto_analysis: bool = True
) -> Generator[None]:
    """Open an idalib database for the ``with`` block, then close it."""
    # open_database waits for auto-analysis when run_auto_analysis is True (idapro docs).
    idapro.open_database(input_binary, run_auto_analysis)
    try:
        yield
    finally:
        with suppress(Exception):
            idapro.close_database()


def lift_binary_to_llvm(
    input_binary: str,
    output_llvm_ir: str,
    target_mode: str = "host",
    verbose: bool = False,
    annotate_concurrency: bool = False,
    ir_passes: tuple[str, ...] = (),
) -> bool:
    """Lift a binary to LLVM IR via idalib (headless CLI entry point)."""
    from .ida2llvm import BIN2LLVMController, ptext, refreshed_funcs

    if target_mode not in ("host", "ida"):
        logging.error("Invalid target_mode '%s'. Must be 'host' or 'ida'", target_mode)
        return False

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)

    try:
        ptext.clear()
        refreshed_funcs.clear()

        with idapro_database(input_binary):
            bin2llvm = BIN2LLVMController(target_mode=target_mode)
            bin2llvm.initialize()
            bin2llvm.insertAllFunctions()
            bin2llvm.save_to_file(
                output_llvm_ir,
                annotate_concurrency=annotate_concurrency,
                ir_passes=ir_passes,
                source_binary=input_binary,
            )

        logging.info("Successfully lifted binary to LLVM IR: %s", output_llvm_ir)
        return True

    except Exception as e:
        logging.exception(
            "Failed to lift binary: %s\n%s", e, "".join(traceback.format_exc())
        )
        return False


@app.command()
def ida2llvm(
    ctx: typer.Context,
    file: str = typer.Option(..., "-f", "--file", help="Binary file to be analyzed"),
    output: str = typer.Option(
        ..., "-o", "--output", help="Output file for LLVM IR (.ll)"
    ),
    target: str = typer.Option(
        "host", "--target", help="Target triple source: host (default) or ida"
    ),
    annotate_concurrency: bool = typer.Option(
        False,
        "--annotate-concurrency",
        help="Append TLS/syscall/futex concurrency metadata to the lifted LLVM IR",
    ),
    ir_passes: str | None = typer.Option(
        None,
        "--ir-pass",
        "--ir-passes",
        help="Comma-separated IR pass pipeline (available: concurrency, verify)",
    ),
    log_type: str | None = typer.Option(
        None,
        "--log-type",
        help="Log destination: file or console (stderr)",
    ),
    log_file: str | None = typer.Option(
        None,
        "--log-file",
        help="Log file path when --log-type=file",
    ),
    verbose: bool | None = typer.Option(
        None,
        "-v",
        "--verbose",
        help="Enable verbose (DEBUG) logging",
    ),
):
    """Convert Hex-Rays microcode into LLVM IR using IDA Pro 9+ idalib."""
    if target not in ("host", "ida"):
        raise typer.BadParameter("must be one of: host, ida")

    verbose_flag = apply_subcommand_logging(
        ctx, log_type=log_type, log_file=log_file, verbose=verbose
    )
    from .ir_passes import parse_ir_passes

    success = lift_binary_to_llvm(
        input_binary=file,
        output_llvm_ir=output,
        target_mode=target,
        verbose=verbose_flag,
        annotate_concurrency=annotate_concurrency,
        ir_passes=parse_ir_passes(ir_passes),
    )
    if not success:
        sys.exit(1)


@app.command("itanium-to-msvc")
def itanium_to_msvc(
    ctx: typer.Context,
    input: str = typer.Option(..., "-i", "--input", help="Input LLVM IR (.ll)"),
    output: str = typer.Option(..., "-o", "--output", help="Output LLVM IR (.ll)"),
    msvc: bool = typer.Option(
        False,
        "--msvc",
        help="Attempt MSVC remangling via cl.exe/dumpbin.exe (Windows only)",
    ),
    log_type: str | None = typer.Option(
        None,
        "--log-type",
        help="Log destination: file or console (stderr)",
    ),
    log_file: str | None = typer.Option(
        None,
        "--log-file",
        help="Log file path when --log-type=file",
    ),
    verbose: bool | None = typer.Option(
        None,
        "-v",
        "--verbose",
        help="Enable verbose (DEBUG) logging",
    ),
):
    """Rewrite Itanium _Z symbols in LLVM IR; optional MSVC mangling on Windows."""
    from .itanium_to_msvc import CxxfiltNotFoundError, convert_itanium_to_msvc

    apply_subcommand_logging(ctx, log_type=log_type, log_file=log_file, verbose=verbose)
    try:
        convert_itanium_to_msvc(input, output, use_msvc=msvc)
    except CxxfiltNotFoundError as exc:
        print(exc, file=sys.stderr)
        raise typer.Exit(1) from exc
