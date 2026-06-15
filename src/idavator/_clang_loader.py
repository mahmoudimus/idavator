"""Shared libclang discovery/loading helpers.

Centralizes libclang lookup so runtime code and tests use the same logic. The
clang Python bindings normally come from ``idavator._vendor.clang``; this module
is otherwise self-contained (stdlib only).

A loaded ``Index`` is only useful to the oracle if its libclang can actually
parse a translation unit. IDA's macOS libclang can; IDA's Linux ``libclang.so``
(idalib) exposes only the type-info C API and returns a *null* TU from
``clang_parseTranslationUnit`` even for a trivial function. So
:func:`load_clang_index` SMOKE-TESTS a parse and, if IDA's libclang fails it,
falls back to the pip ``libclang`` wheel -- which bundles a parse-capable
libclang shared library *and* its own matching ``clang.cindex`` bindings. The
fallback uses that wheel's own bindings (not the vendored ones) because a
shared library only works with the binding version it was built for.
"""

from __future__ import annotations

import pathlib
import platform
import typing

# A trivial, header-free TU. If parsing this raises (null TU), the libclang is
# not usable for the oracle and we fall back to a pip-provided one.
_SMOKE_SRC = "int __idavator_probe(int x){ return x + 1; }"


def _index_can_parse(index: typing.Any, cindex_mod: typing.Any) -> bool:
    """True iff ``index`` can parse a trivial TU without raising.

    ``cindex_mod`` is the ``clang.cindex`` module the index came from (so we
    catch *its* TranslationUnitLoadError, not a foreign one)."""
    load_error = getattr(cindex_mod, "TranslationUnitLoadError", Exception)
    try:
        tu = index.parse(
            "p.c", args=["-x", "c", "-w"],
            unsaved_files=[("p.c", _SMOKE_SRC)])
    except load_error:
        return False
    except Exception:  # noqa: BLE001 - any parse failure => unusable
        return False
    # A non-null TU that exposes the function decl confirms the frontend works.
    try:
        return any(c.kind.name == "FUNCTION_DECL"
                   for c in tu.cursor.get_children())
    except Exception:  # noqa: BLE001
        return False


def _load_pip_libclang_index() -> typing.Any | None:
    """Load an Index from the pip ``libclang`` wheel using ITS OWN bindings.

    The wheel ships ``clang/native/libclang.so`` plus a ``clang.cindex`` matched
    to it. We must use that matched binding (a library only registers against the
    binding version it was built for), so this deliberately imports the top-level
    ``clang.cindex`` rather than the vendored copy. Returns a parse-verified
    Index or None."""
    try:
        import clang.cindex as pip_cindex
    except Exception:  # noqa: BLE001 - wheel not installed
        return None

    # The wheel exposes its bundled .so via conf.get_filename(); pointing the
    # Config at it explicitly avoids depending on system library search paths.
    try:
        lib_path = pip_cindex.conf.get_filename()
        if lib_path:
            pip_cindex.Config.set_library_file(lib_path)
    except Exception:  # noqa: BLE001 - already loaded / no bundled lib
        pass

    try:
        index = pip_cindex.Index.create()
    except Exception:  # noqa: BLE001 - library load failed
        return None

    return index if _index_can_parse(index, pip_cindex) else None


def _platform_lib_name(system_name: str | None = None) -> str:
    system = system_name or platform.system()
    return {
        "Linux": "libclang.so",
        "Darwin": "libclang.dylib",
        "Windows": "libclang.dll",
    }.get(system, "libclang.so")


def discover_libclang_candidates(
    *,
    ida_directory: str | pathlib.Path | None = None,
    project_root: str | pathlib.Path | None = None,
    system_name: str | None = None,
) -> list[pathlib.Path]:
    """Return ordered candidate paths for libclang."""
    system = system_name or platform.system()
    lib_name = _platform_lib_name(system)
    candidates: list[pathlib.Path] = []
    seen: set[pathlib.Path] = set()

    def add(path: pathlib.Path) -> None:
        p = path.expanduser()
        if p in seen:
            return
        seen.add(p)
        candidates.append(p)

    if ida_directory:
        ida_dir = pathlib.Path(ida_directory)
        add(ida_dir / lib_name)
        add(ida_dir / "Contents" / "MacOS" / lib_name)

    # Useful in headless test/runtime environments.
    env_ida_dir = __import__("os").environ.get("IDA_INSTALL_DIR")
    if env_ida_dir:
        env_dir = pathlib.Path(env_ida_dir)
        add(env_dir / lib_name)
        add(env_dir / "Contents" / "MacOS" / lib_name)

    # Project-local development copy.
    if project_root:
        add(pathlib.Path(project_root) / lib_name)

    # macOS app bundle fallbacks.
    if system == "Darwin":
        add(pathlib.Path("/Applications/IDA Professional 9.2.app/Contents/MacOS") / lib_name)
        add(pathlib.Path("/Applications/IDA Professional 9.1.app/Contents/MacOS") / lib_name)
        app_root = pathlib.Path("/Applications")
        if app_root.exists():
            for base in sorted(app_root.glob("IDA Professional *.app/Contents/MacOS")):
                add(base / lib_name)

    # Explicit override path.
    env_libclang = __import__("os").environ.get("IDAVATOR_LIBCLANG_PATH")
    if env_libclang:
        add(pathlib.Path(env_libclang))

    return candidates


def load_clang_index(
    *,
    ida_directory: str | pathlib.Path | None = None,
    project_root: str | pathlib.Path | None = None,
    system_name: str | None = None,
    allow_default_loader: bool = False,
) -> tuple[typing.Any | None, pathlib.Path | None, list[pathlib.Path]]:
    """Load clang Index from discovered libclang path.

    Returns:
        (index_or_none, loaded_path_or_none, tried_paths)
    """
    try:
        from idavator._vendor.clang import cindex as vendored_cindex
        from idavator._vendor.clang.cindex import Config, Index
    except ImportError:
        # Vendored bindings unavailable: still try the pip libclang wheel.
        return _load_pip_libclang_index(), None, []

    candidates = discover_libclang_candidates(
        ida_directory=ida_directory,
        project_root=project_root,
        system_name=system_name,
    )

    for path in candidates:
        if not path.exists():
            continue

        # set_library_file can fail if libclang is already loaded; in that case,
        # try Index.create() anyway against the currently loaded library.
        try:
            Config.set_library_file(str(path.resolve()))
        except Exception:
            pass

        try:
            index = Index.create()
        except Exception:
            continue
        # Only accept IDA's libclang if it can actually PARSE -- the Linux idalib
        # libclang loads and creates an Index but returns a null TU. If it can't
        # parse, keep trying other candidates, then the pip fallback below.
        if _index_can_parse(index, vendored_cindex):
            return index, path, candidates

    # IDA's libclang is unusable for parsing (or none was found): fall back to a
    # pip-provided, parse-capable libclang (its own matched bindings).
    pip_index = _load_pip_libclang_index()
    if pip_index is not None:
        return pip_index, None, candidates

    if allow_default_loader:
        # Optional last resort for environments that intentionally do not use
        # IDA's packaged libclang.
        try:
            return Index.create(), None, candidates
        except Exception:
            pass

    return None, None, candidates
