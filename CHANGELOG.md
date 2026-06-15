# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Release guard: tagging a release now fails fast if the git tag does not match
  `idavator.__version__`, preventing a silent version-mismatch publish.

### Changed

- The AST-equivalence oracle is now fully self-contained: the libclang loader and
  clang Python bindings are vendored under `idavator._vendor` instead of imported
  from a sibling checkout, removing the cross-repo dependency (and the conftest
  path-injection that crashed in container CI).

### Fixed

- `clang_available()` now reflects whether IDA's native libclang actually loads,
  not merely whether the vendored loader imports, so the oracle tests skip
  cleanly where IDA is absent (e.g. CI) instead of erroring.
- The AST oracle now works on Linux idalib, where IDA's bundled libclang creates
  an `Index` but cannot parse a translation unit (its parsing frontend returns a
  null TU). The loader smoke-tests a parse and falls back to the pip `libclang`
  wheel (with its own matched bindings) when IDA's libclang cannot parse; macOS
  keeps using IDA's own libclang. The operator-equivalence canonicalization no
  longer depends on a libclang >= 19 feature (it derives binary operators from
  tokens), and an un-parseable pseudocode body is reported as inconclusive rather
  than a false divergence.
- The idalib test suite no longer crashes mid-run under idalib's database
  open/close cycle limit: CI runs it with `pytest-forked` (a fresh process per
  test).

## [0.1.0] - 2026-06-15

Initial release: a bi-directional bridge between IDA Pro's Hex-Rays microcode and
LLVM IR.

### Added

- **Lift** (`idavator ida2llvm`): headless microcode (`mba_t`) → LLVM IR via
  idalib, with an optional post-lift IR pass pipeline (`concurrency`, `verify`).
- **Drop**: lower LLVM IR back into Hex-Rays microcode and render it through
  `decompile()`, with a native-decompile round-trip fidelity harness.
- IDA GUI plugin: **Lifting Viewer** (`Ctrl+Alt+L`) and **Apply LLVM IR…**.
- `typer` CLI entry point (`idavator`) and an `ida-plugin.json` manifest for the
  IDA Plugin Repository / `hcli`.
- Packaging & release workflow: PyPI trusted publishing, GitHub releases, and an
  offline + idalib (Docker) CI matrix.
- `ida-plugin.json` ↔ `idavator.__version__` version-sync (`tools/sync_plugin_version.py`,
  a `.githooks/pre-commit` hook, and a `tests/test_plugin_manifest.py` CI backstop).

[Unreleased]: https://github.com/mahmoudimus/idavator/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/mahmoudimus/idavator/releases/tag/v0.1.0
