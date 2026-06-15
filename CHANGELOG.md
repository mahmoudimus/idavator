# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
