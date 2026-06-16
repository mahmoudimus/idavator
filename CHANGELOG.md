# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Release guard: tagging a release now fails fast if the git tag does not match
  `idavator.__version__`, preventing a silent version-mismatch publish.

### Changed

- The AST-equivalence oracle now treats a leading-underscore-count skew on libc
  symbols as cosmetic (`__errno_location` == `_errno_location`, `__assert_fail` ==
  `_assert_fail`), so the drop decline gate no longer false-declines a faithful
  body when IDA renders a libc callee with a different underscore count across
  builds. Conservative: it collapses only a run of 2+ leading underscores to one
  and never merges two genuinely different symbols (`foo` and `_foo` stay
  distinct).
- The AST-equivalence oracle is now fully self-contained: the libclang loader and
  clang Python bindings are vendored under `idavator._vendor` instead of imported
  from a sibling checkout, removing the cross-repo dependency (and the conftest
  path-injection that crashed in container CI).
- Drop-vs-native pseudocode tests now compare with a build-tolerant structural
  matcher (`tests/render_tolerance.structural_equiv`) instead of a raw
  name-renamed string equality. The amd64 idalib build is a richer/stricter native
  oracle than arm64 — its native decompile carries DWARF param names + types and a
  `__readfsqword` stack canary that a weakly-typed LLVM-IR drop cannot reproduce —
  so the SAME faithful drop that is byte-identical to arm64-native diverges from
  amd64-native on those benign, type-driven rendering axes. `structural_equiv`
  collapses exactly those axes (types/casts, the canary read + the BYREF `= 0;`
  init it guards, leading-underscore count, a weak-`int` return materialization,
  and a single-use register-temp copy) and compares only the statement / call /
  constant / control-flow skeleton, via a one-directional drop→native identifier
  homomorphism that tolerates a benign weak-typing value SPLIT while still
  rejecting a wrong callee/constant/string, a missing/extra/reordered statement, a
  value MERGE, or a struct-field-vs-raw-offset access.

### Fixed

- Two drop-vs-native idalib tests now also tolerate the amd64 IDA 9.2 build (a
  different decompiler version than 9.3, with benign per-build renderings).
  (1) `test_drop_cursor_struct_define::test_extent_scan_read_ioctl_has_buffer_arg`
  broadens its ioctl detection to match BOTH the plain `ioctl(...)` (9.3) and the
  cast-wrapped `((signed __int32 (__fastcall *)(...))ioctl)(...)` (9.2) renderings
  of the SAME 3-arg call, while STILL verifying the `&fiemap_buf` buffer operand is
  a call argument (not weakened to match any ioctl; a buffer-dropped ioctl still
  fails). (2) `test_drop_stackargs`'s `create_hard_link` xfail signature now keys
  on the build-invariant weak-`int`-vs-`bool` return-materialization axis (native
  returns a bare two-arm `return 1;`/`return 0;`; the drop materializes
  `LOBYTE(<tmp>) = 1/0; return <tmp>`), so it recognizes BOTH the 9.3 nested-guard
  rendering AND the 9.2 full-suite rendering where the drop also combines the guard
  and Hex-Rays aliases one register temp across the (intact) variadic `printf`/
  `error` args. The cp.ll IR genuinely carries the 5-arg `error` / 2-arg `printf`
  and the deterministic converter lifts them in full, so the collapsed visible arg
  list is a Hex-Rays lvar-aliasing render artifact, not a dropped arg. The
  signature stays specific: a wrong callee/constant, a dropped IR arg, a corrupted
  body, a return into a global, or a native that also materializes still FAIL.
- The `set_program_name` and `hash_lookup` idalib drop tests are now build-
  conditional, recognizing two amd64 behaviours the arm64-centric assertions
  previously misflagged as failures. (1) `set_program_name` stores to the glibc
  interposition externs `__progname` / `__progname_full`, which amd64 IDA exposes
  only as type-library display names with no get_name_ea-resolvable address -- so
  BOTH the drop AND Hex-Rays' own native decompile render those stores as
  `*(_QWORD *)0xFFFFFFFFFFFFFFFF` (faithful, not a crash; full-body faithfulness is
  asserted by `test_drop_global_reloc::test_set_program_name_drop_equals_native`),
  so the no-BADADDR assertion now applies only where the extern is nameable
  (arm64). (2) `hash_lookup` carries a pre-existing SROA-residual divergence severe
  enough on amd64 to trip the B5 self-verify decline gate, so the drop correctly
  falls back to native there -- the over-deref invariant is asserted only where the
  drop survives.
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
- The round-trip fidelity ledger no longer reports a FALSE divergence on the amd64
  idalib build when the fallback `libclang` (clang-18) silently drops an `if` /
  `while` / `do` controlling expression to nil on a Hex-Rays construct it cannot
  parse (e.g. a comma-operator/assignment embedded in an `||` guard, as in
  `rpl_fclose`). Such a degenerate drop-side parse is now treated as INCONCLUSIVE
  (the round trip reports the body "unparseable", fidelity unverified) rather than
  a divergence. The change is scoped to `fidelity_ledger`; the B5 decline gate
  (which consumes `matches`) keeps declining a positively-divergent degraded body
  unchanged, so `setlocale_null_unlocked` / `do_copy` still decline correctly.
- The five amd64-only "drop diverges from native" typing-class test failures are
  resolved: `copy` and `quotearg_buffer` now pass via the build-tolerant matcher
  (their only residuals were the benign type/canary/underscore/value-split axes),
  while `create_hard_link`, `extent_copy`, and `transfer_entries` — which carry a
  GENUINE per-build structural divergence on amd64 (a combined-vs-nested guard, an
  `extent_scan` struct scalarized to raw offset arithmetic, and a DWARF
  struct-field walk rendered as raw pointer-offset arithmetic, respectively) — are
  marked xfail under a divergence-specific signature that fires ONLY on that known
  shape (any other divergence still fails) and only on the build whose native
  diverges (they pass on a build whose native matches the drop).
- The two private-string-constant tests locate the reference IDB literal by
  CONTENT (scanning the string table for the exact bytes) instead of the
  hard-coded arm64 auto-name `aValidOptionsOptions`, which the amd64 IDA truncates
  to `aValidOptionsOp`; the tests are now build-agnostic.

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
