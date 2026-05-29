<h1 align="center">🧠 IDAvator - Ride the elevator to lift between microcode and machine.</h1>

<p align="center">
<a href="https://github.com/mahmoudimus/idavator/actions/workflows/python.yml"><img src="https://github.com/mahmoudimus/idavator/actions/workflows/python.yml/badge.svg" alt="idavator tests" /></a>
</p>

<h4 align="center">
<p>
<a href=#about>About</a> |
<a href=#quickstart>QuickStart</a> |
<a href=#acknowledgements>Acknowledgements</a> |
<p>
</h4>

## About

**IDAvator** is a bi-directional bridge between **IDA Pro’s Hex-Rays microcode** and **LLVM IR**. It lets you **lift** decompiler microcode into LLVM for analysis, optimization, or deobfuscation. Then, **drop** it back into IDA, patched and ready for further exploration.

### Features

| Action | Command | Description |
| :-- | :-- | :-- |
| **Lift** | `idavator ida2llvm` | Headless: microcode (`mba_t`) → LLVM IR via idalib. |
| **Drop** | IDA plugin (GUI) | **Edit → IDAvator → Apply LLVM IR...** (microcode drop; patch/export WIP). |
| **Lift (interactive)** | IDA plugin (GUI) | **Edit → IDAvator → Lifting Viewer** (`Ctrl+Alt+L`). |
| **Optimize** | Use `opt` or any LLVM pass pipeline | Apply LLVM analyses or transformations (e.g., constant propagation, CFG cleanup). |
| **Deobfuscate** | Combine with IDAvator’s switch-flattening or simplification passes | Simplify complex control flow graphs. |
| **Patch / Rebuild** | Patch directly in IDA or export `.o` / `.bin` | Choose live patching or external reconstruction. |

---

### Architecture

```text
  +-----------+           +------------------+           +-------------+
  |  IDA Pro  |  ida2llvm |     LLVM IR      | llvm2ida  |  Patched IDA|
  | (microcode) +---------> (optimize, deobf) +----------> (clean code) |
  +-----------+           +------------------+           +-------------+
         ^                        |
         |        idavator        |
         +------------------------+
```

## QuickStart

### Install

```bash
pip install -e .
```

Requires Python `>= 3.10`, IDA Pro 9+ with idalib, and dependencies from `pyproject.toml` (llvmlite, numpy, typer).

### Lift (ida2llvm)

Lift a binary to LLVM IR (headless via idalib):

```bash
idavator ida2llvm -f binary -o output.ll
```

| Option | Description |
| :-- | :-- |
| `-f`, `--file` | Input binary to analyze |
| `-o`, `--output` | Output LLVM IR path (`.ll`) |
| `--target` | Target triple: `host` (default) or `ida` |
| `--ir-pass`, `--ir-passes` | Comma-separated post-lift IR pass pipeline |
| `--annotate-concurrency` | Compatibility alias for `--ir-pass concurrency` |
| `--log-type` | Log destination: `file` (default) or `console` (stderr) |
| `--log-file` | Log file path when `--log-type=file` (default: `idavator.log`) |
| `-v`, `--verbose` | Enable DEBUG logging |

```bash
idavator ida2llvm -f binary -o output.ll --log-type console -v
```

Logging is configured before the lift module loads, so early messages use the chosen destination.

Run post-lift IR passes while writing the output:

```bash
idavator ida2llvm -f binary -o output.ll --ir-pass concurrency,verify
```

Available passes:

| Pass | Description |
| :-- | :-- |
| `concurrency` | Appends metadata for recognized TLS helper calls, syscalls, and futex syscalls |
| `verify` | Parses and verifies the final LLVM IR with llvmlite |

For older scripts, `--annotate-concurrency` is still accepted and enables the `concurrency` pass.

### Regression baselines (pytest)

Metric baselines live under `tests/artifacts/` and compare lift/pass output without
requiring full IR text diffs. Refresh them after intentional improvements:

```bash
pip install -e ".[dev]"
pytest --baseline-update
pytest
```

IDA-backed lift checks run only when idalib is available:

```bash
pytest -m ida
```

### IDA plugin (GUI)

Install the package into IDA’s Python (`pip install -e .` from this repo), then load the plugin via `ida-plugin.json` (IDA 9+).

| Menu | Hotkey | Purpose |
| :-- | :-- | :-- |
| **Edit → IDAvator → Lifting Viewer** | `Ctrl+Alt+L` | Interactive lift: add functions, declare-only toggle, preview IR, save `.ll` |
| **Edit → IDAvator → Apply LLVM IR...** | | Drop optimized `.ll` back into the open database (microcode) |

Headless batch lift remains CLI-only (`idavator ida2llvm`). Drop is not on the CLI.

Workflow:

1. Lift in IDA (viewer) or headless (`idavator ida2llvm -f binary -o output.ll`).
2. Optimize LLVM offline (`opt`, custom passes).
3. **Edit → IDAvator → Apply LLVM IR...** on the same database.

```bash
pip install -e .
python -m idavator ida2llvm -f binary -o output.ll
```

### Requirements

- Python `>= 3.10`, llvmlite, and IDA Pro 9+ with idalib

## Acknowledgements

- [sandspeare](https://github.com/Sandspeare)'s [ida2llvm](https://github.com/Sandspeare/ida2llvm) continuation - Thank you for such a great tool!
- [loyaltypollution](https://github.com/loyaltypolution)'s original [ida2llvm](https://github.com/loyaltypollution/ida2llvm): The codebase sandspeare built on, fixing most of the bugs (float, unsupport inst, unsupport typecast and structure) and transforming it from an experimental toy to a stable tool.
