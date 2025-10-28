<h1 align="center">🧠 IDAvator - Ride the elevator to lift between microcode and machine.</h1>

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
|:--|:--|:--|
| **Lift** | `idavator ida2llvm` | Convert Hex-Rays microcode (`mba_t`) into LLVM IR. |
| **Drop** | `idavator llvm2ida` | Translate LLVM IR back into IDA, patching binary code or updating microcode. |
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

### ida2llvm

```bash
idat -c -A -S"ida2llvm.py [binary].ll" binary
```

```bash
python llvm2ida.py binary [binary].ll
```

### Requirements

- Ensure you have Python (`>= 3.11`) and llvmlite installed on your system.

```bash
pip install llvmlite
```

- ida2llvm and llvm2ida are tested only in IDA-9.0+

## Acknowledgements

- [sandspeare](https://github.com/Sandspeare)'s [ida2llvm](https://github.com/Sandspeare/ida2llvm) continuation - Thank you for such a great tool!
- [loyaltypollution](https://github.com/loyaltypolution)'s original [ida2llvm](https://github.com/loyaltypollution/ida2llvm): The codebase sandspeare built on, fixing most of the bugs (float, unsupport inst, unsupport typecast and structure) and transforming it from an experimental toy to a stable tool.
