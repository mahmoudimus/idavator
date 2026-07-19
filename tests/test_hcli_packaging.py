"""Regression tests for the source-only HCLI plugin archive."""

from __future__ import annotations

import ast
import json
import pathlib
import runpy
import zipfile


ROOT = pathlib.Path(__file__).resolve().parents[1]
INIT = ROOT / "src" / "idavator" / "__init__.py"
MANIFEST = ROOT / "ida-plugin.json"


def _package_version() -> str:
    tree = ast.parse(INIT.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__version__"
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"could not find __version__ in {INIT}")


def _build_hcli_archive(output: pathlib.Path) -> pathlib.Path:
    namespace = runpy.run_path(str(ROOT / "tools" / "build_hcli_archive.py"))
    return namespace["build_hcli_archive"](output)


def test_manifest_uses_the_exact_pypi_distribution() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert manifest["plugin"]["entryPoint"] == "entry_stub.py"
    assert manifest["plugin"]["pythonDependencies"] == [
        f"idavator=={_package_version()}"
    ]


def test_entry_stub_loads_the_real_plugin_entry() -> None:
    assert (ROOT / "entry_stub.py").read_text(encoding="utf-8") == (
        '"""Load Idavator after HCLI installs its Python distribution."""\n\n'
        "from idavator.gui import PLUGIN_ENTRY\n"
    )


def test_hcli_archive_contains_only_runtime_plugin_files(tmp_path: pathlib.Path) -> None:
    archive = _build_hcli_archive(tmp_path / "idavator.zip")

    with zipfile.ZipFile(archive) as package:
        assert set(package.namelist()) == {
            "ida-plugin.json",
            "entry_stub.py",
            "LICENSE",
            "README.md",
        }
