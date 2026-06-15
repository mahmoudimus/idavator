"""Guard against ``ida-plugin.json`` drifting from ``idavator.__version__``.

The IDA Plugin Repository and ``hcli`` read the plugin version out of
``ida-plugin.json``; ``pyproject.toml`` derives the package version from
``idavator.__version__``. This test keeps the two in lockstep. It parses
``__version__`` with ``ast`` rather than importing ``idavator`` (whose package
pulls in third-party / IDA-only dependencies), mirroring
``tools/sync_plugin_version.py``. It carries no ``ida`` marker, so it runs in
the offline CI job (``pytest -m "not ida"``).
"""
import ast
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
INIT = ROOT / "src" / "idavator" / "__init__.py"
MANIFEST = ROOT / "ida-plugin.json"


def _package_version() -> str:
    tree = ast.parse(INIT.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__version__" for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"could not find __version__ in {INIT}")


def test_manifest_version_matches_package() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["plugin"]["version"] == _package_version(), (
        "ida-plugin.json version is out of sync with idavator.__version__; "
        "run: python tools/sync_plugin_version.py"
    )
