"""
Repository rules: no link to other repositories, no legacy loaders, contract A at schema_version 1 only.

    python -m pytest -q tests/test_compliance.py
"""
import ast
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

OTHER_REPOS = ("aerolink", "aeroplan", "propalyser")
THIS_FILE = Path(__file__).resolve()


def _tracked(*patterns):
    out = subprocess.run(["git", "ls-files", *patterns], cwd=ROOT, capture_output=True, text=True, check=True)
    return [ROOT / p for p in out.stdout.splitlines() if (ROOT / p).resolve() != THIS_FILE]


def _python_files():
    return _tracked("*.py")


def test_no_import_of_another_repository():
    for path in _python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                assert name.split(".")[0].lower() not in OTHER_REPOS, f"{path.name} imports {name}"


def test_no_path_to_another_repository():
    # A path names the repository folder, for example ../Aeroplan/, GitHub\Propalyser or /aerolink/
    pattern = re.compile(r"[\\/]\s*(%s)\b|\b(%s)\s*[\\/]" % ("|".join(OTHER_REPOS), "|".join(OTHER_REPOS)), re.I)
    for path in _tracked("*.py", "*.yaml", "*.yml", "*.json", "*.md", "*.txt", "*.cfg", "*.toml", "*.ini"):
        text = path.read_text(encoding="utf-8", errors="replace")
        match = pattern.search(text)
        assert match is None, f"{path.relative_to(ROOT)} holds a path to another repository: {match.group(0)!r}"


def test_no_legacy_loader_remains():
    for path in _python_files():
        text = path.read_text(encoding="utf-8")
        assert ".eqx" not in text, f"{path.name} mentions .eqx files"
        assert not re.search(r"format_version\W{1,6}2\b|\(2, *FORMAT_VERSION\)", text), \
            f"{path.name} handles format_version 2"
        assert not re.search(r"legacy_eqx|_LEGACY_|_RESET_FROM_V2|loadprop|_OldCase", text), \
            f"{path.name} holds a legacy loader name"


def test_saved_files_use_format_version_1():
    import json
    import yaml

    for path in _tracked("designs/*.yaml"):
        assert yaml.safe_load(path.read_text(encoding="utf-8"))["format_version"] == 1, path.name
    for path in _tracked("propellers/*"):
        assert path.name.endswith(".state.json"), f"{path.name} is not a state file"
        assert json.loads(path.read_text(encoding="utf-8"))["format_version"] == 1, path.name


def test_exporter_writes_schema_version_1_only():
    from proptimiser import export_performance as ep

    assert ep.SCHEMA == "prop-performance" and ep.SCHEMA_VERSION == 1
    source = (ROOT / "src" / "proptimiser" / "export_performance.py").read_text(encoding="utf-8")
    assert len(re.findall(r'attrs\["schema_version"\]', source)) == 1
    assert "schema_version = 2" not in source and "SCHEMA_VERSION = 2" not in source
