"""Architecture invariants: world/ and agent/ never import each other; only
engine/save_store.py imports sqlite3; every json.dumps uses ensure_ascii=False."""

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "llm_vil"


def imports_of(path: Path, pkg: str | None = None) -> set[str]:
    if pkg is None:
        pkg = ".".join(path.relative_to(SRC.parent).with_suffix("").parts[:-1])
    out: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = pkg.split(".")
                base = base[: len(base) - node.level + 1]
                mod = ".".join(base + ([node.module] if node.module else []))
            else:
                mod = node.module or ""
            out.add(mod)
            out.update(f"{mod}.{a.name}" for a in node.names)
    return out


def files(sub: str) -> list[Path]:
    return sorted((SRC / sub).rglob("*.py"))


def test_world_never_imports_agent():
    for f in files("world"):
        bad = {m for m in imports_of(f) if m.startswith("llm_vil.agent")}
        assert not bad, f"{f}: {bad}"


def test_agent_never_imports_world():
    for f in files("agent"):
        bad = {m for m in imports_of(f) if m.startswith("llm_vil.world")}
        assert not bad, f"{f}: {bad}"


def test_only_save_store_uses_sqlite():
    users = sorted(str(f.relative_to(SRC)) for f in SRC.rglob("*.py")
                   if any(m.split(".")[0] == "sqlite3" for m in imports_of(f)))
    assert users == ["engine/save_store.py"]


def test_json_dumps_always_ensure_ascii_false():
    for f in SRC.rglob("*.py"):
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "dumps"
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == "json"):
                kws = {k.arg: k.value for k in node.keywords}
                assert "ensure_ascii" in kws and kws["ensure_ascii"].value is False, f


def test_boundary_checker_detects_violations(tmp_path):
    probe = tmp_path / "probe.py"  # never write into src/
    probe.write_text("from ..agent import memory\nimport sqlite3\n", encoding="utf-8")
    found = imports_of(probe, pkg="llm_vil.world")
    assert "llm_vil.agent" in found and "sqlite3" in found
