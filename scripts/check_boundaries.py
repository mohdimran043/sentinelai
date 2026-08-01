#!/usr/bin/env python3
"""Fitness function for spec's system boundary: web/ and ai-engine/ never import
each other. `contracts/` is the one sanctioned seam (the UI derives its types from
the OpenAPI document and the event JSON Schema there) and is exempt in both
directions.

This lives outside both trees on purpose — not in `ai-engine/tests/` (that suite
is Python-only and pytest would never discover a Node violation, and a second
team is actively adding tests there) and not under `web/` (three agents are
writing there right now). A repo-root script that needs nothing but the
standard library can run in CI without installing either toolchain, so the
boundary gate never waits on `pip install` or `npm ci`.

Mirrors the rigor of `ai-engine/tests/test_architecture.py`: real parsers
(the `ast` module for Python, import-statement regexes for JS/TS) rather than a
substring grep, so that prose mentioning "ai-engine" in a comment or a JSX
string (there is plenty of it — see web/src/api/config.ts) is documentation,
not a dependency, exactly as that file's docstring says of `sentinel_ai.adapters`
mentions. And per that file's own thesis — "a fitness function that has never
been observed to fail is a hypothesis, not a guard" — `--self-test` below runs
this checker against a scratch fixture containing a deliberate violation on
each side and asserts it is caught, plus a clean fixture and asserts it is not.

Usage:
    python3 scripts/check_boundaries.py             # check the real repo
    python3 scripts/check_boundaries.py --root DIR   # check a fixture instead
    python3 scripts/check_boundaries.py --self-test  # prove the checker works
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
import tempfile
from pathlib import Path

WEB_DIR_NAME = "web"
AI_DIR_NAME = "ai-engine"

WEB_EXCLUDE_DIRS = {"node_modules", "dist", ".git", "build", "coverage"}
WEB_EXTENSIONS = {".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"}

AI_EXCLUDE_DIRS = {".venv", "venv", "__pycache__", ".git", "var", ".pytest_cache", ".mypy_cache", ".ruff_cache"}

# Matches the specifier string in every JS/TS import shape that can name a
# module: `import x from '...'`, bare `import '...'`, `export ... from '...'`,
# `require('...')` and dynamic `import('...')`. Deliberately anchored to the
# keyword so a JSX text node or a `//`/`/* */` comment containing the word
# "ai-engine" is never mistaken for a module specifier.
JS_IMPORT_SPECIFIER_RE = re.compile(
    r"""(?:^|\bimport\b\s*(?:type\s+)?(?:[\w$*{}\s,]+\bfrom\s*)?|\bexport\b\s*(?:\*|\{[^}]*\}|type\s*\{[^}]*\})\s*\bfrom\s*|\brequire\s*\(\s*|\bimport\s*\(\s*)['"]([^'"]+)['"]""",
    re.MULTILINE,
)

# A crude but sufficient path-literal detector for the Python side: a quoted
# string that, once leading `../` segments are stripped, starts with `web/`
# or is exactly `web`. Ordinary imports are caught by the AST walk below;
# this catches the `open("../web/foo")` / `Path(__file__).parents[1] / "web"`
# style of reference that never appears as an `import` statement at all.
PY_WEB_PATH_LITERAL_RE = re.compile(r"^(?:\.\./)*web(?:/.*)?$")


def _iter_files(root: Path, exclude_dirs: set[str], extensions: set[str] | None) -> list[Path]:
    out: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in exclude_dirs for part in path.relative_to(root).parts[:-1]):
            continue
        if extensions is not None and path.suffix not in extensions:
            continue
        out.append(path)
    return out


def _resolves_into(specifier: str, from_file: Path, repo_root: Path, forbidden_top: str) -> bool:
    """True if a JS/TS import specifier reaches into `forbidden_top` at repo root."""
    if specifier.startswith("."):
        resolved = (from_file.parent / specifier).resolve()
        try:
            relative = resolved.relative_to(repo_root.resolve())
        except ValueError:
            # Escapes the repo entirely (e.g. `../../../etc/passwd`) — not this
            # checker's concern, but not a same-repo boundary crossing either.
            return False
        return relative.parts[0:1] == (forbidden_top,) or (
            len(relative.parts) > 0 and relative.parts[0] == forbidden_top
        )
    # Bare specifier: only a concern if it *names* the forbidden tree, e.g. a
    # workspace package literally called "ai-engine" or "ai-engine/whatever".
    return specifier == forbidden_top or specifier.startswith(f"{forbidden_top}/")


def check_web_does_not_import_ai_engine(repo_root: Path) -> list[str]:
    web_root = repo_root / WEB_DIR_NAME
    if not web_root.is_dir():
        return [f"expected {web_root} to exist — nothing to check"]
    violations: list[str] = []
    for path in _iter_files(web_root, WEB_EXCLUDE_DIRS, WEB_EXTENSIONS):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for match in JS_IMPORT_SPECIFIER_RE.finditer(text):
            specifier = match.group(1)
            if _resolves_into(specifier, path, repo_root, AI_DIR_NAME):
                line = text[: match.start()].count("\n") + 1
                violations.append(
                    f"{path.relative_to(repo_root)}:{line}: imports {specifier!r}, "
                    f"which reaches into {AI_DIR_NAME}/"
                )
    return violations


def _py_import_roots(tree: ast.AST) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def _py_web_path_literals(tree: ast.AST) -> list[str]:
    hits: list[str] = []
    for node in ast.walk(tree):
        value = None
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value
        if value and PY_WEB_PATH_LITERAL_RE.match(value.strip()):
            hits.append(value)
    return hits


def check_ai_engine_does_not_import_web(repo_root: Path) -> list[str]:
    ai_root = repo_root / AI_DIR_NAME
    if not ai_root.is_dir():
        return [f"expected {ai_root} to exist — nothing to check"]
    violations: list[str] = []
    for path in _iter_files(ai_root, AI_EXCLUDE_DIRS, {".py"}):
        source = path.read_text(encoding="utf-8", errors="ignore")
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            violations.append(f"{path.relative_to(repo_root)}: could not parse ({exc})")
            continue
        bad_imports = _py_import_roots(tree) & {WEB_DIR_NAME}
        if bad_imports:
            violations.append(f"{path.relative_to(repo_root)}: imports {sorted(bad_imports)}")
        for literal in _py_web_path_literals(tree):
            violations.append(f"{path.relative_to(repo_root)}: references path literal {literal!r}")
    return violations


def check(repo_root: Path) -> list[str]:
    return check_web_does_not_import_ai_engine(repo_root) + check_ai_engine_does_not_import_web(repo_root)


# --------------------------------------------------------------------------
# Self-test: prove the checker actually fails on a real violation, and does
# not false-positive on the legitimate contracts/ seam or on prose that merely
# mentions the other tree's name. Builds two throwaway fixture trees under a
# tempdir; never touches the real web/ or ai-engine/.
# --------------------------------------------------------------------------


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _build_clean_fixture(root: Path) -> None:
    _write(
        root / "contracts" / "openapi" / "ai-engine.yaml",
        "openapi: 3.1.0\ninfo:\n  title: fixture\n  version: '0'\npaths: {}\n",
    )
    _write(
        root / "web" / "src" / "api" / "config.ts",
        "// The engine (`ai-engine/sentinel_ai/api/app.py`) sets no CORS headers.\n"
        "// This comment mentions ai-engine in prose only — not an import.\n"
        "import { z } from 'zod'\n"
        "import spec from '../../contracts/openapi/ai-engine.yaml'\n"
        "export const ENGINE_BASE = '/engine'\n",
    )
    _write(
        root / "web" / "src" / "routes" / "DashboardPage.tsx"
        , "export const Hint = () => <code>ai-engine/cameras.json</code>\n",
    )
    _write(
        root / "ai-engine" / "sentinel_ai" / "domain" / "entities.py",
        "from dataclasses import dataclass\n\n\n@dataclass(frozen=True)\nclass Scene:\n    id: str\n",
    )


def _build_violation_fixture(root: Path) -> None:
    _build_clean_fixture(root)
    # A web module actually importing engine source, not just mentioning it.
    _write(
        root / "web" / "src" / "bad" / "leaky.ts",
        "import { Detector } from '../../../ai-engine/sentinel_ai/ports/detector'\n",
    )
    # An ai-engine module actually importing the UI package.
    _write(
        root / "ai-engine" / "sentinel_ai" / "bad" / "leaky.py",
        "import web\n\nweb.do_something()\n",
    )


def self_test() -> int:
    with tempfile.TemporaryDirectory(prefix="boundary-check-clean-") as clean_dir:
        clean_root = Path(clean_dir)
        _build_clean_fixture(clean_root)
        clean_violations = check(clean_root)
        if clean_violations:
            print("SELF-TEST FAILED: clean fixture raised false positives:")
            for v in clean_violations:
                print(f"  - {v}")
            return 1
        print("self-test: clean fixture (contracts/ import + prose mentions) -> 0 violations. OK")

    with tempfile.TemporaryDirectory(prefix="boundary-check-dirty-") as dirty_dir:
        dirty_root = Path(dirty_dir)
        _build_violation_fixture(dirty_root)
        dirty_violations = check(dirty_root)
        found_web_side = any("leaky.ts" in v for v in dirty_violations)
        found_ai_side = any("leaky.py" in v for v in dirty_violations)
        if not (found_web_side and found_ai_side):
            print("SELF-TEST FAILED: planted violations were not detected.")
            print(f"  web-side caught: {found_web_side}, ai-engine-side caught: {found_ai_side}")
            print("  violations reported:")
            for v in dirty_violations:
                print(f"  - {v}")
            return 1
        print("self-test: planted web->ai-engine import -> caught. OK")
        print("self-test: planted ai-engine->web import -> caught. OK")

    print("SELF-TEST PASSED: the checker fails on a real violation and is silent on a clean tree.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repo root to check (default: this script's repo)",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run the positive-control self-test instead of checking --root",
    )
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    violations = check(args.root.resolve())
    if violations:
        print("Boundary violation: web/ and ai-engine/ must not import each other.")
        print(f"({AI_DIR_NAME}/ and {WEB_DIR_NAME}/ may both read contracts/ — that is the sanctioned seam.)\n")
        for v in violations:
            print(f"  - {v}")
        return 1

    print(f"OK: no boundary violations between {WEB_DIR_NAME}/ and {AI_DIR_NAME}/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
