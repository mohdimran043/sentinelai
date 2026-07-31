"""Architecture fitness functions — these encode spec §3.3 and §6.1 as tests."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "sentinel_ai"

FORBIDDEN_IN_PURE_LAYERS = {
    "torch",
    "torchvision",
    "cv2",
    "av",
    "numpy",
    "ultralytics",
    "supervision",
    "transformers",
    "aio_pika",
    "pika",
    "fastapi",
    "minio",
    "httpx",
    "requests",
    "grpc",
    "sqlalchemy",
    "redis",
}

PURE_LAYERS = ("domain", "ports")


def _module_files(layer: str) -> list[Path]:
    """Modules in a pure layer.

    Raises if the layer is missing or empty: otherwise every check below would
    pass vacuously against zero files and the fitness tests would be worthless.
    """
    directory = PACKAGE_ROOT / layer
    assert directory.is_dir(), f"pure layer '{layer}' does not exist at {directory}"
    modules = sorted(directory.rglob("*.py"))
    assert modules, f"pure layer '{layer}' contains no modules — nothing to verify"
    return modules


def _imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


@pytest.mark.parametrize("layer", PURE_LAYERS)
def test_pure_layers_have_no_io_dependencies(layer: str) -> None:
    offenders: dict[str, set[str]] = {}
    for path in _module_files(layer):
        bad = _imported_roots(path) & FORBIDDEN_IN_PURE_LAYERS
        if bad:
            offenders[str(path.relative_to(PACKAGE_ROOT))] = bad
    assert offenders == {}, f"pure layer '{layer}' imports I/O libraries: {offenders}"


@pytest.mark.parametrize("layer", PURE_LAYERS)
def test_pure_layers_do_not_read_the_clock(layer: str) -> None:
    """Clocks must be arguments so policy is deterministic under test."""
    offenders: dict[str, list[str]] = {}
    for path in _module_files(layer):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        hits = [
            f"{node.func.value.id}.{node.func.attr}"
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "time"
            and node.func.attr in {"time", "monotonic", "monotonic_ns", "time_ns"}
        ]
        if hits:
            offenders[str(path.relative_to(PACKAGE_ROOT))] = hits
    assert offenders == {}, f"pure layer '{layer}' reads the clock: {offenders}"


def test_domain_does_not_import_adapters_or_orchestrator() -> None:
    offenders: dict[str, set[str]] = {}
    for layer in PURE_LAYERS:
        for path in _module_files(layer):
            text = path.read_text(encoding="utf-8")
            bad = {
                name
                for name in ("adapters", "orchestrator", "pipeline", "api")
                if f"sentinel_ai.{name}" in text
            }
            if bad:
                offenders[str(path.relative_to(PACKAGE_ROOT))] = bad
    assert offenders == {}, f"pure layers depend on outer layers: {offenders}"
