"""Architecture fitness functions — these encode spec §3.3 and §6.1 as tests."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "sentinel_ai"

FORBIDDEN_IN_PURE_LAYERS = frozenset(
    {
        # Third-party I/O and model runtimes.
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
        # Standard-library ambient state. Banning the *module* makes the alias,
        # from-import and never-heard-of-it-yet variants (time.perf_counter,
        # datetime.datetime.now, os.environ, random.random) unrepresentable
        # rather than merely undetected — see the clock-call matcher below,
        # which is defence in depth, not the primary guard.
        "time",
        "datetime",
        "random",
        "os",
        "importlib",
    }
)

OUTER_LAYERS = ("adapters", "orchestrator", "pipeline", "api", "config")
"""Layers a pure module may not depend on.

`config` is here because `Settings()` reads `.env` from disk and pulls in
pydantic-settings: importing it from `domain/` would make policy depend on
process environment.
"""

CLOCK_READERS = frozenset(
    {
        "time",
        "monotonic",
        "monotonic_ns",
        "time_ns",
        "perf_counter",
        "perf_counter_ns",
        "process_time",
        "process_time_ns",
        "thread_time",
        "clock_gettime",
        "clock_gettime_ns",
        "sleep",
        "now",
        "utcnow",
        "today",
        "gmtime",
        "localtime",
    }
)
"""Names that read wall/monotonic time, matched as bare calls or attributes."""

DYNAMIC_IMPORTERS = frozenset({"__import__", "import_module"})

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


def _package_of(path: Path) -> str:
    """The dotted package a module file lives in, e.g. `sentinel_ai.domain.policy`."""
    relative = path.relative_to(PACKAGE_ROOT.parent)
    return ".".join(relative.parts[:-1])


def _forbidden_import_roots(tree: ast.AST) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots & FORBIDDEN_IN_PURE_LAYERS


def _called_name(node: ast.Call) -> str | None:
    """The trailing identifier of a call target, however it was reached.

    `monotonic()`, `t.monotonic()` and `datetime.datetime.now()` all reduce to
    their final name, so aliasing and attribute chains cannot hide a clock read.
    """
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _clock_reads(tree: ast.AST) -> list[str]:
    return [
        ast.unparse(node.func)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _called_name(node) in CLOCK_READERS
    ]


def _dynamic_imports(tree: ast.AST) -> list[str]:
    """`importlib.import_module("torch")` and `__import__("torch")` are imports."""
    return [
        ast.unparse(node)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _called_name(node) in DYNAMIC_IMPORTERS
    ]


def _absolute_module(package: str, level: int, module: str | None) -> str:
    """Resolve a (possibly relative) `from ... import` target to a dotted path."""
    if level == 0:
        return module or ""
    parts = package.split(".")
    assert level <= len(parts), f"relative import level {level} escapes package '{package}'"
    base = ".".join(parts[: len(parts) - level + 1])
    return f"{base}.{module}" if module else base


def _import_targets(tree: ast.AST, package: str) -> set[str]:
    """Every dotted module path this source imports, relative imports resolved."""
    targets: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            targets.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = _absolute_module(package, node.level, node.module)
            targets.add(module)
            # `from sentinel_ai import adapters` names the layer in the alias.
            targets.update(f"{module}.{alias.name}" for alias in node.names)
    return targets


def _outer_layer_dependencies(source: str, package: str) -> set[str]:
    """Outer layers imported by this module.

    AST-based, not a substring search: prose in a docstring that happens to
    mention `sentinel_ai.adapters` is documentation, not a dependency, and a
    relative `from ..adapters import x` is a dependency even though the string
    `sentinel_ai.adapters` never appears in the file.
    """
    targets = _import_targets(ast.parse(source), package)
    return {
        qualified
        for layer in OUTER_LAYERS
        if (qualified := f"sentinel_ai.{layer}")
        and any(target == qualified or target.startswith(f"{qualified}.") for target in targets)
    }


def _purity_offences(source: str, package: str) -> list[str]:
    """Every purity violation the fitness functions above can see in one module.

    Composed from the same detectors the layer scans use, so the self-tests
    below exercise the real matchers rather than a parallel implementation.
    """
    tree = ast.parse(source)
    offences = [f"forbidden-import:{root}" for root in sorted(_forbidden_import_roots(tree))]
    offences += [f"clock-read:{hit}" for hit in _clock_reads(tree)]
    offences += [f"dynamic-import:{hit}" for hit in _dynamic_imports(tree)]
    offences += [f"outer-layer:{dep}" for dep in sorted(_outer_layer_dependencies(source, package))]
    return offences


@pytest.mark.parametrize("layer", PURE_LAYERS)
def test_pure_layers_have_no_io_dependencies(layer: str) -> None:
    offenders: dict[str, set[str]] = {}
    for path in _module_files(layer):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        bad = _forbidden_import_roots(tree)
        if bad:
            offenders[str(path.relative_to(PACKAGE_ROOT))] = bad
    assert offenders == {}, f"pure layer '{layer}' imports I/O libraries: {offenders}"


@pytest.mark.parametrize("layer", PURE_LAYERS)
def test_pure_layers_do_not_read_the_clock(layer: str) -> None:
    """Clocks must be arguments so policy is deterministic under test."""
    offenders: dict[str, list[str]] = {}
    for path in _module_files(layer):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        hits = _clock_reads(tree)
        if hits:
            offenders[str(path.relative_to(PACKAGE_ROOT))] = hits
    assert offenders == {}, f"pure layer '{layer}' reads the clock: {offenders}"


@pytest.mark.parametrize("layer", PURE_LAYERS)
def test_pure_layers_do_not_import_dynamically(layer: str) -> None:
    """A runtime import is still an import, and hides from the static ban above."""
    offenders: dict[str, list[str]] = {}
    for path in _module_files(layer):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        hits = _dynamic_imports(tree)
        if hits:
            offenders[str(path.relative_to(PACKAGE_ROOT))] = hits
    assert offenders == {}, f"pure layer '{layer}' imports dynamically: {offenders}"


def test_pure_layers_do_not_import_outer_layers() -> None:
    offenders: dict[str, set[str]] = {}
    for layer in PURE_LAYERS:
        for path in _module_files(layer):
            source = path.read_text(encoding="utf-8")
            bad = _outer_layer_dependencies(source, _package_of(path))
            if bad:
                offenders[str(path.relative_to(PACKAGE_ROOT))] = bad
    assert offenders == {}, f"pure layers depend on outer layers: {offenders}"


# --------------------------------------------------------------------------
# Self-tests for the detectors above.
#
# A fitness function that has never been observed to fail is a hypothesis, not
# a guard. Each case below is a known-bad module source that a reasonable
# contributor could plausibly write. They are parsed as strings — no bad file
# is ever written into the package.
# --------------------------------------------------------------------------

DOMAIN_PACKAGE = "sentinel_ai.domain"
POLICY_PACKAGE = "sentinel_ai.domain.policy"

# (case name, the module's own package, known-bad source, expected offence prefix)
KNOWN_BAD_MODULES: tuple[tuple[str, str, str, str], ...] = (
    (
        "from-import clock read",
        POLICY_PACKAGE,
        "from time import monotonic\n\ndef stamp() -> float:\n    return monotonic()\n",
        "forbidden-import:time",
    ),
    (
        "aliased clock read",
        POLICY_PACKAGE,
        "import time as t\n\ndef stamp() -> float:\n    return t.monotonic()\n",
        "forbidden-import:time",
    ),
    (
        "perf_counter is still a clock",
        POLICY_PACKAGE,
        "import time\n\ndef stamp() -> float:\n    return time.perf_counter()\n",
        "forbidden-import:time",
    ),
    (
        "datetime is a clock too",
        POLICY_PACKAGE,
        "import datetime\n\ndef stamp():\n    return datetime.datetime.now()\n",
        "forbidden-import:datetime",
    ),
    (
        # The one case whose ONLY offence is a clock read. Every other clock case
        # above trips the forbidden-import ban first and asserts on *that*, so
        # `_clock_reads` was never the thing under test: mutating it to `return []`
        # left every control passing. `asyncio` is deliberately not on the forbidden
        # list — the pure layers may not read a clock, but they are allowed to be
        # async — which is what makes this case pin the matcher itself.
        "an awaited sleep is a clock read, and asyncio is not import-banned",
        POLICY_PACKAGE,
        "import asyncio\n\nasync def settle() -> None:\n    await asyncio.sleep(1.0)\n",
        "clock-read:asyncio.sleep",
    ),
    (
        "dynamic import via importlib",
        POLICY_PACKAGE,
        'import importlib\n\ntorch = importlib.import_module("torch")\n',
        "dynamic-import",
    ),
    (
        "dynamic import via __import__",
        POLICY_PACKAGE,
        'torch = __import__("torch")\n',
        "dynamic-import",
    ),
    (
        "relative outer-layer import from domain/",
        DOMAIN_PACKAGE,
        "from ..adapters.serialization import event_codec\n",
        "outer-layer:sentinel_ai.adapters",
    ),
    (
        "relative outer-layer import from domain/policy/",
        POLICY_PACKAGE,
        "from ...adapters.serialization import event_codec\n",
        "outer-layer:sentinel_ai.adapters",
    ),
    (
        "relative import of the layer by alias",
        DOMAIN_PACKAGE,
        "from .. import adapters\n",
        "outer-layer:sentinel_ai.adapters",
    ),
    (
        "settings import reads .env from disk",
        POLICY_PACKAGE,
        "from sentinel_ai.config import get_settings\n",
        "outer-layer:sentinel_ai.config",
    ),
    (
        "absolute outer-layer import",
        POLICY_PACKAGE,
        "from sentinel_ai.adapters.serialization import event_codec\n",
        "outer-layer:sentinel_ai.adapters",
    ),
    (
        "plain import of an outer layer",
        POLICY_PACKAGE,
        "import sentinel_ai.orchestrator.runner\n",
        "outer-layer:sentinel_ai.orchestrator",
    ),
    (
        "numpy stays forbidden",
        POLICY_PACKAGE,
        "import numpy as np\n",
        "forbidden-import:numpy",
    ),
    (
        "randomness is non-determinism",
        POLICY_PACKAGE,
        "import random\n\ndef pick():\n    return random.random()\n",
        "forbidden-import:random",
    ),
    (
        "os is I/O",
        POLICY_PACKAGE,
        'import os\n\nHOME = os.environ["HOME"]\n',
        "forbidden-import:os",
    ),
)


@pytest.mark.parametrize(
    ("package", "source", "expected"),
    [(package, source, expected) for _, package, source, expected in KNOWN_BAD_MODULES],
    ids=[name for name, _, _, _ in KNOWN_BAD_MODULES],
)
def test_the_detectors_flag_known_bad_module_sources(
    package: str, source: str, expected: str
) -> None:
    offences = _purity_offences(source, package)
    assert any(offence.startswith(expected) for offence in offences), (
        f"expected an offence starting with {expected!r}, got {offences}"
    )


CLEAN_MODULE = '''\
"""A pure module. Mentions sentinel_ai.adapters in prose, which is allowed."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot

from sentinel_ai.domain.entities import SceneState

from .triggers import DwellAnchor


@dataclass(frozen=True)
class Window:
    """Serialized by sentinel_ai.adapters.serialization elsewhere."""

    opened_at: float

    def elapsed(self, scene: SceneState, now: float) -> float:
        return now - self.opened_at

    def drift(self, anchor: DwellAnchor) -> float:
        return hypot(anchor.cx, anchor.cy)
'''


def test_the_detectors_do_not_flag_a_clean_module() -> None:
    """No false positives: prose about an outer layer is documentation, and a
    relative import of a sibling pure module is legitimate."""
    assert _purity_offences(CLEAN_MODULE, POLICY_PACKAGE) == []
