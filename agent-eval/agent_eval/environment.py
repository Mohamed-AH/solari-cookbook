"""Prepared environments: build the expensive setup once, fork it per task.

A task's fixture is cheap — a few files. The expensive part of a real suite is
everything that comes before it: cloning your repo, installing dependencies,
warming a toolchain. Doing that inside every task's sandbox multiplies it by
the size of your suite, which is what makes people run evals nightly instead
of on every PR.

So it runs once, in one sandbox, which is then snapshotted. Every task forks
from that snapshot and starts with the work already done.

The snapshot is keyed by a fingerprint of the preparation itself, so changing
`prepare()` builds a new one and stale environments are never silently reused
— a suite that passes against last week's dependencies is not a signal.
"""

from __future__ import annotations

import hashlib
import inspect
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

from .sandbox import DEFAULT_TEMPLATE, DEFAULT_TIMEOUT_MS, sandbox_session

PrepareFn = Callable[[Any], Awaitable[None]]

# Task modules starting with "_" are not tasks; this one describes the shared
# environment they run in.
ENVIRONMENT_MODULE = "_environment.py"
SNAPSHOT_PREFIX = "agent-eval-env"


@dataclass
class Environment:
    """A preparation step and the fingerprint that identifies its output."""

    fingerprint: str
    prepare: PrepareFn
    description: str = ""

    @property
    def snapshot_name(self) -> str:
        return f"{SNAPSHOT_PREFIX}-{self.fingerprint}"


@dataclass
class PreparedEnvironment:
    """A snapshot ready to be forked, and what it cost to get one."""

    snapshot_id: str
    name: str
    built: bool
    build_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "name": self.name,
            "built_this_run": self.built,
            "build_s": round(self.build_s, 2),
        }


def fingerprint_of(prepare: PrepareFn, extra: str = "") -> str:
    """Hash the preparation's own source, so editing it invalidates the snapshot."""
    try:
        source = inspect.getsource(prepare)
    except (OSError, TypeError):  # pragma: no cover - builtins, C functions
        source = repr(prepare)
    return hashlib.sha256((source + extra).encode("utf-8")).hexdigest()[:16]


def load_environment(tasks_dir: Path) -> Environment | None:
    """Load `_environment.py` from a tasks directory, if it defines one."""
    path = Path(tasks_dir) / ENVIRONMENT_MODULE
    if not path.is_file():
        return None

    import importlib.util  # noqa: PLC0415

    spec = importlib.util.spec_from_file_location("agent_eval_environment", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import environment module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    prepare = getattr(module, "prepare", None)
    if prepare is None:
        raise RuntimeError(f"{path} defines no `prepare` coroutine")

    return Environment(
        fingerprint=fingerprint_of(prepare, getattr(module, "FINGERPRINT_SALT", "")),
        prepare=prepare,
        description=getattr(module, "DESCRIPTION", "").strip(),
    )


async def find_snapshot(client: Any, name: str) -> str | None:
    """Return the id of an existing snapshot with this name, if any."""
    for snapshot in await client.list_snapshots():
        if getattr(snapshot, "name", None) == name:
            return snapshot.id
    return None


async def build_snapshot(
    client: Any,
    environment: Environment,
    *,
    template: str = DEFAULT_TEMPLATE,
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
) -> tuple[str, float]:
    """Run the preparation once in a fresh sandbox and snapshot the result."""
    started = time.monotonic()
    async with sandbox_session(client, template=template, timeout_ms=timeout_ms) as sandbox:
        await environment.prepare(sandbox)
        snapshot_id = await sandbox.snapshot(environment.snapshot_name)
    return snapshot_id, time.monotonic() - started


async def ensure_prepared(
    client: Any,
    environment: Environment,
    *,
    rebuild: bool = False,
    template: str = DEFAULT_TEMPLATE,
) -> PreparedEnvironment:
    """Find the snapshot for this preparation, or build it.

    Reuse is looked up by name on the server rather than cached on disk, so a
    CI runner with no persistent state still gets the fast path.
    """
    if not rebuild:
        existing = await find_snapshot(client, environment.snapshot_name)
        if existing:
            return PreparedEnvironment(
                snapshot_id=existing, name=environment.snapshot_name, built=False
            )

    snapshot_id, build_s = await build_snapshot(client, environment, template=template)
    return PreparedEnvironment(
        snapshot_id=snapshot_id,
        name=environment.snapshot_name,
        built=True,
        build_s=build_s,
    )


@dataclass
class SnapshotBenchmark:
    """Cost of getting one task-ready sandbox, prepared vs from scratch."""

    cold_s: list[float]
    forked_s: list[float]
    tasks: int = 0

    @staticmethod
    def _median(values: list[float]) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        mid = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[mid]
        return (ordered[mid - 1] + ordered[mid]) / 2

    @property
    def cold_median_s(self) -> float:
        return self._median(self.cold_s)

    @property
    def forked_median_s(self) -> float:
        return self._median(self.forked_s)

    @property
    def saving_per_sandbox_s(self) -> float:
        return self.cold_median_s - self.forked_median_s

    @property
    def speedup(self) -> float:
        return self.cold_median_s / self.forked_median_s if self.forked_median_s else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "samples": len(self.cold_s),
            "cold_s": [round(v, 2) for v in self.cold_s],
            "forked_s": [round(v, 2) for v in self.forked_s],
            "cold_median_s": round(self.cold_median_s, 2),
            "forked_median_s": round(self.forked_median_s, 2),
            "saving_per_sandbox_s": round(self.saving_per_sandbox_s, 2),
            "speedup": round(self.speedup, 2),
            "tasks": self.tasks,
            "saving_per_suite_s": round(self.saving_per_sandbox_s * self.tasks, 2),
        }

    def render(self) -> str:
        lines = [
            "",
            f"snapshot benchmark — {len(self.cold_s)} sample(s) each",
            "=" * 72,
            f"  from template + prepare : {self.cold_median_s:6.1f}s  (median)",
            f"  forked from snapshot    : {self.forked_median_s:6.1f}s  (median)",
            "=" * 72,
            f"  {self.saving_per_sandbox_s:.1f}s saved per sandbox, {self.speedup:.1f}x faster to ready",
        ]
        if self.tasks:
            lines.append(
                f"  across a {self.tasks}-task suite that is "
                f"{self.saving_per_sandbox_s * self.tasks:.1f}s a run"
            )
        lines.append("")
        return "\n".join(lines)


async def benchmark_snapshot(
    client: Any,
    environment: Environment,
    prepared: PreparedEnvironment,
    *,
    samples: int = 3,
    tasks: int = 0,
    template: str = DEFAULT_TEMPLATE,
) -> SnapshotBenchmark:
    """Time getting one task-ready sandbox both ways.

    "Ready" means the same thing on both sides — a connected sandbox with the
    preparation done — which is the only comparison worth quoting. The cold
    path pays for the preparation every time; the forked path paid once.
    """
    cold: list[float] = []
    forked: list[float] = []

    for _ in range(samples):
        started = time.monotonic()
        async with sandbox_session(client, template=template) as sandbox:
            await environment.prepare(sandbox)
            cold.append(time.monotonic() - started)

        started = time.monotonic()
        async with sandbox_session(client, from_snapshot=prepared.snapshot_id):
            forked.append(time.monotonic() - started)

    return SnapshotBenchmark(cold_s=cold, forked_s=forked, tasks=tasks)
