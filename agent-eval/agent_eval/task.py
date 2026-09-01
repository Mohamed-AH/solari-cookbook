"""Task definition and discovery.

A task is four separable things:

  prompt   the only text an agent is ever given
  setup    builds the fixture inside a fresh sandbox
  check    decides pass/fail from the machine's end state
  agents   reference implementations, used to test the harness itself

The separation between `prompt` and `setup` is the point, not an accident.
Fixture contents must never reach the model: hand an agent the setup commands
"for context" and it can emit the expected answer without ever reading the
file, and the task silently stops measuring anything. That is enforced
structurally — an agent is handed an Observation carrying `task.prompt` and
the results of its own actions, never the task object.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from .agent import Agent
from .assertions import Checker

SetupFn = Callable[[Any], Awaitable[None]]
CheckFn = Callable[[Checker], Awaitable[None]]
AgentFactory = Callable[[], Agent]

TASKS_DIR = Path(__file__).resolve().parent.parent / "tasks"


@dataclass(frozen=True)
class Task:
    """One evaluation task."""

    id: str
    summary: str
    prompt: str
    setup: SetupFn
    check: CheckFn
    agents: dict[str, AgentFactory] = field(default_factory=dict)
    template: str = "base"
    timeout_ms: int = 5 * 60_000
    max_steps: int = 8
    # Tools this task needs beyond the ones every assertion uses. Checked
    # before the agent runs, so a missing binary is an ERROR about the image
    # rather than a FAIL blamed on the agent.
    required_tools: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("task id must not be empty")
        if not self.prompt.strip():
            raise ValueError(f"task {self.id!r} has an empty prompt")


class TaskLoadError(RuntimeError):
    """Raised when a task module cannot be loaded or is malformed."""


def _load_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(f"agent_eval_task_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise TaskLoadError(f"cannot import task module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_task(path: Path) -> Task:
    """Load a single task module, which must define a module-level TASK."""
    module = _load_module(path)
    task = getattr(module, "TASK", None)
    if task is None:
        raise TaskLoadError(f"{path} defines no module-level TASK")
    if not isinstance(task, Task):
        raise TaskLoadError(f"{path}: TASK is {type(task).__name__}, expected Task")
    return task


def discover(tasks_dir: Path | None = None, only: list[str] | None = None) -> list[Task]:
    """Load every task in `tasks_dir`, optionally filtered to a set of ids.

    An unknown id in `only` is an error, not an empty run — asking for a task
    that does not exist must never look like a suite that passed.
    """
    directory = tasks_dir or TASKS_DIR
    if not directory.is_dir():
        raise TaskLoadError(f"tasks directory not found: {directory}")

    paths = sorted(p for p in directory.glob("*.py") if not p.name.startswith("_"))
    tasks = [load_task(p) for p in paths]

    seen: dict[str, Path] = {}
    for path, task in zip(paths, tasks):
        if task.id in seen:
            raise TaskLoadError(f"duplicate task id {task.id!r}: {seen[task.id]} and {path}")
        seen[task.id] = path

    if only:
        wanted = list(dict.fromkeys(only))
        by_id = {t.id: t for t in tasks}
        missing = [i for i in wanted if i not in by_id]
        if missing:
            known = ", ".join(sorted(by_id)) or "<none>"
            raise TaskLoadError(f"unknown task id(s): {', '.join(missing)}. known: {known}")
        return [by_id[i] for i in wanted]

    return tasks
