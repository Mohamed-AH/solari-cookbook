"""Resolving an agent name to a factory.

A task's own agents are its reference implementations — the correct one and
the saboteur that proves the task can fail. Built-in agents are the ones that
work on *any* task, which is what you actually point at your own suite.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable

from .task import Task

AgentFactory = Callable[[], Any]

BUILTIN_NAMES = ("claude", "gemini")


def _claude_factory() -> Any:
    # Imported lazily: the anthropic SDK is an optional extra, and a suite run
    # with scripted agents must not require it.
    from .claude_agent import ClaudeAgent  # noqa: PLC0415

    return ClaudeAgent()


def _gemini_factory() -> Any:
    from .gemini_agent import GeminiAgent  # noqa: PLC0415

    return GeminiAgent()


_BUILTINS: dict[str, AgentFactory] = {"claude": _claude_factory, "gemini": _gemini_factory}


class AgentImportError(RuntimeError):
    """Raised when an agent given as an import path cannot be loaded."""


def load_agent_path(spec: str) -> AgentFactory:
    """Load an agent factory from `module.path:attribute`.

    This is how you point the harness at your own agent without editing any
    task: the attribute is anything callable that returns an object with a
    `next_action` method — a class, or a factory function.
    """
    module_name, separator, attribute = spec.partition(":")
    if not separator or not module_name or not attribute:
        raise AgentImportError(
            f"agent path {spec!r} must look like 'my_package.my_module:MyAgent'"
        )
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise AgentImportError(f"cannot import {module_name!r}: {exc}") from exc
    try:
        factory = getattr(module, attribute)
    except AttributeError as exc:
        raise AgentImportError(f"{module_name!r} has no attribute {attribute!r}") from exc
    if not callable(factory):
        raise AgentImportError(f"{spec!r} is not callable; it must build an agent")
    return factory


def resolve_agent(task: Task, name: str) -> AgentFactory | None:
    """A task's own agents win, then built-ins, then an import path.

    Checking the task first means a task's reference agents cannot be shadowed
    by something on the import path, and an import path is only consulted when
    it is unambiguously one — it has to contain a colon.
    """
    if name in task.agents:
        return task.agents[name]
    if name in _BUILTINS:
        return _BUILTINS[name]
    if ":" in name:
        return load_agent_path(name)
    return None


def known_agents(task: Task) -> list[str]:
    return sorted(set(task.agents) | set(_BUILTINS))
