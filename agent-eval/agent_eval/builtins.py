"""Resolving an agent name to a factory.

A task's own agents are its reference implementations — the correct one and
the saboteur that proves the task can fail. Built-in agents are the ones that
work on *any* task, which is what you actually point at your own suite.
"""

from __future__ import annotations

from typing import Any, Callable

from .task import Task

AgentFactory = Callable[[], Any]

BUILTIN_NAMES = ("claude",)


def _claude_factory() -> Any:
    # Imported lazily: the anthropic SDK is an optional extra, and a suite run
    # with scripted agents must not require it.
    from .claude_agent import ClaudeAgent  # noqa: PLC0415

    return ClaudeAgent()


_BUILTINS: dict[str, AgentFactory] = {"claude": _claude_factory}


def resolve_agent(task: Task, name: str) -> AgentFactory | None:
    """A task's own agents win; built-ins fill in for any task."""
    if name in task.agents:
        return task.agents[name]
    return _BUILTINS.get(name)


def known_agents(task: Task) -> list[str]:
    return sorted(set(task.agents) | set(_BUILTINS))
