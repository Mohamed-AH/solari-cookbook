"""The agent loop: observation -> action -> result, bounded by max_steps.

The harness owns this loop, not the agent. An agent only decides the next
action; executing it, capturing what came back, and recording the step are the
harness's job. That is deliberate — it means the trajectory is fully populated
for *any* agent plugged in here, including someone else's, rather than
depending on that agent to report itself honestly. An agent that lies about
what it did still cannot fabricate the exit codes recorded here.

An action is one of five things: run a command, write a file, read a file,
list a directory, or finish. Everything the agent learns about the machine it
is working in arrives as an `ActionResult` from one of those.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Protocol

# How much of a command's output the agent is shown, and how much is kept in
# the trajectory. Enough to diagnose a failure, not enough to bury it.
OUTPUT_LIMIT = 4000

DEFAULT_WORKDIR = "/workspace"


def _clip(text: str, limit: int = OUTPUT_LIMIT) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    dropped = len(text) - limit
    return f"{text[:half]}\n...[{dropped} chars omitted]...\n{text[-half:]}"


# -- actions -----------------------------------------------------------------


@dataclass(frozen=True)
class Action:
    """One thing an agent does. `kind` decides which fields matter."""

    kind: str  # run | write | read | list | finish
    cmd: str = ""
    args: tuple[str, ...] = ()
    cwd: str = ""
    path: str = ""
    content: str = ""
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        base: dict[str, Any] = {"kind": self.kind}
        if self.kind == "run":
            base.update(cmd=self.cmd, args=list(self.args), cwd=self.cwd or DEFAULT_WORKDIR)
        elif self.kind == "write":
            base.update(path=self.path, bytes=len(self.content.encode("utf-8")))
        elif self.kind in ("read", "list"):
            base.update(path=self.path)
        elif self.kind == "finish":
            base.update(summary=self.summary)
        return base

    def render(self) -> str:
        if self.kind == "run":
            return f"run: {self.cmd} {' '.join(self.args)}".strip()
        if self.kind == "write":
            return f"write: {self.path} ({len(self.content)} chars)"
        if self.kind in ("read", "list"):
            return f"{self.kind}: {self.path}"
        return f"finish: {self.summary}"


def run(cmd: str, args: list[str] | tuple[str, ...] = (), cwd: str = "") -> Action:
    return Action(kind="run", cmd=cmd, args=tuple(args), cwd=cwd)


def write(path: str, content: str) -> Action:
    return Action(kind="write", path=path, content=content)


def read(path: str) -> Action:
    return Action(kind="read", path=path)


def list_dir(path: str) -> Action:
    return Action(kind="list", path=path)


def finish(summary: str = "") -> Action:
    return Action(kind="finish", summary=summary)


@dataclass
class ActionResult:
    """What the machine did in response to an action."""

    ok: bool
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "exit_code": self.exit_code,
            "stdout": _clip(self.stdout),
            "stderr": _clip(self.stderr),
            "note": self.note,
        }

    def render(self) -> str:
        """The text an agent is shown after acting."""
        parts: list[str] = []
        if self.exit_code is not None:
            parts.append(f"exit code: {self.exit_code}")
        if self.note:
            parts.append(self.note)
        if self.stdout.strip():
            parts.append(f"stdout:\n{_clip(self.stdout).rstrip()}")
        if self.stderr.strip():
            parts.append(f"stderr:\n{_clip(self.stderr).rstrip()}")
        if not parts:
            parts.append("(no output)")
        return "\n".join(parts)


# -- observations and steps --------------------------------------------------


@dataclass(frozen=True)
class Observation:
    """Everything the agent knows going into a step.

    Note what is absent: the task object, the fixture, the expected answer.
    An agent sees the prompt and what its own actions returned. Nothing else.
    """

    step: int
    max_steps: int
    prompt: str
    workdir: str
    last_action: Action | None = None
    last_result: ActionResult | None = None

    @property
    def steps_left(self) -> int:
        return self.max_steps - self.step

    def render(self) -> str:
        if self.last_action is None:
            return (
                f"Task:\n{self.prompt.strip()}\n\n"
                f"You are in {self.workdir}. You have {self.max_steps} steps."
            )
        result = self.last_result.render() if self.last_result else "(no result)"
        return (
            f"Previous action: {self.last_action.render()}\n"
            f"Result:\n{result}\n\n"
            f"Steps remaining: {self.steps_left}"
        )


@dataclass
class Step:
    """One observation -> action -> result triple, recorded in full."""

    index: int
    observation: str
    action: dict[str, Any]
    result: dict[str, Any]
    duration_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.index,
            "observation": _clip(self.observation),
            "action": self.action,
            "result": self.result,
            "duration_s": round(self.duration_s, 2),
        }


@dataclass
class AgentRun:
    """The outcome of one agent loop."""

    steps: list[Step] = field(default_factory=list)
    stop_reason: str = ""  # finished | max_steps | agent_error
    error: str | None = None

    def to_trajectory(self) -> list[dict[str, Any]]:
        return [s.to_dict() for s in self.steps]


# -- the agent interface -----------------------------------------------------


class Agent(Protocol):
    """Anything that can choose a next action given an observation.

    This is the whole contract. Bring your own agent by implementing it.
    """

    name: str

    async def next_action(self, obs: Observation) -> Action: ...


class FunctionAgent:
    """Wraps an async function as an Agent.

    Used for the scripted reference agents: they branch on what they actually
    observed, so a task that needs "run, read the error, fix, re-run" is
    genuinely exercising the loop rather than replaying a fixed list.
    """

    def __init__(self, name: str, fn: Callable[[Observation], Awaitable[Action]]) -> None:
        self.name = name
        self._fn = fn

    async def next_action(self, obs: Observation) -> Action:
        return await self._fn(obs)


class ScriptedAgent:
    """Replays a fixed list of actions, then finishes.

    Deterministic and offline — the harness's own regression test. It ignores
    observations by design; anything that needs to react uses FunctionAgent.
    """

    def __init__(self, name: str, actions: list[Action]) -> None:
        self.name = name
        self._actions = list(actions)
        self._i = 0

    async def next_action(self, obs: Observation) -> Action:
        if self._i >= len(self._actions):
            return finish("scripted actions exhausted")
        action = self._actions[self._i]
        self._i += 1
        return action


# -- execution ---------------------------------------------------------------


async def execute(sandbox: Any, action: Action, *, workdir: str = DEFAULT_WORKDIR) -> ActionResult:
    """Perform one action in the sandbox and capture what came back."""
    if action.kind == "run":
        res = await sandbox.commands.run(
            action.cmd, args=list(action.args), cwd=action.cwd or workdir
        )
        return ActionResult(
            ok=res.exitCode == 0,
            exit_code=res.exitCode,
            stdout=res.stdout,
            stderr=res.stderr,
        )

    if action.kind == "write":
        await sandbox.files.write(action.path, action.content)
        return ActionResult(ok=True, note=f"wrote {len(action.content)} chars to {action.path}")

    if action.kind == "read":
        try:
            text = await sandbox.files.read_text(action.path)
        except Exception as exc:  # noqa: BLE001 - reported to the agent, not hidden
            return ActionResult(ok=False, note=f"could not read {action.path}: {exc}")
        return ActionResult(ok=True, stdout=text, note=f"read {action.path}")

    if action.kind == "list":
        try:
            entries = await sandbox.files.list(action.path)
        except Exception as exc:  # noqa: BLE001 - reported to the agent, not hidden
            return ActionResult(ok=False, note=f"could not list {action.path}: {exc}")
        listing = "\n".join(
            f"{'dir ' if e.dir else 'file'} {e.name}" for e in entries
        )
        return ActionResult(ok=True, stdout=listing, note=f"listed {action.path}")

    if action.kind == "finish":
        return ActionResult(ok=True, note="agent finished")

    return ActionResult(ok=False, note=f"unknown action kind: {action.kind!r}")


async def run_agent_loop(
    sandbox: Any,
    agent: Agent,
    prompt: str,
    *,
    max_steps: int,
    workdir: str = DEFAULT_WORKDIR,
) -> AgentRun:
    """Drive an agent until it finishes, errors, or runs out of steps."""
    run_record = AgentRun()
    last_action: Action | None = None
    last_result: ActionResult | None = None

    for index in range(max_steps):
        obs = Observation(
            step=index,
            max_steps=max_steps,
            prompt=prompt,
            workdir=workdir,
            last_action=last_action,
            last_result=last_result,
        )
        started = time.monotonic()

        try:
            action = await agent.next_action(obs)
        except Exception as exc:  # noqa: BLE001 - recorded; end state is still scored
            run_record.stop_reason = "agent_error"
            run_record.error = f"{type(exc).__name__}: {exc}"
            run_record.steps.append(
                Step(
                    index=index,
                    observation=obs.render(),
                    action={"kind": "none"},
                    result={"ok": False, "note": run_record.error},
                    duration_s=time.monotonic() - started,
                )
            )
            return run_record

        try:
            result = await execute(sandbox, action, workdir=workdir)
        except Exception as exc:  # noqa: BLE001 - a broken action ends the run loudly
            run_record.stop_reason = "agent_error"
            run_record.error = f"executing {action.render()}: {type(exc).__name__}: {exc}"
            result = ActionResult(ok=False, note=run_record.error)

        run_record.steps.append(
            Step(
                index=index,
                observation=obs.render(),
                action=action.to_dict(),
                result=result.to_dict(),
                duration_s=time.monotonic() - started,
            )
        )

        if run_record.stop_reason == "agent_error":
            return run_record

        last_action, last_result = action, result

        if action.kind == "finish":
            run_record.stop_reason = "finished"
            return run_record

    run_record.stop_reason = "max_steps"
    return run_record
