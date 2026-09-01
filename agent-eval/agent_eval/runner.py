"""The run loop.

One task attempt is: fresh sandbox -> setup -> agent loop -> checks -> destroy.

Two rules shape the error handling here:

* An agent that crashes or runs out of steps is still scored. What is being
  measured is the end state of a real machine, not whether the agent returned
  cleanly, so the checks run either way and the reason it stopped is recorded
  beside them.
* A failure inside the harness (setup blew up, a check raised, the image is
  missing a tool the assertions need) is reported as ERROR, never as FAIL.
  Confusing "the harness broke" with "the agent was wrong" would poison every
  number this project produces.
"""

from __future__ import annotations

import time
import traceback
from typing import Any

from .agent import run_agent_loop
from .assertions import REQUIRED_TOOLS, Checker, preflight
from .report import ERROR, FAIL, PASS, RunReport, TaskResult, now_iso
from .sandbox import sandbox_session
from .task import Task


def _describe(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


async def run_task(client: Any, task: Task, agent_name: str) -> TaskResult:
    """Run one task attempt in its own sandbox and score it."""
    factory = task.agents.get(agent_name)
    if factory is None:
        known = ", ".join(sorted(task.agents)) or "<none>"
        return TaskResult(
            task_id=task.id,
            agent=agent_name,
            status=ERROR,
            error=f"task {task.id!r} defines no agent {agent_name!r} (has: {known})",
        )

    result = TaskResult(task_id=task.id, agent=agent_name, status=ERROR)
    stages: list[dict[str, Any]] = []
    started = time.monotonic()

    def stage(name: str, t0: float, **extra: Any) -> None:
        stages.append({"stage": name, "duration_s": round(time.monotonic() - t0, 2), **extra})

    try:
        boot = time.monotonic()
        async with sandbox_session(
            client,
            template=task.template,
            timeout_ms=task.timeout_ms,
            metadata={"agent_eval_task": task.id, "agent_eval_agent": agent_name},
        ) as sandbox:
            result.sandbox_id = sandbox.sandboxId
            stage("boot", boot, sandbox_id=sandbox.sandboxId)

            t0 = time.monotonic()
            missing = await preflight(sandbox, REQUIRED_TOOLS + tuple(task.required_tools))
            stage("preflight", t0, missing_tools=missing)
            if missing:
                result.status = ERROR
                result.error = (
                    f"sandbox image is missing tools the assertions need: {', '.join(missing)}"
                )
                return result

            t0 = time.monotonic()
            await task.setup(sandbox)
            stage("setup", t0)

            # The agent is constructed here and handed the prompt string via
            # its observations. It never sees the task, the fixture, or the
            # checks.
            t0 = time.monotonic()
            agent_run = await run_agent_loop(
                sandbox, factory(), task.prompt, max_steps=task.max_steps
            )
            stage("agent", t0, steps=len(agent_run.steps), stop_reason=agent_run.stop_reason)
            result.trajectory = agent_run.to_trajectory()
            result.steps = len(agent_run.steps)
            result.stop_reason = agent_run.stop_reason

            t0 = time.monotonic()
            checker = Checker(sandbox)
            await task.check(checker)
            stage(
                "check",
                t0,
                checks=len(checker.results),
                failed=[c.name for c in checker.results if not c.passed],
            )

            result.checks = checker.results
            result.status = PASS if checker.passed else FAIL
            if not checker.results:
                result.status = ERROR
                result.error = "check function recorded no assertions"
            elif agent_run.error:
                result.error = f"agent raised: {agent_run.error}"
            return result

    except Exception as exc:  # noqa: BLE001 - harness failure, reported loudly
        result.status = ERROR
        result.error = _describe(exc)
        stages.append({"stage": "harness_error", "traceback": traceback.format_exc()})
        return result
    finally:
        result.duration_s = time.monotonic() - started
        result.stages = stages


async def run_suite(client: Any, tasks: list[Task], agent_name: str) -> RunReport:
    """Run every task in sequence. Parallel execution arrives in phase 2."""
    report = RunReport(started_at=now_iso(), agent=agent_name)
    wall = time.monotonic()
    for task in tasks:
        report.results.append(await run_task(client, task, agent_name))
    report.wall_clock_s = time.monotonic() - wall
    return report
