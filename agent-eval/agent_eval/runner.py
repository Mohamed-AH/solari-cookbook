"""The run loop.

One task attempt is: fresh sandbox -> setup -> solver -> checks -> destroy.

Two rules shape the error handling here:

* A solver that crashes still gets scored. What is being measured is the end
  state of a real machine, not whether the agent returned cleanly, so the
  checks run either way and the crash is recorded alongside them.
* A failure inside the harness (setup blew up, a check raised, the image is
  missing a tool the assertions need) is reported as ERROR, never as FAIL.
  Confusing "the harness broke" with "the agent was wrong" would poison every
  number this project produces.
"""

from __future__ import annotations

import time
import traceback
from typing import Any

from .assertions import Checker, preflight
from .report import ERROR, FAIL, PASS, RunReport, TaskResult, now_iso
from .sandbox import sandbox_session
from .task import Task


def _describe(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


async def run_task(client: Any, task: Task, solver_name: str) -> TaskResult:
    """Run one task attempt in its own sandbox and score it."""
    solver = task.solvers.get(solver_name)
    if solver is None:
        known = ", ".join(sorted(task.solvers)) or "<none>"
        return TaskResult(
            task_id=task.id,
            solver=solver_name,
            status=ERROR,
            error=f"task {task.id!r} defines no solver {solver_name!r} (has: {known})",
        )

    result = TaskResult(task_id=task.id, solver=solver_name, status=ERROR)
    trajectory: list[dict[str, Any]] = []
    started = time.monotonic()

    def stage(name: str, t0: float, **extra: Any) -> None:
        trajectory.append(
            {"stage": name, "duration_s": round(time.monotonic() - t0, 2), **extra}
        )

    try:
        boot = time.monotonic()
        async with sandbox_session(
            client,
            template=task.template,
            timeout_ms=task.timeout_ms,
            metadata={"agent_eval_task": task.id, "agent_eval_solver": solver_name},
        ) as sandbox:
            result.sandbox_id = sandbox.sandboxId
            stage("boot", boot, sandbox_id=sandbox.sandboxId)

            t0 = time.monotonic()
            missing = await preflight(sandbox)
            stage("preflight", t0, missing_tools=missing)
            if missing:
                result.status = ERROR
                result.error = (
                    f"sandbox image is missing tools the assertions need: "
                    f"{', '.join(missing)}"
                )
                return result

            t0 = time.monotonic()
            await task.setup(sandbox)
            stage("setup", t0)

            # The solver receives the prompt string and nothing else. It never
            # sees the task object, so fixture contents cannot leak into it.
            t0 = time.monotonic()
            solver_error: str | None = None
            try:
                await solver(sandbox, task.prompt)
            except Exception as exc:  # noqa: BLE001 - recorded, then scored anyway
                solver_error = _describe(exc)
            stage("solve", t0, solver=solver_name, error=solver_error)

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
            if solver_error:
                result.error = f"solver raised: {solver_error}"
            elif not checker.results:
                result.status = ERROR
                result.error = "check function recorded no assertions"
            return result

    except Exception as exc:  # noqa: BLE001 - harness failure, reported loudly
        result.status = ERROR
        result.error = _describe(exc)
        trajectory.append({"stage": "harness_error", "traceback": traceback.format_exc()})
        return result
    finally:
        result.duration_s = time.monotonic() - started
        result.trajectory = trajectory


async def run_suite(client: Any, tasks: list[Task], solver_name: str) -> RunReport:
    """Run every task in sequence. Parallel execution arrives in phase 2."""
    report = RunReport(started_at=now_iso(), solver=solver_name)
    wall = time.monotonic()
    for task in tasks:
        report.results.append(await run_task(client, task, solver_name))
    report.wall_clock_s = time.monotonic() - wall
    return report
