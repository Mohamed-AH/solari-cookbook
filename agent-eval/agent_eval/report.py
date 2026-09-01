"""Run results: the data model, the JSON artifact, and the terminal summary."""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .assertions import CheckResult

PASS = "PASS"
FAIL = "FAIL"
ERROR = "ERROR"  # the harness itself broke — not a verdict on the agent


@dataclass
class TaskResult:
    """The outcome of one task attempt."""

    task_id: str
    agent: str
    status: str
    checks: list[CheckResult] = field(default_factory=list)
    duration_s: float = 0.0
    sandbox_id: str = ""
    error: str | None = None
    stop_reason: str = ""
    steps: int = 0
    trajectory: list[dict[str, Any]] = field(default_factory=list)
    stages: list[dict[str, Any]] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.status == PASS

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "agent": self.agent,
            "status": self.status,
            "duration_s": round(self.duration_s, 2),
            "sandbox_id": self.sandbox_id,
            "error": self.error,
            "stop_reason": self.stop_reason,
            "steps": self.steps,
            "checks": [c.to_dict() for c in self.checks],
            "trajectory": self.trajectory,
            "stages": self.stages,
        }


@dataclass
class RunReport:
    """Everything one invocation of the harness produced."""

    results: list[TaskResult] = field(default_factory=list)
    wall_clock_s: float = 0.0
    started_at: str = ""
    agent: str = ""
    parallel: int = 1
    concurrency_ceiling: int = 0

    @property
    def summed_latency_s(self) -> float:
        return sum(r.duration_s for r in self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.status == PASS)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if r.status == FAIL)

    @property
    def errored(self) -> int:
        return sum(1 for r in self.results if r.status == ERROR)

    @property
    def pass_rate(self) -> float:
        return self.passed / len(self.results) if self.results else 0.0

    @property
    def speedup(self) -> float:
        """Summed sandbox latency over wall clock.

        Not a throughput claim — it is how much of the suite's time was spent
        waiting on sandboxes that could have been waited on at the same time.
        """
        return self.summed_latency_s / self.wall_clock_s if self.wall_clock_s else 0.0

    @property
    def exit_code(self) -> int:
        """0 only when every task passed. Anything else is a red build."""
        return self.exit_code_for("pass")

    def exit_code_for(self, expect: str) -> int:
        """Exit code for the expected outcome.

        `expect="fail"` is how the harness tests itself: run the sabotage
        agents and require every task to fail. A suite that cannot go red is
        not measuring anything, so "everything failed" is the success
        condition there. An ERROR is never success under either expectation.
        """
        if not self.results or self.errored:
            return 1
        if expect == "fail":
            return 0 if self.passed == 0 else 1
        return 0 if self.failed == 0 else 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "agent-eval/v1",
            "started_at": self.started_at,
            "agent": self.agent,
            "python": platform.python_version(),
            "parallel": self.parallel,
            "concurrency_ceiling": self.concurrency_ceiling,
            "summary": {
                "total": len(self.results),
                "passed": self.passed,
                "failed": self.failed,
                "errored": self.errored,
                "pass_rate": round(self.pass_rate, 4),
                "wall_clock_s": round(self.wall_clock_s, 2),
                "summed_latency_s": round(self.summed_latency_s, 2),
                "speedup": round(self.speedup, 2),
            },
            "results": [r.to_dict() for r in self.results],
        }

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


_MARK = {PASS: "PASS", FAIL: "FAIL", ERROR: "ERR "}


def render(report: RunReport, *, verbose: bool = False) -> str:
    """Render a run as terminal text. Failing checks always show their detail."""
    lines: list[str] = []
    lines.append("")
    lines.append(
        f"agent-eval — agent={report.agent}  parallel={report.parallel}  "
        f"started={report.started_at}"
    )
    lines.append("=" * 72)

    for result in report.results:
        mark = _MARK.get(result.status, result.status)
        steps_note = f", {result.steps} steps, {result.stop_reason}" if result.steps else ""
        lines.append(f"[{mark}] {result.task_id}  ({result.duration_s:.1f}s{steps_note})")
        if result.error:
            lines.append(f"       error: {result.error}")
        for check in result.checks:
            failed = not check.passed
            if failed or verbose:
                bullet = "x" if failed else "+"
                lines.append(f"       {bullet} {check.name}")
                lines.append(f"         {check.detail}")
                if failed and check.command:
                    lines.append(f"         cmd: {check.command}  -> exit {check.exit_code}")
                if failed and check.stderr.strip():
                    first = check.stderr.strip().splitlines()[0]
                    lines.append(f"         stderr: {first}")

    lines.append("=" * 72)
    lines.append(
        f"{report.passed}/{len(report.results)} passed"
        f"  ({report.pass_rate * 100:.0f}%)"
        f"  failed={report.failed} errored={report.errored}"
    )
    timing = (
        f"wall clock {report.wall_clock_s:.1f}s"
        f"  |  summed sandbox latency {report.summed_latency_s:.1f}s"
    )
    if report.parallel > 1:
        ran_at = report.concurrency_ceiling or report.parallel
        timing += f"  |  {report.speedup:.1f}x on {ran_at} at a time"
        if ran_at < report.parallel:
            timing += f" (asked for {report.parallel}; the account allowed {ran_at})"
    lines.append(timing)
    lines.append("")
    return "\n".join(lines)


def render_markdown(report: RunReport) -> str:
    """A summary for a CI job summary or a PR comment.

    Failing checks are shown with the command that decided them, because the
    person reading this in CI cannot re-run anything from where they are
    sitting.
    """
    verdict = "all tasks passed" if report.exit_code == 0 else "regressions detected"
    out: list[str] = []
    out.append(f"## agent-eval — {verdict}")
    out.append("")
    out.append(
        f"**{report.passed}/{len(report.results)} passed** "
        f"({report.pass_rate * 100:.0f}%) · agent `{report.agent}` · "
        f"{report.wall_clock_s:.1f}s wall clock"
    )
    if report.errored:
        out.append("")
        out.append(
            f"> {report.errored} task(s) reported ERROR — the harness failed, "
            f"which is not a verdict on the agent."
        )
    out.append("")
    out.append("| | task | steps | time | checks |")
    out.append("| --- | --- | --- | --- | --- |")
    for r in report.results:
        icon = {PASS: "PASS", FAIL: "FAIL", ERROR: "ERROR"}.get(r.status, r.status)
        failed = sum(1 for c in r.checks if not c.passed)
        checks = f"{len(r.checks) - failed}/{len(r.checks)}" if r.checks else "—"
        out.append(
            f"| {icon} | `{r.task_id}` | {r.steps} | {r.duration_s:.1f}s | {checks} |"
        )

    problems = [r for r in report.results if r.status != PASS]
    if problems:
        out.append("")
        out.append("<details><summary>What failed</summary>")
        out.append("")
        for r in problems:
            out.append(f"**`{r.task_id}`** — {r.status}, stopped: {r.stop_reason or 'n/a'}")
            if r.error:
                out.append(f"- harness: {r.error}")
            for c in r.checks:
                if not c.passed:
                    out.append(f"- {c.name} — {c.detail}")
                    if c.command:
                        out.append(f"  - `{c.command}` exited {c.exit_code}")
            out.append("")
        out.append("</details>")

    out.append("")
    out.append(
        f"_summed sandbox latency {report.summed_latency_s:.1f}s across "
        f"{len(report.results)} sandboxes, {report.parallel} at a time_"
    )
    out.append("")
    return "\n".join(out)
