"""Tests for repeated runs — turning a verdict into a rate.

The motivating observation: the same agent scored 4/6 and then 6/6 on
consecutive runs of the same suite, with one task flipping for no reason but
the model's own non-determinism. A single run reports either number with equal
confidence, which is exactly the problem this project says it exists to solve.
"""

from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tasks"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent_eval.report import ERROR, FAIL, PASS, RunReport, TaskResult, render  # noqa: E402
from agent_eval.runner import run_suite  # noqa: E402
from agent_eval.task import discover  # noqa: E402
from capabilities import supported  # noqa: E402
from local_sandbox import LocalSandboxClient  # noqa: E402

TASKS = supported(discover(ROOT / "tasks"))


def _report(repeat: int, outcomes: dict[str, list[str]]) -> RunReport:
    report = RunReport(started_at="t", agent="a", repeat=repeat)
    report.wall_clock_s = 1.0
    for task_id, statuses in outcomes.items():
        for status in statuses:
            report.results.append(
                TaskResult(task_id=task_id, agent="a", status=status, duration_s=1.0)
            )
    return report


class TestRunnerRepeats(unittest.TestCase):
    def test_each_task_runs_the_requested_number_of_times(self):
        async def go():
            async with LocalSandboxClient() as client:
                return await run_suite(client, TASKS, "correct", repeat=3)

        report = asyncio.run(go())
        self.assertEqual(len(report.results), len(TASKS) * 3)
        self.assertEqual(report.repeat, 3)
        for task_id, attempts in report.attempts_by_task.items():
            self.assertEqual(len(attempts), 3, task_id)

    def test_attempts_are_grouped_by_task(self):
        async def go():
            async with LocalSandboxClient() as client:
                return await run_suite(client, TASKS, "correct", repeat=2)

        report = asyncio.run(go())
        self.assertEqual(list(report.attempts_by_task), [t.id for t in TASKS])

    def test_repeat_zero_is_rejected(self):
        async def go():
            async with LocalSandboxClient() as client:
                return await run_suite(client, TASKS, "correct", repeat=0)

        with self.assertRaises(ValueError):
            asyncio.run(go())

    def test_repeating_works_in_parallel_too(self):
        async def go():
            async with LocalSandboxClient(max_concurrent=2) as client:
                return await run_suite(client, TASKS, "correct", repeat=2, parallel=4)

        report = asyncio.run(go())
        self.assertEqual(len(report.results), len(TASKS) * 2)
        self.assertEqual(report.errored, 0)


class TestPassRates(unittest.TestCase):
    def test_per_task_rates(self):
        report = _report(2, {"a": [PASS, PASS], "b": [PASS, FAIL], "c": [FAIL, FAIL]})
        self.assertEqual(report.task_pass_rates, {"a": 1.0, "b": 0.5, "c": 0.0})

    def test_flaky_is_neither_always_passing_nor_always_failing(self):
        report = _report(2, {"a": [PASS, PASS], "b": [PASS, FAIL], "c": [FAIL, FAIL]})
        self.assertEqual(report.flaky_tasks, ["b"])

    def test_a_flaky_task_is_called_out_in_the_summary(self):
        report = _report(2, {"a": [PASS, PASS], "b": [PASS, FAIL]})
        text = render(report)
        self.assertIn("FLAKY", text)
        self.assertIn("flaky: b", text)
        self.assertIn("[1/2] b", text)

    def test_a_stable_run_is_not_called_flaky(self):
        report = _report(2, {"a": [PASS, PASS], "b": [FAIL, FAIL]})
        self.assertEqual(report.flaky_tasks, [])
        self.assertNotIn("FLAKY", render(report))


class TestExitCodeWithRates(unittest.TestCase):
    def test_all_attempts_must_pass_by_default(self):
        report = _report(2, {"a": [PASS, PASS], "b": [PASS, FAIL]})
        self.assertEqual(report.exit_code_for("pass"), 1)

    def test_a_threshold_tolerates_known_flakiness(self):
        report = _report(2, {"a": [PASS, PASS], "b": [PASS, FAIL]})
        self.assertEqual(report.exit_code_for("pass", min_pass_rate=0.75), 0)
        self.assertEqual(report.exit_code_for("pass", min_pass_rate=0.8), 1)

    def test_an_error_is_never_success_whatever_the_threshold(self):
        """A broken harness must not be tolerated by a loose gate."""
        report = _report(2, {"a": [PASS, PASS], "b": [PASS, ERROR]})
        self.assertEqual(report.exit_code_for("pass", min_pass_rate=0.0), 1)

    def test_the_self_test_still_requires_every_attempt_to_fail(self):
        self.assertEqual(_report(2, {"a": [FAIL, FAIL]}).exit_code_for("fail"), 0)
        self.assertEqual(_report(2, {"a": [FAIL, PASS]}).exit_code_for("fail"), 1)


if __name__ == "__main__":
    unittest.main()
