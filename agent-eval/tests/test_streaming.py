"""Tests for live progress while a run is in flight.

A six-task suite can sit for a minute on one slow sandbox boot. Printing only
at the end means a blank screen for that whole time and then a wall of text,
which tells you nothing about what is happening.
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

from agent_eval.report import ERROR, FAIL, PASS, RunReport, TaskResult, render, stream_line  # noqa: E402
from agent_eval.assertions import CheckResult  # noqa: E402
from agent_eval.runner import run_suite  # noqa: E402
from agent_eval.task import discover  # noqa: E402
from capabilities import supported  # noqa: E402
from local_sandbox import LocalSandboxClient  # noqa: E402
from test_cli import run_cli  # noqa: E402

TASKS = supported(discover(ROOT / "tasks"))


class TestRunnerAnnounces(unittest.TestCase):
    def _collect(self, **kwargs):
        seen = []

        async def go():
            async with LocalSandboxClient() as client:
                return await run_suite(
                    client, TASKS, "correct", on_result=seen.append, **kwargs
                )

        report = asyncio.run(go())
        return seen, report

    def test_one_announcement_per_attempt_sequentially(self):
        seen, report = self._collect()
        self.assertEqual(len(seen), len(TASKS))
        self.assertEqual([r.task_id for r in seen], [t.id for t in TASKS])
        self.assertEqual(len(seen), len(report.results))

    def test_one_announcement_per_attempt_in_parallel(self):
        seen, report = self._collect(parallel=3)
        self.assertEqual(len(seen), len(TASKS))
        self.assertEqual({r.task_id for r in seen}, {t.id for t in TASKS})

    def test_repeats_are_announced_individually(self):
        seen, _ = self._collect(repeat=2)
        self.assertEqual(len(seen), len(TASKS) * 2)

    def test_a_run_without_a_callback_still_works(self):
        async def go():
            async with LocalSandboxClient() as client:
                return await run_suite(client, TASKS, "correct")

        self.assertEqual(asyncio.run(go()).passed, len(TASKS))


class TestStreamLine(unittest.TestCase):
    def _result(self, status, checks=(), steps=3):
        return TaskResult(
            task_id="csv_error_rate",
            agent="correct",
            status=status,
            duration_s=1.9,
            steps=steps,
            checks=[CheckResult(name=f"c{i}", passed=p) for i, p in enumerate(checks)],
        )

    def test_pass_shows_steps(self):
        line = stream_line(self._result(PASS))
        self.assertIn("[PASS]", line)
        self.assertIn("csv_error_rate", line)
        self.assertIn("1.9s", line)
        self.assertIn("3 steps", line)

    def test_fail_shows_how_many_checks_failed(self):
        line = stream_line(self._result(FAIL, checks=(True, False, False)))
        self.assertIn("[FAIL]", line)
        self.assertIn("2 of 3 checks failed", line)

    def test_error_is_not_reported_as_a_check_failure(self):
        line = stream_line(self._result(ERROR))
        self.assertIn("harness error", line)
        self.assertNotIn("checks failed", line)

    def test_width_aligns_the_names(self):
        line = stream_line(self._result(PASS), width=30)
        self.assertIn("csv_error_rate" + " " * 16, line)


class TestFinalBlockDoesNotRepeatItself(unittest.TestCase):
    def _report(self, *statuses):
        report = RunReport(started_at="t", agent="correct")
        report.wall_clock_s = 5.0
        for i, status in enumerate(statuses):
            checks = [CheckResult(name="a check", passed=status == PASS, detail="why")]
            report.results.append(
                TaskResult(task_id=f"task_{i}", agent="correct", status=status,
                           duration_s=1.0, steps=3, checks=checks)
            )
        return report

    def test_streamed_passes_are_not_listed_again(self):
        text = render(self._report(PASS, PASS), streamed=True)
        self.assertNotIn("[PASS] task_0", text)
        self.assertIn("2/2 attempts passed", text)

    def test_failures_still_get_their_detail(self):
        text = render(self._report(PASS, FAIL), streamed=True)
        self.assertNotIn("[PASS] task_0", text)
        self.assertIn("[FAIL] task_1", text)
        self.assertIn("a check", text)

    def test_verbose_still_shows_everything(self):
        text = render(self._report(PASS, FAIL), streamed=True, verbose=True)
        self.assertIn("[PASS] task_0", text)

    def test_without_streaming_nothing_is_hidden(self):
        text = render(self._report(PASS, FAIL))
        self.assertIn("[PASS] task_0", text)


class TestCliStreams(unittest.TestCase):
    def test_lines_appear_before_the_final_report(self):
        ids = [t.id for t in TASKS]
        code, out = run_cli(
            ["--tasks", ",".join(ids[:2]), "--tasks-dir", str(ROOT / "tasks")]
        )
        self.assertEqual(code, 0, out)
        lines = out.splitlines()
        streamed = next(i for i, l in enumerate(lines) if l.startswith("[PASS]"))
        header = next(i for i, l in enumerate(lines) if l.startswith("agent-eval —"))
        self.assertLess(streamed, header, "results were not printed until the end")

    def test_every_task_reports_as_it_lands(self):
        ids = [t.id for t in TASKS]
        _, out = run_cli(["--tasks", ",".join(ids), "--tasks-dir", str(ROOT / "tasks")])
        header = next(i for i, l in enumerate(out.splitlines()) if l.startswith("agent-eval —"))
        live = out.splitlines()[:header]
        for task_id in ids:
            self.assertTrue(
                any(task_id in line and line.startswith("[") for line in live),
                f"{task_id} never reported while the run was in flight",
            )


if __name__ == "__main__":
    unittest.main()
