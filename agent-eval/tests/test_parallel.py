"""Tests for the parallel runner, the retry policy, and the CI summary.

The concurrency bound is asserted by watching how many sandboxes are alive at
once, not by reading the source. A semaphore that is constructed but not
awaited looks identical in a diff and behaves very differently against an
account concurrency cap.
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

from solari_sandbox import ConcurrencyLimitError  # noqa: E402

from agent_eval import sandbox as sandbox_module  # noqa: E402
from agent_eval.report import ERROR, FAIL, PASS, RunReport, TaskResult, render_markdown  # noqa: E402
from agent_eval.runner import run_suite  # noqa: E402
from agent_eval.task import discover  # noqa: E402
from capabilities import supported  # noqa: E402
from local_sandbox import LocalSandboxClient  # noqa: E402

ALL_TASKS = discover(ROOT / "tasks")
# Concurrency and timing hold for any task; pass/fail verdicts only mean
# something for tasks whose tools this host actually has.
TASKS = supported(ALL_TASKS)


class TestBoundedConcurrency(unittest.TestCase):
    def _run(self, parallel: int, boot_delay_s: float = 0.05):
        client = LocalSandboxClient(boot_delay_s=boot_delay_s)
        report = asyncio.run(run_suite(client, TASKS, "correct", parallel=parallel))
        asyncio.run(client.aclose())
        return client, report

    def test_never_exceeds_the_requested_parallelism(self):
        for parallel in (1, 2, 3):
            with self.subTest(parallel=parallel):
                client, _ = self._run(parallel)
                self.assertLessEqual(
                    client.peak_live,
                    parallel,
                    f"ran {client.peak_live} sandboxes at once with --parallel {parallel}",
                )

    def test_parallelism_is_actually_used(self):
        """Bounded is not the same as serialised."""
        client, _ = self._run(3)
        self.assertGreater(client.peak_live, 1, "no two tasks ever overlapped")

    def test_every_sandbox_is_destroyed(self):
        client, report = self._run(3)
        self.assertEqual(client.live, 0, "a sandbox outlived the run")
        self.assertEqual(client.killed, len(report.results))

    def test_results_come_back_in_task_order(self):
        """Two runs of the same suite must produce comparable reports."""
        _, report = self._run(3)
        self.assertEqual([r.task_id for r in report.results], [t.id for t in TASKS])

    def test_something_is_runnable_here(self):
        """Guard against the suite silently skipping everything."""
        self.assertGreater(len(TASKS), 0, "no task can run on this host")

    def test_parallel_run_still_passes_every_task(self):
        _, report = self._run(3)
        failures = [(r.task_id, r.status, r.error) for r in report.results if r.status != PASS]
        self.assertEqual(failures, [], f"tasks failed under parallel execution: {failures}")

    def test_wall_clock_beats_summed_latency(self):
        _, report = self._run(3, boot_delay_s=0.1)
        self.assertLess(report.wall_clock_s, report.summed_latency_s)
        self.assertGreater(report.speedup, 1.0)
        self.assertEqual(report.parallel, 3)

    def test_parallel_zero_is_rejected(self):
        client = LocalSandboxClient()
        with self.assertRaises(ValueError):
            asyncio.run(run_suite(client, TASKS, "correct", parallel=0))


class TestConcurrencyCapRetry(unittest.TestCase):
    """A busy account is not a failing agent."""

    def setUp(self):
        self._backoff = sandbox_module.CREATE_BACKOFF_S
        sandbox_module.CREATE_BACKOFF_S = 0.01

    def tearDown(self):
        sandbox_module.CREATE_BACKOFF_S = self._backoff

    def test_retries_past_a_concurrency_limit_and_still_passes(self):
        client = LocalSandboxClient(
            fail_creates_with=ConcurrencyLimitError(), fail_times=2
        )
        one = [t for t in TASKS if t.id == "csv_error_rate"]
        report = asyncio.run(run_suite(client, one, "correct"))
        asyncio.run(client.aclose())
        self.assertEqual(client.create_calls, 3, "did not retry the refused creates")
        self.assertEqual(report.results[0].status, PASS)

    def test_gives_up_loudly_rather_than_forever(self):
        client = LocalSandboxClient(
            fail_creates_with=ConcurrencyLimitError(), fail_times=99
        )
        one = [t for t in TASKS if t.id == "csv_error_rate"]
        report = asyncio.run(run_suite(client, one, "correct"))
        asyncio.run(client.aclose())
        self.assertEqual(client.create_calls, sandbox_module.CREATE_ATTEMPTS)
        # A harness problem, not a verdict on the agent.
        self.assertEqual(report.results[0].status, ERROR)
        self.assertIn("ConcurrencyLimit", report.results[0].error)


class TestMarkdownSummary(unittest.TestCase):
    def _report(self, *statuses):
        report = RunReport(started_at="t", agent="correct", parallel=3)
        report.wall_clock_s = 10.0
        report.results = [
            TaskResult(task_id=f"task_{i}", agent="correct", status=s, duration_s=5.0, steps=3)
            for i, s in enumerate(statuses)
        ]
        return report

    def test_green_run_says_so(self):
        text = render_markdown(self._report(PASS, PASS))
        self.assertIn("all tasks passed", text)
        self.assertIn("2/2 passed", text)

    def test_red_run_names_the_regression(self):
        text = render_markdown(self._report(PASS, FAIL))
        self.assertIn("regressions detected", text)
        self.assertIn("task_1", text)
        self.assertIn("What failed", text)

    def test_harness_error_is_called_out_separately(self):
        text = render_markdown(self._report(PASS, ERROR))
        self.assertIn("not a verdict on the agent", text)


if __name__ == "__main__":
    unittest.main()


class TestAgentResolution(unittest.TestCase):
    """Pointing the harness at your own agent, without editing any task."""

    def setUp(self):
        from agent_eval.builtins import resolve_agent  # noqa: PLC0415

        self.resolve = resolve_agent
        self.task = TASKS[0]

    def test_task_agents_win(self):
        self.assertIsNotNone(self.resolve(self.task, "correct"))

    def test_builtin_resolves_for_any_task(self):
        for task in TASKS:
            self.assertIsNotNone(self.resolve(task, "claude"), task.id)

    def test_unknown_plain_name_is_not_silently_accepted(self):
        self.assertIsNone(self.resolve(self.task, "does_not_exist"))

    def test_import_path_loads_a_factory(self):
        factory = self.resolve(self.task, "local_sandbox:LocalSandboxClient")
        self.assertIsNotNone(factory)
        self.assertTrue(callable(factory))

    def test_bad_import_path_says_what_is_wrong(self):
        from agent_eval.builtins import AgentImportError, load_agent_path  # noqa: PLC0415

        with self.assertRaises(AgentImportError) as ctx:
            load_agent_path("no_such_module_xyz:Thing")
        self.assertIn("no_such_module_xyz", str(ctx.exception))

        with self.assertRaises(AgentImportError) as ctx:
            load_agent_path("local_sandbox:NoSuchAttribute")
        self.assertIn("NoSuchAttribute", str(ctx.exception))

        with self.assertRaises(AgentImportError) as ctx:
            load_agent_path("not_an_import_path")
        self.assertIn("my_package.my_module:MyAgent", str(ctx.exception))
