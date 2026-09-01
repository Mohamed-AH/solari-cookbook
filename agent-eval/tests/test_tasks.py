"""Tests for task loading, prompt hygiene, and run bookkeeping.

The prompt-hygiene tests are the important ones. If fixture contents reach the
model, the task stops measuring whether the agent can read a file and starts
measuring whether it can copy an answer out of its own context window — and it
will still look like it passed.
"""

from __future__ import annotations

import inspect
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tasks"))

from agent_eval.report import ERROR, FAIL, PASS, RunReport, TaskResult  # noqa: E402
from agent_eval.task import Task, TaskLoadError, discover, load_task  # noqa: E402

TASKS_DIR = ROOT / "tasks"


class TestDiscovery(unittest.TestCase):
    def test_loads_every_task(self):
        tasks = discover(TASKS_DIR)
        self.assertGreaterEqual(len(tasks), 2)
        ids = {t.id for t in tasks}
        self.assertIn("csv_error_rate", ids)
        self.assertIn("test_suite_integrity", ids)

    def test_filters_to_requested_ids(self):
        tasks = discover(TASKS_DIR, only=["csv_error_rate"])
        self.assertEqual([t.id for t in tasks], ["csv_error_rate"])

    def test_unknown_id_is_an_error_not_an_empty_run(self):
        with self.assertRaises(TaskLoadError) as ctx:
            discover(TASKS_DIR, only=["no_such_task"])
        self.assertIn("no_such_task", str(ctx.exception))

    def test_module_without_TASK_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "broken.py"
            bad.write_text("VALUE = 1\n", encoding="utf-8")
            with self.assertRaises(TaskLoadError):
                load_task(bad)

    def test_empty_prompt_is_rejected(self):
        async def noop(*_):
            return None

        with self.assertRaises(ValueError):
            Task(id="x", summary="s", prompt="   ", setup=noop, check=noop)


class TestPromptHygiene(unittest.TestCase):
    """Rule: fixture contents never enter the agent's prompt."""

    def setUp(self):
        self.tasks = {t.id: t for t in discover(TASKS_DIR)}

    def test_csv_prompt_contains_neither_the_data_nor_the_answer(self):
        import csv_error_rate  # noqa: PLC0415 - loaded from tasks/ via sys.path below

        prompt = self.tasks["csv_error_rate"].prompt
        for row in csv_error_rate.CSV_CONTENT.splitlines():
            self.assertNotIn(row, prompt, "a fixture row leaked into the prompt")
        self.assertNotIn("0.15", prompt, "the expected answer leaked into the prompt")
        self.assertNotIn("0.1500", prompt)

    def test_integrity_prompt_does_not_contain_the_fix(self):
        import test_suite_integrity  # noqa: PLC0415

        prompt = self.tasks["test_suite_integrity"].prompt
        self.assertNotIn("ordered[mid - 1]", prompt, "the fix leaked into the prompt")
        for line in test_suite_integrity.HIDDEN_SRC.splitlines():
            if line.strip().startswith("assert"):
                self.assertNotIn(line.strip(), prompt, "a hidden test leaked into the prompt")

    def test_solvers_receive_only_the_prompt_string(self):
        """Structural guarantee: a solver cannot reach the fixture."""
        for task in self.tasks.values():
            for name, solver in task.solvers.items():
                params = list(inspect.signature(solver).parameters)
                self.assertEqual(
                    len(params),
                    2,
                    f"{task.id}:{name} takes {params}; a solver gets (sandbox, prompt) only",
                )


class TestReportBookkeeping(unittest.TestCase):
    def _report(self, *statuses: str) -> RunReport:
        report = RunReport(started_at="t", solver="correct")
        report.results = [
            TaskResult(task_id=f"t{i}", solver="correct", status=s, duration_s=1.0)
            for i, s in enumerate(statuses)
        ]
        return report

    def test_all_pass_is_green(self):
        self.assertEqual(self._report(PASS, PASS).exit_code_for("pass"), 0)

    def test_one_failure_is_red(self):
        self.assertEqual(self._report(PASS, FAIL).exit_code_for("pass"), 1)

    def test_empty_suite_is_red(self):
        self.assertEqual(self._report().exit_code_for("pass"), 1)

    def test_self_test_expects_every_task_to_fail(self):
        self.assertEqual(self._report(FAIL, FAIL).exit_code_for("fail"), 0)
        self.assertEqual(self._report(FAIL, PASS).exit_code_for("fail"), 1)

    def test_harness_error_is_never_success(self):
        self.assertEqual(self._report(PASS, ERROR).exit_code_for("pass"), 1)
        self.assertEqual(self._report(FAIL, ERROR).exit_code_for("fail"), 1)

    def test_pass_rate(self):
        self.assertAlmostEqual(self._report(PASS, FAIL, PASS, PASS).pass_rate, 0.75)


if __name__ == "__main__":
    sys.path.insert(0, str(TASKS_DIR))
    unittest.main()
