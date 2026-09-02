"""Tests that actually execute the CLI's own code path.

Regression: `--repeat` shipped with a NameError in the line that prints the
run header. 150 tests passed, because every one of them called run_suite
directly and nothing ever ran `_run`. A flag that crashes the moment a user
types it is not covered by testing the machinery underneath it.
"""

from __future__ import annotations

import asyncio
import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tasks"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent_eval import cli  # noqa: E402
from capabilities import supported  # noqa: E402
from local_sandbox import LocalSandboxClient  # noqa: E402

SUPPORTED_IDS = [t.id for t in supported(cli.discover(ROOT / "tasks"))]


class _ClientFactory:
    """Stands in for make_client(), which would otherwise need a real key."""

    def __init__(self) -> None:
        self.clients: list[LocalSandboxClient] = []

    def __call__(self) -> LocalSandboxClient:
        client = LocalSandboxClient()
        self.clients.append(client)
        return client


def run_cli(argv: list[str]) -> tuple[int, str]:
    """Run the CLI end to end against the local double, capturing stdout."""
    factory = _ClientFactory()
    saved = cli.make_client
    cli.make_client = factory
    buffer = io.StringIO()
    try:
        args = cli.build_parser().parse_args(argv)
        with redirect_stdout(buffer):
            code = asyncio.run(cli._run(args))
    finally:
        cli.make_client = saved
        for client in factory.clients:
            asyncio.run(client.aclose())
    return code, buffer.getvalue()


class TestCliRuns(unittest.TestCase):
    def test_a_single_task_run_succeeds(self):
        code, out = run_cli(["--tasks", SUPPORTED_IDS[0], "--tasks-dir", str(ROOT / "tasks")])
        self.assertEqual(code, 0, out)
        self.assertIn("1/1", out)

    def test_repeat_prints_a_header_and_per_task_rates(self):
        """The exact invocation that crashed with a NameError."""
        code, out = run_cli(
            ["--tasks", SUPPORTED_IDS[0], "--repeat", "2", "--tasks-dir", str(ROOT / "tasks")]
        )
        self.assertEqual(code, 0, out)
        self.assertIn("2 attempts each", out)
        self.assertIn("[2/2]", out)

    def test_repeat_with_a_threshold(self):
        code, out = run_cli(
            [
                "--tasks", SUPPORTED_IDS[0],
                "--repeat", "2",
                "--min-pass-rate", "0.5",
                "--tasks-dir", str(ROOT / "tasks"),
            ]
        )
        self.assertEqual(code, 0, out)

    def test_parallel_path_runs(self):
        code, out = run_cli(
            ["--parallel", "3", "--tasks-dir", str(ROOT / "tasks"),
             "--tasks", ",".join(SUPPORTED_IDS)]
        )
        self.assertEqual(code, 0, out)

    def test_the_self_test_invocation_runs(self):
        code, out = run_cli(
            ["--agent", "sabotage", "--expect", "fail", "--tasks-dir", str(ROOT / "tasks"),
             "--tasks", ",".join(SUPPORTED_IDS)]
        )
        self.assertEqual(code, 0, out)
        self.assertIn("self-test OK", out)

    def test_list_needs_no_client(self):
        code, out = run_cli(["--list", "--tasks-dir", str(ROOT / "tasks")])
        self.assertEqual(code, 0)
        self.assertIn("csv_error_rate", out)


class TestCliRejectsBadFlags(unittest.TestCase):
    def test_repeat_zero(self):
        code, _ = run_cli(["--repeat", "0", "--tasks-dir", str(ROOT / "tasks")])
        self.assertEqual(code, 2)

    def test_parallel_zero(self):
        code, _ = run_cli(["--parallel", "0", "--tasks-dir", str(ROOT / "tasks")])
        self.assertEqual(code, 2)

    def test_min_pass_rate_out_of_range(self):
        for value in ("-0.1", "1.5"):
            with self.subTest(value=value):
                code, _ = run_cli(["--min-pass-rate", value, "--tasks-dir", str(ROOT / "tasks")])
                self.assertEqual(code, 2)

    def test_unknown_task_id_is_an_error(self):
        from agent_eval.task import TaskLoadError  # noqa: PLC0415

        with self.assertRaises(TaskLoadError):
            run_cli(["--tasks", "no_such_task", "--tasks-dir", str(ROOT / "tasks")])


class TestEveryFlagIsExercised(unittest.TestCase):
    """Guard against another flag shipping with an unbound name in its path."""

    def test_no_flag_crashes_on_a_minimal_run(self):
        base = ["--tasks", SUPPORTED_IDS[0], "--tasks-dir", str(ROOT / "tasks")]
        variants = [
            ["--verbose"],
            ["--repeat", "2"],
            ["--min-pass-rate", "0.0"],
            ["--parallel", "2"],
            ["--agent", "correct"],
        ]
        for extra in variants:
            with self.subTest(flag=extra[0]):
                code, out = run_cli(base + extra)
                self.assertIn(code, (0, 1), f"{extra} crashed: {out}")


if __name__ == "__main__":
    unittest.main()
