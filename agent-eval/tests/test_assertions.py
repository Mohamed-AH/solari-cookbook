"""Tests for the assertion layer. No sandbox, no credits, no network."""

from __future__ import annotations

import asyncio
import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent_eval.assertions import _NUMBER_VERIFIER, Checker, preflight  # noqa: E402
from fake_sandbox import FakeResult, FakeSandbox  # noqa: E402


def run(coro):
    return asyncio.run(coro)


class TestExitCodeAssertions(unittest.TestCase):
    def test_command_succeeds_reads_exit_code(self):
        sb = FakeSandbox({("pytest",): FakeResult(0, "ok")})
        c = Checker(sb)
        run(c.command_succeeds("pytest"))
        self.assertTrue(c.passed)

    def test_command_succeeds_fails_on_nonzero(self):
        sb = FakeSandbox({("pytest",): FakeResult(1, "", "boom")})
        c = Checker(sb)
        result = run(c.command_succeeds("pytest"))
        self.assertFalse(result.passed)
        self.assertEqual(result.exit_code, 1)
        self.assertEqual(result.stderr, "boom")
        self.assertFalse(c.passed)

    def test_command_fails_is_the_inverse(self):
        sb = FakeSandbox({("false",): FakeResult(1)})
        c = Checker(sb)
        self.assertTrue(run(c.command_fails("false")).passed)

    def test_file_exists_uses_test_f(self):
        sb = FakeSandbox({("test", "-f", "/workspace/out.txt"): FakeResult(0)})
        c = Checker(sb)
        self.assertTrue(run(c.file_exists("/workspace/out.txt")).passed)
        self.assertEqual(sb.calls[-1], ("test", "-f", "/workspace/out.txt"))

    def test_file_exists_fails_when_path_is_wrong(self):
        """The misplacement failure: right content, wrong directory."""
        sb = FakeSandbox({("test", "-f", "/workspace/output/answer.txt"): FakeResult(1)})
        c = Checker(sb)
        self.assertFalse(run(c.file_exists("/workspace/output/answer.txt")).passed)


class TestContentAssertions(unittest.TestCase):
    def test_file_equals_ignores_trailing_newline(self):
        sb = FakeSandbox({("cat", "/a"): FakeResult(0, "0.1500\n")})
        c = Checker(sb)
        self.assertTrue(run(c.file_equals("/a", "0.1500")).passed)

    def test_file_equals_reports_both_sides_on_mismatch(self):
        sb = FakeSandbox({("cat", "/a"): FakeResult(0, "0.3000")})
        c = Checker(sb)
        result = run(c.file_equals("/a", "0.1500"))
        self.assertFalse(result.passed)
        self.assertIn("0.1500", result.detail)
        self.assertIn("0.3000", result.detail)

    def test_file_equals_fails_when_cat_fails(self):
        sb = FakeSandbox({("cat", "/a"): FakeResult(1, "", "No such file")})
        c = Checker(sb)
        self.assertFalse(run(c.file_equals("/a", "")).passed)

    def test_stdout_equals_requires_exit_zero_too(self):
        """Matching text from a command that failed is not a pass."""
        sb = FakeSandbox({("echo", "hi"): FakeResult(3, "hi")})
        c = Checker(sb)
        self.assertFalse(run(c.stdout_equals("echo", ["hi"], "hi")).passed)

    def test_stdout_contains(self):
        sb = FakeSandbox({("cat", "/log"): FakeResult(0, "line one\nline two\n")})
        c = Checker(sb)
        self.assertTrue(run(c.stdout_contains("cat", ["/log"], "line two")).passed)


class TestTamperDetection(unittest.TestCase):
    def setUp(self):
        self.content = "hello\n"
        self.digest = hashlib.sha256(self.content.encode()).hexdigest()

    def test_sha256_matches(self):
        sb = FakeSandbox({("sha256sum", "/t"): FakeResult(0, f"{self.digest}  /t\n")})
        c = Checker(sb)
        self.assertTrue(run(c.file_sha256("/t", self.digest)).passed)

    def test_sha256_detects_modification(self):
        other = hashlib.sha256(b"tampered").hexdigest()
        sb = FakeSandbox({("sha256sum", "/t"): FakeResult(0, f"{other}  /t\n")})
        c = Checker(sb)
        result = run(c.file_sha256("/t", self.digest))
        self.assertFalse(result.passed)
        self.assertIn("modified", result.detail)

    def test_sha256_fails_when_file_is_gone(self):
        sb = FakeSandbox({("sha256sum", "/t"): FakeResult(1, "", "No such file")})
        c = Checker(sb)
        self.assertFalse(run(c.file_sha256("/t", self.digest)).passed)


class TestNumberVerifier(unittest.TestCase):
    """The verifier script is executed for real, against real files."""

    def _run_verifier(self, file_contents: str, expected: str, places: str = "4"):
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "verify.py"
            script.write_text(_NUMBER_VERIFIER, encoding="utf-8")
            target = Path(tmp) / "answer.txt"
            if file_contents is not None:
                target.write_text(file_contents, encoding="utf-8")
            return subprocess.run(
                [sys.executable, str(script), str(target), expected, places],
                capture_output=True,
                text=True,
            )

    def test_equal_exits_zero(self):
        self.assertEqual(self._run_verifier("0.1500\n", "0.15").returncode, 0)

    def test_equal_despite_formatting(self):
        self.assertEqual(self._run_verifier("0.15", "0.15").returncode, 0)

    def test_wrong_value_exits_one(self):
        self.assertEqual(self._run_verifier("0.3000\n", "0.15").returncode, 1)

    def test_missing_file_exits_two(self):
        self.assertEqual(self._run_verifier(None, "0.15").returncode, 2)

    def test_non_numeric_exits_three(self):
        self.assertEqual(self._run_verifier("error rate: 0.15\n", "0.15").returncode, 3)

    def test_rounding_boundary(self):
        self.assertEqual(self._run_verifier("0.15004\n", "0.15").returncode, 0)
        self.assertEqual(self._run_verifier("0.15006\n", "0.15").returncode, 1)


class TestNumberAssertionWiring(unittest.TestCase):
    def test_writes_verifier_and_passes_argv(self):
        sb = FakeSandbox(default_by_cmd={"mkdir": FakeResult(0), "python3": FakeResult(0, "ok")})
        c = Checker(sb)
        run(c.number_in_file_equals("/workspace/output/answer.txt", 0.15))
        self.assertTrue(c.passed)
        self.assertIn("/opt/agent_eval/verify_number.py", sb.writes)
        python_call = [call for call in sb.calls if call[0] == "python3"][0]
        # Values arrive as argv, never interpolated into the script source.
        self.assertIn("/workspace/output/answer.txt", python_call)
        self.assertIn("0.15", python_call)
        self.assertNotIn("0.15", sb.writes["/opt/agent_eval/verify_number.py"])

    def test_nonzero_verifier_exit_fails_the_check(self):
        sb = FakeSandbox(default_by_cmd={"mkdir": FakeResult(0), "python3": FakeResult(1, "")})
        c = Checker(sb)
        self.assertFalse(run(c.number_in_file_equals("/a", 0.15)).passed)


class TestCheckerVerdict(unittest.TestCase):
    def test_no_checks_is_not_a_pass(self):
        """A check function that asserted nothing must never report green."""
        self.assertFalse(Checker(FakeSandbox()).passed)

    def test_one_failure_sinks_the_task(self):
        sb = FakeSandbox({("true",): FakeResult(0), ("false",): FakeResult(1)})
        c = Checker(sb)
        run(c.command_succeeds("true"))
        run(c.command_succeeds("false"))
        self.assertFalse(c.passed)


class TestPreflight(unittest.TestCase):
    def test_reports_missing_tools(self):
        sb = FakeSandbox(
            {
                ("which", "test"): FakeResult(0, "/usr/bin/test"),
                ("which", "cat"): FakeResult(0, "/bin/cat"),
                ("which", "sha256sum"): FakeResult(1),
                ("which", "python3"): FakeResult(0, "/usr/bin/python3"),
            }
        )
        self.assertEqual(run(preflight(sb)), ["sha256sum"])


if __name__ == "__main__":
    unittest.main()
