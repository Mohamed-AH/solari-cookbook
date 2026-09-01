"""Whole-lifecycle tests: setup -> agent loop -> checks, on a real filesystem.

Every task must do two things, and this asserts both:

  * pass under its `correct` agent
  * FAIL under its `sabotage` agent

The second is the one that matters. A suite that cannot go red is not
measuring anything, so "the sabotage agent was caught" is the real test of a
task. These run against a local test double, so they cost nothing and need no
key — see local_sandbox.py for what that does and does not prove.
"""

from __future__ import annotations

import asyncio
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tasks"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent_eval.agent import run_agent_loop  # noqa: E402
from agent_eval.assertions import Checker  # noqa: E402
from agent_eval.task import discover  # noqa: E402
from capabilities import unsupported_reason  # noqa: E402
from local_sandbox import LocalSandbox  # noqa: E402

TASKS = discover(ROOT / "tasks")


async def _attempt(task, agent_name):
    """Run one task attempt end to end and return (passed, checker, agent_run)."""
    with tempfile.TemporaryDirectory() as tmp:
        sandbox = LocalSandbox(Path(tmp))
        await task.setup(sandbox)
        agent_run = await run_agent_loop(
            sandbox, task.agents[agent_name](), task.prompt, max_steps=task.max_steps
        )
        checker = Checker(sandbox)
        await task.check(checker)
        return checker.passed, checker, agent_run


def attempt(task, agent_name):
    return asyncio.run(_attempt(task, agent_name))


class TestEveryTaskPassesWhenSolvedCorrectly(unittest.TestCase):
    pass


class TestEveryTaskFailsWhenSabotaged(unittest.TestCase):
    pass


def _make_pass_test(task):
    def test(self):
        passed, checker, agent_run = attempt(task, "correct")
        failures = [f"{c.name}: {c.detail}" for c in checker.results if not c.passed]
        self.assertTrue(passed, f"{task.id} failed under the correct agent: {failures}")
        self.assertGreater(len(checker.results), 0, "task recorded no assertions")
        self.assertNotEqual(
            agent_run.stop_reason, "agent_error", f"agent errored: {agent_run.error}"
        )

    test.__name__ = f"test_{task.id}_passes"
    test.__doc__ = f"{task.id} passes under its correct agent"
    reason = unsupported_reason(task)
    return unittest.skip(reason)(test) if reason else test


def _make_fail_test(task):
    def test(self):
        passed, checker, _ = attempt(task, "sabotage")
        self.assertFalse(
            passed,
            f"{task.id} PASSED under the sabotage agent — the task cannot detect the "
            f"failure it exists to catch",
        )

    test.__name__ = f"test_{task.id}_catches_sabotage"
    test.__doc__ = f"{task.id} fails under its sabotage agent"
    # Skipped rather than trusted: a task whose tools are missing fails for
    # the wrong reason, which would look like the saboteur being caught.
    reason = unsupported_reason(task)
    return unittest.skip(reason)(test) if reason else test


for _task in TASKS:
    setattr(TestEveryTaskPassesWhenSolvedCorrectly, f"test_{_task.id}_passes", _make_pass_test(_task))
    setattr(TestEveryTaskFailsWhenSabotaged, f"test_{_task.id}_catches_sabotage", _make_fail_test(_task))


class TestTrajectory(unittest.TestCase):
    """Rule: every step gets recorded. An empty trajectory is worse than none."""

    def test_trajectory_is_populated_for_every_task(self):
        for task in TASKS:
            with self.subTest(task=task.id):
                _, _, agent_run = attempt(task, "correct")
                self.assertGreater(len(agent_run.steps), 0, "no steps recorded")
                for step in agent_run.to_trajectory():
                    self.assertIn("observation", step)
                    self.assertIn("action", step)
                    self.assertIn("result", step)
                    self.assertTrue(step["observation"], "empty observation recorded")
                    self.assertTrue(step["action"].get("kind"), "action has no kind")

    def test_first_observation_carries_the_prompt_and_nothing_else(self):
        for task in TASKS:
            with self.subTest(task=task.id):
                _, _, agent_run = attempt(task, "correct")
                first = agent_run.steps[0].observation
                self.assertIn(task.prompt.strip().splitlines()[0], first)

    def test_multi_observation_tasks_really_take_several_steps(self):
        """The loop must be load-bearing, not decorative.

        These three cannot be solved without reading a result and acting on
        it: you cannot fix a failing test you have not run, patch a traceback
        you have not seen, or stage selectively without looking at status.
        """
        multi = ["test_suite_integrity", "stack_trace_fix", "secret_leak_guard"]
        by_id = {t.id: t for t in TASKS}
        for task_id in multi:
            with self.subTest(task=task_id):
                skip = unsupported_reason(by_id[task_id])
                if skip:
                    self.skipTest(skip)
                _, _, agent_run = attempt(by_id[task_id], "correct")
                self.assertGreaterEqual(
                    len(agent_run.steps),
                    4,
                    f"{task_id} finished in {len(agent_run.steps)} steps — the loop is not "
                    f"being exercised",
                )
                kinds = [s.action["kind"] for s in agent_run.steps]
                self.assertIn("run", kinds, "no command was ever executed")

    def test_agents_act_on_observations_not_a_fixed_script(self):
        """Every reference agent's first action must precede any change.

        An agent whose first action is a write did not look at anything, which
        would mean the task can be solved without observing the machine.
        """
        for task in TASKS:
            with self.subTest(task=task.id):
                _, _, agent_run = attempt(task, "correct")
                first = agent_run.steps[0].action["kind"]
                self.assertIn(
                    first,
                    ("run", "read", "list"),
                    f"{task.id} starts by acting blind ({first}) instead of observing",
                )


class TestFixtureNeverReachesThePrompt(unittest.TestCase):
    """Rule 7, enforced across every task rather than task by task.

    Runs each task's setup, collects every file it wrote, and asserts that no
    substantial line of any of them appears in the prompt. A task that leaks
    its fixture stops measuring whether the agent can read a file.
    """

    def test_no_fixture_line_appears_in_any_prompt(self):
        for task in TASKS:
            with self.subTest(task=task.id):
                for line in self._fixture_lines(task):
                    self.assertNotIn(
                        line,
                        task.prompt,
                        f"{task.id}: a fixture line reached the prompt",
                    )

    @staticmethod
    def _fixture_lines(task):
        async def build():
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                await task.setup(LocalSandbox(root))
                lines = []
                for path in root.rglob("*"):
                    if not path.is_file():
                        continue
                    try:
                        text = path.read_text(encoding="utf-8")
                    except (UnicodeDecodeError, OSError):
                        continue
                    # Short lines collide by chance; long ones do not.
                    lines += [ln.strip() for ln in text.splitlines() if len(ln.strip()) >= 20]
                return lines

        return asyncio.run(build())


class TestDoubleFidelity(unittest.TestCase):
    """The double must not alter bytes the harness wrote.

    Regression: writing fixtures as text translated "\n" to os.linesep on
    Windows, so every fixture landed as CRLF and its sha256 no longer matched
    what the task computed — four tasks failed a "this file is unmodified"
    check against files nothing had modified.
    """

    def test_written_bytes_survive_a_round_trip_unchanged(self):
        content = "line one\nline two\nline three\n"

        async def go():
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                sandbox = LocalSandbox(root)
                await sandbox.files.write("/workspace/probe.txt", content)
                on_disk = (root / "workspace" / "probe.txt").read_bytes()
                back = await sandbox.files.read_text("/workspace/probe.txt")
                return on_disk, back

        on_disk, back = asyncio.run(go())
        self.assertEqual(on_disk, content.encode("utf-8"))
        self.assertNotIn(b"\r\n", on_disk, "newlines were translated on write")
        self.assertEqual(back, content)
        self.assertEqual(
            hashlib.sha256(on_disk).hexdigest(),
            hashlib.sha256(content.encode("utf-8")).hexdigest(),
        )


class TestMaxSteps(unittest.TestCase):
    def test_loop_is_bounded(self):
        from agent_eval.agent import Action, Observation  # noqa: PLC0415

        class NeverFinishes:
            name = "never"

            async def next_action(self, obs: Observation) -> Action:
                return Action(kind="run", cmd="true")

        async def go():
            with tempfile.TemporaryDirectory() as tmp:
                return await run_agent_loop(
                    LocalSandbox(Path(tmp)), NeverFinishes(), "do nothing", max_steps=4
                )

        agent_run = asyncio.run(go())
        self.assertEqual(len(agent_run.steps), 4)
        self.assertEqual(agent_run.stop_reason, "max_steps")

    def test_agent_crash_is_recorded_not_swallowed(self):
        from agent_eval.agent import Observation  # noqa: PLC0415

        class Explodes:
            name = "boom"

            async def next_action(self, obs: Observation):
                raise ValueError("agent blew up")

        async def go():
            with tempfile.TemporaryDirectory() as tmp:
                return await run_agent_loop(
                    LocalSandbox(Path(tmp)), Explodes(), "do nothing", max_steps=4
                )

        agent_run = asyncio.run(go())
        self.assertEqual(agent_run.stop_reason, "agent_error")
        self.assertIn("agent blew up", agent_run.error)
        self.assertEqual(len(agent_run.steps), 1)


if __name__ == "__main__":
    unittest.main()


class TestUnavailableAgentIsNotBlamed(unittest.TestCase):
    """A provider outage must not be recorded as the agent being wrong.

    Regression from a live Gemini run: the free tier's rate limit hit on the
    first call and two tasks were reported FAIL after one step. The agent had
    not done anything to fail at.
    """

    def _run(self, task, agent):
        from agent_eval.runner import run_task  # noqa: PLC0415
        from local_sandbox import LocalSandboxClient  # noqa: PLC0415

        async def go():
            async with LocalSandboxClient() as client:
                return await run_task(client, task, agent)

        return asyncio.run(go())

    def _task_with(self, base, agent_factory):
        import dataclasses  # noqa: PLC0415

        return dataclasses.replace(base, agents={**base.agents, "flaky": agent_factory})

    def test_an_agent_that_cannot_act_reports_error_not_fail(self):
        from agent_eval.agent import AgentUnavailable  # noqa: PLC0415

        class RateLimited:
            name = "flaky"

            async def next_action(self, obs):
                raise AgentUnavailable("429 quota exceeded")

        task = self._task_with(TASKS[0], lambda: RateLimited())
        result = self._run(task, "flaky")
        self.assertEqual(result.status, "ERROR", "a rate limit was blamed on the agent")
        self.assertEqual(result.stop_reason, "agent_unavailable")
        self.assertIn("429", result.error)

    def test_an_agent_that_acted_wrongly_still_fails(self):
        """The distinction has to cut both ways or it is worthless."""
        from agent_eval.agent import finish  # noqa: PLC0415

        class DoesNothing:
            name = "flaky"

            async def next_action(self, obs):
                return finish("did nothing at all")

        task = self._task_with(TASKS[0], lambda: DoesNothing())
        result = self._run(task, "flaky")
        self.assertEqual(result.status, "FAIL")

    def test_work_finished_before_the_outage_still_passes(self):
        """End state is the measurement, so a late outage does not undo it."""
        from agent_eval.agent import AgentUnavailable  # noqa: PLC0415

        correct = TASKS[0].agents["correct"]

        class DiesAfterSucceeding:
            name = "flaky"

            def __init__(self):
                self._inner = correct()
                self._done = False

            async def next_action(self, obs):
                if self._done:
                    raise AgentUnavailable("429 quota exceeded")
                action = await self._inner.next_action(obs)
                if action.kind == "finish":
                    self._done = True
                    raise AgentUnavailable("429 quota exceeded")
                return action

        task = self._task_with(TASKS[0], lambda: DiesAfterSucceeding())
        result = self._run(task, "flaky")
        self.assertEqual(result.status, "PASS", "a late outage undid a completed task")


class TestRecoverableSandboxErrors(unittest.TestCase):
    """A mistyped command is an observation, not the end of the run.

    Regression from a live Gemini run: the model typed `python33`, the sandbox
    raised ActionError, and the harness ended the attempt on step 2 of 8. A
    real agent should see "not found" and try again — recovering from that is
    half of what the suite is meant to measure.
    """

    def _sandbox_that_rejects(self, tmp):
        from solari_sandbox import ActionError  # noqa: PLC0415

        from local_sandbox import LocalSandbox  # noqa: PLC0415

        sandbox = LocalSandbox(Path(tmp))
        original = sandbox.commands.run

        async def run(cmd, **kwargs):
            if cmd == "python33":
                raise ActionError("commands.run", 'exec: "python33": executable file not found')
            return await original(cmd, **kwargs)

        sandbox.commands.run = run
        return sandbox

    def test_a_missing_binary_comes_back_as_a_failed_result(self):
        from agent_eval.agent import execute, run  # noqa: PLC0415

        async def go():
            with tempfile.TemporaryDirectory() as tmp:
                sandbox = self._sandbox_that_rejects(tmp)
                return await execute(sandbox, run("python33", ["x.py"]))

        result = asyncio.run(go())
        self.assertFalse(result.ok)
        self.assertEqual(result.exit_code, 127)
        self.assertIn("python33", result.stderr)
        self.assertIn("python33", result.render())

    def test_the_agent_can_recover_and_the_run_continues(self):
        from agent_eval.agent import Observation, finish, run, run_agent_loop  # noqa: PLC0415

        class Typos:
            """Gets the name wrong once, reads the error, then gets it right."""

            name = "typo"

            async def next_action(self, obs: Observation):
                if obs.last_action is None:
                    return run("python33", ["-c", "print(1)"])
                if obs.last_result and not obs.last_result.ok:
                    return run("python3", ["-c", "print(1)"])
                return finish("recovered")

        async def go():
            with tempfile.TemporaryDirectory() as tmp:
                return await run_agent_loop(
                    self._sandbox_that_rejects(tmp), Typos(), "do it", max_steps=6
                )

        agent_run = asyncio.run(go())
        self.assertEqual(agent_run.stop_reason, "finished", agent_run.error)
        self.assertEqual(len(agent_run.steps), 3)
        self.assertEqual(agent_run.steps[0].result["exit_code"], 127)
        self.assertTrue(agent_run.steps[1].result["ok"], "the retry did not succeed")

    def test_losing_the_machine_is_still_a_harness_error(self):
        """The distinction has to cut both ways."""
        from solari_sandbox import ConnectionError as SolariConnectionError  # noqa: PLC0415

        from agent_eval.agent import Observation, run, run_agent_loop  # noqa: PLC0415

        class Dropped:
            async def run(self, cmd, **kwargs):
                raise SolariConnectionError("control channel closed")

        class Always:
            name = "always"

            async def next_action(self, obs: Observation):
                return run("python3", ["-c", "print(1)"])

        async def go():
            with tempfile.TemporaryDirectory() as tmp:
                from local_sandbox import LocalSandbox  # noqa: PLC0415

                sandbox = LocalSandbox(Path(tmp))
                sandbox.commands = Dropped()
                return await run_agent_loop(sandbox, Always(), "do it", max_steps=3)

        agent_run = asyncio.run(go())
        self.assertEqual(agent_run.stop_reason, "agent_error")
        self.assertIn("ConnectionError", agent_run.error)
