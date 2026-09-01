"""Tests for prepared environments: build once, fork per task.

The claim being tested is not "snapshots are faster" — that is the API's job.
It is that forking gives a task the prepared filesystem, that a stale
preparation is never silently reused, and that a preparation which fails is
never snapshotted and handed to every task in the run.
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

from agent_eval.environment import (  # noqa: E402
    Environment,
    ensure_prepared,
    fingerprint_of,
    load_environment,
)
from agent_eval.runner import run_suite  # noqa: E402
from agent_eval.task import discover  # noqa: E402
from capabilities import supported  # noqa: E402
from local_sandbox import LocalSandboxClient  # noqa: E402

TASKS = supported(discover(ROOT / "tasks"))

MARKER = "/opt/prepared/marker.txt"


async def _prepare_marker(sb):
    await sb.commands.run("mkdir", args=["-p", "/opt/prepared"])
    await sb.files.write(MARKER, "prepared once\n")


def _env(prepare=_prepare_marker, salt="") -> Environment:
    return Environment(fingerprint=fingerprint_of(prepare, salt), prepare=prepare)


class TestFingerprint(unittest.TestCase):
    def test_same_preparation_gives_the_same_fingerprint(self):
        self.assertEqual(fingerprint_of(_prepare_marker), fingerprint_of(_prepare_marker))

    def test_editing_the_preparation_changes_it(self):
        async def other(sb):
            await sb.commands.run("mkdir", args=["-p", "/opt/prepared"])
            await sb.files.write(MARKER, "something else\n")

        self.assertNotEqual(fingerprint_of(_prepare_marker), fingerprint_of(other))

    def test_the_salt_changes_it(self):
        self.assertNotEqual(
            fingerprint_of(_prepare_marker), fingerprint_of(_prepare_marker, "v2")
        )

    def test_snapshot_name_is_derived_from_the_fingerprint(self):
        env = _env()
        self.assertIn(env.fingerprint, env.snapshot_name)


class TestBuildAndReuse(unittest.TestCase):
    def test_builds_once_then_reuses_by_name(self):
        async def go():
            async with LocalSandboxClient() as client:
                env = _env()
                first = await ensure_prepared(client, env)
                second = await ensure_prepared(client, env)
                return first, second, client.create_calls, len(client.snapshots)

        first, second, creates, snapshots = asyncio.run(go())
        self.assertTrue(first.built, "the first call should have built it")
        self.assertGreater(first.build_s, 0)
        self.assertFalse(second.built, "the second call rebuilt instead of reusing")
        self.assertEqual(first.snapshot_id, second.snapshot_id)
        self.assertEqual(snapshots, 1, "reuse still created a second snapshot")
        self.assertEqual(creates, 1, "reuse still booted a sandbox")

    def test_a_changed_preparation_is_not_reused(self):
        """A suite passing against last week's dependencies is not a signal."""

        async def go():
            async with LocalSandboxClient() as client:
                await ensure_prepared(client, _env(salt="v1"))
                second = await ensure_prepared(client, _env(salt="v2"))
                return second, len(client.snapshots)

        second, snapshots = asyncio.run(go())
        self.assertTrue(second.built)
        self.assertEqual(snapshots, 2)

    def test_rebuild_forces_a_new_build(self):
        async def go():
            async with LocalSandboxClient() as client:
                env = _env()
                await ensure_prepared(client, env)
                return await ensure_prepared(client, env, rebuild=True)

        self.assertTrue(asyncio.run(go()).built)

    def test_a_failing_preparation_is_never_snapshotted(self):
        """Otherwise a broken environment forks into every task in the run."""

        async def broken(sb):
            raise RuntimeError("pip install failed")

        async def go():
            async with LocalSandboxClient() as client:
                try:
                    await ensure_prepared(client, _env(broken))
                except RuntimeError as exc:
                    return str(exc), len(client.snapshots), client.live
                return "", len(client.snapshots), client.live

        message, snapshots, live = asyncio.run(go())
        self.assertIn("pip install failed", message)
        self.assertEqual(snapshots, 0, "a broken environment was snapshotted")
        self.assertEqual(live, 0, "the build sandbox was left running")


class TestForkedTasksInheritThePreparation(unittest.TestCase):
    def test_a_forked_sandbox_starts_with_the_prepared_files(self):
        async def go():
            async with LocalSandboxClient() as client:
                prepared = await ensure_prepared(client, _env())
                sandbox = await client.create(from_snapshot=prepared.snapshot_id)
                return await sandbox.files.read_text(MARKER)

        self.assertEqual(asyncio.run(go()).strip(), "prepared once")

    def test_the_whole_suite_passes_when_forked(self):
        async def go():
            async with LocalSandboxClient() as client:
                prepared = await ensure_prepared(client, _env())
                report = await run_suite(client, TASKS, "correct", prepared=prepared)
                return report

        report = asyncio.run(go())
        failures = [(r.task_id, r.status, r.error) for r in report.results if r.status != "PASS"]
        self.assertEqual(failures, [], f"tasks failed when forked: {failures}")
        self.assertIsNotNone(report.environment)
        self.assertEqual(report.environment["built_this_run"], True)

    def test_tasks_do_not_depend_on_the_preparation(self):
        """--no-snapshot has to stay a fair baseline, not a different suite."""

        async def go():
            async with LocalSandboxClient() as client:
                return await run_suite(client, TASKS, "correct", prepared=None)

        report = asyncio.run(go())
        failures = [(r.task_id, r.status) for r in report.results if r.status != "PASS"]
        self.assertEqual(failures, [], f"tasks need the prepared environment: {failures}")


class TestShippedEnvironment(unittest.TestCase):
    def test_the_repo_ships_one_and_it_loads(self):
        env = load_environment(ROOT / "tasks")
        self.assertIsNotNone(env, "tasks/_environment.py did not load")
        self.assertTrue(env.description)
        self.assertTrue(env.snapshot_name.startswith("agent-eval-env-"))

    def test_a_directory_without_one_is_fine(self):
        self.assertIsNone(load_environment(ROOT / "agent_eval"))

    def test_the_environment_module_is_not_loaded_as_a_task(self):
        self.assertNotIn("_environment", [t.id for t in discover(ROOT / "tasks")])


if __name__ == "__main__":
    unittest.main()
