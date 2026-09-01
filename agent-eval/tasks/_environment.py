"""The environment every task in this directory runs in.

Built once per run (or reused across runs), snapshotted, and forked per task.
Files starting with `_` are not tasks — the loader skips them.

In your own repo this is where you would clone the project and install its
dependencies. Here it installs a test runner and pre-creates the directory the
harness writes its verifier scripts into: small enough to stay honest about,
expensive enough that paying for it six times instead of once is the whole
reason snapshots exist.

Tasks must not *require* anything installed here. They run against the plain
`base` template too, so `--no-snapshot` stays a fair comparison rather than a
different suite.
"""

from __future__ import annotations

from typing import Any

DESCRIPTION = "pytest + the harness's verifier directory"

# Bump to force a rebuild without editing prepare() itself.
FINGERPRINT_SALT = "v1"


async def prepare(sb: Any) -> None:
    """Runs once, in one sandbox, before any task."""
    result = await sb.commands.run(
        "pip", args=["install", "--quiet", "--disable-pip-version-check", "pytest"]
    )
    if result.exitCode != 0:
        # Loudly, not silently: a half-prepared environment snapshotted here
        # would be forked into every task for the rest of the run.
        raise RuntimeError(
            f"preparing the environment failed (exit {result.exitCode}): "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )

    await sb.commands.run("mkdir", args=["-p", "/opt/agent_eval"])
