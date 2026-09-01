"""The environment every task in this directory runs in.

Built once per run (or reused across runs), snapshotted, and forked per task.
Files starting with `_` are not tasks — the loader skips them.

In your own repo this is where you would clone the project and install its
dependencies. What is here deliberately stays cheap — and that means you should
NOT run this suite with `--snapshot`.

Measured on a real account, restoring a snapshot costs ~9.5s more than booting
the plain template, so a preparation has to be more expensive than that before
forking wins. Varying only this file:

    preparation                                    cost    verdict
    pytest only                                     1.7s   skip  (+46s a run)
    a few mid-size wheels                           7.7s   skip  (+11s a run)
    pandas, scipy, scikit-learn, matplotlib, ...   15.5s   USE   (-28s a run)

Run `agent-eval --bench-snapshot 3` against your own preparation rather than
copying anyone's answer, including this one.

Tasks must not *require* anything installed here. They run against the plain
`base` template too, so running with and without a snapshot compares the same
suite rather than two different ones.
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
