"""Probes for the host tools the local test double relies on.

The tasks target a Linux sandbox. Run the offline suite on a machine whose
`find` is Windows' `find.exe` — a string search, not findutils — and
`log_cleanup_precision` fails for a reason that has nothing to do with the
harness or the agent.

That is precisely the confusion this project exists to prevent: an
environment problem reported as a verdict on the agent. So probe the tools,
and skip with a message that names what is missing.
"""

from __future__ import annotations

import subprocess
import tempfile
import time
from functools import lru_cache
from pathlib import Path
from typing import Any


def _run(argv: list[str]) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None


@lru_cache(maxsize=None)
def probe_git() -> str:
    """Return '' if git is usable, else why not."""
    res = _run(["git", "--version"])
    if res is None or res.returncode != 0:
        return "git is not available on PATH"
    return ""


@lru_cache(maxsize=None)
def probe_find_and_touch() -> str:
    """Return '' if GNU-style find and touch are usable, else why not.

    Checked by behaviour, not by version string: backdate a file, then ask
    find to match it by age. Windows' find.exe passes neither.
    """
    with tempfile.TemporaryDirectory() as tmp:
        probe = Path(tmp) / "probe.log"
        probe.write_bytes(b"probe\n")

        res = _run(["touch", "-d", "10 days ago", str(probe)])
        if res is None:
            return "touch is not available on PATH (needs GNU coreutils)"
        if res.returncode != 0:
            return f"touch -d is not supported here: {res.stderr.strip() or 'non-zero exit'}"

        age_days = (time.time() - probe.stat().st_mtime) / 86400
        if age_days < 7:
            return "touch -d did not backdate the file (needs GNU coreutils)"

        res = _run(["find", str(tmp), "-name", "*.log", "-mtime", "+7"])
        if res is None:
            return "find is not available on PATH (needs GNU findutils)"
        if res.returncode != 0:
            return (
                "find does not support -name/-mtime here — on Windows this is "
                "usually System32\\\\find.exe shadowing GNU findutils: "
                f"{res.stderr.strip() or 'non-zero exit'}"
            )
        if "probe.log" not in res.stdout:
            return "find -mtime +7 did not match a backdated file (needs GNU findutils)"
    return ""


_PROBES = {
    "git": probe_git,
    "find": probe_find_and_touch,
    "touch": probe_find_and_touch,
}


def unsupported_reason(task: Any) -> str:
    """Why this task cannot run against the local double here, or ''."""
    checked = set()
    for tool in getattr(task, "required_tools", ()):
        probe = _PROBES.get(tool)
        if probe is None or probe in checked:
            continue
        checked.add(probe)
        reason = probe()
        if reason:
            return f"{task.id} needs `{tool}`: {reason}"
    return ""


def supported(tasks: list[Any]) -> list[Any]:
    """The subset of tasks this host can actually run."""
    return [t for t in tasks if not unsupported_reason(t)]
