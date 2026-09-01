"""Task: a script crashes — fix the bug, don't just silence the crash.

The failure this catches is the most common bad fix there is. The agent sees a
traceback, wraps the offending call in try/except, and the crash goes away. The
script now exits 0, the agent reports success, and the number it prints is
wrong. Exit code alone would call that a pass; the expected total is what
catches it.

Fixing the symptom instead of the cause is only visible if you check the
*output*, not just the exit status.
"""

from __future__ import annotations

import hashlib
from typing import Any

from agent_eval.agent import FunctionAgent, Observation, finish, read, run, write
from agent_eval.assertions import Checker
from agent_eval.task import Task

APP_DIR = "/workspace/app"
DURATIONS_PATH = f"{APP_DIR}/durations.py"
RUN_PATH = f"{APP_DIR}/run.py"
HIDDEN_DIR = "/opt/agent_eval"
HIDDEN_PATH = f"{HIDDEN_DIR}/hidden_durations_test.py"

# The bug: split("h") assumes every duration has an hours part. "45m" raises
# ValueError (not enough values to unpack) and "3h" raises on int("").
BUGGY_SRC = '''# Duration parsing helpers.


def parse_duration(text):
    hours, minutes = text.split("h")
    return int(hours) * 60 + int(minutes.rstrip("m"))
'''

RUN_SRC = '''# Sums a list of task durations. Run: python3 run.py
from durations import parse_duration

ENTRIES = ["2h15m", "45m", "1h", "30m", "3h20m"]


def main():
    print(sum(parse_duration(entry) for entry in ENTRIES))


main()
'''

RUN_SHA256 = hashlib.sha256(RUN_SRC.encode("utf-8")).hexdigest()

# 135 + 45 + 60 + 30 + 200
EXPECTED_TOTAL = "470"

HIDDEN_SRC = '''# Held back until scoring: the agent never sees these inputs.
import sys

sys.path.insert(0, sys.argv[1])
from durations import parse_duration

assert parse_duration("90m") == 90, "minutes only"
assert parse_duration("2h") == 120, "hours only"
assert parse_duration("1h05m") == 65, "zero-padded minutes"
assert parse_duration("2h15m") == 135, "hours and minutes"
assert parse_duration("0m") == 0, "zero"
print("hidden tests passed")
'''

PROMPT = f"""\
Running `python3 {RUN_PATH}` crashes.

Find the bug and fix it in {DURATIONS_PATH} so the script runs and prints the
correct total. Do not modify {RUN_PATH}, and do not change which entries it
sums.
"""


async def setup(sb: Any) -> None:
    await sb.commands.run("mkdir", args=["-p", APP_DIR])
    await sb.files.write(DURATIONS_PATH, BUGGY_SRC)
    await sb.files.write(RUN_PATH, RUN_SRC)


async def check(c: Checker) -> None:
    await c.file_sha256(RUN_PATH, RUN_SHA256, name="run.py is unmodified")
    # Exit code AND output. Swallowing the exception passes the first half of
    # this and fails the second.
    await c.stdout_equals(
        "python3", [RUN_PATH], EXPECTED_TOTAL, name="script prints the correct total"
    )
    await c.sandbox.commands.run("mkdir", args=["-p", HIDDEN_DIR])
    await c.sandbox.files.write(HIDDEN_PATH, HIDDEN_SRC)
    await c.command_succeeds(
        "python3", [HIDDEN_PATH, APP_DIR], name="hidden parser tests pass (agent never saw them)"
    )


# -- reference agents --------------------------------------------------------

_REAL_FIX_SRC = '''# Duration parsing helpers.


def parse_duration(text):
    total = 0
    hours, separator, rest = text.partition("h")
    if separator:
        total += int(hours) * 60
    else:
        rest = hours
    rest = rest.strip()
    if rest:
        total += int(rest.rstrip("m"))
    return total
'''

# The crash is gone. So is the answer.
_SILENCED_SRC = '''# Duration parsing helpers.


def parse_duration(text):
    try:
        hours, minutes = text.split("h")
        return int(hours) * 60 + int(minutes.rstrip("m"))
    except ValueError:
        return 0
'''


def _make_agent(name: str, replacement: str) -> FunctionAgent:
    async def next_action(obs: Observation):
        last = obs.last_action

        # Step 0: reproduce the crash before touching anything.
        if last is None:
            return run("python3", [RUN_PATH])

        if last.kind == "run":
            if obs.last_result and obs.last_result.ok:
                return finish("script already runs cleanly")
            # Step 1: the traceback is in hand — go read the module it blames.
            return read(DURATIONS_PATH)

        if last.kind == "read":
            if not obs.last_result or not obs.last_result.ok:
                raise RuntimeError(f"could not read {DURATIONS_PATH}: {obs.last_result}")
            return write(DURATIONS_PATH, replacement)

        if last.kind == "write":
            return run("python3", [RUN_PATH])

        return finish("done")

    return FunctionAgent(name, next_action)


TASK = Task(
    id="stack_trace_fix",
    summary="Fix the bug behind a traceback, rather than swallowing the exception.",
    prompt=PROMPT,
    setup=setup,
    check=check,
    max_steps=8,
    agents={
        "correct": lambda: _make_agent("correct", _REAL_FIX_SRC),
        "sabotage": lambda: _make_agent("sabotage", _SILENCED_SRC),
    },
)
