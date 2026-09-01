"""Task: aggregate an event log into a JSON summary with an exact schema.

The failure here is the quietest one in the suite. The output is valid JSON,
every required key is present, every value is a number of the right type, and
one of them counts the wrong thing: rows instead of distinct users. Nothing
short of comparing against an independently computed answer catches it.

This is the "the JSON is valid but `error_rate` is computed wrong" case, and
it is why the assertion is a document comparison rather than a schema check.
"""

from __future__ import annotations

import json
from typing import Any

from agent_eval.agent import FunctionAgent, Observation, finish, read, write
from agent_eval.assertions import Checker
from agent_eval.task import Task

DATA_DIR = "/workspace/data"
OUTPUT_DIR = "/workspace/output"
EVENTS_PATH = f"{DATA_DIR}/events.jsonl"
SUMMARY_PATH = f"{OUTPUT_DIR}/summary.json"

# (user, action, ms). Users repeat — that is the whole point.
_EVENTS = [
    ("alice", "open", 120),
    ("bob", "open", 340),
    ("alice", "edit", 90),
    ("carol", "open", 210),
    ("alice", "save", 450),
    ("dave", "open", 180),
    ("bob", "edit", 260),
    ("erin", "open", 130),
    ("carol", "save", 620),
    ("alice", "open", 75),
    ("dave", "edit", 310),
    ("bob", "save", 155),
]

EVENTS_CONTENT = (
    "\n".join(
        json.dumps({"user": u, "action": a, "ms": ms}, sort_keys=True) for u, a, ms in _EVENTS
    )
    + "\n"
)


def _median(values: list[int]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2


_ACTIONS: dict[str, int] = {}
for _u, _a, _ms in _EVENTS:
    _ACTIONS[_a] = _ACTIONS.get(_a, 0) + 1

# Ground truth, computed from the same fixture the sandbox receives.
EXPECTED_SUMMARY = {
    "total_events": len(_EVENTS),
    "unique_users": len({u for u, _, _ in _EVENTS}),
    "p50_ms": _median([ms for _, _, ms in _EVENTS]),
    "actions": dict(sorted(_ACTIONS.items())),
}

assert EXPECTED_SUMMARY["total_events"] == 12, "fixture drifted"
assert EXPECTED_SUMMARY["unique_users"] == 5, "fixture drifted"
assert EXPECTED_SUMMARY["unique_users"] != EXPECTED_SUMMARY["total_events"], (
    "fixture is not discriminating: counting rows would give the right answer"
)

PROMPT = f"""\
{EVENTS_PATH} is a JSON Lines file: one JSON object per line, each with the
keys "user", "action" and "ms".

Write a summary to {SUMMARY_PATH} as a JSON object with exactly these four
keys and no others:

  total_events  how many events the file contains
  unique_users  how many distinct users appear
  p50_ms        the median of the "ms" values; with an even number of events,
                the mean of the two middle values
  actions       an object mapping each action name to how many times it occurs

Write nothing else to that file.
"""


async def setup(sb: Any) -> None:
    await sb.commands.run("mkdir", args=["-p", DATA_DIR, OUTPUT_DIR])
    await sb.files.write(EVENTS_PATH, EVENTS_CONTENT)


async def check(c: Checker) -> None:
    await c.file_exists(SUMMARY_PATH, name="summary written to the requested path")
    await c.json_matches(
        SUMMARY_PATH, EXPECTED_SUMMARY, name="summary matches the independently computed answer"
    )


# -- reference agents --------------------------------------------------------
# Both parse the log they observed and compute a summary. They differ in one
# expression: whether "unique users" means distinct users or rows.


def _summarise(text: str, *, dedupe_users: bool) -> dict[str, Any]:
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not rows:
        raise ValueError("observed an empty event log — the read returned nothing")
    actions: dict[str, int] = {}
    for row in rows:
        actions[row["action"]] = actions.get(row["action"], 0) + 1
    users = {r["user"] for r in rows} if dedupe_users else rows
    return {
        "total_events": len(rows),
        "unique_users": len(users),
        "p50_ms": _median([r["ms"] for r in rows]),
        "actions": dict(sorted(actions.items())),
    }


def _make_agent(name: str, *, dedupe_users: bool) -> FunctionAgent:
    async def next_action(obs: Observation):
        if obs.last_action is None:
            return read(EVENTS_PATH)
        if obs.last_action.kind == "read":
            if not obs.last_result or not obs.last_result.ok:
                raise RuntimeError(f"could not read the event log: {obs.last_result}")
            summary = _summarise(obs.last_result.stdout, dedupe_users=dedupe_users)
            return write(SUMMARY_PATH, json.dumps(summary, indent=2, sort_keys=True))
        return finish(f"wrote the summary to {SUMMARY_PATH}")

    return FunctionAgent(name, next_action)


TASK = Task(
    id="json_report_schema",
    summary="Aggregate a JSONL event log into a JSON summary with an exact schema and values.",
    prompt=PROMPT,
    setup=setup,
    check=check,
    max_steps=6,
    agents={
        "correct": lambda: _make_agent("correct", dedupe_users=True),
        # Counts rows, not distinct users. Valid JSON, right keys, wrong number.
        "sabotage": lambda: _make_agent("sabotage", dedupe_users=False),
    },
)
