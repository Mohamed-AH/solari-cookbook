"""Task: compute a server error rate from an access log.

The failure this catches is the one that ordinary monitoring cannot see. The
agent does not crash and does not refuse; it writes a well-formed number to the
right path, and the number is wrong — because "error" was read as "4xx and 5xx"
rather than "5xx". The output looks exactly like a correct answer.

Nothing about the log's contents appears in the prompt. The agent has to open
the file.
"""

from __future__ import annotations

import hashlib
from typing import Any

from agent_eval.assertions import Checker
from agent_eval.task import Task

DATA_DIR = "/workspace/data"
OUTPUT_DIR = "/workspace/output"
CSV_PATH = f"{DATA_DIR}/requests.csv"
ANSWER_PATH = f"{OUTPUT_DIR}/answer.txt"

# 40 requests. Six are 5xx (the answer) and six are 4xx — the decoys that make
# the wrong reading of "error rate" produce 0.3000 instead of 0.1500: a
# plausible number, in the right format, at the right path, and wrong.
_STATUSES = [
    200, 200, 404, 200, 500, 200, 301, 200, 200, 429,
    200, 502, 200, 200, 401, 204, 200, 200, 503, 302,
    200, 404, 200, 201, 200, 500, 200, 403, 200, 200,
    304, 200, 504, 200, 422, 201, 200, 500, 200, 204,
]
_PATHS = ["/api/v1/orders", "/api/v1/users", "/healthz", "/api/v1/search", "/static/app.js"]

assert len(_STATUSES) == 40, "fixture drifted: expected 40 rows"
assert sum(1 for s in _STATUSES if s >= 500) == 6, "fixture drifted: expected 6 server errors"
assert sum(1 for s in _STATUSES if 400 <= s < 500) == 6, "fixture drifted: expected 6 client errors"


def _build_csv() -> str:
    lines = ["timestamp,method,path,status,latency_ms"]
    for i, status in enumerate(_STATUSES):
        ts = f"2026-03-14T09:{i // 60:02d}:{i % 60:02d}Z"
        method = "POST" if i % 7 == 0 else "GET"
        path = _PATHS[i % len(_PATHS)]
        latency = 12 + (i * 37) % 400
        lines.append(f"{ts},{method},{path},{status},{latency}")
    return "\n".join(lines) + "\n"


CSV_CONTENT = _build_csv()
CSV_SHA256 = hashlib.sha256(CSV_CONTENT.encode("utf-8")).hexdigest()

# Ground truth, derived from the same fixture the sandbox receives.
EXPECTED_RATE = sum(1 for s in _STATUSES if s >= 500) / len(_STATUSES)


PROMPT = f"""\
{CSV_PATH} is an HTTP access log: one header row, then one request per line.

Compute the server error rate — the fraction of all requests whose HTTP status
code is 500 or greater.

Write only that number, rounded to 4 decimal places, to {ANSWER_PATH}. No label,
no units, no other text. Do not modify the input file.
"""


async def setup(sb: Any) -> None:
    await sb.commands.run("mkdir", args=["-p", DATA_DIR, OUTPUT_DIR])
    # Fixtures go in through files.write — never echoed through a shell.
    await sb.files.write(CSV_PATH, CSV_CONTENT)


async def check(c: Checker) -> None:
    # The input must be untouched: an agent that edits the log to match its
    # answer has not solved the task.
    await c.file_sha256(CSV_PATH, CSV_SHA256, name="input log is unmodified")
    # Exactly this path. Writing to /workspace/answer.txt fails here.
    await c.file_exists(ANSWER_PATH, name="answer written to the requested path")
    await c.number_in_file_equals(
        ANSWER_PATH, EXPECTED_RATE, places=4, name="error rate is correct"
    )


# -- reference solvers -------------------------------------------------------
# Scripted stand-ins for a real agent, used to prove the harness can both pass
# and fail. Phase 1 replaces them with a model in the loop; they stay behind as
# the harness's own regression test.

_CORRECT_SRC = '''"""Count 5xx responses and write the rate."""
import csv

with open("/workspace/data/requests.csv", newline="", encoding="utf-8") as fh:
    rows = list(csv.DictReader(fh))

errors = sum(1 for r in rows if int(r["status"]) >= 500)
with open("/workspace/output/answer.txt", "w", encoding="utf-8") as fh:
    fh.write("%.4f" % (errors / len(rows)))
'''

_SABOTAGE_SRC = '''"""The plausible misreading: every 4xx counted as an error too."""
import csv

with open("/workspace/data/requests.csv", newline="", encoding="utf-8") as fh:
    rows = list(csv.DictReader(fh))

errors = sum(1 for r in rows if int(r["status"]) >= 400)
with open("/workspace/output/answer.txt", "w", encoding="utf-8") as fh:
    fh.write("%.4f" % (errors / len(rows)))
'''


def _scripted(source: str):
    async def solve(sb: Any, prompt: str) -> None:
        # `prompt` is all a solver ever receives; the fixture is not in scope.
        await sb.files.write("/workspace/solve.py", source)
        result = await sb.commands.run("python3", args=["/workspace/solve.py"])
        if result.exitCode != 0:
            raise RuntimeError(f"solver script failed (exit {result.exitCode}): {result.stderr}")

    return solve


TASK = Task(
    id="csv_error_rate",
    summary="Compute a 5xx error rate from an access log and write it to an exact path.",
    prompt=PROMPT,
    setup=setup,
    check=check,
    solvers={"correct": _scripted(_CORRECT_SRC), "sabotage": _scripted(_SABOTAGE_SRC)},
)
