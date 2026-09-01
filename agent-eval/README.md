# agent-eval

Execution-based regression testing for LLM agents, on Solari sandboxes.

> Drop this into your agent's repo, and every PR tells you whether your agent
> still works.

**Status: phase 1 — the agent loop.** Six tasks, a bounded observation ->
action -> result loop, and a pluggable agent interface. The CLI flags, the
GitHub Action and the snapshot work land in later phases; this README grows
with them.

## Why

Agents rarely fail by producing nothing — ordinary error monitoring catches
that. They fail by producing something fluent, confident and wrong: the JSON is
valid but the rate is computed wrong; the suite goes green because the agent
deleted the assertion; the file lands in `/workspace/` instead of
`/workspace/output/`.

None of that is visible in the agent's output text. It is only visible in the
end state of a real machine after the agent has worked in it.

## How a task is scored

Every assertion is decided by a command actually executed in the sandbox — an
integer exit code, or the exact stdout that command produced. No LLM judge.

A task is four separable pieces:

| piece | what it is |
| --- | --- |
| `prompt` | the only text an agent ever receives |
| `setup` | builds the fixture in a fresh sandbox, via `files.write` |
| `check` | decides pass/fail from the machine's end state |
| `agents` | reference implementations, used to test the harness itself |

`prompt` and `setup` are separate on purpose. Hand an agent the setup commands
"for context" and it can emit the expected answer without opening the file —
and the task still reports a pass while measuring nothing.

## The agent loop

The harness owns the loop. An agent only decides the next action:

```python
class Agent(Protocol):
    name: str

    async def next_action(self, obs: Observation) -> Action: ...
```

An `Observation` carries the task prompt and the result of the agent's own
previous action. Nothing else — there is no parameter through which a task, a
fixture, or an expected answer could arrive.

An `Action` is one of five things: run a command, write a file, read a file,
list a directory, or finish. The harness executes it, captures the exit code
and output, records the step, and hands the result back as the next
observation, until the agent finishes or `max_steps` runs out.

Keeping the loop on this side is what makes the trajectory worth reading: every
step is recorded by the harness from the sandbox's own responses, so an agent
that misreports what it did cannot fabricate the exit codes in the record.

`agent_eval.claude_agent.ClaudeAgent` is a reference implementation of that
interface in about 80 lines. Replace it with yours.

## The suite

| task | the failure it catches |
| --- | --- |
| `csv_error_rate` | Reads "error rate" as 4xx *and* 5xx: `0.3000` instead of `0.1500`. Plausible number, right format, right path, wrong. |
| `test_suite_integrity` | Makes the suite green by deleting the failing assertion. |
| `stack_trace_fix` | Wraps the crash in try/except. The script now exits 0 and prints `335` instead of `470`. |
| `secret_leak_guard` | `git add -A` — right commit message, right code change, and a credential in git history. |
| `json_report_schema` | Valid JSON, correct keys, `unique_users` counting rows instead of distinct users. |
| `log_cleanup_precision` | Deletes every log rather than only the aged ones. "Deleted 5 files" reads exactly like "deleted 3 files". |

Three of them — `test_suite_integrity`, `stack_trace_fix`, `secret_leak_guard`
— cannot be solved without reading a result and acting on it. You cannot fix a
failing test you have not run, patch a traceback you have not seen, or stage
selectively without looking at `git status`.

Every task ships two reference agents: one that solves it, and a saboteur that
fails it in the way described above. The suite asserts both, because a task
that cannot go red is not measuring anything.

## Running

```bash
cd agent-eval
pip install -e .
export SOLARI_API_KEY=slr_live_...   # https://console.getsolari.com

agent-eval --list
agent-eval                                  # every task, correct solver
agent-eval --agent sabotage --expect fail  # the harness's own self-test
```

`--expect fail` is how the harness proves it can go red: it runs the sabotage
solvers and requires *every* task to fail. A suite that cannot fail is not
measuring anything.

## Tests

The harness's own tests need no sandbox, no key and no network — so CI can run
them on every PR without spending credits:

```bash
cd agent-eval && python3 -m unittest discover -s tests
```

They run each task's full lifecycle — setup, agent loop, checks — against a
local test double, asserting that every task passes under its correct agent and
**fails** under its saboteur. That double is not a backend and is not reachable
from the CLI: it proves the harness logic is sound and says nothing about
whether an agent works in a real sandbox. That is what the sandbox run is for.
