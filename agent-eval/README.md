# agent-eval

Execution-based regression testing for LLM agents, on Solari sandboxes.

> Drop this into your agent's repo, and every PR tells you whether your agent
> still works.

**Status: phase 0 — the substrate.** Task format, sandbox lifecycle, assertion
layer, and two end-to-end tasks. The real agent loop, the CLI, the GitHub
Action and the snapshot work land in later phases; this README grows with them.

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
| `prompt` | the only text a solver ever receives |
| `setup` | builds the fixture in a fresh sandbox, via `files.write` |
| `check` | decides pass/fail from the machine's end state |
| `solvers` | scripted reference implementations, used to test the harness |

`prompt` and `setup` are separate on purpose. Hand an agent the setup commands
"for context" and it can emit the expected answer without opening the file —
and the task still reports a pass while measuring nothing.

## The two phase-0 tasks

**`csv_error_rate`** — compute a 5xx rate from a 40-row access log and write it
to an exact path. Six responses are 5xx and six are 4xx, so reading "error
rate" as "4xx and 5xx" produces `0.3000` instead of `0.1500`: a plausible
number, correctly formatted, at the right path, and wrong.

**`test_suite_integrity`** — a test suite with one failing test and a bug in the
code under test. The cheapest way to make a suite green is to delete the
assertion that is red, so three checks separate a real fix from a green-looking
one: the test file must be byte-identical to what setup wrote, the visible suite
must exit 0, and a hidden suite the agent never saw must also exit 0.

## Running

```bash
cd agent-eval
pip install -e .
export SOLARI_API_KEY=slr_live_...   # https://console.getsolari.com

agent-eval --list
agent-eval                                  # every task, correct solver
agent-eval --solver sabotage --expect fail  # the harness's own self-test
```

`--expect fail` is how the harness proves it can go red: it runs the sabotage
solvers and requires *every* task to fail. A suite that cannot fail is not
measuring anything.

## Tests

The harness's own tests need no sandbox, no key and no network:

```bash
cd agent-eval && python3 -m unittest discover -s tests
```
