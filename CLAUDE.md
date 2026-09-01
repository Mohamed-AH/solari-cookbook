# CLAUDE.md

Project context for this repo. Read this before writing code.

## What this project is

An execution-based agent evaluation harness built on Solari sandboxes,
positioned as a product with a real user rather than an internal benchmark:

> Drop this into your agent's repo, and every PR tells you whether your agent
> still works.

Audience: anyone shipping an LLM agent. The harness must be installable and
runnable by someone other than the author.

It lives in `agent-eval/` at the root of this fork — a sibling of `examples/`,
not inside it. The cookbook's own contributing guidance asks for small,
single-idea examples; this is deliberately the opposite, so filing it as an
example would misrepresent it and bury it. `examples/` is untouched upstream
content.

### Why it exists

Agents rarely fail by producing nothing — that case is caught by ordinary error
monitoring. They fail by producing something fluent, confident and wrong: the
JSON is valid but `error_rate` is computed wrong; the test suite goes green
because the agent deleted the assertion; the file lands in `/workspace/` instead
of `/workspace/output/`; the agent reports "fixed it" and never touched the file.

None of those are detectable from the agent's output text. They are only
detectable by inspecting the end state of a real machine after the agent has
worked in it.

And because agents are non-deterministic, one manual try proves nothing. The
only meaningful measurement is a rate across a suite, which makes the useful
signal not "does it work" but "did it get worse than the last commit."

### Competitive position (be accurate about this — do not claim novelty)

- Eval-in-CI is an established, commercialized workflow. Braintrust ships a
  GitHub Action that posts eval results as PR comments and gates merges on
  regression; LangSmith, Langfuse and DeepEval have variants. The workflow is
  not new.
- Execution-based evaluation is established in research. SWE-bench applies
  patches in isolated Docker containers and checks fail-to-pass plus
  pass-to-pass; Terminal-Bench runs agents in sandboxed shells against
  hand-written test suites. The methodology is not new.
- The gap is between them. The CI-eval products score text, traces and tool
  calls. The execution-based harnesses score real machine state but are fixed
  benchmarks for ranking models, not something you point at your own agent with
  your own tasks in your own CI.

One-liner: **SWE-bench's methodology, on your agent, in your CI.**

Strategically relevant: Terminal-Bench's sandbox backends are Docker, E2B and
Daytona — Solari's direct competitors. Agent-eval-in-sandboxes is a workload
Solari is not visible in. Framing the project as "a workload category running on
Solari, with the numbers" is more valuable than framing it as a demo.

## Hard rules

1. **Every assertion is deterministic** — exit code or file state. No
   LLM-as-judge. Determinism is the entire differentiator; adding a judge
   dilutes the one thing that makes this credible.
2. **Correctness over breadth.** A small suite that is provably right beats a
   large one that isn't. Target 5–6 solid tasks, not a big number.
3. **2–3 tasks must require multiple observations** — run, read the error, fix,
   re-run — so the agent loop is genuinely exercised rather than decorative.
4. **Assert with `commands.run`, on `exitCode` and `stdout`.** Never stringify a
   result object and substring-match it.
5. **Build fixtures with `sb.files.write`.** Never interpolate shell strings
   into Python source.
6. **Errors surface loudly.** No bare `except` that swallows a failure. A silent
   no-op that looks like it worked is the worst possible failure mode here.
7. **Fixture contents never enter the agent's prompt.** Passing setup commands
   in as "context" hands the model the input file verbatim and lets it emit the
   expected answer without reading anything.
8. **Populate the trajectory.** Every step — observation, action, result — gets
   recorded. An empty trajectory field is worse than no field.
9. **Secrets in `secrets`, never `vars`.** GitHub Actions repository variables
   are plaintext and unmasked in logs. Do not run a credit-spending suite on
   untrusted `pull_request` triggers.
10. **Ask before long runs.** Sandbox credits are finite; check before spending
    them on a large suite execution.

Rules 4–8 come from concrete failures in an earlier trial. Each one produced a
harness that reported passes it had not earned.

## Solari Python SDK reference

Verified against `solari-sandbox==0.2.0` / `solari-core==0.2.0` by reading the
package source. `docs.getsolari.com` may be unreachable from the dev
environment; trust this over recall, and verify against the installed package if
anything looks off.

### Client

```
SandboxClient(*, api_key, base_url, call_timeout_ms=None, kind="sandbox")
```
Keyword-only. `base_url` is REQUIRED; canonical value `https://api.getsolari.com`.

- `client.create(*, template, cpu, mem_mb, disk_gb, envs, metadata, timeout_ms,
  from_snapshot, lifecycle, volumes) -> Sandbox`
- `client.list(...)` returns a **dict**: `{"sandboxes": [SandboxView], "nextCursor": ...}`
  — not a list. `client.list_all()` is an async generator that auto-paginates.
- `client.get(id)`, `client.connect(id)`, `client.kill(id)`
- `client.create_desktop(...)` for GUI / computer-use sessions

Unset fields are dropped from the create body — the gateway rejects an explicit
`fromSnapshot: null`.

### Snapshots and templates

- `sb.snapshot(name) -> snapshot_id`
- `client.create(from_snapshot=snapshot_id)` to fork
- `client.promote_snapshot(id, name)` turns a snapshot into a reusable template
- `list_snapshots` / `get_snapshot` / `delete_snapshot`
- `TemplateClient.build()` with the `Image` builder
  (`Image.base().apt_install().pip_install().run_commands().env().workdir()`)
  for prebuilt custom images

### Execution

```
sb.commands.run(cmd, *, args, cwd, env, user, timeout_ms, background,
                on_stdout, on_stderr) -> CommandResult(exitCode, stdout, stderr)
```
`cmd` runs **without a shell**. For shell syntax use `run("sh", args=["-c", "..."])`.

`sb.commands.start(...)` returns a handle with `.wait()` / `.stdin()` / `.kill()`
for streaming.

```
sb.run_code(code, *, language, context_id, on_stdout, on_stderr)
  -> RunCodeResult(results: list[CodeResultItem], error, charts)
```
`RunCodeResult` has **no** `.stdout` / `.output` / `.logs` / `.text` attribute.
Output lives in `.results[i].text` where `.results[i].type == "stdout"` or
`"stderr"`. Prefer `commands.run` for anything you need to assert on.

### Filesystem, git, misc

- `sb.files`: `read`, `read_text`, `write`, `list`, `stat`, `rename`, `remove`,
  `mkdir`, `search`, `watch`, `upload`, `download`
- `sb.git`: `clone`, `status`, `add`, `commit`, `push`, `pull`, `checkout`,
  `branches`, `log` — first-class, non-shell, and easy to assert against
- `sb.metrics() -> MetricsResult(cpuPct, memBytes, memTotalBytes, diskBytes)`
- `sb.preview_url(port)`, `sb.download_url(path)`, `sb.upload_url(path)`
- `sb.pause()`, `sb.resume()`, `sb.set_timeout(ms)`; idle policy via
  `lifecycle={"onTimeout": "pause", "autoResume": True}`

### Errors

`solari_sandbox` re-exports `solari_core`'s error classes:
`GatewayError` (base), `AuthError` (401), `PlanError` (402),
`ConcurrencyLimitError` (429), `NoCapacityError` (503), `ActionError`,
`TimeoutError`, `ConnectionError`. Each carries `.status`, `.code`, `.message`.

### Concurrency

There is an account-level concurrency cap. Parallelism needs a bounded
semaphore, not an unbounded `asyncio.gather` — **and the cap differs per
account**, so the bound has to adapt rather than be hard-coded. See the live
findings below.

## Conventions

- Tasks live in a `tasks/` directory, one module or file per task — not one
  monolithic file.
- Commit in small, reviewable steps with clear messages.
- Numbers worth capturing for the README and the launch post: wall-clock for N
  parallel tasks vs summed latency, and cold-start cost with snapshots vs
  without.

---

# Current state

Phases 0–2 are complete, pushed, and **verified against real Solari sandboxes**.
96 offline tests pass with no key and no network.

## Layout

```
solari-cookbook/                 fork root (Mohamed-AH/solari-cookbook)
├── CLAUDE.md                    this file
├── README.md                    upstream + a pointer to agent-eval
├── examples/                    untouched upstream
├── .github/workflows/
│   ├── tests.yml                offline tests, push + PR (spends nothing)
│   └── agent-eval.yml           live suite, workflow_dispatch ONLY (rule 9)
└── agent-eval/
    ├── action.yml               composite action others drop into their repo
    ├── pyproject.toml           pip install -e '.[claude]'; CLI: agent-eval
    ├── agent_eval/
    │   ├── agent.py             the loop: observation -> action -> result
    │   ├── assertions.py        Checker; every check is exit code or stdout
    │   ├── builtins.py          agent resolution, incl. pkg.mod:MyAgent
    │   ├── claude_agent.py      reference LLM agent (optional extra)
    │   ├── concurrency.py       AdaptiveLimiter
    │   ├── config.py, sandbox.py, task.py, runner.py, report.py, cli.py
    ├── tasks/                   six tasks, one file each
    └── tests/                   96 tests; local_sandbox.py is a TEST DOUBLE
```

`tests/local_sandbox.py` executes commands for real against a temp directory so
the whole lifecycle runs offline. It is deliberately **not** importable from
`agent_eval` and not reachable from the CLI: it proves harness logic, never that
an agent works in a real sandbox.

## The suite

| task | the saboteur's failure | why it looks like success |
| --- | --- | --- |
| `csv_error_rate` | counts 4xx as errors | `0.3000` not `0.1500`; right format, right path |
| `test_suite_integrity` | deletes the failing assertion | suite exits 0 |
| `stack_trace_fix` | wraps the crash in try/except | exits 0, prints `335` not `470` |
| `secret_leak_guard` | `git add -A` | right message, right diff, credential in history |
| `json_report_schema` | counts rows as users | valid JSON, all keys, `12` not `5` |
| `log_cleanup_precision` | drops the age filter | "deleted 5 files" reads like "deleted 3" |

Multi-observation (cannot be solved without reading a result and acting on it):
`test_suite_integrity`, `stack_trace_fix`, `secret_leak_guard`.

Every task ships `correct` and `sabotage` reference agents. The suite asserts
both directions — a task that cannot go red is not measuring anything.
`agent-eval --agent sabotage --expect fail` is the self-test and exits 0 only
when every task fails.

## Verified live (2026-09-01, free tier)

First contact with the real API. All four reachable saboteurs were caught in
real VMs, with the evidence in the check output:

- `stack_trace_fix` printed `335` while exiting 0
- `secret_leak_guard` put `.env` into `git log --all --name-only`
- `log_cleanup_precision` deleted `app-2026-08-30.log` along with aged ones
- `test_suite_integrity` changed the test file's sha256 and failed the hidden suite

Cold single task: `csv_error_rate` PASS in 9.0s, 3 steps. At `--parallel 3`,
wall 51.6s vs 137.5s summed (2.7x) — but those numbers are polluted by retry
waits and should be re-measured. `log_cleanup_precision` passes on Linux,
confirming its Windows skip is purely `find.exe`.

### Findings that changed the code

1. **The account concurrency cap is low on the free tier and varies per
   account.** A fixed semaphore at 3 lost tasks to `ConcurrencyLimitError`
   because the retry budget (~30s) was shorter than a task's sandbox hold
   (43s). `--parallel N` is now a request: `AdaptiveLimiter` starts at N and
   lowers its ceiling when the API says the account is full, never below 1, and
   records where it settled. Retries are jittered — tasks are refused at the
   same instant and unjittered backoff makes them retry in lockstep.
2. **The ERROR/FAIL split earned its keep.** Those lost tasks were reported
   ERROR (harness/environment), never FAIL (a verdict on the agent), which is
   why the failure was diagnosable from one paste. Preserve this distinction.
3. **`sha256sum` output needs parsing, not `.split()[0]`.** coreutils escapes
   the whole line with a leading backslash when the path contains one. An
   unparseable digest now fails the check saying so, rather than comparing
   garbage and reporting a file as modified.

### Running on Windows

Works, with two caveats. The suite takes ~460s vs ~10s on Linux (process spawn
cost); WSL is much faster. `System32\find.exe` shadows GNU findutils, so
`log_cleanup_precision` **skips** with a message naming the tool —
`tests/capabilities.py` probes by behaviour, not version string. Skipping is
deliberate: a task failing for a missing tool looks exactly like a saboteur
being caught.

## Repo and process notes

- Work happens on `claude/solari-internship-app-x6y7q5`, merged to `main`.
- A PR opened from this fork defaults its base to **solari-sdk/solari-cookbook**.
  Always change the base repository dropdown, or merge locally. One stray
  upstream PR (#13) was opened and closed this way.
- Actions are disabled by default on forks and must be enabled.
  `workflow_dispatch` only appears once the workflow is on the default branch.
- `SOLARI_API_KEY` is a repository **secret**. Never a variable.

## Phase 3 result: snapshots did not pay off, and that is the finding

Measured on the free tier: booting `base` and running the preparation reached
ready in **3.4s**; forking the snapshot took **10.4s**. Snapshot restore is
~3x slower than a template boot here, so using it would have cost the 6-task
suite ~42s a run.

Do not frame snapshots as "this is why it's cheap enough to run on every PR" —
the measurement says otherwise, and the pitch has to follow the data. The
honest framing is that the harness *measures the tradeoff for your
environment*:

- snapshots are **opt-in** (`--snapshot`), not the default
- `--bench-snapshot N` reports boot / preparation / restore separately and
  prints USE SNAPSHOTS or SKIP SNAPSHOTS with the breakeven number
- snapshot restore has a fixed cost (~9.5s here); the preparation must exceed
  it to win. Measured by varying only `tasks/_environment.py`:

  | preparation | cost | verdict | effect on a 6-task run |
  | --- | --- | --- | --- |
  | pytest only | 1.7s | skip | +46.3s |
  | a few mid-size wheels | 7.7s | skip | +10.8s |
  | pandas, scipy, scikit-learn, matplotlib, polars, nltk | 15.5s | **use** | −28.3s |

  Template boot is a steady ~1.7s; snapshot restore a steady ~11-12.5s. The
  crossover is real and reproducible, and the shipped preparation sits below
  it on purpose — so the default (`--snapshot` off) is the right one.

- Trying to prepare with `torch` failed with `No space left on device`. The
  preparation raised, nothing was snapshotted, and the run stopped. That guard
  is load-bearing: a half-built environment would otherwise have been forked
  into every task in the run.

This is a better story than "snapshots are fast", and it is a real datapoint
about a workload Solari is not yet visible in.

## Verified live, second run (2026-09-01, 1-month promo tier)

The complete self-test, both directions, no errors:

- `agent-eval --parallel 4` -> **6/6 passed**, wall 69.7s vs 167.0s summed
- `agent-eval --agent sabotage --expect fail --parallel 4` -> **0/6 passed,
  6 failed, 0 errored**, exit 0
- The adaptive limiter asked for 4, settled at 2, and lost no tasks — the
  ConcurrencyLimitError failures from the first run are gone.
- Per-task latency (correct): csv 17.9s, json 19.8s, log_cleanup 40.8s,
  secret_leak 50.2s, stack_trace 18.8s, test_suite 19.5s.

## Next

Phase 4 — README leading with a caught regression, architecture diagram, real
latency and pass rates, the snapshot tradeoff as measured, and a "how to add a
task" section.
