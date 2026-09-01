"""Command line entry point.

    agent-eval --list
    agent-eval                              # run every task with the correct agent
    agent-eval --tasks csv_error_rate
    agent-eval --agent sabotage --expect fail      # the harness's own self-test
    agent-eval --agent claude                     # the reference LLM agent
    agent-eval --agent my_pkg.agents:MyAgent      # your own agent
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from .config import MissingApiKey, load_dotenv
from .environment import benchmark_snapshot, ensure_prepared, load_environment
from .report import render, render_markdown
from .runner import run_suite
from .sandbox import make_client
from .builtins import AgentImportError, known_agents
from .task import TaskLoadError, discover

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BUNDLED_TASKS = PROJECT_ROOT / "tasks"


def default_tasks_dir() -> Path:
    """Prefer a `tasks/` directory in the current project, then the bundled one.

    The point of this harness is your tasks, not ours, so a `tasks/` beside the
    repo you run it from wins.
    """
    local = Path.cwd() / "tasks"
    return local if local.is_dir() else BUNDLED_TASKS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-eval",
        description="Execution-based regression testing for LLM agents.",
    )
    parser.add_argument(
        "--tasks",
        default="",
        help="comma-separated task ids to run (default: every task in tasks/)",
    )
    parser.add_argument(
        "--agent",
        default="correct",
        help=(
            "which agent to run: a name defined by the task, a built-in "
            "('claude'), or an import path to your own — "
            "'my_package.my_module:MyAgent' (default: correct)"
        ),
    )
    parser.add_argument(
        "--expect",
        choices=("pass", "fail"),
        default="pass",
        help="expected outcome for every task; 'fail' is the harness self-test",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        metavar="N",
        help=(
            "run up to N tasks at once, each in its own sandbox (default: 1). "
            "Bounded by a semaphore; raise it until you hit your account's "
            "concurrency cap"
        ),
    )
    parser.add_argument(
        "--output",
        default="",
        help="write the JSON report to this path (default: none)",
    )
    parser.add_argument(
        "--markdown",
        default="",
        help=(
            "write a markdown summary to this path — point it at "
            "$GITHUB_STEP_SUMMARY in CI"
        ),
    )
    parser.add_argument(
        "--tasks-dir",
        default="",
        help="directory to load tasks from (default: ./tasks, else the bundled tasks)",
    )
    parser.add_argument(
        "--no-snapshot",
        action="store_true",
        help=(
            "boot every task from the plain template instead of forking a "
            "prepared snapshot (slower; useful as a baseline)"
        ),
    )
    parser.add_argument(
        "--rebuild-snapshot",
        action="store_true",
        help="rebuild the prepared environment even if a matching snapshot exists",
    )
    parser.add_argument(
        "--bench-snapshot",
        type=int,
        default=0,
        metavar="N",
        help="measure prepared-vs-cold sandbox start over N samples, then exit",
    )
    parser.add_argument("--list", action="store_true", help="list tasks and exit")
    parser.add_argument("-v", "--verbose", action="store_true", help="show passing checks too")
    return parser


async def _run(args: argparse.Namespace) -> int:
    only = [t.strip() for t in args.tasks.split(",") if t.strip()]
    tasks_dir = Path(args.tasks_dir) if args.tasks_dir else default_tasks_dir()
    tasks = discover(tasks_dir, only=only or None)

    if args.list:
        for task in tasks:
            agents = ", ".join(known_agents(task))
            print(f"{task.id}\n    {task.summary}\n    agents: {agents}  max_steps: {task.max_steps}")
        return 0

    if args.parallel < 1:
        print("error: --parallel must be at least 1", file=sys.stderr)
        return 2

    concurrency = "one at a time" if args.parallel == 1 else f"{args.parallel} at a time"
    environment = load_environment(tasks_dir)

    async with make_client() as client:
        prepared = None
        if environment and not args.no_snapshot:
            what = environment.description or "the prepared environment"
            print(f"preparing the environment ({what})...")
            prepared = await ensure_prepared(
                client, environment, rebuild=args.rebuild_snapshot
            )
            if prepared.built:
                print(
                    f"built snapshot {prepared.name} in {prepared.build_s:.1f}s "
                    f"— later runs reuse it"
                )
            else:
                print(f"reusing snapshot {prepared.name} (nothing to build)")

        if args.bench_snapshot:
            if not environment:
                print(
                    f"error: no {tasks_dir / '_environment.py'} to benchmark",
                    file=sys.stderr,
                )
                return 2
            if prepared is None:
                prepared = await ensure_prepared(client, environment)
            bench = await benchmark_snapshot(
                client, environment, prepared, samples=args.bench_snapshot, tasks=len(tasks)
            )
            print(bench.render())
            if args.output:
                Path(args.output).write_text(
                    json.dumps(bench.to_dict(), indent=2) + "\n", encoding="utf-8"
                )
                print(f"benchmark written to {args.output}")
            return 0

        print(
            f"running {len(tasks)} task(s) from {tasks_dir} with agent={args.agent}, "
            f"{concurrency}, expecting every task to {args.expect}"
        )
        report = await run_suite(
            client, tasks, args.agent, parallel=args.parallel, prepared=prepared
        )

    print(render(report, verbose=args.verbose))

    if args.output:
        path = Path(args.output)
        report.write_json(path)
        print(f"report written to {path}")

    if args.markdown:
        path = Path(args.markdown)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Appended, not overwritten: $GITHUB_STEP_SUMMARY accumulates.
        with path.open("a", encoding="utf-8") as fh:
            fh.write(render_markdown(report) + "\n")
        print(f"markdown summary written to {path}")

    code = report.exit_code_for(args.expect)
    if code == 0 and args.expect == "fail":
        print("self-test OK: every task failed under the sabotage agent, as required.")
    return code


def main() -> int:
    args = build_parser().parse_args()
    load_dotenv(PROJECT_ROOT / ".env")
    try:
        return asyncio.run(_run(args))
    except (MissingApiKey, TaskLoadError, AgentImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
