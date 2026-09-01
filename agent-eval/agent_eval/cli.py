"""Command line entry point.

    agent-eval --list
    agent-eval                              # run every task with the correct agent
    agent-eval --tasks csv_error_rate
    agent-eval --agent sabotage --expect fail      # the harness's own self-test
    agent-eval --agent claude                     # the reference LLM agent
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from .config import MissingApiKey, load_dotenv
from .report import render
from .runner import run_suite
from .sandbox import make_client
from .builtins import known_agents
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
        help="which agent to run (default: correct)",
    )
    parser.add_argument(
        "--expect",
        choices=("pass", "fail"),
        default="pass",
        help="expected outcome for every task; 'fail' is the harness self-test",
    )
    parser.add_argument(
        "--output",
        default="",
        help="write the JSON report to this path (default: none)",
    )
    parser.add_argument(
        "--tasks-dir",
        default="",
        help="directory to load tasks from (default: ./tasks, else the bundled tasks)",
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

    print(
        f"running {len(tasks)} task(s) from {tasks_dir} "
        f"with agent={args.agent}, expecting every task to {args.expect}"
    )

    async with make_client() as client:
        report = await run_suite(client, tasks, args.agent)

    print(render(report, verbose=args.verbose))

    if args.output:
        path = Path(args.output)
        report.write_json(path)
        print(f"report written to {path}")

    code = report.exit_code_for(args.expect)
    if code == 0 and args.expect == "fail":
        print("self-test OK: every task failed under the sabotage agent, as required.")
    return code


def main() -> int:
    args = build_parser().parse_args()
    load_dotenv(PROJECT_ROOT / ".env")
    try:
        return asyncio.run(_run(args))
    except (MissingApiKey, TaskLoadError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
