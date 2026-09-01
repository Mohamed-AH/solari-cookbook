"""The tool surface an agent sees, defined once, provider-neutral.

Both reference agents — Claude and Gemini — build their provider-specific tool
definitions from this list and translate calls back through `to_action`. Keeping
it in one place is not just deduplication: it is the check that the Agent
interface is not secretly shaped around one vendor's SDK. If adding a provider
required changing the actions, the abstraction would be wrong.
"""

from __future__ import annotations

from typing import Any

from .agent import Action, finish

SYSTEM_PROMPT = """\
You are an autonomous software agent working inside a Linux sandbox.

Complete the task using the tools provided. You cannot see the machine except
through them: run commands to observe state, read files before changing them,
and check your work before finishing.

Rules:
- Use exactly one tool per turn.
- Follow the task's instructions precisely, including exact file paths and
  exact output formats. A correct value written to the wrong path is wrong.
- If the task tells you not to modify something, do not modify it.
- Call `finish` only once the task is genuinely complete.
"""

# name -> (description, JSON schema for the arguments)
TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "run_command",
        "description": (
            "Run a command in the sandbox and get back its exit code, stdout and "
            "stderr. The command is NOT shell-interpreted: pass the program in "
            "`cmd` and each argument separately in `args`. For shell syntax such "
            "as pipes or globs, use cmd='sh' with args=['-c', '<script>']."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "cmd": {"type": "string", "description": "Program to run, e.g. 'python3'."},
                "args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Arguments, one per element.",
                },
                "cwd": {"type": "string", "description": "Working directory (optional)."},
            },
            "required": ["cmd"],
        },
    },
    {
        "name": "write_file",
        "description": "Write text to a path, creating or overwriting it.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute path to write."},
                "content": {"type": "string", "description": "Full contents of the file."},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "read_file",
        "description": "Read a text file and get its contents back.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Absolute path to read."}},
            "required": ["path"],
        },
    },
    {
        "name": "list_dir",
        "description": "List the entries in a directory.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Absolute directory path."}},
            "required": ["path"],
        },
    },
    {
        "name": "finish",
        "description": "Declare the task complete. Call this only when it actually is.",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "What you did, in one sentence."}
            },
            "required": ["summary"],
        },
    },
]

TOOL_NAMES = tuple(spec["name"] for spec in TOOL_SPECS)


def to_action(name: str, payload: dict[str, Any]) -> Action:
    """Translate one tool call into a harness Action."""
    if name == "run_command":
        return Action(
            kind="run",
            cmd=str(payload.get("cmd", "")),
            args=tuple(str(a) for a in payload.get("args", []) or []),
            cwd=str(payload.get("cwd", "") or ""),
        )
    if name == "write_file":
        return Action(
            kind="write",
            path=str(payload.get("path", "")),
            content=str(payload.get("content", "")),
        )
    if name == "read_file":
        return Action(kind="read", path=str(payload.get("path", "")))
    if name == "list_dir":
        return Action(kind="list", path=str(payload.get("path", "")))
    if name == "finish":
        return finish(str(payload.get("summary", "")))
    raise ValueError(f"model called an unknown tool: {name!r}")


def anthropic_tools() -> list[dict[str, Any]]:
    """The same surface in Anthropic's shape."""
    return [
        {
            "name": spec["name"],
            "description": spec["description"],
            "input_schema": spec["parameters"],
        }
        for spec in TOOL_SPECS
    ]
