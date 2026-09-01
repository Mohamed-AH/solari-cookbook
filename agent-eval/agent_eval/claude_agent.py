"""A reference agent backed by Claude, to demonstrate the Agent interface.

This is the adapter you would replace with your own agent. It is deliberately
thin: `next_action` is one model call that returns one tool call, which the
harness then executes and reports back. The harness keeps the loop and the
trajectory; this class keeps only the conversation.

Parallel tool use is disabled on purpose. The harness scores one action per
step, and returning several tool calls at once would either desynchronise the
trajectory or require batching results in a way that trains the model out of
parallel calls anyway.

Optional dependency:

    pip install 'agent-eval[claude]'
    export ANTHROPIC_API_KEY=sk-ant-...
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from .agent import Action, Observation, finish

MODEL = os.environ.get("AGENT_EVAL_MODEL", "claude-opus-5")
MAX_TOKENS = 16000

# Refusal fallbacks: on a policy decline the API re-runs the same request on a
# fallback model inside the same call, instead of the turn simply stopping.
FALLBACK_BETA = "server-side-fallback-2026-07-01"

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

TOOLS: list[dict[str, Any]] = [
    {
        "name": "run_command",
        "description": (
            "Run a command in the sandbox and get back its exit code, stdout and "
            "stderr. The command is NOT shell-interpreted: pass the program in "
            "`cmd` and each argument separately in `args`. For shell syntax such "
            "as pipes or globs, use cmd='sh' with args=['-c', '<script>']."
        ),
        "input_schema": {
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
        "input_schema": {
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
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Absolute path to read."}},
            "required": ["path"],
        },
    },
    {
        "name": "list_dir",
        "description": "List the entries in a directory.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Absolute directory path."}},
            "required": ["path"],
        },
    },
    {
        "name": "finish",
        "description": "Declare the task complete. Call this only when it actually is.",
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "What you did, in one sentence."}
            },
            "required": ["summary"],
        },
    },
]


class MissingDependency(RuntimeError):
    """Raised when the optional Claude extra is not installed."""


def _load_sdk() -> Any:
    try:
        import anthropic  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise MissingDependency(
            "the claude agent needs the anthropic SDK:\n"
            "  pip install 'agent-eval[claude]'\n"
            "  export ANTHROPIC_API_KEY=sk-ant-..."
        ) from exc
    return anthropic


def _to_action(name: str, payload: dict[str, Any]) -> Action:
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
            kind="write", path=str(payload.get("path", "")), content=str(payload.get("content", ""))
        )
    if name == "read_file":
        return Action(kind="read", path=str(payload.get("path", "")))
    if name == "list_dir":
        return Action(kind="list", path=str(payload.get("path", "")))
    if name == "finish":
        return finish(str(payload.get("summary", "")))
    raise ValueError(f"model called an unknown tool: {name!r}")


class ClaudeAgent:
    """Chooses the next action by asking Claude, one tool call at a time."""

    name = "claude"

    def __init__(self, model: str = MODEL, *, use_fallbacks: bool = True) -> None:
        anthropic = _load_sdk()
        self._anthropic = anthropic
        self._client = anthropic.AsyncAnthropic()
        self._model = model
        self._use_fallbacks = use_fallbacks
        self._messages: list[dict[str, Any]] = []
        self._pending_tool_use_id: str | None = None

    # -- conversation bookkeeping ----------------------------------------

    def _record_observation(self, obs: Observation) -> None:
        """Feed the harness's observation back as the model's next input."""
        text = obs.render()
        if self._pending_tool_use_id is None:
            self._messages.append({"role": "user", "content": text})
            return
        # The result of the tool call the harness just executed for us.
        self._messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": self._pending_tool_use_id,
                        "content": text,
                        "is_error": bool(obs.last_result and not obs.last_result.ok),
                    }
                ],
            }
        )
        self._pending_tool_use_id = None

    async def _create(self, **extra: Any) -> Any:
        return await self._client.beta.messages.create(
            model=self._model,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            thinking={"type": "adaptive"},
            tools=TOOLS,
            # One action per step is the harness's contract.
            tool_choice={"type": "auto", "disable_parallel_tool_use": True},
            messages=self._messages,
            **extra,
        )

    async def _call_model(self) -> Any:
        anthropic = self._anthropic
        try:
            if self._use_fallbacks:
                try:
                    return await self._create(betas=[FALLBACK_BETA], fallbacks="default")
                except (anthropic.BadRequestError, TypeError) as exc:
                    # The SDK or the account does not know this beta. Say so
                    # loudly, once, and continue without it rather than
                    # failing every task for a resilience feature.
                    print(
                        f"!! refusal fallbacks unavailable ({type(exc).__name__}: {exc}); "
                        f"continuing without them",
                        file=sys.stderr,
                    )
                    self._use_fallbacks = False
            return await self._create()
        except anthropic.NotFoundError as exc:
            raise RuntimeError(f"model {self._model!r} not found: {exc}") from exc
        except anthropic.RateLimitError as exc:
            raise RuntimeError(f"rate limited after the SDK's own retries: {exc}") from exc
        except anthropic.APIStatusError as exc:
            raise RuntimeError(f"API error {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise RuntimeError(f"could not reach the API: {exc}") from exc

    # -- the Agent interface ---------------------------------------------

    async def next_action(self, obs: Observation) -> Action:
        self._record_observation(obs)
        response = await self._call_model()

        # A policy decline stops the turn; treat it as the end of the run
        # rather than reading content that is not there.
        if getattr(response, "stop_reason", None) == "refusal":
            detail = getattr(response, "stop_details", None)
            return finish(f"model declined to continue: {detail}")

        self._messages.append({"role": "assistant", "content": response.content})

        tool_calls = [b for b in response.content if getattr(b, "type", None) == "tool_use"]
        if not tool_calls:
            said = " ".join(
                b.text for b in response.content if getattr(b, "type", None) == "text"
            ).strip()
            return finish(said[:200] or "model returned no tool call")

        call = tool_calls[0]
        self._pending_tool_use_id = call.id
        # Tool inputs are parsed JSON objects from the SDK; never string-match
        # the serialized form.
        payload = call.input if isinstance(call.input, dict) else json.loads(call.input)
        return _to_action(call.name, payload)
