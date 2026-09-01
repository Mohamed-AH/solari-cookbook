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
from .tool_surface import SYSTEM_PROMPT, anthropic_tools, to_action

MODEL = os.environ.get("AGENT_EVAL_MODEL", "claude-opus-5")
MAX_TOKENS = 16000

# Refusal fallbacks: on a policy decline the API re-runs the same request on a
# fallback model inside the same call, instead of the turn simply stopping.
FALLBACK_BETA = "server-side-fallback-2026-07-01"

TOOLS = anthropic_tools()


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
        return to_action(call.name, payload)
