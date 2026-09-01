"""A reference agent backed by Gemini, alongside the Claude one.

Two providers on one Agent interface is the point of having this file. The
harness's actions, trajectory and scoring are unchanged between them; only the
adapter differs. If adding a second provider had required changing the action
set, the abstraction would have been wrong.

Gemini's free tier also makes it the cheapest way to run the suite against a
real model rather than scripted agents.

Optional dependency:

    pip install 'agent-eval[gemini]'
    export GEMINI_API_KEY=...        # or GOOGLE_API_KEY

Model defaults to `gemini-flash-lite-latest`; override with
AGENT_EVAL_GEMINI_MODEL. A stronger model will do better on the
multi-observation tasks — and the harness will tell you by how much, which is
rather the point.
"""

from __future__ import annotations

import asyncio
import os
import random
import re
import sys
from typing import Any

from .agent import Action, AgentUnavailable, Observation, finish
from .tool_surface import SYSTEM_PROMPT, TOOL_SPECS, to_action

MODEL = os.environ.get("AGENT_EVAL_GEMINI_MODEL", "gemini-flash-lite-latest")
API_KEY_ENVS = ("GEMINI_API_KEY", "GOOGLE_API_KEY")

# The free tier allows 15 requests per minute per model. A six-task suite makes
# roughly five calls per task, so a sequential run walks straight into it. The
# API says how long to wait; honour that rather than guessing.
RETRY_ATTEMPTS = 5
RETRY_BACKOFF_S = 5.0
RETRY_MAX_WAIT_S = 75.0

_RETRY_DELAY_PATTERNS = (
    re.compile(r"retryDelay['\"]?\s*:\s*['\"]?(\d+(?:\.\d+)?)s"),
    re.compile(r"retry in (\d+(?:\.\d+)?)s"),
)


def _retry_after_s(exc: BaseException) -> float | None:
    """Pull the server's own retry delay out of the error, if it gave one."""
    text = str(exc)
    for pattern in _RETRY_DELAY_PATTERNS:
        match = pattern.search(text)
        if match:
            return float(match.group(1))
    return None


class MissingDependency(RuntimeError):
    """Raised when the optional Gemini extra is not installed."""


class MissingApiKey(RuntimeError):
    """Raised when no Gemini API key is available."""


def _load_sdk() -> Any:
    try:
        from google import genai  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise MissingDependency(
            "the gemini agent needs the google-genai SDK:\n"
            "  pip install 'agent-eval[gemini]'\n"
            "  export GEMINI_API_KEY=..."
        ) from exc
    return genai


def _api_key() -> str:
    for name in API_KEY_ENVS:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    raise MissingApiKey(
        f"none of {', '.join(API_KEY_ENVS)} is set.\n"
        f"  export GEMINI_API_KEY=...   # https://aistudio.google.com/apikey"
    )


def build_tools(types_module: Any) -> list[Any]:
    """The shared tool surface, in Gemini's shape.

    The schemas are the same dicts the Claude adapter uses; the SDK coerces
    them into its own Schema type.
    """
    declarations = [
        types_module.FunctionDeclaration(
            name=spec["name"],
            description=spec["description"],
            parameters=spec["parameters"],
        )
        for spec in TOOL_SPECS
    ]
    return [types_module.Tool(function_declarations=declarations)]


class GeminiAgent:
    """Chooses the next action by asking Gemini, one function call at a time."""

    name = "gemini"

    def __init__(self, model: str = MODEL) -> None:
        genai = _load_sdk()
        from google.genai import types  # noqa: PLC0415

        self._types = types
        self._client = genai.Client(api_key=_api_key())
        self._model = model
        self._history: list[Any] = []
        self._pending_call_name: str | None = None
        self._config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=build_tools(types),
            # The harness executes the tools, not the SDK. Without this the
            # SDK would try to call them itself and the trajectory would stop
            # matching what actually ran.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

    # -- conversation bookkeeping ----------------------------------------

    def _record_observation(self, obs: Observation) -> None:
        types = self._types
        if self._pending_call_name is None:
            self._history.append(
                types.Content(role="user", parts=[types.Part.from_text(text=obs.render())])
            )
            return

        result = obs.last_result
        self._history.append(
            types.Content(
                role="user",
                parts=[
                    types.Part.from_function_response(
                        name=self._pending_call_name,
                        response={
                            "ok": bool(result.ok) if result else False,
                            "output": obs.render(),
                        },
                    )
                ],
            )
        )
        self._pending_call_name = None

    async def _call_model(self) -> Any:
        errors = self._sdk_errors()
        delay = RETRY_BACKOFF_S

        for attempt in range(1, RETRY_ATTEMPTS + 1):
            try:
                return await self._client.aio.models.generate_content(
                    model=self._model, contents=self._history, config=self._config
                )
            except errors as exc:  # noqa: PERF203 - retry loop, not a hot path
                status = getattr(exc, "code", None)
                transient = status == 429 or (isinstance(status, int) and status >= 500)
                if not transient:
                    raise AgentUnavailable(
                        f"Gemini API error {status}: {exc}"
                    ) from exc
                if attempt == RETRY_ATTEMPTS:
                    raise AgentUnavailable(
                        f"Gemini still rate limited after {RETRY_ATTEMPTS} attempts: {exc}"
                    ) from exc

                # The server usually names its own delay; jitter it so several
                # tasks refused at once do not retry in lockstep.
                wait = _retry_after_s(exc) or delay
                wait = min(wait, RETRY_MAX_WAIT_S) * (0.9 + 0.4 * random.random())
                print(
                    f"!! Gemini rate limited (attempt {attempt}/{RETRY_ATTEMPTS}); "
                    f"waiting {wait:.0f}s",
                    file=sys.stderr,
                )
                await asyncio.sleep(wait)
                delay = min(delay * 2, RETRY_MAX_WAIT_S)

        raise AssertionError("unreachable")  # pragma: no cover

    def _sdk_errors(self) -> tuple[type[BaseException], ...]:
        try:
            from google.genai import errors  # noqa: PLC0415

            return (errors.APIError,)
        except ImportError:  # pragma: no cover
            return ()

    # -- the Agent interface ---------------------------------------------

    async def next_action(self, obs: Observation) -> Action:
        self._record_observation(obs)
        response = await self._call_model()

        calls = response.function_calls or []
        if not calls:
            # No call means the model thinks it is done, or it was blocked.
            said = (getattr(response, "text", None) or "").strip()
            return finish(said[:200] or "model returned no function call")

        candidate = response.candidates[0] if response.candidates else None
        if candidate is not None and candidate.content is not None:
            self._history.append(candidate.content)

        # One action per step is the harness's contract; if the model asked for
        # several, the rest are dropped rather than silently reordered.
        call = calls[0]
        self._pending_call_name = call.name
        return to_action(call.name, dict(call.args or {}))
