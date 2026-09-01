"""Tests for the reference Gemini agent.

These do not call the API. They pin the parts that would otherwise only fail
with a key in hand: that the shared tool surface is accepted by the real SDK's
own types, that every tool the model can call maps to an executable action, and
that function calls and their responses stay paired across turns.
"""

from __future__ import annotations

import asyncio
import os
import sys
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from google.genai import types as genai_types

    HAS_SDK = True
except ImportError:  # pragma: no cover
    HAS_SDK = False

from agent_eval.agent import ActionResult, Observation  # noqa: E402
from agent_eval.tool_surface import TOOL_NAMES, TOOL_SPECS, to_action  # noqa: E402

if HAS_SDK:
    from agent_eval.gemini_agent import GeminiAgent, build_tools  # noqa: E402


@dataclass
class FakeCall:
    name: str
    args: dict = field(default_factory=dict)


@dataclass
class FakeCandidate:
    content: Any = None


@dataclass
class FakeResponse:
    function_calls: list = field(default_factory=list)
    candidates: list = field(default_factory=list)
    text: str = ""


class FakeModels:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self._responses = responses
        self.requests: list[dict] = []

    async def generate_content(self, **kwargs: Any) -> FakeResponse:
        # Snapshot the history: the agent keeps appending to the same list.
        self.requests.append({**kwargs, "contents": list(kwargs["contents"])})
        return self._responses.pop(0)


class FakeClient:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.aio = type("Aio", (), {"models": FakeModels(responses)})()


def make_agent(responses: list[FakeResponse]) -> "GeminiAgent":
    agent = GeminiAgent.__new__(GeminiAgent)  # skip __init__'s key lookup
    agent._types = genai_types
    agent._client = FakeClient(responses)
    agent._model = "gemini-flash-lite-latest"
    agent._history = []
    agent._pending_call_name = None
    agent._config = genai_types.GenerateContentConfig()
    return agent


@unittest.skipUnless(HAS_SDK, "google-genai not installed")
class TestToolSurfaceIsAcceptedBySdk(unittest.TestCase):
    def test_the_shared_schemas_build_real_function_declarations(self):
        """A schema the SDK rejects must fail here, not with a key in hand."""
        tools = build_tools(genai_types)
        self.assertEqual(len(tools), 1)
        declared = [d.name for d in tools[0].function_declarations]
        self.assertEqual(sorted(declared), sorted(TOOL_NAMES))

    def test_every_declaration_keeps_its_parameters(self):
        tools = build_tools(genai_types)
        by_name = {d.name: d for d in tools[0].function_declarations}
        run = by_name["run_command"]
        self.assertEqual(run.parameters.type, genai_types.Type.OBJECT)
        self.assertIn("cmd", run.parameters.properties)
        self.assertIn("cmd", run.parameters.required)
        self.assertTrue(run.description.strip())


class TestSharedSurface(unittest.TestCase):
    """The interface must not be shaped around one vendor."""

    def test_both_providers_expose_the_same_tools(self):
        from agent_eval.tool_surface import anthropic_tools  # noqa: PLC0415

        anthropic = sorted(t["name"] for t in anthropic_tools())
        self.assertEqual(anthropic, sorted(TOOL_NAMES))

    def test_every_tool_maps_to_an_executable_action(self):
        samples = {
            "run_command": {"cmd": "ls", "args": ["-la"]},
            "write_file": {"path": "/tmp/x", "content": "hi"},
            "read_file": {"path": "/tmp/x"},
            "list_dir": {"path": "/tmp"},
            "finish": {"summary": "done"},
        }
        self.assertEqual(sorted(samples), sorted(TOOL_NAMES))
        for name, payload in samples.items():
            with self.subTest(tool=name):
                self.assertIn(
                    to_action(name, payload).kind, ("run", "write", "read", "list", "finish")
                )

    def test_every_spec_declares_its_required_fields(self):
        for spec in TOOL_SPECS:
            with self.subTest(tool=spec["name"]):
                schema = spec["parameters"]
                for required in schema.get("required", []):
                    self.assertIn(required, schema["properties"])


@unittest.skipUnless(HAS_SDK, "google-genai not installed")
class TestConversation(unittest.TestCase):
    def _obs(self, step=0, last_action=None, last_result=None):
        return Observation(
            step=step,
            max_steps=5,
            prompt="Write 1 to /workspace/out.txt",
            workdir="/workspace",
            last_action=last_action,
            last_result=last_result,
        )

    def test_first_turn_sends_the_prompt_and_returns_the_call(self):
        agent = make_agent(
            [FakeResponse(function_calls=[FakeCall("read_file", {"path": "/workspace/in.txt"})])]
        )
        action = asyncio.run(agent.next_action(self._obs()))
        self.assertEqual(action.kind, "read")
        self.assertEqual(action.path, "/workspace/in.txt")
        sent = agent._client.aio.models.requests[0]["contents"]
        self.assertEqual(sent[0].role, "user")
        self.assertIn("Write 1 to /workspace/out.txt", sent[0].parts[0].text)

    def test_the_result_comes_back_as_a_function_response_with_the_same_name(self):
        agent = make_agent(
            [
                FakeResponse(function_calls=[FakeCall("read_file", {"path": "/workspace/in.txt"})]),
                FakeResponse(function_calls=[FakeCall("finish", {"summary": "done"})]),
            ]
        )
        first = asyncio.run(agent.next_action(self._obs()))
        result = ActionResult(ok=True, stdout="file contents", note="read")
        asyncio.run(agent.next_action(self._obs(1, last_action=first, last_result=result)))

        contents = agent._client.aio.models.requests[1]["contents"]
        response_part = contents[-1].parts[0].function_response
        self.assertEqual(response_part.name, "read_file")
        self.assertTrue(response_part.response["ok"])
        self.assertIn("file contents", response_part.response["output"])

    def test_a_failed_action_is_reported_as_not_ok(self):
        agent = make_agent(
            [
                FakeResponse(function_calls=[FakeCall("run_command", {"cmd": "false"})]),
                FakeResponse(function_calls=[FakeCall("finish", {"summary": "done"})]),
            ]
        )
        first = asyncio.run(agent.next_action(self._obs()))
        failed = ActionResult(ok=False, exit_code=1, stderr="boom")
        asyncio.run(agent.next_action(self._obs(1, last_action=first, last_result=failed)))
        part = agent._client.aio.models.requests[1]["contents"][-1].parts[0].function_response
        self.assertFalse(part.response["ok"])

    def test_no_function_call_ends_the_run(self):
        agent = make_agent([FakeResponse(function_calls=[], text="I think I am done.")])
        action = asyncio.run(agent.next_action(self._obs()))
        self.assertEqual(action.kind, "finish")
        self.assertIn("done", action.summary)

    def test_only_the_first_call_is_taken(self):
        """One action per step is the harness's contract."""
        agent = make_agent(
            [
                FakeResponse(
                    function_calls=[
                        FakeCall("read_file", {"path": "/a"}),
                        FakeCall("read_file", {"path": "/b"}),
                    ]
                )
            ]
        )
        action = asyncio.run(agent.next_action(self._obs()))
        self.assertEqual(action.path, "/a")
        self.assertEqual(agent._pending_call_name, "read_file")


class TestMissingKey(unittest.TestCase):
    def test_a_missing_key_says_which_variables_it_looked_at(self):
        from agent_eval.gemini_agent import API_KEY_ENVS, MissingApiKey, _api_key  # noqa: PLC0415

        saved = {name: os.environ.pop(name, None) for name in API_KEY_ENVS}
        try:
            with self.assertRaises(MissingApiKey) as ctx:
                _api_key()
            for name in API_KEY_ENVS:
                self.assertIn(name, str(ctx.exception))
        finally:
            for name, value in saved.items():
                if value is not None:
                    os.environ[name] = value


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(HAS_SDK, "google-genai not installed")
class TestRateLimitHandling(unittest.TestCase):
    """Regression: a 429 was scored as a failing agent.

    In the first live run the free tier's 15-requests-per-minute limit hit on
    the agent's first call. Two tasks were reported FAIL with one step each —
    a verdict on an agent that had not yet done anything.
    """

    def test_the_servers_own_retry_delay_is_parsed(self):
        from agent_eval.gemini_agent import _retry_after_s  # noqa: PLC0415

        real = (
            "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded "
            "your current quota. Please retry in 12.486266166s.', 'details': "
            "[{'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '58s'}]}}"
        )
        self.assertEqual(_retry_after_s(real), 58.0)

    def test_a_message_without_a_delay_falls_back(self):
        from agent_eval.gemini_agent import _retry_after_s  # noqa: PLC0415

        self.assertIsNone(_retry_after_s(Exception("500 internal error")))

    def test_a_persistent_rate_limit_raises_agent_unavailable(self):
        from google.genai import errors  # noqa: PLC0415

        from agent_eval import gemini_agent as mod  # noqa: PLC0415
        from agent_eval.agent import AgentUnavailable  # noqa: PLC0415

        class AlwaysLimited:
            requests: list = []

            async def generate_content(self, **kwargs):
                raise errors.ClientError(429, {"error": {"message": "quota"}})

        agent = make_agent([])
        agent._client.aio.models = AlwaysLimited()
        saved = mod.RETRY_BACKOFF_S, mod.RETRY_MAX_WAIT_S
        mod.RETRY_BACKOFF_S, mod.RETRY_MAX_WAIT_S = 0.001, 0.001
        try:
            with self.assertRaises(AgentUnavailable):
                asyncio.run(agent.next_action(self._obs()))
        finally:
            mod.RETRY_BACKOFF_S, mod.RETRY_MAX_WAIT_S = saved

    def test_it_recovers_when_the_limit_clears(self):
        from google.genai import errors  # noqa: PLC0415

        from agent_eval import gemini_agent as mod  # noqa: PLC0415

        class LimitedOnce:
            def __init__(self):
                self.calls = 0

            async def generate_content(self, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    raise errors.ClientError(429, {"error": {"message": "quota"}})
                return FakeResponse(function_calls=[FakeCall("finish", {"summary": "ok"})])

        agent = make_agent([])
        agent._client.aio.models = LimitedOnce()
        saved = mod.RETRY_BACKOFF_S, mod.RETRY_MAX_WAIT_S
        mod.RETRY_BACKOFF_S, mod.RETRY_MAX_WAIT_S = 0.001, 0.001
        try:
            action = asyncio.run(agent.next_action(self._obs()))
        finally:
            mod.RETRY_BACKOFF_S, mod.RETRY_MAX_WAIT_S = saved
        self.assertEqual(action.kind, "finish")

    def test_a_non_transient_error_is_still_unavailable_not_a_verdict(self):
        from google.genai import errors  # noqa: PLC0415

        from agent_eval.agent import AgentUnavailable  # noqa: PLC0415

        class BadKey:
            async def generate_content(self, **kwargs):
                raise errors.ClientError(403, {"error": {"message": "bad key"}})

        agent = make_agent([])
        agent._client.aio.models = BadKey()
        with self.assertRaises(AgentUnavailable):
            asyncio.run(agent.next_action(self._obs()))

    def _obs(self):
        return Observation(
            step=0, max_steps=5, prompt="do something", workdir="/workspace"
        )
