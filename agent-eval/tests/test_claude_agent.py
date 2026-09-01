"""Tests for the reference Claude agent.

These do not call the API. They pin the parts that would otherwise only be
discovered at runtime with a key: that the tool schemas are well formed, that
every tool the model can call translates to an action the harness understands,
and that the conversation bookkeeping keeps tool_use and tool_result paired.
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
    import anthropic  # noqa: F401

    HAS_SDK = True
except ImportError:  # pragma: no cover
    HAS_SDK = False

from agent_eval.agent import ActionResult, Observation  # noqa: E402

if HAS_SDK:
    os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test-not-a-real-key")
    from agent_eval.claude_agent import TOOLS, ClaudeAgent, _to_action  # noqa: E402


@dataclass
class FakeBlock:
    type: str
    id: str = ""
    name: str = ""
    input: dict = field(default_factory=dict)
    text: str = ""


@dataclass
class FakeResponse:
    content: list
    stop_reason: str = "tool_use"
    stop_details: Any = None


class FakeMessages:
    """Stands in for client.beta.messages, recording what it was sent."""

    def __init__(self, responses: list[FakeResponse]) -> None:
        self._responses = responses
        self.requests: list[dict] = []

    async def create(self, **kwargs: Any) -> FakeResponse:
        # Snapshot the message list: the agent keeps mutating the same object
        # after the call, and a recorded request must show what was sent.
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        return self._responses.pop(0)


class FakeClient:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.beta = type("Beta", (), {"messages": FakeMessages(responses)})()


def make_agent(responses: list[FakeResponse]) -> "ClaudeAgent":
    agent = ClaudeAgent.__new__(ClaudeAgent)  # skip __init__'s real client
    import anthropic as _anthropic  # noqa: PLC0415

    agent._anthropic = _anthropic
    agent._client = FakeClient(responses)
    agent._model = "claude-opus-5"
    agent._use_fallbacks = False
    agent._messages = []
    agent._pending_tool_use_id = None
    return agent


@unittest.skipUnless(HAS_SDK, "anthropic SDK not installed")
class TestToolSchemas(unittest.TestCase):
    def test_every_tool_is_well_formed(self):
        for tool in TOOLS:
            with self.subTest(tool=tool["name"]):
                self.assertTrue(tool["description"].strip(), "tool has no description")
                schema = tool["input_schema"]
                self.assertEqual(schema["type"], "object")
                for required in schema.get("required", []):
                    self.assertIn(
                        required,
                        schema["properties"],
                        f"{required!r} is required but not declared in properties",
                    )

    def test_every_tool_maps_to_an_action(self):
        """A tool the model can call but the harness cannot execute is a trap."""
        samples = {
            "run_command": {"cmd": "ls", "args": ["-la"]},
            "write_file": {"path": "/tmp/x", "content": "hi"},
            "read_file": {"path": "/tmp/x"},
            "list_dir": {"path": "/tmp"},
            "finish": {"summary": "done"},
        }
        self.assertEqual(sorted(samples), sorted(t["name"] for t in TOOLS))
        for name, payload in samples.items():
            with self.subTest(tool=name):
                action = _to_action(name, payload)
                self.assertIn(action.kind, ("run", "write", "read", "list", "finish"))

    def test_run_command_translation_keeps_argv_separate(self):
        action = _to_action("run_command", {"cmd": "python3", "args": ["-c", "print(1)"]})
        self.assertEqual(action.cmd, "python3")
        self.assertEqual(action.args, ("-c", "print(1)"))

    def test_unknown_tool_raises(self):
        with self.assertRaises(ValueError):
            _to_action("rm_minus_rf", {})


@unittest.skipUnless(HAS_SDK, "anthropic SDK not installed")
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

    def test_first_turn_sends_the_prompt_and_returns_the_tool_call(self):
        agent = make_agent(
            [FakeResponse([FakeBlock(type="tool_use", id="tu_1", name="read_file",
                                     input={"path": "/workspace/in.txt"})])]
        )
        action = asyncio.run(agent.next_action(self._obs()))
        self.assertEqual(action.kind, "read")
        sent = agent._client.beta.messages.requests[0]
        self.assertIn("Write 1 to /workspace/out.txt", sent["messages"][0]["content"])
        self.assertEqual(sent["model"], "claude-opus-5")
        self.assertTrue(sent["tool_choice"]["disable_parallel_tool_use"])

    def test_second_turn_returns_the_result_against_the_right_tool_use_id(self):
        agent = make_agent(
            [
                FakeResponse([FakeBlock(type="tool_use", id="tu_1", name="read_file",
                                        input={"path": "/workspace/in.txt"})]),
                FakeResponse([FakeBlock(type="tool_use", id="tu_2", name="finish",
                                        input={"summary": "done"})]),
            ]
        )
        first = asyncio.run(agent.next_action(self._obs()))
        result = ActionResult(ok=True, stdout="file contents", note="read")
        asyncio.run(agent.next_action(self._obs(step=1, last_action=first, last_result=result)))

        second_request = agent._client.beta.messages.requests[1]
        tool_result = second_request["messages"][-1]["content"][0]
        self.assertEqual(tool_result["type"], "tool_result")
        self.assertEqual(tool_result["tool_use_id"], "tu_1")
        self.assertFalse(tool_result["is_error"])
        self.assertIn("file contents", tool_result["content"])

    def test_failed_action_is_flagged_as_an_error_result(self):
        agent = make_agent(
            [
                FakeResponse([FakeBlock(type="tool_use", id="tu_1", name="run_command",
                                        input={"cmd": "false"})]),
                FakeResponse([FakeBlock(type="tool_use", id="tu_2", name="finish",
                                        input={"summary": "done"})]),
            ]
        )
        first = asyncio.run(agent.next_action(self._obs()))
        failed = ActionResult(ok=False, exit_code=1, stderr="boom")
        asyncio.run(agent.next_action(self._obs(step=1, last_action=first, last_result=failed)))
        tool_result = agent._client.beta.messages.requests[1]["messages"][-1]["content"][0]
        self.assertTrue(tool_result["is_error"])

    def test_no_tool_call_ends_the_run(self):
        agent = make_agent([FakeResponse([FakeBlock(type="text", text="I think I am done.")],
                                         stop_reason="end_turn")])
        action = asyncio.run(agent.next_action(self._obs()))
        self.assertEqual(action.kind, "finish")
        self.assertIn("done", action.summary)

    def test_refusal_ends_the_run_without_reading_content(self):
        agent = make_agent([FakeResponse([], stop_reason="refusal", stop_details="cyber")])
        action = asyncio.run(agent.next_action(self._obs()))
        self.assertEqual(action.kind, "finish")
        self.assertIn("declined", action.summary)


if __name__ == "__main__":
    unittest.main()
