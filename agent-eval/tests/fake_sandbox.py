"""A scripted stand-in for a Solari sandbox, for tests that spend no credits.

Commands are answered from a table keyed by the exact argv the code under test
issued. An unexpected command raises rather than returning a default: a test
that silently accepted the wrong command would be testing nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass
class FakeResult:
    exitCode: int
    stdout: str = ""
    stderr: str = ""


class _FakeCommands:
    def __init__(self, sb: "FakeSandbox") -> None:
        self._sb = sb

    async def run(self, cmd: str, *, args: Iterable[str] | None = None, **_: Any) -> FakeResult:
        argv = (cmd, *(args or ()))
        self._sb.calls.append(argv)
        if argv in self._sb.responses:
            return self._sb.responses[argv]
        if cmd in self._sb.default_by_cmd:
            return self._sb.default_by_cmd[cmd]
        raise AssertionError(f"FakeSandbox got an unscripted command: {argv}")


class _FakeFiles:
    def __init__(self, sb: "FakeSandbox") -> None:
        self._sb = sb

    async def write(self, path: str, data: str, mode: int | None = None) -> None:
        self._sb.writes[path] = data


class FakeSandbox:
    sandboxId = "sbx_fake"

    def __init__(
        self,
        responses: dict[tuple[str, ...], FakeResult] | None = None,
        default_by_cmd: dict[str, FakeResult] | None = None,
    ) -> None:
        self.responses = responses or {}
        self.default_by_cmd = default_by_cmd or {}
        self.calls: list[tuple[str, ...]] = []
        self.writes: dict[str, str] = {}
        self.commands = _FakeCommands(self)
        self.files = _FakeFiles(self)
