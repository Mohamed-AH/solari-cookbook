"""A local test double for a Solari sandbox.

Commands really execute, files really land on disk — inside a temp directory
that absolute sandbox paths are mapped into. That makes it possible to run a
whole task end to end (setup -> agent loop -> checks) with no credits and no
network, which is what lets CI run the suite on every PR.

This is a TEST DOUBLE, not a backend. It is deliberately not importable from
`agent_eval` and not reachable from the CLI. Results from here say the harness
logic is sound; they say nothing about whether an agent works in a real
sandbox, because the isolation, the image and the kernel are all different. The
product measures real machines — that is the entire point of it.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass
class LocalCommandResult:
    exitCode: int
    stdout: str
    stderr: str


@dataclass
class LocalEntry:
    name: str
    dir: bool
    size: int


class _LocalCommands:
    def __init__(self, sb: "LocalSandbox") -> None:
        self._sb = sb

    async def run(
        self,
        cmd: str,
        *,
        args: Iterable[str] | None = None,
        cwd: str | None = None,
        **_: Any,
    ) -> LocalCommandResult:
        argv = [cmd] + [self._sb.map_arg(a) for a in (args or [])]
        workdir = self._sb.map_path(cwd) if cwd else self._sb.default_cwd
        workdir.mkdir(parents=True, exist_ok=True)
        self._sb.calls.append(tuple([cmd, *(args or [])]))
        try:
            proc = subprocess.run(
                argv, cwd=str(workdir), capture_output=True, text=True, timeout=120
            )
        except FileNotFoundError:
            return LocalCommandResult(127, "", f"{cmd}: command not found")
        except subprocess.TimeoutExpired:
            return LocalCommandResult(124, "", f"{cmd}: timed out")
        return LocalCommandResult(proc.returncode, proc.stdout, proc.stderr)


class _LocalFiles:
    def __init__(self, sb: "LocalSandbox") -> None:
        self._sb = sb

    async def write(self, path: str, data: str, mode: int | None = None) -> None:
        target = self._sb.map_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(data, encoding="utf-8")

    async def read_text(self, path: str) -> str:
        return self._sb.map_path(path).read_text(encoding="utf-8")

    async def read(self, path: str) -> bytes:
        return self._sb.map_path(path).read_bytes()

    async def list(self, path: str) -> list[LocalEntry]:
        target = self._sb.map_path(path)
        return [
            LocalEntry(name=p.name, dir=p.is_dir(), size=p.stat().st_size if p.is_file() else 0)
            for p in sorted(target.iterdir())
        ]

    async def mkdir(self, path: str) -> None:
        self._sb.map_path(path).mkdir(parents=True, exist_ok=True)


class LocalSandbox:
    """Maps absolute sandbox paths into `root` and executes for real."""

    sandboxId = "local"

    def __init__(self, root: Path) -> None:
        self.root = root
        self.calls: list[tuple[str, ...]] = []
        self.commands = _LocalCommands(self)
        self.files = _LocalFiles(self)
        self.default_cwd = self.map_path("/workspace")
        self.default_cwd.mkdir(parents=True, exist_ok=True)

    def map_path(self, path: str) -> Path:
        if path.startswith("/"):
            return self.root / path.lstrip("/")
        return self.default_cwd / path

    def map_arg(self, arg: str) -> str:
        """Rewrite absolute paths in argv; leave flags and plain words alone."""
        return str(self.map_path(arg)) if arg.startswith("/") else arg
