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

import asyncio
import shutil
import subprocess
import tempfile
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
            # Off the event loop: a blocking subprocess here would serialise
            # every task and hide whether the runner overlaps them at all.
            proc = await asyncio.to_thread(
                subprocess.run,
                argv,
                cwd=str(workdir),
                capture_output=True,
                text=True,
                timeout=120,
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
        # Bytes, never text. `write_text` translates "\n" to os.linesep, which
        # on Windows silently turns every fixture into CRLF and changes its
        # sha256 — so a task's "this file is unmodified" check fails against a
        # file nothing modified. The real sandbox writes bytes over the wire
        # and does no such thing; the double must not either.
        target.write_bytes(data.encode("utf-8"))

    async def read_text(self, path: str) -> str:
        return self._sb.map_path(path).read_bytes().decode("utf-8")

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
        self.connected = False
        self.killed = False
        self._client: Any = None

    def map_path(self, path: str) -> Path:
        if path.startswith("/"):
            return self.root / path.lstrip("/")
        return self.default_cwd / path

    def map_arg(self, arg: str) -> str:
        """Rewrite absolute paths in argv; leave flags and plain words alone."""
        return str(self.map_path(arg)) if arg.startswith("/") else arg

    async def connect(self) -> None:
        self.connected = True

    async def kill(self) -> None:
        self.killed = True
        if self._client is not None:
            self._client.live -= 1
            self._client.killed += 1


class LocalSandboxClient:
    """Stands in for SandboxClient, handing out LocalSandboxes.

    Tracks how many sandboxes are alive at once, which is what makes it
    possible to assert that the parallel runner's semaphore actually bounds
    concurrency instead of merely being present in the source.
    """

    def __init__(self, *, boot_delay_s: float = 0.0, fail_creates_with: Any = None,
                 fail_times: int = 0) -> None:
        self.boot_delay_s = boot_delay_s
        self._fail_with = fail_creates_with
        self._fail_times = fail_times
        self.create_calls = 0
        self.live = 0
        self.peak_live = 0
        self.killed = 0
        self._roots: list[Path] = []

    async def create(self, **_: Any) -> LocalSandbox:
        self.create_calls += 1
        if self._fail_times > 0 and self._fail_with is not None:
            self._fail_times -= 1
            raise self._fail_with
        if self.boot_delay_s:
            await asyncio.sleep(self.boot_delay_s)
        root = Path(tempfile.mkdtemp(prefix="agent-eval-local-"))
        self._roots.append(root)
        sandbox = LocalSandbox(root)
        sandbox._client = self
        self.live += 1
        self.peak_live = max(self.peak_live, self.live)
        return sandbox

    async def aclose(self) -> None:
        for root in self._roots:
            shutil.rmtree(root, ignore_errors=True)

    async def __aenter__(self) -> "LocalSandboxClient":
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        await self.aclose()
