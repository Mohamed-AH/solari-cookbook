"""The assertion layer.

Every assertion is deterministic and is decided by a real command executed in
the sandbox: an integer `exitCode`, or the exact `stdout` bytes that command
produced. There is no LLM judge, and no assertion is ever decided by
string-matching a stringified result object.

Each check records what it actually observed — the command, its exit code, its
stdout and stderr — so a failure in the report is debuggable without re-running
anything.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import Any, Sequence

# Tools every check in this module depends on. The runner verifies these exist
# before a task is scored, so a missing binary fails loudly as an environment
# error instead of quietly failing a check the agent actually passed.
REQUIRED_TOOLS = ("test", "cat", "sha256sum", "python3")

_VERIFIER_DIR = "/opt/agent_eval"
_VERIFIER_PATH = _VERIFIER_DIR + "/verify_number.py"

# Written into the sandbox by `Checker.number_in_file_equals`. Everything it
# needs arrives as argv, so nothing is ever interpolated into this source.
_NUMBER_VERIFIER = '''"""Compare a number stored in a file against an expected value."""
import sys

path, expected, places = sys.argv[1], sys.argv[2], int(sys.argv[3])

try:
    raw = open(path, encoding="utf-8").read().strip()
except OSError as exc:
    print("cannot read %s: %s" % (path, exc))
    sys.exit(2)

try:
    actual = round(float(raw), places)
except ValueError:
    print("not a number: %r" % (raw,))
    sys.exit(3)

want = round(float(expected), places)
print("expected=%s actual=%s raw=%r" % (want, actual, raw))
sys.exit(0 if actual == want else 1)
'''


@dataclass
class CheckResult:
    """One assertion, and the machine state that decided it."""

    name: str
    passed: bool
    detail: str = ""
    command: str = ""
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "detail": self.detail,
            "command": self.command,
            "exit_code": self.exit_code,
            "stdout": _truncate(self.stdout),
            "stderr": _truncate(self.stderr),
        }


def _truncate(text: str, limit: int = 2000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated, {len(text)} chars total]"


class Checker:
    """Runs assertions against a live sandbox and accumulates their results.

    A task's check function receives one of these and calls its methods. The
    task passes only if every recorded check passed.
    """

    def __init__(self, sandbox: Any) -> None:
        self._sb = sandbox
        self.results: list[CheckResult] = []

    @property
    def sandbox(self) -> Any:
        """The live sandbox, for checks that install a verifier script first."""
        return self._sb

    # -- plumbing ---------------------------------------------------------

    async def _exec(self, cmd: str, args: Sequence[str] = ()) -> Any:
        """Run a command in the sandbox. `cmd` is not shell-interpreted."""
        return await self._sb.commands.run(cmd, args=list(args))

    def _record(self, result: CheckResult) -> CheckResult:
        self.results.append(result)
        return result

    @staticmethod
    def _render(cmd: str, args: Sequence[str]) -> str:
        return " ".join(shlex.quote(part) for part in (cmd, *args))

    # -- assertions -------------------------------------------------------

    async def command_succeeds(
        self, cmd: str, args: Sequence[str] = (), *, name: str | None = None
    ) -> CheckResult:
        """Assert a command exits 0."""
        rendered = self._render(cmd, args)
        res = await self._exec(cmd, args)
        return self._record(
            CheckResult(
                name=name or f"`{rendered}` exits 0",
                passed=res.exitCode == 0,
                detail=f"exit code {res.exitCode}, wanted 0",
                command=rendered,
                exit_code=res.exitCode,
                stdout=res.stdout,
                stderr=res.stderr,
            )
        )

    async def command_fails(
        self, cmd: str, args: Sequence[str] = (), *, name: str | None = None
    ) -> CheckResult:
        """Assert a command exits non-zero.

        Used to prove a task's failing state really is failing before the agent
        touches it — a check that cannot fail is worthless.
        """
        rendered = self._render(cmd, args)
        res = await self._exec(cmd, args)
        return self._record(
            CheckResult(
                name=name or f"`{rendered}` exits non-zero",
                passed=res.exitCode != 0,
                detail=f"exit code {res.exitCode}, wanted non-zero",
                command=rendered,
                exit_code=res.exitCode,
                stdout=res.stdout,
                stderr=res.stderr,
            )
        )

    async def file_exists(self, path: str, *, name: str | None = None) -> CheckResult:
        """Assert a regular file exists at exactly this path."""
        res = await self._exec("test", ["-f", path])
        return self._record(
            CheckResult(
                name=name or f"file exists: {path}",
                passed=res.exitCode == 0,
                detail="present" if res.exitCode == 0 else "missing (or not a regular file)",
                command=self._render("test", ["-f", path]),
                exit_code=res.exitCode,
                stdout=res.stdout,
                stderr=res.stderr,
            )
        )

    async def file_absent(self, path: str, *, name: str | None = None) -> CheckResult:
        """Assert nothing exists at this path."""
        res = await self._exec("test", ["-e", path])
        return self._record(
            CheckResult(
                name=name or f"nothing at: {path}",
                passed=res.exitCode != 0,
                detail="absent" if res.exitCode != 0 else "unexpectedly present",
                command=self._render("test", ["-e", path]),
                exit_code=res.exitCode,
                stdout=res.stdout,
                stderr=res.stderr,
            )
        )

    async def file_equals(
        self, path: str, expected: str, *, name: str | None = None
    ) -> CheckResult:
        """Assert a file's contents equal `expected`, ignoring trailing whitespace."""
        res = await self._exec("cat", [path])
        actual = res.stdout.rstrip()
        want = expected.rstrip()
        ok = res.exitCode == 0 and actual == want
        if res.exitCode != 0:
            detail = f"could not read file (exit {res.exitCode})"
        elif ok:
            detail = "contents match"
        else:
            detail = f"wanted {want!r}, got {actual!r}"
        return self._record(
            CheckResult(
                name=name or f"contents of {path}",
                passed=ok,
                detail=detail,
                command=self._render("cat", [path]),
                exit_code=res.exitCode,
                stdout=res.stdout,
                stderr=res.stderr,
            )
        )

    async def file_sha256(
        self, path: str, expected_hex: str, *, name: str | None = None
    ) -> CheckResult:
        """Assert a file is byte-identical to a known hash.

        This is how "did the agent tamper with something it was told not to
        touch" is decided: an input fixture, or a test file it must leave alone.
        """
        res = await self._exec("sha256sum", [path])
        parts = res.stdout.split()
        actual = parts[0] if parts else ""
        ok = res.exitCode == 0 and actual == expected_hex
        if res.exitCode != 0:
            detail = f"could not hash file (exit {res.exitCode})"
        elif ok:
            detail = "unmodified"
        else:
            detail = f"modified — wanted sha256 {expected_hex}, got {actual or '<none>'}"
        return self._record(
            CheckResult(
                name=name or f"{path} is unmodified",
                passed=ok,
                detail=detail,
                command=self._render("sha256sum", [path]),
                exit_code=res.exitCode,
                stdout=res.stdout,
                stderr=res.stderr,
            )
        )

    async def stdout_equals(
        self,
        cmd: str,
        args: Sequence[str],
        expected: str,
        *,
        name: str | None = None,
    ) -> CheckResult:
        """Assert a command exits 0 and its stdout equals `expected`."""
        rendered = self._render(cmd, args)
        res = await self._exec(cmd, args)
        actual = res.stdout.rstrip()
        want = expected.rstrip()
        ok = res.exitCode == 0 and actual == want
        return self._record(
            CheckResult(
                name=name or f"stdout of `{rendered}`",
                passed=ok,
                detail=(
                    "matches" if ok else f"exit {res.exitCode}; wanted {want!r}, got {actual!r}"
                ),
                command=rendered,
                exit_code=res.exitCode,
                stdout=res.stdout,
                stderr=res.stderr,
            )
        )

    async def stdout_contains(
        self,
        cmd: str,
        args: Sequence[str],
        needle: str,
        *,
        name: str | None = None,
    ) -> CheckResult:
        """Assert a command exits 0 and its stdout contains `needle`."""
        rendered = self._render(cmd, args)
        res = await self._exec(cmd, args)
        ok = res.exitCode == 0 and needle in res.stdout
        return self._record(
            CheckResult(
                name=name or f"stdout of `{rendered}` contains {needle!r}",
                passed=ok,
                detail="found" if ok else f"exit {res.exitCode}; {needle!r} not in stdout",
                command=rendered,
                exit_code=res.exitCode,
                stdout=res.stdout,
                stderr=res.stderr,
            )
        )

    async def number_in_file_equals(
        self,
        path: str,
        expected: float,
        *,
        places: int = 4,
        name: str | None = None,
    ) -> CheckResult:
        """Assert the number written in `path` equals `expected` to `places` dp.

        Decided by a verifier script's exit code, not by string comparison:
        0 equal, 1 different, 2 unreadable, 3 not a number. The script is
        written with `files.write` and takes its arguments as argv, so no
        value is ever interpolated into source text.
        """
        # Written fresh on every call, so a file the agent left at this path
        # is overwritten rather than trusted.
        await self._exec("mkdir", ["-p", _VERIFIER_DIR])
        await self._sb.files.write(_VERIFIER_PATH, _NUMBER_VERIFIER)
        args = [_VERIFIER_PATH, path, repr(float(expected)), str(places)]
        rendered = self._render("python3", args)
        res = await self._exec("python3", args)
        reason = {
            0: "matches",
            1: "wrong value",
            2: "file unreadable",
            3: "file does not contain a number",
        }.get(res.exitCode, f"verifier exited {res.exitCode}")
        return self._record(
            CheckResult(
                name=name or f"{path} == {expected} (to {places} dp)",
                passed=res.exitCode == 0,
                detail=f"{reason}: {res.stdout.strip() or res.stderr.strip()}",
                command=rendered,
                exit_code=res.exitCode,
                stdout=res.stdout,
                stderr=res.stderr,
            )
        )

    # -- verdict ----------------------------------------------------------

    @property
    def passed(self) -> bool:
        """True only if at least one check ran and every one of them passed.

        Zero checks is a failure, not a pass: a task whose check function did
        nothing must never be reported as green.
        """
        return bool(self.results) and all(r.passed for r in self.results)


async def preflight(sandbox: Any, tools: Sequence[str] = REQUIRED_TOOLS) -> list[str]:
    """Return the names of assertion tools missing from the sandbox image."""
    missing = []
    for tool in tools:
        res = await sandbox.commands.run("which", args=[tool])
        if res.exitCode != 0:
            missing.append(tool)
    return missing
