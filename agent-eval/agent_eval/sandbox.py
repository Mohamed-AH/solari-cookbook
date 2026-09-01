"""Sandbox lifecycle.

One sandbox per task attempt, always destroyed. `kill()` is what ends a VM —
`close()` only drops the local control channel and leaves the machine running
until its idle timeout, which is how you quietly burn credits.

Nothing here swallows an exception. If teardown fails while the body also
failed, the body's error propagates and the teardown error is reported beside
it; a silent no-op that looks like it worked is the worst failure mode this
project has.
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from solari_sandbox import SandboxClient

from .config import BASE_URL, api_key

DEFAULT_TEMPLATE = "base"
DEFAULT_TIMEOUT_MS = 5 * 60_000


def make_client() -> SandboxClient:
    """Build a SandboxClient. Keyword-only; base_url is required."""
    return SandboxClient(api_key=api_key(), base_url=BASE_URL)


@asynccontextmanager
async def sandbox_session(
    client: SandboxClient,
    *,
    template: str = DEFAULT_TEMPLATE,
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
    from_snapshot: str | None = None,
    metadata: dict[str, str] | None = None,
) -> AsyncIterator[Any]:
    """Create a sandbox, connect it, hand it over, and always destroy it."""
    create_kwargs: dict[str, Any] = {"timeout_ms": timeout_ms}
    if from_snapshot is not None:
        # A forked sandbox inherits its template from the snapshot; sending
        # both is redundant and `fromSnapshot: null` is rejected outright.
        create_kwargs["from_snapshot"] = from_snapshot
    else:
        create_kwargs["template"] = template
    if metadata:
        create_kwargs["metadata"] = metadata

    sandbox = await client.create(**create_kwargs)
    body_failed = False
    try:
        await sandbox.connect()
        yield sandbox
    except BaseException:
        body_failed = True
        raise
    finally:
        try:
            await sandbox.kill()
        except Exception as teardown_error:  # noqa: BLE001 - reported, never hidden
            print(
                f"!! failed to kill sandbox {sandbox.sandboxId}: "
                f"{type(teardown_error).__name__}: {teardown_error}\n"
                f"!! the VM may still be running and billing — check "
                f"https://console.getsolari.com",
                file=sys.stderr,
            )
            # Only raise if it is not masking a real failure from the body.
            if not body_failed:
                raise
