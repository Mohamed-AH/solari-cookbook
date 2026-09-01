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

import asyncio
import random
import sys
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Awaitable, Callable

from solari_sandbox import ConcurrencyLimitError, NoCapacityError, SandboxClient

from .config import BASE_URL, api_key

DEFAULT_TEMPLATE = "base"
DEFAULT_TIMEOUT_MS = 5 * 60_000

# Creating a sandbox can be refused for reasons that are nothing to do with
# the task: the account's concurrency cap (429) or no host free right now
# (503). Both clear on their own, so they are retried with backoff rather than
# scored as a failure of the agent. Everything else propagates immediately.
# The budget has to outlast a task that is already holding a sandbox, or a
# queued task gives up while a slot is about to free. The longest task in the
# reference suite holds one for ~45s, so the ceiling per wait is 30s and the
# total budget is several minutes.
CREATE_ATTEMPTS = 8
CREATE_BACKOFF_S = 2.0
CREATE_BACKOFF_MAX_S = 30.0


def make_client() -> SandboxClient:
    """Build a SandboxClient. Keyword-only; base_url is required."""
    return SandboxClient(api_key=api_key(), base_url=BASE_URL)


async def _create_with_retry(
    client: SandboxClient,
    kwargs: dict[str, Any],
    on_busy: Callable[[], Awaitable[int]] | None = None,
) -> Any:
    """Create a sandbox, waiting out a busy account rather than failing a task.

    The first refusal tells the runner the account is at its ceiling so it can
    stop asking for more than it can have; later refusals in the same attempt
    are just waiting for a slot. Waits are jittered because several tasks are
    typically refused at the same instant, and unjittered backoff makes them
    retry in lockstep forever.
    """
    delay = CREATE_BACKOFF_S
    for attempt in range(1, CREATE_ATTEMPTS + 1):
        try:
            return await client.create(**kwargs)
        except (ConcurrencyLimitError, NoCapacityError) as exc:
            if attempt == CREATE_ATTEMPTS:
                raise
            ceiling = ""
            if attempt == 1 and on_busy is not None:
                ceiling = f" concurrency lowered to {await on_busy()}."
            wait = min(delay, CREATE_BACKOFF_MAX_S) * (0.5 + random.random())
            print(
                f"!! account at capacity ({type(exc).__name__}), "
                f"attempt {attempt}/{CREATE_ATTEMPTS}; waiting {wait:.0f}s.{ceiling}",
                file=sys.stderr,
            )
            await asyncio.sleep(wait)
            delay = min(delay * 2, CREATE_BACKOFF_MAX_S)
    raise AssertionError("unreachable")  # pragma: no cover


@asynccontextmanager
async def sandbox_session(
    client: SandboxClient,
    *,
    template: str = DEFAULT_TEMPLATE,
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
    from_snapshot: str | None = None,
    metadata: dict[str, str] | None = None,
    on_busy: Callable[[], Awaitable[int]] | None = None,
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

    sandbox = await _create_with_retry(client, create_kwargs, on_busy)
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
