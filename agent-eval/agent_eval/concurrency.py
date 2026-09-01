"""Concurrency that adapts to the account it is running against.

`--parallel N` is a request, not a fact. Accounts differ — a free tier allows
very few live sandboxes, a paid one many — and the number is not something a
user should have to look up and hard-code into CI.

So the limiter starts at N and shrinks whenever the API says the account is at
its ceiling, never below one. The suite then runs at whatever concurrency the
account actually supports, instead of spending its retry budget arguing about
a number it was never going to get.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator


class AdaptiveLimiter:
    """A semaphore whose ceiling can be lowered while tasks are in flight.

    Shrinking never revokes a slot already held — in-flight tasks finish
    normally and the lower ceiling applies to whoever is waiting. That is what
    keeps it deadlock-free: `active` may briefly exceed `limit`, and the wait
    condition tolerates it.
    """

    def __init__(self, limit: int, *, floor: int = 1) -> None:
        if limit < 1:
            raise ValueError(f"limit must be at least 1, got {limit}")
        self._requested = limit
        self._limit = limit
        self._floor = max(1, floor)
        self._active = 0
        self._condition = asyncio.Condition()
        self.shrinks = 0

    @property
    def requested(self) -> int:
        """What the caller asked for."""
        return self._requested

    @property
    def limit(self) -> int:
        """What the account turned out to allow."""
        return self._limit

    @property
    def active(self) -> int:
        return self._active

    async def acquire(self) -> None:
        async with self._condition:
            await self._condition.wait_for(lambda: self._active < self._limit)
            self._active += 1

    async def release(self) -> None:
        async with self._condition:
            self._active -= 1
            self._condition.notify_all()

    async def shrink(self) -> int:
        """Lower the ceiling by one and report the new value."""
        async with self._condition:
            if self._limit > self._floor:
                self._limit -= 1
                self.shrinks += 1
                self._condition.notify_all()
            return self._limit

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        await self.acquire()
        try:
            yield
        finally:
            await self.release()
