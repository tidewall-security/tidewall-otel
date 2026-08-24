"""Bounded execution with a total wall-clock deadline.

The default executor has bounded workers and an UNBOUNDED queue, so a slow
guard turns into unbounded memory rather than an error. And ``asyncio.to_thread``
stops event-loop blocking without bounding anything.

THE DEADLINE BOUNDS CALLER LATENCY, NOT THE UNDERLYING WORK. ``urllib`` is not
cancellable mid-call, so on deadline the caller proceeds per the mode contract
while a worker may still be in flight. The pool bound is what stops those
accumulating.

This executor is the single admission layer for both execution modes: the
synchronous path blocks on :meth:`BoundedExecutor.submit`, the asynchronous
path awaits :meth:`BoundedExecutor.submit_awaitable`, and both go through the
same queue, the same capacity check and the same fork hook. Bridging to async
through the event loop's default executor would add a second, unbounded queue
outside this accounting.
"""

from __future__ import annotations

import asyncio
import os
import threading
import weakref
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable


class ExecutorSaturated(Exception):
    """The bounded queue is full. An overload condition, not a guard verdict."""


class DeadlineExceeded(Exception):
    """The caller's wall-clock bound elapsed. Work may still be in flight."""


class BoundedExecutor:
    """A thread pool with a bounded queue and a caller-latency deadline."""

    def __init__(self, max_workers: int = 4, queue_size: int = 32) -> None:
        self._max_workers = max_workers
        self._queue_size = queue_size
        self._lock = threading.Lock()
        self._inflight = 0
        # Admitted futures, for `outstanding()`. Held weakly so a completed
        # future is not kept alive by this set alone.
        self._admitted: weakref.WeakSet[Future] = weakref.WeakSet()
        self._shutdown = False
        self._pool: ThreadPoolExecutor | None = ThreadPoolExecutor(
            max_workers=max_workers
        )
        if hasattr(os, "register_at_fork"):
            os.register_at_fork(after_in_child=self._reset_after_fork)

    # -- lifecycle ---------------------------------------------------------

    def _reset_after_fork(self) -> None:
        """A pool inherited across fork is not usable in the child."""
        # A lock held by a thread that does not exist in the child would
        # deadlock the first acquire. Re-create it, do not inherit it.
        self._lock = threading.Lock()
        self._admitted = weakref.WeakSet()   # the parent's futures are gone
        self._inflight = 0
        if self._shutdown:
            # DO NOT REVIVE. Rebuilding unconditionally would give the child a
            # working executor while `is_shutdown()` still reported True,
            # contradicting the lifecycle contract under which uninstrument()
            # kills it for good.
            self._pool = None
            return
        self._pool = ThreadPoolExecutor(max_workers=self._max_workers)

    def shutdown(self) -> None:
        """Idempotent. Does not wait: in-flight guard calls are abandoned,
        exactly as a cancelled dispatch abandons them."""
        self._shutdown = True
        if self._pool is not None:
            self._pool.shutdown(wait=False)

    def is_shutdown(self) -> bool:
        return self._shutdown

    def __enter__(self) -> "BoundedExecutor":
        return self

    def __exit__(self, *exc: object) -> None:
        self.shutdown()

    # -- admission ---------------------------------------------------------

    @property
    def _capacity(self) -> int:
        """Running workers PLUS queued work -- the spec's two bounds."""
        return self._max_workers + self._queue_size

    def _acquire(self) -> None:
        with self._lock:
            if self._inflight >= self._capacity:
                raise ExecutorSaturated(
                    f"{self._inflight} in flight, capacity {self._capacity} "
                    f"({self._max_workers} workers + {self._queue_size} queued)"
                )
            self._inflight += 1

    def _release(self, _future: Future | None = None) -> None:
        with self._lock:
            self._inflight -= 1

    def outstanding(self) -> int:
        """Jobs admitted and not yet finished (queued + running).

        Exists so a test can prove cancelled work stays bounded and drains,
        which is the achievable property: a future that has started running
        cannot be cancelled, so it cannot reach zero immediately.
        """
        with self._lock:
            return sum(1 for future in self._admitted if not future.done())

    # -- submission --------------------------------------------------------

    def submit_nowait(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future:
        """Admit and submit, without waiting. Raises ExecutorSaturated."""
        if self._pool is None:
            raise RuntimeError("executor is shut down")
        self._acquire()
        try:
            future = self._pool.submit(fn, *args, **kwargs)
            # REQUIRED by `outstanding()`: without this the count is always
            # zero and every bounded/drains assertion observes nothing.
            self._admitted.add(future)
        except BaseException:
            # The releasing callback is attached below, so a raising submit
            # would otherwise consume this slot permanently.
            self._release()
            raise
        future.add_done_callback(self._release)
        return future

    def submit(self, fn: Callable[..., Any], *args: Any, deadline: float, **kwargs: Any) -> Any:
        """BLOCKS and returns the callable's value -- not a Future.

        ``deadline`` is keyword-only and required; ``**kwargs`` are forwarded
        to ``fn``, so no wrapped callable may take a parameter of that name.
        """
        future = self.submit_nowait(fn, *args, **kwargs)
        try:
            return future.result(timeout=deadline)
        except TimeoutError as exc:
            raise DeadlineExceeded(f"exceeded {deadline}s") from exc

    async def submit_awaitable(
        self, fn: Callable[..., Any], *args: Any, deadline: float, **kwargs: Any
    ) -> Any:
        """Awaitable twin of :meth:`submit`, backed by the SAME pool.

        ``wrap_future`` adapts this executor's own future; it does not
        introduce another executor, so admission, capacity, the fork hook and
        shutdown all stay in one place.
        """
        future = self.submit_nowait(fn, *args, **kwargs)
        try:
            return await asyncio.wait_for(asyncio.wrap_future(future), deadline)
        except asyncio.TimeoutError as exc:
            future.cancel()                 # no-op if it already started
            raise DeadlineExceeded(f"exceeded {deadline}s") from exc
        except asyncio.CancelledError:
            # External cancellation: the caller went away. Abandon the result;
            # a job already running cannot be stopped and is left to finish.
            # Capacity is released when it does -- see `outstanding`.
            future.cancel()
            raise
