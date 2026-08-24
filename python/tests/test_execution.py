"""Bounded execution. Task 2 of the P0 remediation plan."""

import asyncio
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from tests._fixtures import MAX_WORKERS, QUEUE_SIZE, wait_until
from tidewall_otel._execution import (
    BoundedExecutor,
    DeadlineExceeded,
    ExecutorSaturated,
)

requires_fork = pytest.mark.skipif(not hasattr(os, "fork"), reason="POSIX only")


def test_a_call_within_the_deadline_returns_its_value():
    with BoundedExecutor(max_workers=2, queue_size=2) as ex:
        assert ex.submit(lambda: 7, deadline=1.0) == 7


def test_the_deadline_bounds_CALLER_latency():
    """It does not terminate the underlying work -- urllib is not cancellable
    mid-call. What is bounded is how long the caller waits."""
    with BoundedExecutor(max_workers=1, queue_size=1) as ex:
        started = time.monotonic()
        with pytest.raises(DeadlineExceeded):
            ex.submit(lambda: time.sleep(2.0), deadline=0.1)
        assert time.monotonic() - started < 1.0


def test_capacity_is_running_workers_PLUS_queued_work():
    """The spec asks for a bounded pool AND a bounded queue -- two numbers."""
    with BoundedExecutor(max_workers=1, queue_size=1) as ex:
        blocker = threading.Event()
        ex.submit_nowait(blocker.wait, 30)              # running
        ex.submit_nowait(blocker.wait, 30)              # queued
        with pytest.raises(ExecutorSaturated):
            ex.submit_nowait(lambda: None)          # over capacity
        blocker.set()


def test_a_failed_submission_does_not_leak_capacity():
    """The releasing callback is attached only on SUCCESS, so a raising
    _pool.submit would consume a slot forever.

    CAPACITY 1 IS LOAD-BEARING. With max_workers=1 and queue_size=1 the
    capacity is 2, so leaking one slot still leaves one free and the
    assertion below passes either way -- mutation-testing the release arm
    showed exactly that: the guard survived. One slot total means a single
    leak is immediately observable.
    """
    ex = BoundedExecutor(max_workers=1, queue_size=0)
    assert ex._capacity == 1

    # _pool.submit raises after shutdown, AFTER a slot has been acquired.
    ex._pool.shutdown(wait=False)
    with pytest.raises(RuntimeError):
        ex.submit_nowait(lambda: None)
    assert ex.outstanding() == 0

    ex._pool = ThreadPoolExecutor(max_workers=1)
    ex.submit_nowait(lambda: None)                  # must not raise Saturated
    ex.shutdown()


def test_outstanding_ACTUALLY_TRACKS_admitted_work():
    """An earlier draft had the registering line only in a comment, so this
    returned 0 with two futures in flight and every bounded/drains assertion
    passed while observing nothing."""
    ex = BoundedExecutor(max_workers=1, queue_size=4)
    blocker = threading.Event()
    ex.submit_nowait(blocker.wait, 30)
    ex.submit_nowait(lambda: None)

    assert ex.outstanding() == 2, "outstanding() is not tracking anything"
    blocker.set()
    wait_until(lambda: ex.outstanding() == 0, timeout=5)
    ex.shutdown()


@requires_fork
def test_a_fork_after_shutdown_does_NOT_revive_the_executor():
    """An earlier fork hook rebuilt the pool unconditionally, so the child
    submitted successfully while is_shutdown() reported True."""
    ex = BoundedExecutor()
    ex.shutdown()

    pid = os.fork()
    if pid == 0:                                    # child
        try:
            ex.submit_nowait(lambda: 42)
            os._exit(1)                             # revived -- FAIL
        except Exception:
            os._exit(0)                             # correctly dead
    _, status = os.waitpid(pid, 0)
    assert os.WEXITSTATUS(status) == 0, "fork revived a shut-down executor"
    assert ex.is_shutdown()


@requires_fork
def test_the_pool_works_in_a_REAL_forked_child():
    """Calling _reset_after_fork() directly tests the method rather than the
    behaviour: a pool inherited across a real fork can deadlock on a lock held
    by a thread that does not exist in the child."""
    ex = BoundedExecutor(max_workers=1, queue_size=1)
    ex.submit_nowait(lambda: None)

    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        try:
            ex.submit(lambda: 42, deadline=5.0)
            os.write(write_fd, b"ok")
        except BaseException:
            os.write(write_fd, b"no")
        finally:
            os._exit(0)

    os.close(write_fd)
    assert os.read(read_fd, 2) == b"ok"
    os.waitpid(pid, 0)
    ex.shutdown()


def test_the_fork_hook_is_registered_EXACTLY_ONCE_per_executor():
    """Repeated activate/deactivate cycles must not accumulate fork hooks,
    which would rebuild the pool N times in a child."""
    from unittest import mock

    registered = []
    with mock.patch.object(os, "register_at_fork",
                           side_effect=lambda **kw: registered.append(kw)):
        ex = BoundedExecutor()
    assert len(registered) == 1, registered
    assert "after_in_child" in registered[0]
    ex.shutdown()


def test_shutdown_is_observable_and_idempotent():
    ex = BoundedExecutor()
    assert not ex.is_shutdown()
    ex.shutdown()
    assert ex.is_shutdown()
    ex.shutdown()                                   # must not raise


@pytest.mark.asyncio
async def test_submit_awaitable_returns_its_value_on_the_SAME_pool():
    """One admission layer for both execution modes: the async path must not
    create the event loop's default executor."""
    loop = asyncio.get_running_loop()
    with BoundedExecutor(max_workers=2, queue_size=2) as ex:
        assert await ex.submit_awaitable(lambda: 7, deadline=1.0) == 7
    assert loop._default_executor is None, "a second pool was created"


@pytest.mark.asyncio
async def test_submit_awaitable_raises_DeadlineExceeded():
    with BoundedExecutor(max_workers=1, queue_size=1) as ex:
        with pytest.raises(DeadlineExceeded):
            await ex.submit_awaitable(lambda: time.sleep(2.0), deadline=0.1)


@pytest.mark.asyncio
async def test_cancelled_work_stays_BOUNDED_and_then_DRAINS():
    """A started concurrent.futures.Future cannot be cancelled, so asserting
    outstanding() == 0 immediately after cancellation is impossible. What must
    be true: the backlog stays bounded, and clears once work unblocks."""
    blocker = threading.Event()
    ex = BoundedExecutor(max_workers=MAX_WORKERS, queue_size=QUEUE_SIZE)
    capacity = MAX_WORKERS + QUEUE_SIZE

    tasks = [asyncio.create_task(ex.submit_awaitable(blocker.wait, 30, deadline=10.0))
             for _ in range(capacity)]
    await asyncio.sleep(0.05)
    for task in tasks:
        task.cancel()
    for task in tasks:
        with pytest.raises(asyncio.CancelledError):
            await task

    assert ex.outstanding() <= capacity

    blocker.set()
    wait_until(lambda: ex.outstanding() == 0, timeout=5)
    assert await ex.submit_awaitable(lambda: "ok", deadline=1.0) == "ok"
    ex.shutdown()


@pytest.mark.asyncio
async def test_saturation_is_reported_from_the_SAME_pool():
    """Beyond capacity the async path must raise ExecutorSaturated from the
    bounded executor, not queue silently elsewhere."""
    blocker = threading.Event()
    ex = BoundedExecutor(max_workers=1, queue_size=1)
    tasks = [asyncio.create_task(ex.submit_awaitable(blocker.wait, 30, deadline=10.0))
             for _ in range(2)]
    await asyncio.sleep(0.05)

    with pytest.raises(ExecutorSaturated):
        await ex.submit_awaitable(lambda: None, deadline=1.0)

    for task in tasks:
        task.cancel()
    blocker.set()
    ex.shutdown()
