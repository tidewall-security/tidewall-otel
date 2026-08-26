"""The async adapter and executor lifecycle. Task 12c."""

import asyncio

import pytest

from tests._fixtures import MAX_WORKERS, QUEUE_SIZE, wait_until
from tidewall_otel._config import TidewallConfig
from tidewall_otel._dispatch import LossyInputError, dispatch_async
from tidewall_otel._exceptions import TidewallBlockedError, TidewallError
from tidewall_otel._execution import BoundedExecutor
from tidewall_otel._manifest import OPENAI_CHAT_ASYNC
from tidewall_otel._state import State

from tests.test_dispatch import (
    FAILURE_KINDS,
    FakeInstance,
    GuardRaising,
    GuardReturning,
    blocked_body,
    clean_body,
    minimal,
)


@pytest.fixture
def config(monkeypatch):
    monkeypatch.setenv("TIDEWALL_BASE_URL", "https://guard.example")
    monkeypatch.setenv("TIDEWALL_TOKEN", "t")
    return TidewallConfig()


@pytest.fixture
def executor():
    ex = BoundedExecutor(max_workers=MAX_WORKERS, queue_size=QUEUE_SIZE)
    yield ex
    ex.shutdown()


class AsyncRecordingProvider:
    def __init__(self, result="provider-result"):
        self.calls = []
        self.awaited = []
        self._result = result

    async def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        await asyncio.sleep(0)
        self.awaited.append(True)
        return self._result


@pytest.mark.parametrize("kind", FAILURE_KINDS)
async def test_enforce_never_invokes_the_ASYNC_provider_on_any_failure(
        config, executor, kind):
    provider = AsyncRecordingProvider()
    guard = GuardRaising(kind)
    with pytest.raises(TidewallError) as raised:
        await dispatch_async(OPENAI_CHAT_ASYNC, provider, FakeInstance(), (),
                             minimal(OPENAI_CHAT_ASYNC), config, guard, executor)
    assert provider.calls == [], f"contacted the provider on {kind}"
    # See the sync twin: catching the base class alone is a false green,
    # because `LossyInputError` satisfies it without the guard being called.
    assert guard.calls, f"the guard was never called on {kind}"
    assert raised.value.outcome_kind == kind, \
        f"refused as {raised.value.outcome_kind}, not {kind}"


async def test_the_async_adapter_actually_AWAITS_the_provider(config, executor):
    """Invoking a coroutine function is not evidence it was awaited: an
    un-awaited coroutine returns an object and the body never runs."""
    provider = AsyncRecordingProvider()
    result = await dispatch_async(OPENAI_CHAT_ASYNC, provider, FakeInstance(), (),
                                  minimal(OPENAI_CHAT_ASYNC), config,
                                  GuardReturning(clean_body()), executor)
    assert provider.awaited == [True]
    assert result == "provider-result"


async def test_the_async_bridge_adds_NO_second_pool(config, executor):
    """run_in_executor(None, ...) would use the loop's DEFAULT executor: a
    second, unbounded queue outside this admission layer."""
    loop = asyncio.get_running_loop()
    await dispatch_async(OPENAI_CHAT_ASYNC, AsyncRecordingProvider(),
                         FakeInstance(), (), minimal(OPENAI_CHAT_ASYNC),
                         config, GuardReturning(clean_body()), executor)
    assert loop._default_executor is None, "a default executor was created"


async def test_awaiting_the_guard_does_not_BLOCK_the_event_loop(config, executor):
    """The reason the executor exists at all: a ticker must keep running
    while the guard call is outstanding."""
    ticks = []

    async def ticker():
        while True:
            ticks.append(1)
            await asyncio.sleep(0.001)

    class SlowGuard:
        def check_raw(self, *, guard_input, **kwargs):
            import time
            time.sleep(0.05)
            return clean_body()

    task = asyncio.create_task(ticker())
    await dispatch_async(OPENAI_CHAT_ASYNC, AsyncRecordingProvider(),
                         FakeInstance(), (), minimal(OPENAI_CHAT_ASYNC),
                         config, SlowGuard(), executor)
    task.cancel()
    assert len(ticks) > 1, "the loop was blocked for the guard call"


async def test_cancelling_the_caller_propagates_CancelledError(config, executor):
    import threading

    blocker = threading.Event()

    class BlockingGuard:
        def check_raw(self, *, guard_input, **kwargs):
            blocker.wait(30)
            return clean_body()

    task = asyncio.create_task(
        dispatch_async(OPENAI_CHAT_ASYNC, AsyncRecordingProvider(),
                       FakeInstance(), (), minimal(OPENAI_CHAT_ASYNC),
                       config, BlockingGuard(), executor))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    blocker.set()
    wait_until(lambda: executor.outstanding() == 0, timeout=5)


async def test_a_BLOCKED_verdict_in_enforce_never_reaches_the_async_provider(
        config, executor):
    provider = AsyncRecordingProvider()
    guard = GuardReturning(blocked_body())
    with pytest.raises(TidewallBlockedError):
        await dispatch_async(OPENAI_CHAT_ASYNC, provider, FakeInstance(), (),
                             minimal(OPENAI_CHAT_ASYNC), config, guard, executor)
    assert provider.calls == []
    assert guard.calls, "nothing was blocked -- the guard was never called"


async def test_a_typed_call_in_enforce_reaches_NEITHER_side(config, executor):
    provider, guard = AsyncRecordingProvider(), GuardReturning(clean_body())
    typed = {"model": "gpt-4o", "messages": [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]}]}

    with pytest.raises(LossyInputError):
        await dispatch_async(OPENAI_CHAT_ASYNC, provider, FakeInstance(), (),
                             typed, config, guard, executor)
    assert guard.calls == [] and provider.calls == []


async def test_a_lossy_call_in_MONITOR_is_recorded_and_proceeds(
        monkeypatch, executor):
    monkeypatch.setenv("TIDEWALL_BASE_URL", "https://g.example")
    monkeypatch.setenv("TIDEWALL_TOKEN", "t")
    monkeypatch.setenv("TIDEWALL_MODE", "monitor")
    provider, guard = AsyncRecordingProvider(), GuardReturning(clean_body())
    state = State(lifecycle="installed", mode="monitor")
    typed = {"model": "gpt-4o", "messages": [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]}]}

    await dispatch_async(OPENAI_CHAT_ASYNC, provider, FakeInstance(), (), typed,
                         TidewallConfig(), guard, executor, state=state)

    assert provider.calls and guard.calls == []
    assert any("lossy" in str(event) for event in state.events)


# -- wiring: the components must be CALLED, not merely correct -------------

def test_all_four_factories_accept_the_executor():
    """Activation constructs sync and async closures together, so passing the
    executor to only some signatures raises TypeError at activation."""
    import inspect

    from tidewall_otel._anthropic_wrapper import (
        make_anthropic_async_wrapper,
        make_anthropic_sync_wrapper,
    )
    from tidewall_otel._openai_wrapper import (
        make_openai_async_wrapper,
        make_openai_sync_wrapper,
    )

    factories = {
        "openai_sync": (make_openai_sync_wrapper, "_openai_wrapper", "dispatch_sync"),
        "openai_async": (make_openai_async_wrapper, "_openai_wrapper", "dispatch_async"),
        "anthropic_sync": (make_anthropic_sync_wrapper, "_anthropic_wrapper", "dispatch_sync"),
        "anthropic_async": (make_anthropic_async_wrapper, "_anthropic_wrapper", "dispatch_async"),
    }

    for name, (factory, _module, _dispatch) in factories.items():
        assert "executor" in inspect.signature(factory).parameters, name

    # The signature check above is SPELLING. This is WIRING: keeping the
    # parameter and passing `None` to dispatch left the old assertion green --
    # the executor accepted and discarded, which is exactly the defect that
    # made every enforce call an invariant violation in round 1.
    import asyncio
    import importlib

    sentinel = object()
    for name, (factory, module_name, dispatch_name) in factories.items():
        module = importlib.import_module(f"tidewall_otel.{module_name}")
        original = getattr(module, dispatch_name)
        received = []

        def record(*args, **kwargs):
            # positional: surface, wrapped, instance, args, kwargs, config,
            #             guard, executor
            received.append(args[7] if len(args) > 7 else kwargs.get("executor"))
            return None

        async def record_async(*args, **kwargs):
            record(*args, **kwargs)
            return None

        setattr(module, dispatch_name,
                record_async if dispatch_name.endswith("async") else record)
        try:
            wrapper = factory(object(), object(), sentinel, object())
            result = wrapper(lambda **kw: None, None, (), {})
            if inspect.iscoroutine(result):
                asyncio.run(result)
        finally:
            setattr(module, dispatch_name, original)

        assert received == [sentinel], (
            f"{name} did not pass its executor through to {dispatch_name}: "
            f"{received}")


def test_the_SYNC_wrappers_route_through_dispatch(monkeypatch, config, executor):
    """Correct components that nothing calls is the defect this detects: an
    earlier arrangement left the synchronous path -- which most applications
    use -- with its original behaviour entirely."""
    from tidewall_otel import _openai_wrapper

    seen = {}
    monkeypatch.setattr(_openai_wrapper, "dispatch_sync",
                        lambda surface, wrapped, instance, args, kwargs, *a, **k:
                            seen.update(surface=surface.attribute, instance=instance)
                            or "dispatched")

    wrapper = _openai_wrapper.make_openai_sync_wrapper(None, config, executor)
    result = wrapper(lambda **k: "provider", FakeInstance(), (), minimal(OPENAI_CHAT_ASYNC))

    assert result == "dispatched", "the wrapper ignored dispatch's return value"
    assert seen["surface"] == "Completions.create"
    assert seen["instance"] is not None, "wrapt's instance never reached dispatch"


async def test_the_ASYNC_wrappers_route_through_dispatch(monkeypatch, config, executor):
    """The double must be a REAL coroutine function: a synchronous lambda
    returning None makes `await` raise TypeError, so the test would fail for
    an unrelated reason or pass while the body never ran."""
    from tidewall_otel import _openai_wrapper

    seen = {}

    async def fake_dispatch_async(surface, wrapped, instance, args, kwargs, *a, **k):
        seen.update(surface=surface.attribute, instance=instance)
        return "dispatched"

    monkeypatch.setattr(_openai_wrapper, "dispatch_async", fake_dispatch_async)

    wrapper = _openai_wrapper.make_openai_async_wrapper(None, config, executor)
    result = await wrapper(None, FakeInstance(), (), minimal(OPENAI_CHAT_ASYNC))

    assert result == "dispatched"
    assert seen["surface"] == "AsyncCompletions.create"


def test_the_fail_open_check_is_GONE():
    """A guard returning None on transport failure is the defect this
    programme exists to remove. It survived 12b only because the async
    wrappers still called it."""
    from tidewall_otel._guard import TidewallGuard

    assert not hasattr(TidewallGuard, "check")
    assert hasattr(TidewallGuard, "check_raw")
