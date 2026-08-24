"""The refuser state machine. Task 5 of the P0 remediation plan."""

import asyncio

import pytest

from tidewall_otel._manifest import (
    ANTHROPIC_MESSAGES_ASYNC,
    OPENAI_CHAT_ASYNC,
    OPENAI_CHAT_SYNC,
    SURFACES,
)
from tidewall_otel._refuser import (
    TidewallActivationRefusedError,
    is_refuser,
    make_refuser,
)


def test_a_sync_refuser_RAISES_rather_than_calling_through():
    called = []
    refuser = make_refuser(OPENAI_CHAT_SYNC, reason="coverage unverified")

    with pytest.raises(TidewallActivationRefusedError, match="coverage unverified"):
        refuser(lambda *a, **k: called.append(1), None, (), {})

    assert called == [], "the refuser passed the call through"


@pytest.mark.asyncio
async def test_an_async_refuser_is_AWAITABLE_and_raises():
    """An async surface's refuser must be awaitable. A synchronous refuser
    installed on an async method makes `await` raise TypeError instead of the
    refusal -- the caller sees the wrong error and the provider is still never
    called, so a test asserting only 'something raised' cannot tell them
    apart."""
    called = []

    async def wrapped(*a, **k):
        called.append(1)

    refuser = make_refuser(OPENAI_CHAT_ASYNC, reason="coverage unverified")
    with pytest.raises(TidewallActivationRefusedError, match="coverage unverified"):
        await refuser(wrapped, None, (), {})

    assert called == []


@pytest.mark.parametrize("surface", SURFACES, ids=lambda s: s.attribute)
def test_a_refuser_is_installable_for_every_manifest_kind(surface):
    """The manifest declares `sync` and `async`; each needs its own shape."""
    refuser = make_refuser(surface, reason="x")
    assert is_refuser(refuser)
    assert asyncio.iscoroutinefunction(refuser) is (surface.kind == "async")


def test_is_refuser_identifies_one_and_rejects_anything_else():
    """The manager needs to tell its own refusers from a foreign wrapper: it
    may remove the former and must never remove the latter."""
    assert is_refuser(make_refuser(OPENAI_CHAT_SYNC, reason="x"))
    assert not is_refuser(lambda *a, **k: None)
    assert not is_refuser(object())

    async def foreign(*a, **k): ...
    assert not is_refuser(foreign)


def test_the_refusal_names_the_surface_and_the_reason():
    """An operator seeing this in a traceback needs to know which boundary
    refused and why, not merely that something refused."""
    refuser = make_refuser(ANTHROPIC_MESSAGES_ASYNC, reason="guard unreachable")
    with pytest.raises(TidewallActivationRefusedError) as caught:
        asyncio.run(refuser(None, None, (), {}))

    message = str(caught.value)
    assert "Messages.create" in message
    assert "guard unreachable" in message
