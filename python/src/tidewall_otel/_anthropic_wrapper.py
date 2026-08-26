"""Anthropic Messages wrappers.

Thin: resolve the surface, hand everything to the shared dispatch component.
Every enforcement decision lives in one place, so the sync and async paths
cannot drift -- an earlier arrangement had correct components that nothing
called, and the synchronous path, which most applications use, kept its
original behaviour entirely.
"""

from __future__ import annotations

from typing import Any

from tidewall_otel._config import TidewallConfig
from tidewall_otel._dispatch import dispatch_async, dispatch_sync
from tidewall_otel._manifest import ANTHROPIC_MESSAGES_ASYNC, ANTHROPIC_MESSAGES_SYNC


def make_anthropic_sync_wrapper(guard: Any, config: TidewallConfig,
                             executor: Any = None, state: Any = None) -> Any:
    """Build a wrapt-compatible sync wrapper for ``Messages.create``."""

    def wrapper(wrapped: Any, instance: Any, args: tuple, kwargs: dict) -> Any:
        return dispatch_sync(ANTHROPIC_MESSAGES_SYNC, wrapped, instance, args, kwargs,
                             config, guard, executor, state=state)

    return wrapper


def make_anthropic_async_wrapper(guard: Any, config: TidewallConfig,
                              executor: Any = None, state: Any = None) -> Any:
    """Build a wrapt-compatible async wrapper for ``AsyncMessages.create``."""

    async def wrapper(wrapped: Any, instance: Any, args: tuple, kwargs: dict) -> Any:
        return await dispatch_async(ANTHROPIC_MESSAGES_ASYNC, wrapped, instance, args,
                                    kwargs, config, guard, executor, state=state)

    return wrapper
