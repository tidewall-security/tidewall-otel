"""OpenAI Chat Completions wrappers.

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
from tidewall_otel._manifest import OPENAI_CHAT_ASYNC, OPENAI_CHAT_SYNC


def make_openai_sync_wrapper(guard: Any, config: TidewallConfig,
                             executor: Any = None, state: Any = None) -> Any:
    """Build a wrapt-compatible sync wrapper for ``Completions.create``."""

    def wrapper(wrapped: Any, instance: Any, args: tuple, kwargs: dict) -> Any:
        return dispatch_sync(OPENAI_CHAT_SYNC, wrapped, instance, args, kwargs,
                             config, guard, executor, state=state)

    return wrapper


def make_openai_async_wrapper(guard: Any, config: TidewallConfig,
                              executor: Any = None, state: Any = None) -> Any:
    """Build a wrapt-compatible async wrapper for ``AsyncCompletions.create``."""

    async def wrapper(wrapped: Any, instance: Any, args: tuple, kwargs: dict) -> Any:
        return await dispatch_async(OPENAI_CHAT_ASYNC, wrapped, instance, args,
                                    kwargs, config, guard, executor, state=state)

    return wrapper
