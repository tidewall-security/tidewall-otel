"""OpenTelemetry instrumentor that activates Tidewall guard enforcement.

This is the main orchestrator. It inherits from OpenTelemetry's
:class:`BaseInstrumentor` and applies ``wrapt`` patches to OpenAI and
Anthropic SDK methods so that every chat-completion call is intercepted
by the Tidewall guard before reaching the underlying provider.

Activation paths (any of these will reach :class:`TidewallInstrumentor`):

1. OTel auto-discovery: ``opentelemetry-instrument python app.py``
2. Programmatic: ``TidewallInstrumentor().instrument()``
3. CLI wrapper: ``tidewall-instrument python app.py``
4. Direct import: ``import tidewall_otel; tidewall_otel.activate()``

The instrumentor degrades gracefully — if neither OpenAI nor Anthropic
SDKs are installed, it logs a warning and applies no patches rather
than raising, so adding ``tidewall-otel`` as a dependency never breaks
an unrelated app.
"""

from __future__ import annotations

import logging
from typing import Any, Collection

from tidewall_otel._config import TidewallConfig
from tidewall_otel._guard import TidewallGuard

logger = logging.getLogger("tidewall.otel.instrumentor")

# Module paths used by wrapt to locate the methods to patch. These match
# the module structure of the upstream openai and anthropic SDKs (verified
# against the published packages on PyPI).
_OPENAI_MODULE = "openai.resources.chat.completions.completions"
_ANTHROPIC_MODULE = "anthropic.resources.messages.messages"

# Try to import OpenTelemetry's BaseInstrumentor — if OTel isn't installed,
# substitute a minimal stub so the package still works for the explicit
# `tidewall_otel.activate()` path.
try:
    from opentelemetry.instrumentation.instrumentor import BaseInstrumentor

    _HAS_OTEL_INSTRUMENTOR = True
except ImportError:
    _HAS_OTEL_INSTRUMENTOR = False

    class BaseInstrumentor:  # type: ignore[no-redef]
        """Minimal stub used when OpenTelemetry isn't installed.

        Mirrors the shape of the real BaseInstrumentor closely enough that
        :class:`TidewallInstrumentor` works the same way regardless of
        whether OTel is present.
        """

        _is_instrumented = False

        def instrumentation_dependencies(self) -> Collection[str]:
            return []

        def instrument(self, **kwargs: Any) -> None:
            if not self._is_instrumented:
                self._instrument(**kwargs)
                self._is_instrumented = True

        def uninstrument(self, **kwargs: Any) -> None:
            if self._is_instrumented:
                self._uninstrument(**kwargs)
                self._is_instrumented = False

        def _instrument(self, **kwargs: Any) -> None:
            raise NotImplementedError

        def _uninstrument(self, **kwargs: Any) -> None:
            raise NotImplementedError


def _sdk_available(module_path: str) -> bool:
    """Return True if a module can be imported without errors."""
    try:
        __import__(module_path)
        return True
    except ImportError:
        return False


class TidewallInstrumentor(BaseInstrumentor):
    """OpenTelemetry instrumentor that adds Tidewall security enforcement.

    Wraps OpenAI and Anthropic SDK chat-completion methods so that:

    1. Every prompt is sent to the Tidewall guard for policy evaluation
       before reaching the AI provider.
    2. Blocking and transformation decisions are applied inline in
       ``enforce`` mode (see :class:`TidewallConfig`).
    3. ``gen_ai.*`` OpenTelemetry spans are emitted regardless of mode,
       so observability works even when the guard is in monitor mode.
    """

    _guard: TidewallGuard | None = None
    _config: TidewallConfig | None = None

    def instrumentation_dependencies(self) -> Collection[str]:
        # Returning an empty collection means OTel will load this
        # instrumentor regardless of which AI SDK (if any) is installed,
        # and we decide what to patch at runtime. This is intentional —
        # the alternative would be to require either openai or anthropic
        # as a dependency, which would force unnecessary installs.
        return []

    def _instrument(self, **kwargs: Any) -> None:
        config = kwargs.get("config") or TidewallConfig()
        self._config = config
        self._guard = TidewallGuard(config)
        self._patched: list[tuple[str, str]] = []

        try:
            from wrapt import wrap_function_wrapper
        except ImportError:
            logger.error(
                "wrapt package not installed. Install with: pip install wrapt"
            )
            return

        if _sdk_available(_OPENAI_MODULE):
            from tidewall_otel._openai_wrapper import (
                make_openai_async_wrapper,
                make_openai_sync_wrapper,
            )

            sync_wrapper = make_openai_sync_wrapper(self._guard, config)
            async_wrapper = make_openai_async_wrapper(self._guard, config)

            wrap_function_wrapper(
                _OPENAI_MODULE, "Completions.create", sync_wrapper
            )
            self._patched.append((_OPENAI_MODULE, "Completions.create"))

            wrap_function_wrapper(
                _OPENAI_MODULE, "AsyncCompletions.create", async_wrapper
            )
            self._patched.append((_OPENAI_MODULE, "AsyncCompletions.create"))

            logger.info("OpenAI SDK instrumented with Tidewall guard")
        else:
            logger.debug("OpenAI SDK not available, skipping")

        if _sdk_available(_ANTHROPIC_MODULE):
            from tidewall_otel._anthropic_wrapper import (
                make_anthropic_async_wrapper,
                make_anthropic_sync_wrapper,
            )

            sync_wrapper = make_anthropic_sync_wrapper(self._guard, config)
            async_wrapper = make_anthropic_async_wrapper(self._guard, config)

            wrap_function_wrapper(
                _ANTHROPIC_MODULE, "Messages.create", sync_wrapper
            )
            self._patched.append((_ANTHROPIC_MODULE, "Messages.create"))

            wrap_function_wrapper(
                _ANTHROPIC_MODULE, "AsyncMessages.create", async_wrapper
            )
            self._patched.append((_ANTHROPIC_MODULE, "AsyncMessages.create"))

            logger.info("Anthropic SDK instrumented with Tidewall guard")
        else:
            logger.debug("Anthropic SDK not available, skipping")

        if not self._patched:
            logger.warning(
                "No supported AI SDKs found (openai, anthropic). "
                "Nothing to instrument."
            )
        else:
            logger.info(
                "Tidewall instrumentation active (mode=%s, targets=%s)",
                config.mode,
                [f"{m}.{c}" for m, c in self._patched],
            )

    def _uninstrument(self, **kwargs: Any) -> None:
        """Remove all applied patches and restore the original SDK methods."""
        for module_path, class_method in self._patched:
            try:
                import importlib

                mod = importlib.import_module(module_path)
                parts = class_method.split(".")
                parent = mod
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                attr_name = parts[-1]
                original = getattr(parent, attr_name)
                # wrapt stores the original callable on __wrapped__; if it
                # isn't there we assume the patch was already removed.
                if hasattr(original, "__wrapped__"):
                    setattr(parent, attr_name, original.__wrapped__)
                    logger.debug("Unpatched %s.%s", module_path, class_method)
            except Exception:
                logger.debug(
                    "Could not unpatch %s.%s",
                    module_path, class_method, exc_info=True,
                )

        self._patched.clear()
        logger.info("Tidewall instrumentation deactivated")
