"""Dispatch: the enforcement boundary, as pure decisions plus thin adapters.

The decisions are pure functions with the I/O between them, because selecting
Proceed, Transform or Refuse depends on a CLASSIFIED outcome that does not
exist until the guard has answered. One function taking the call and returning
a decision has nowhere to receive that answer.

Two adapters, not one: a single ``def`` cannot both await and not await.
They own only blocking versus awaiting; everything else is shared.

WHY THE ORIGINAL CALL IS CARRIED THROUGHOUT. A transform must return every
untouched provider kwarg, and a classified response carries only what the
guard rewrote. Without the original bound call, Proceed and Transform would
have to reconstruct arguments they never saw.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from ._bound import bound_nodes
from ._coverage import classify_input
from ._execution import DeadlineExceeded, ExecutorSaturated
from ._span_helper import gen_ai_span, record_response_in_span
from ._exceptions import LossyInputError, TidewallBlockedError, TidewallRefusedError
from ._http import GuardSchemaInvalid, GuardTimeout, GuardUnreachable
from ._manifest import Surface, client_escapes
from ._response import Outcome, classify_response


# -- state carried through dispatch ---------------------------------------

@dataclass(frozen=True)
class BoundCall:
    """The ORIGINAL call, preserved end to end.

    Without this no decision can return provider kwargs, because most of them
    were never inspected and the guard never saw them.
    """

    wrapped: Callable
    instance: Any
    args: tuple
    kwargs: dict
    nodes: dict[str, Any]


# -- pre-I/O ---------------------------------------------------------------

@dataclass(frozen=True)
class SendToGuard:
    guard_input: dict


@dataclass(frozen=True)
class RefuseLossy:
    paths: tuple[str, ...]


@dataclass(frozen=True)
class SkipGuard:
    reason: str
    detail: Any = None


InputDecision = SendToGuard | RefuseLossy | SkipGuard


class GuardPort(Protocol):
    """The injectable guard.

    ``check_raw`` returns the RAW decoded body -- not a Decision, not an
    Outcome. Classification is the response module's job and happens in the
    adapter, so the transport cannot quietly decide policy.

    Raises GuardUnreachable, GuardTimeout or GuardSchemaInvalid. Saturation is
    NOT here: it comes from the executor, which raises ExecutorSaturated, and
    dispatch translates it. Naming it on the port would misstate which layer
    owns overload.
    """

    def check_raw(self, *, guard_input: dict) -> dict: ...


#: Exception to outcome. The adapter's whole classification responsibility,
#: stated as a table so it cannot be improvised differently in each adapter.
_EXCEPTION_OUTCOMES: dict[type, str] = {
    GuardUnreachable: "unreachable",
    GuardTimeout: "timeout",
    GuardSchemaInvalid: "schema_invalid",
    ExecutorSaturated: "saturated",
    DeadlineExceeded: "timeout",
}


def dispatch_outcome_for(exc: BaseException) -> Outcome:
    """Map a raised exception to an outcome. Anything unmapped is
    invariant_violated -- fail-closed, never swallowed."""
    for exc_type, kind in _EXCEPTION_OUTCOMES.items():
        if isinstance(exc, exc_type):
            return Outcome(kind=kind, detail=str(exc))
    return Outcome(kind="invariant_violated", detail=repr(exc))


# -- post-I/O --------------------------------------------------------------

@dataclass(frozen=True)
class Proceed:
    kwargs: dict


@dataclass(frozen=True)
class Transform:
    kwargs: dict


@dataclass(frozen=True)
class Refuse:
    error: BaseException


Decision = Proceed | Transform | Refuse

#: Outcomes that are guard FAILURES rather than verdicts.
#: NO "incomplete". Nothing in this package ever constructs an Outcome with
#: that kind -- the producible set is blocked/clean/degraded/transformed/lossy
#: /schema_invalid/invariant_violated plus unreachable/timeout/saturated from
#: `_EXCEPTION_OUTCOMES`. A member that cannot occur implies a protection that
#: is not operating, which is the same reasoning that removed the dead
#: bool-masquerading-as-int guard from `_response.py`.
#:
#: Its presence was invisible because the dispatch tests retyped this set by
#: hand and omitted it. Deriving the test cases from here surfaced it
#: immediately: four tests failed on a kind no code can emit.
_FAILURES = frozenset({
    "unreachable", "timeout", "saturated", "schema_invalid",
    "invariant_violated",
})


def decide_input(surface: Surface, call: BoundCall, config) -> InputDecision:
    """Pure. Whether the guard may be called at all, and with what."""
    if config.mode == "dry-run":
        return SkipGuard(reason="dry-run")

    coverage = classify_input(surface, call.nodes)
    if not coverage.is_lossless:
        if config.mode == "enforce":
            return RefuseLossy(paths=coverage.lossy_paths)
        # monitor: record and proceed. The guard is NOT called with input it
        # cannot faithfully represent.
        return SkipGuard(reason="lossy", detail=coverage.lossy_paths)

    from ._normalizer import normalize

    return SendToGuard(guard_input=normalize(surface, call.kwargs))


def decide_outcome(surface: Surface, call: BoundCall, pre: InputDecision,
                   outcome: Outcome, config) -> Decision:
    """Pure. Consumes an ALREADY-CLASSIFIED outcome."""
    if outcome.kind == "blocked":
        if config.mode == "enforce":
            return Refuse(TidewallBlockedError(outcome.detail or "blocked by policy", {}))
        return Proceed(call.kwargs)

    if outcome.kind == "transformed":
        if config.mode != "enforce":
            return Proceed(call.kwargs)
        messages = (outcome.guard_output or {}).get("messages") or []
        # The write-back is the manifest entry's, not branched on inline:
        # Anthropic must split the synthetic head back out to its own kwarg.
        return Transform(surface.transform_into(call.kwargs, messages))

    if outcome.kind in _FAILURES:
        if config.mode == "enforce":
            return Refuse(TidewallRefusedError(
                f"refusing: guard {outcome.kind} ({outcome.detail})",
                outcome_kind=outcome.kind,
            ))
        return Proceed(call.kwargs)

    return Proceed(call.kwargs)          # clean, degraded


# -- adapters --------------------------------------------------------------

def _prepare(surface, wrapped, instance, args, kwargs, config, state):
    """Shared prologue: client integrity, then the bound call."""
    escapes = client_escapes(getattr(instance, "_client", None))
    if escapes and state is not None:
        state.record_unverified(surface.attribute, reason="client_escapes",
                                detail=escapes)
    return BoundCall(
        wrapped=wrapped, instance=instance, args=args, kwargs=kwargs,
        nodes=bound_nodes(
            wrapped, args, kwargs,
            stop=surface.opaque_subtrees + surface.non_prompt_bearing,
            unset=surface.unset_sentinels,
        ),
    )


def _apply(decision: Decision, call: BoundCall):
    if isinstance(decision, Refuse):
        raise decision.error                # the provider is NEVER invoked
    return decision.kwargs


logger = logging.getLogger("tidewall.otel.dispatch")


def _annotate(span, key: str, value) -> None:
    """Set one Tidewall attribute, never breaking the call if the span cannot."""
    if span is None:
        return
    try:
        span.set_attribute(key, str(value))
    except Exception:                       # pragma: no cover - defensive
        logger.debug("could not annotate span %s", key, exc_info=True)


def _record_refusal(span, kind: str, *, blocked: bool = False) -> None:
    """Mark a span for a call the provider never saw.

    A refusal is the event an operator opens a trace to explain, so it must
    leave one. `blocked` distinguishes a policy verdict from a guard failure,
    because a dashboard that cannot tell them apart cannot triage either.
    """
    _annotate(span, "tidewall.refused", kind)
    record_response_in_span(span, blocked=blocked, include_output=False)


def _invoke_and_record(wrapped, args, kwargs, span, surface):
    """Call the provider inside the span and record what came back."""
    response = wrapped(*args, **kwargs)
    record_response_in_span(
        span,
        content=_response_text(response),
        finish_reason=_finish_reason(response),
        include_output=surface.span_output,
    )
    return response


def _response_text(response):
    """Best-effort completion text. Never raises: a span is not worth a call."""
    try:
        choices = getattr(response, "choices", None)
        if choices:
            return getattr(getattr(choices[0], "message", None), "content", None)
        content = getattr(response, "content", None)
        if content:
            return getattr(content[0], "text", None)
    except Exception:                       # pragma: no cover - defensive
        return None
    return None


def _finish_reason(response):
    try:
        choices = getattr(response, "choices", None)
        if choices:
            return getattr(choices[0], "finish_reason", None)
        return getattr(response, "stop_reason", None)
    except Exception:                       # pragma: no cover - defensive
        return None


def dispatch_sync(surface, wrapped, instance, args, kwargs, config, guard, executor,
                  state=None):
    call = _prepare(surface, wrapped, instance, args, kwargs, config, state)
    pre = decide_input(surface, call, config)

    # THE SPAN WRAPS EVERY EXIT, including the ones that never reach the
    # provider. `gen_ai_span` and `record_response_in_span` were defined,
    # documented and unit-tested while no production path called either, so a
    # fully guarded call emitted nothing at all -- in a package named
    # `tidewall-otel`, whose instrumentor docstring promised spans "regardless
    # of mode". A refusal or a block is exactly the event an operator opens a
    # trace to find, so those exits carry spans too.
    #
    # `include_input` is the surface's own flag, never a default: serialising
    # the conversation is the P0 this programme opened with.
    with gen_ai_span(provider=surface.provider,
                     model=str(call.kwargs.get("model", "")),
                     guard_input=getattr(pre, "guard_input", None),
                     include_input=surface.span_input) as span:
        if isinstance(pre, RefuseLossy):
            _record_refusal(span, "lossy_input")
            raise LossyInputError(pre.paths)

        if isinstance(pre, SkipGuard):
            if state is not None:
                state.record_skip(surface.attribute, reason=pre.reason,
                                  detail=pre.detail)
            _annotate(span, "tidewall.guard.skipped", pre.reason)
            return _invoke_and_record(wrapped, args, kwargs, span, surface)

        try:
            # submit BLOCKS and returns the value; `deadline` is keyword-only.
            raw = executor.submit(guard.check_raw,
                                  guard_input=pre.guard_input,
                                  deadline=config.guard_deadline_s)
            outcome = classify_response(raw)
        except BaseException as exc:
            outcome = dispatch_outcome_for(exc)

        _annotate(span, "tidewall.guard.outcome", outcome.kind)
        decision = decide_outcome(surface, call, pre, outcome, config)

        if isinstance(decision, Refuse):
            _record_refusal(span, outcome.kind, blocked=outcome.kind == "blocked")

        provider_kwargs = _apply(decision, call)
        return _invoke_and_record(wrapped, args, provider_kwargs, span, surface)


async def dispatch_async(surface, wrapped, instance, args, kwargs, config, guard,
                         executor, state=None):
    """The same sequence, differing only in awaiting rather than blocking."""
    call = _prepare(surface, wrapped, instance, args, kwargs, config, state)
    pre = decide_input(surface, call, config)

    # Same span discipline as the sync path, and it has to be duplicated
    # rather than shared: the context manager wraps an `await`, so the two
    # cannot be one function without making the sync path a coroutine.
    with gen_ai_span(provider=surface.provider,
                     model=str(call.kwargs.get("model", "")),
                     guard_input=getattr(pre, "guard_input", None),
                     include_input=surface.span_input) as span:
        if isinstance(pre, RefuseLossy):
            _record_refusal(span, "lossy_input")
            raise LossyInputError(pre.paths)

        if isinstance(pre, SkipGuard):
            if state is not None:
                state.record_skip(surface.attribute, reason=pre.reason,
                                  detail=pre.detail)
            _annotate(span, "tidewall.guard.skipped", pre.reason)
            response = await wrapped(*args, **kwargs)
            record_response_in_span(
                span, content=_response_text(response),
                finish_reason=_finish_reason(response),
                include_output=surface.span_output)
            return response

        try:
            # The SAME bounded pool, awaited. run_in_executor(None, ...) would
            # add the loop's default executor: a second, unbounded queue
            # outside this admission layer.
            raw = await executor.submit_awaitable(
                guard.check_raw,
                guard_input=pre.guard_input,
                deadline=config.guard_deadline_s)
            outcome = classify_response(raw)
        except asyncio.CancelledError:
            raise                           # the caller went away
        except BaseException as exc:
            outcome = dispatch_outcome_for(exc)

        _annotate(span, "tidewall.guard.outcome", outcome.kind)
        decision = decide_outcome(surface, call, pre, outcome, config)

        if isinstance(decision, Refuse):
            _record_refusal(span, outcome.kind, blocked=outcome.kind == "blocked")

        provider_kwargs = _apply(decision, call)
        response = await wrapped(*args, **provider_kwargs)
        record_response_in_span(
            span, content=_response_text(response),
            finish_reason=_finish_reason(response),
            include_output=surface.span_output)
        return response
