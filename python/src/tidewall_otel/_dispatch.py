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
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from ._bound import bound_nodes
from ._coverage import classify_input
from ._execution import DeadlineExceeded, ExecutorSaturated
from ._exceptions import LossyInputError, TidewallRefusedError, TidewallBlockedError, TidewallError
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


def dispatch_sync(surface, wrapped, instance, args, kwargs, config, guard, executor,
                  state=None):
    call = _prepare(surface, wrapped, instance, args, kwargs, config, state)
    pre = decide_input(surface, call, config)

    if isinstance(pre, RefuseLossy):
        raise LossyInputError(pre.paths)
    if isinstance(pre, SkipGuard):
        if state is not None:
            state.record_skip(surface.attribute, reason=pre.reason, detail=pre.detail)
        return wrapped(*args, **kwargs)

    try:
        # submit BLOCKS and returns the value; `deadline` is keyword-only.
        raw = executor.submit(guard.check_raw,
                              guard_input=pre.guard_input,
                              deadline=config.guard_deadline_s)
        outcome = classify_response(raw)
    except BaseException as exc:
        outcome = dispatch_outcome_for(exc)

    return wrapped(*args, **_apply(decide_outcome(surface, call, pre, outcome, config), call))


async def dispatch_async(surface, wrapped, instance, args, kwargs, config, guard,
                         executor, state=None):
    """The same sequence, differing only in awaiting rather than blocking."""
    call = _prepare(surface, wrapped, instance, args, kwargs, config, state)
    pre = decide_input(surface, call, config)

    if isinstance(pre, RefuseLossy):
        raise LossyInputError(pre.paths)
    if isinstance(pre, SkipGuard):
        if state is not None:
            state.record_skip(surface.attribute, reason=pre.reason, detail=pre.detail)
        return await wrapped(*args, **kwargs)

    try:
        # The SAME bounded pool, awaited. run_in_executor(None, ...) would add
        # the loop's default executor: a second, unbounded queue outside this
        # admission layer.
        raw = await executor.submit_awaitable(guard.check_raw,
                                              guard_input=pre.guard_input,
                                              deadline=config.guard_deadline_s)
        outcome = classify_response(raw)
    except asyncio.CancelledError:
        raise                               # the caller went away
    except BaseException as exc:
        outcome = dispatch_outcome_for(exc)

    return await wrapped(*args, **_apply(
        decide_outcome(surface, call, pre, outcome, config), call))
