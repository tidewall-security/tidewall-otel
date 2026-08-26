"""Response schema and the total classifier. Task 10 of the P0 plan."""

import itertools

import pytest

from tidewall_otel._response import Outcome, classify_response


def body(**overrides):
    """A minimal VALID guard response.

    An explicit `result=` REPLACES the default rather than merging into it:
    merging silently restores any field a test deleted, so the
    missing-required-field cases would have passed against a classifier that
    never checked.
    """
    result = overrides.pop("result", None)
    if result is None:
        result = {"blocked": False, "transformed": False, "policy": "default"}
    return {"request_id": "r", "request_time": "t", "summary": "", "result": result,
            **overrides}


# -- the cross-product -----------------------------------------------------

@pytest.mark.parametrize(
    "blocked,transformed,degraded,failed,applicable",
    list(itertools.product([True, False], repeat=5)),
)
def test_every_combination_yields_EXACTLY_ONE_outcome(
    blocked, transformed, degraded, failed, applicable
):
    """Thirty-two combinations, each producing exactly one verdict.

    A classifier with gaps returns None for some input, and dispatch's
    catch-all then files a legitimate guard verdict as invariant_violated.
    """
    result = {"blocked": blocked, "transformed": transformed,
              "policy": "default", "degraded": degraded,
              "failed_detectors": ["pii"] if failed else []}
    if transformed and applicable:
        result["guard_output"] = {"messages": [{"role": "user", "content": "x"}]}

    outcome = classify_response(body(result=result))
    assert isinstance(outcome, Outcome)
    assert outcome.kind in {
        "blocked", "transformed", "clean", "degraded", "schema_invalid",
    }, outcome


@pytest.mark.parametrize("guard_output", [None, {}, {"no_messages": 1}])
def test_blocked_and_transformed_without_usable_output_is_BLOCKED(guard_output):
    """Not invariant_violated. The guard said block; an unusable transform
    payload does not turn a refusal into a bug -- the safe reading of an
    ambiguous verdict is the more restrictive one."""
    result = {"blocked": True, "transformed": True, "policy": "default"}
    if guard_output is not None:
        result["guard_output"] = guard_output

    assert classify_response(body(result=result)).kind == "blocked"


def test_blocked_wins_over_transformed_even_when_the_transform_is_usable():
    result = {"blocked": True, "transformed": True, "policy": "default",
              "guard_output": {"messages": [{"role": "user", "content": "x"}]}}
    assert classify_response(body(result=result)).kind == "blocked"


def test_a_usable_transform_carries_its_output():
    result = {"blocked": False, "transformed": True, "policy": "default",
              "guard_output": {"messages": [{"role": "user", "content": "clean"}]}}
    outcome = classify_response(body(result=result))
    assert outcome.kind == "transformed"
    assert outcome.guard_output["messages"][0]["content"] == "clean"


def test_transformed_without_usable_output_is_SCHEMA_INVALID_not_clean():
    """Reporting clean would let the unmodified prompt proceed while the guard
    believed it had rewritten it."""
    result = {"blocked": False, "transformed": True, "policy": "default"}
    assert classify_response(body(result=result)).kind == "schema_invalid"


def test_degraded_is_distinct_from_clean():
    """'checked, found nothing' and 'could not check' are different answers,
    and a caller that cannot tell them apart cannot act on either."""
    result = {"blocked": False, "transformed": False, "policy": "default",
              "degraded": True, "failed_detectors": ["pii"]}
    outcome = classify_response(body(result=result))
    assert outcome.kind == "degraded"
    assert "pii" in outcome.failed_detectors


# -- schema validity -------------------------------------------------------

@pytest.mark.parametrize("missing", ["blocked", "transformed", "policy"])
def test_a_missing_REQUIRED_field_is_schema_invalid(missing):
    result = {"blocked": False, "transformed": False, "policy": "default"}
    del result[missing]
    assert classify_response(body(result=result)).kind == "schema_invalid"


@pytest.mark.parametrize("optional,default", [
    ("guard_output", None), ("detectors", {}), ("access_rules", {}),
    ("fpe_context", None), ("degraded", False), ("failed_detectors", []),
])
def test_a_missing_OPTIONAL_field_gets_its_documented_default(optional, default):
    """Rejecting these would refuse valid server responses: every one has a
    default in the server model."""
    outcome = classify_response(body())
    assert outcome.kind == "clean"
    assert getattr(outcome, "raw")["result"].get(optional, default) == default


@pytest.mark.parametrize("field,bad", [
    ("blocked", "yes"), ("transformed", 1), ("policy", 3),
    ("degraded", "true"), ("failed_detectors", "pii"),
])
def test_a_WRONG_TYPE_is_schema_invalid(field, bad):
    """For a required field and for a PRESENT optional one alike."""
    result = {"blocked": False, "transformed": False, "policy": "default"}
    result[field] = bad
    assert classify_response(body(result=result)).kind == "schema_invalid"


@pytest.mark.parametrize("bad", [None, [], "text", 3, {}])
def test_a_non_object_or_empty_body_is_schema_invalid(bad):
    assert classify_response(bad).kind == "schema_invalid"


def test_UNKNOWN_FIELDS_are_accepted_at_both_levels():
    """Both server models declare extra='allow', so refusing unknown fields
    would break on a server minor release."""
    payload = body(new_top_level="x")
    payload["result"]["new_nested"] = "y"
    assert classify_response(payload).kind == "clean"


# -- the pin ---------------------------------------------------------------

def test_the_schema_is_PINNED_against_the_server_package():
    """Imported from the server when available, not copied.

    A copy drifts silently; this fails loudly when the server changes its
    required fields. The skip is RECORDED rather than silent, because a pin
    that quietly does not run is not a pin.
    """
    pytest.importorskip(
        "app.models",
        reason="tidewall-server not importable: the schema pin did NOT run",
    )
    from app.models import GuardResult

    from tidewall_otel._response import REQUIRED_RESULT_FIELDS

    server_required = {
        name for name, info in GuardResult.model_fields.items() if info.is_required()
    }
    assert REQUIRED_RESULT_FIELDS == server_required, (
        f"client requires {REQUIRED_RESULT_FIELDS}, server requires {server_required}"
    )


@pytest.mark.parametrize("field", ["policy", "failed_detectors", "detectors"])
def test_a_BOOL_is_rejected_where_a_non_bool_is_expected(field):
    """Caught by the ordinary type check, since isinstance(True, str) is False.

    A separate bool guard is needed only for an int- or float-typed field,
    because bool subclasses int. No field here is numeric, so such a guard
    could never execute, and dead code implies a protection that is not
    operating. Add it with the first numeric field.
    """
    result = {"blocked": False, "transformed": False, "policy": "default"}
    result[field] = True
    assert classify_response(body(result=result)).kind == "schema_invalid"


@pytest.mark.parametrize("guard_output", [{}, {"messages": []}, {"other": 1}])
def test_a_transform_whose_output_has_NO_MESSAGES_is_unusable(guard_output):
    """An empty or shapeless guard_output cannot be written back. Accepting
    any dict would send the caller's ORIGINAL prompt while reporting that it
    had been rewritten -- the transform silently doing nothing."""
    result = {"blocked": False, "transformed": True, "policy": "default",
              "guard_output": guard_output}
    assert classify_response(body(result=result)).kind == "schema_invalid"
