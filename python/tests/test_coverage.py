"""Dispatch-time coverage. Task 8 of the P0 remediation plan."""

import pytest

from tidewall_otel._bound import bound_nodes
from tidewall_otel._coverage import Coverage, canonicalise, classify_input
from tidewall_otel._manifest import OPENAI_CHAT_SYNC


def fake_create(self, *, messages, model, **kwargs):
    """Signature-faithful stand-in with a var-keyword, so the unknown-field
    branch is reachable at all -- neither real SDK method has one."""


def nodes_for(kwargs):
    return bound_nodes(fake_create, (), kwargs,
                       stop=OPENAI_CHAT_SYNC.opaque_subtrees
                            + OPENAI_CHAT_SYNC.non_prompt_bearing)


# -- canonicalisation ------------------------------------------------------

def test_object_KEY_ORDER_is_not_significant():
    assert canonicalise({"a": 1, "b": 2}) == canonicalise({"b": 2, "a": 1})


def test_array_ORDER_IS_significant():
    """Reordering messages changes who said what."""
    assert canonicalise([1, 2]) != canonicalise([2, 1])


def test_a_string_is_compared_exactly_including_whitespace():
    assert canonicalise("a b") != canonicalise("a  b")
    assert canonicalise(" a") != canonicalise("a")


def test_one_and_one_point_zero_and_the_string_one_are_THREE_nodes():
    """Type is part of identity: a guard that treats them alike cannot tell a
    numeric field from a string carrying digits."""
    assert canonicalise(1) != canonicalise(1.0)
    assert canonicalise(1) != canonicalise("1")
    assert canonicalise(1.0) != canonicalise("1")


def test_true_is_not_one():
    """Python makes True == 1; canonicalisation must not inherit that."""
    assert canonicalise(True) != canonicalise(1)


def test_null_is_its_own_node():
    assert canonicalise(None) != canonicalise("")
    assert canonicalise(None) != canonicalise(False)


def test_binary_is_compared_by_its_REFERENCE_FORM_not_decoded_content():
    """Decoding to compare would mean holding attacker-controlled binary in
    memory and normalising it -- the comparison is over the reference."""
    assert canonicalise(b"\x00\x01") == canonicalise(b"\x00\x01")
    assert canonicalise(b"\x00\x01") != canonicalise(b"\x00\x02")
    assert canonicalise(b"1") != canonicalise("1")


def test_nested_structures_compare_recursively():
    left = {"m": [{"role": "user", "content": "hi"}]}
    right = {"m": [{"content": "hi", "role": "user"}]}
    assert canonicalise(left) == canonicalise(right)

    changed = {"m": [{"role": "user", "content": "HI"}]}
    assert canonicalise(left) != canonicalise(changed)


# -- classification --------------------------------------------------------

def test_a_clean_call_is_covered_with_no_lossy_paths():
    coverage = classify_input(OPENAI_CHAT_SYNC, nodes_for(
        {"messages": [{"role": "user", "content": "hi"}], "model": "gpt-4o"}))
    assert isinstance(coverage, Coverage)
    assert coverage.lossy_paths == ()
    assert coverage.is_lossless


def test_an_unlisted_kwarg_is_lossy():
    """The fail-closed default: a new prompt-bearing field in neither list is
    unknown, and unknown is lossy."""
    coverage = classify_input(OPENAI_CHAT_SYNC, nodes_for(
        {"messages": [], "model": "gpt-4o", "brand_new_field": "x"}))
    assert "brand_new_field" in coverage.lossy_paths
    assert not coverage.is_lossless


def test_a_multipart_image_is_lossy_because_the_normalizer_drops_it():
    """This refuses calls that work today. That is the point: they work by
    sending uninspected content while the agent reports enforcement."""
    coverage = classify_input(OPENAI_CHAT_SYNC, nodes_for({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "what is this"},
            {"type": "image_url", "image_url": {"url": "data:..."}},
        ]}]}))
    assert "messages[0].content" in coverage.lossy_paths


def test_classify_input_DELEGATES_to_the_manifest_predicate(monkeypatch):
    """One definition of lossy. If dispatch grew its own copy the two could
    disagree and the manifest's declarations would stop governing anything --
    so this asserts the DELEGATION, not merely an equal answer."""
    import tidewall_otel._coverage as coverage_module

    seen = []
    monkeypatch.setattr(
        coverage_module, "lossy_paths",
        lambda surface, nodes: seen.append((surface, nodes)) or ("sentinel",),
    )
    result = classify_input(OPENAI_CHAT_SYNC, nodes_for(
        {"messages": [], "model": "gpt-4o"}))

    assert seen, "classify_input did not consult the manifest"
    assert result.lossy_paths == ("sentinel",)


def test_coverage_carries_the_surface_it_classified():
    """Dispatch needs to know which contract produced the verdict, not just
    the verdict."""
    coverage = classify_input(OPENAI_CHAT_SYNC, nodes_for(
        {"messages": [], "model": "gpt-4o"}))
    assert coverage.surface is OPENAI_CHAT_SYNC


@pytest.mark.parametrize("value,tag", [
    (1, "int"), (1.0, "float"), ("1", "string"), (True, "bool"),
    (None, "null"), (b"1", "binary"), ([], "array"), ({}, "object"),
])
def test_the_canonical_form_CARRIES_ITS_TYPE(value, tag):
    """Type is part of identity, asserted on the representation itself.

    Comparing pairs is not enough: dropping the tag from ONE scalar still
    leaves it distinguishable from the others that kept theirs, so a
    pairwise test passes while the invariant is broken. Dropping them all
    would silently merge 1, 1.0 and True, since Python considers them equal.
    """
    canonical = canonicalise(value)
    assert isinstance(canonical, tuple) and canonical[0] == tag, canonical


def test_untagged_scalars_would_COLLIDE():
    """Why the tag exists at all, demonstrated rather than asserted: without
    it, Python's own equality merges three distinct nodes."""
    assert 1 == 1.0 == True                     # noqa: E712  (the point)
    assert len({1, 1.0, True}) == 1             # they collapse to one
    assert len({canonicalise(1), canonicalise(1.0), canonicalise(True)}) == 3
