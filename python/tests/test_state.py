"""State dimensions and is_active(). Task 9 of the P0 remediation plan."""

import pytest

from tidewall_otel._state import State


def test_guard_health_starts_UNKNOWN_not_ok():
    """Preflight is optional, so there is no evidence either way until the
    first call. Starting at `ok` would be a claim with nothing behind it."""
    assert State().guard_health == "unknown"


def test_is_active_is_FALSE_when_no_supported_surface_is_present():
    """The vacuous-true case. With no SDK imported, "every present surface is
    covered" is trivially satisfied -- and an agent guarding nothing would
    report itself enforcing."""
    state = State(lifecycle="installed", mode="enforce", surfaces={})
    assert state.is_active() is False


def test_is_active_is_TRUE_when_a_present_surface_is_covered():
    state = State(lifecycle="installed", mode="enforce",
                  surfaces={"openai.chat": "covered"})
    assert state.is_active() is True


@pytest.mark.parametrize("disposition", ["unverified", "uncovered", "refusing"])
def test_is_active_is_FALSE_when_any_present_surface_is_not_covered(disposition):
    """Universal, not existential: one unguarded boundary makes the whole
    claim false, because a caller cannot know which boundary they crossed."""
    state = State(lifecycle="installed", mode="enforce",
                  surfaces={"openai.chat": "covered", "anthropic.messages": disposition})
    assert state.is_active() is False


@pytest.mark.parametrize("lifecycle", ["uninstalled", "installing", "removed"])
def test_is_active_is_FALSE_outside_the_installed_lifecycle(lifecycle):
    state = State(lifecycle=lifecycle, mode="enforce",
                  surfaces={"openai.chat": "covered"})
    assert state.is_active() is False


def test_dry_run_is_never_ACTIVE_even_when_everything_is_covered():
    """dry-run skips guard calls entirely, so nothing is being enforced; the
    dimensions are independent and this is the one that decides."""
    state = State(lifecycle="installed", mode="dry-run",
                  surfaces={"openai.chat": "covered"})
    assert state.is_active() is False


def test_the_dimensions_are_INDEPENDENT():
    """Lifecycle, mode, surface disposition and guard health are four separate
    facts. Collapsing them into one boolean is what let earlier versions
    report `active` while a boundary was unguarded."""
    state = State(lifecycle="installed", mode="enforce",
                  surfaces={"openai.chat": "covered"}, guard_health="unreachable")
    assert state.lifecycle == "installed"
    assert state.mode == "enforce"
    assert state.guard_health == "unreachable"
    assert state.surfaces["openai.chat"] == "covered"


def test_guard_health_does_NOT_by_itself_decide_activity():
    """An unreachable guard is a runtime condition the mode contract handles;
    it does not retroactively mean the boundaries are unguarded."""
    state = State(lifecycle="installed", mode="enforce",
                  surfaces={"openai.chat": "covered"}, guard_health="unreachable")
    assert state.is_active() is True


def test_record_unverified_downgrades_a_surface_and_is_visible():
    state = State(lifecycle="installed", mode="enforce",
                  surfaces={"openai.chat": "covered"})
    assert state.is_active() is True

    state.record_unverified("openai.chat", reason="client_escapes",
                            detail=("middleware",))
    assert state.surfaces["openai.chat"] == "unverified"
    assert state.is_active() is False
    assert any("middleware" in str(event) for event in state.events)


def test_record_skip_is_observable():
    """Monitor mode exists to produce evidence; a skip that records nothing is
    the one call worth recording recording nothing."""
    state = State(lifecycle="installed", mode="monitor",
                  surfaces={"openai.chat": "covered"})
    state.record_skip("openai.chat", reason="lossy", detail=("messages[0].content",))
    assert any("lossy" in str(event) for event in state.events)


def test_the_summary_names_every_dimension():
    """An operator reading one line needs all four, because 'active: false'
    without a reason is not actionable."""
    state = State(lifecycle="installed", mode="enforce",
                  surfaces={"openai.chat": "unverified"}, guard_health="ok")
    summary = state.summary()
    for expected in ("installed", "enforce", "ok", "unverified", "openai.chat"):
        assert expected in summary, summary


# -- caller code must not run under the lock -------------------------------

def test_a_REENTRANT_detail_does_not_deadlock_the_recorder():
    """`detail` is `Any`, so comparing samples runs the CALLER's `__eq__`.

    Doing that inside the critical section deadlocked the thread outright
    when the method recorded anything -- a non-reentrant lock reacquired by
    the same thread. An application whose diagnostic objects touch state
    would hang on an LLM call, permanently, with no error.

    Timed rather than asserted directly: a deadlock has no exception to
    catch, so the only evidence is that the work never finishes.
    """
    import threading

    from tidewall_otel._state import State

    state = State()

    class RecordsWhileComparing:
        def __eq__(self, other):
            state.record_skip("s", reason="recorded from inside __eq__")
            return False

        def __hash__(self):
            return 0

    state.record_skip("s", reason="lossy", detail=("an earlier sample",))

    finished = threading.Event()

    def record():
        state.record_skip("s", reason="lossy", detail=RecordsWhileComparing())
        finished.set()

    worker = threading.Thread(target=record, daemon=True)
    worker.start()

    assert finished.wait(timeout=5), \
        "recording deadlocked on a caller-defined __eq__"


def test_a_SLOW_detail_comparison_does_not_block_health_publication():
    """The same fix, for the non-fatal case: caller equality that merely
    takes time must not serialise every other call's state publication."""
    import threading
    import time

    from tidewall_otel._state import State

    state = State()
    entered = threading.Event()

    class SlowEquality:
        def __eq__(self, other):
            entered.set()
            time.sleep(2)
            return False

        def __hash__(self):
            return 0

    state.record_skip("s", reason="lossy", detail=("an earlier sample",))
    threading.Thread(
        target=lambda: state.record_skip("s", reason="lossy",
                                         detail=SlowEquality()),
        daemon=True).start()

    assert entered.wait(timeout=5), "the comparison never ran"
    started = time.perf_counter()
    state.record_guard_health("clean")
    assert time.perf_counter() - started < 1.0, \
        "health publication waited on a caller's slow __eq__"
    assert state.guard_health == "ok"
