"""Mode-determined failure behaviour. Task 7 of the P0 remediation plan."""

import pytest

from tidewall_otel._config import TidewallConfig


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Every TIDEWALL_ variable cleared, so a developer's shell cannot make a
    test pass or fail for reasons the test never mentions."""
    import os

    for name in list(os.environ):
        if name.startswith("TIDEWALL_"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TIDEWALL_BASE_URL", "https://guard.example")
    monkeypatch.setenv("TIDEWALL_TOKEN", "t")


@pytest.mark.parametrize("removed", [
    "TIDEWALL_ON_GUARD_FAILURE",
    "TIDEWALL_ON_UNSUPPORTED_INPUT",
])
def test_a_removed_variable_is_refused_at_startup(monkeypatch, removed):
    """Silently ignoring a variable an operator deliberately set is its own
    fail-open: they believe a policy is in force that is not.

    A clean cut rather than an alias -- both were replaced by TIDEWALL_MODE,
    and quietly mapping one onto it would pick a policy the operator did not
    choose.
    """
    monkeypatch.setenv(removed, "allow")
    with pytest.raises(ValueError, match="TIDEWALL_MODE"):
        TidewallConfig()


@pytest.mark.parametrize("mode", ["monitor", "dry-run"])
def test_block_is_invalid_in_monitor_and_dry_run(monkeypatch, mode):
    """`block` installs PERSISTENT per-call refusal into a mode whose runtime
    contract is to proceed and record. `exit` is install-time and stays valid
    everywhere."""
    monkeypatch.setenv("TIDEWALL_MODE", mode)
    monkeypatch.setenv("TIDEWALL_ON_ACTIVATION_FAILURE", "block")
    with pytest.raises(ValueError, match="block"):
        TidewallConfig()


def test_block_IS_valid_in_enforce(monkeypatch):
    monkeypatch.setenv("TIDEWALL_MODE", "enforce")
    monkeypatch.setenv("TIDEWALL_ON_ACTIVATION_FAILURE", "block")
    assert TidewallConfig().on_activation_failure == "block"


@pytest.mark.parametrize("mode", ["enforce", "monitor", "dry-run"])
@pytest.mark.parametrize("value", ["exit", "disable"])
def test_exit_and_disable_are_honoured_in_every_mode(monkeypatch, mode, value):
    monkeypatch.setenv("TIDEWALL_MODE", mode)
    monkeypatch.setenv("TIDEWALL_ON_ACTIVATION_FAILURE", value)
    assert TidewallConfig().on_activation_failure == value


def test_an_unknown_activation_failure_value_is_refused(monkeypatch):
    monkeypatch.setenv("TIDEWALL_ON_ACTIVATION_FAILURE", "shrug")
    with pytest.raises(ValueError, match="shrug"):
        TidewallConfig()


def test_an_unknown_mode_is_refused(monkeypatch):
    monkeypatch.setenv("TIDEWALL_MODE", "sometimes")
    with pytest.raises(ValueError, match="sometimes"):
        TidewallConfig()


# -- the two bounds --------------------------------------------------------

def test_the_removed_timeout_variable_fails_LOUDLY(monkeypatch):
    """Two bounds now exist -- a per-connection read and a caller-latency
    deadline -- and silently mapping one old variable onto one of them would
    pick the wrong default for the other."""
    monkeypatch.setenv("TIDEWALL_TIMEOUT", "5")
    with pytest.raises(ValueError, match="TIDEWALL_SOCKET_TIMEOUT"):
        TidewallConfig()


def test_dispatch_reads_fields_that_EXIST():
    """The cross-task check: dispatch reads config.socket_timeout and
    config.guard_deadline_s, and neither existed when they were first used."""
    config = TidewallConfig()
    for attribute in ("socket_timeout", "guard_deadline_s"):
        assert hasattr(config, attribute), f"dispatch reads config.{attribute}"
    assert not hasattr(config, "timeout"), "the old single bound must be gone"


def test_a_socket_timeout_looser_than_the_deadline_is_rejected(monkeypatch):
    """A per-connection bound wider than the caller bound cannot be honoured:
    the caller gives up first and the socket keeps waiting."""
    monkeypatch.setenv("TIDEWALL_SOCKET_TIMEOUT", "30")
    monkeypatch.setenv("TIDEWALL_GUARD_DEADLINE", "10")
    with pytest.raises(ValueError, match="socket"):
        TidewallConfig()


@pytest.mark.parametrize("name", ["TIDEWALL_SOCKET_TIMEOUT", "TIDEWALL_GUARD_DEADLINE"])
@pytest.mark.parametrize("bad", ["0", "-1", "nonsense", "inf"])
def test_a_nonpositive_or_unparseable_bound_is_refused(monkeypatch, name, bad):
    monkeypatch.setenv(name, bad)
    with pytest.raises(ValueError):
        TidewallConfig()
