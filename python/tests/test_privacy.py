"""Privacy and identity defaults. Task 15 of the P0 remediation plan."""

import pytest

from tidewall_otel._config import TidewallConfig
from tidewall_otel._manifest import SURFACES
from tidewall_otel._span_helper import gen_ai_span

CONTENT_ATTRS = ("gen_ai.input.messages", "gen_ai.output.messages")


class RecordingSpan:
    def __init__(self):
        self.attributes = {}

    def set_attribute(self, key, value):
        self.attributes[key] = value


@pytest.fixture
def span(monkeypatch):
    recorded = RecordingSpan()
    import tidewall_otel._span_helper as helper

    monkeypatch.setattr(helper, "_start_span", lambda *a, **k: recorded, raising=False)
    return recorded


@pytest.mark.parametrize("surface", SURFACES, ids=lambda s: s.attribute)
def test_span_content_is_OFF_by_default_for_every_manifest_entry(surface):
    """Declared per entry, so no surface can quietly default to on."""
    assert surface.span_input is False
    assert surface.span_output is False


def test_content_attributes_are_ABSENT_not_blank(span):
    """Absent, not empty. A blank attribute still creates the key, and a
    downstream pipeline that treats presence as consent would export it --
    and an operator auditing for the key would find it."""
    with gen_ai_span(provider="openai", model="gpt-4o",
                     guard_input={"messages": [{"role": "user", "content": "SECRET"}]}):
        pass

    for attribute in CONTENT_ATTRS:
        assert attribute not in span.attributes, (
            f"{attribute} was set without span content being enabled"
        )
    assert not any("SECRET" in str(v) for v in span.attributes.values())


def test_the_user_id_is_OMITTED_when_USER_is_unset(monkeypatch):
    """Omitted, not sent empty or null: the OS account name is not ours to
    disclose, and `""` still asserts a field the operator never set."""
    monkeypatch.delenv("USER", raising=False)
    monkeypatch.delenv("TIDEWALL_USER_ID", raising=False)
    monkeypatch.setenv("TIDEWALL_BASE_URL", "https://g.example")
    monkeypatch.setenv("TIDEWALL_TOKEN", "t")

    from tidewall_otel._config import TidewallConfig
    from tidewall_otel._guard import TidewallGuard

    payload = TidewallGuard(TidewallConfig())._payload_for({"messages": []})

    assert "user_id" not in payload or payload["user_id"], "empty user_id sent"
    assert "user_name" not in payload.get("extra_info", {}) or \
        payload["extra_info"]["user_name"], "empty user_name sent"


def test_the_OS_account_name_is_not_sent_by_default(monkeypatch):
    """The host user is disclosed twice by default today. Sending it must be
    an explicit choice, not a fallback."""
    monkeypatch.setenv("USER", "jane.doe")
    monkeypatch.delenv("TIDEWALL_USER_ID", raising=False)
    monkeypatch.setenv("TIDEWALL_BASE_URL", "https://g.example")
    monkeypatch.setenv("TIDEWALL_TOKEN", "t")

    from tidewall_otel._config import TidewallConfig
    from tidewall_otel._guard import TidewallGuard

    payload = TidewallGuard(TidewallConfig())._payload_for({"messages": []})
    assert "jane.doe" not in str(payload), (
        "the OS account name reached the guard payload without being configured"
    )


def test_an_EXPLICIT_user_id_is_still_sent(monkeypatch):
    """Privacy by default, not privacy by removal: an operator who sets it
    means it."""
    monkeypatch.setenv("TIDEWALL_USER_ID", "service-account-7")
    monkeypatch.setenv("TIDEWALL_BASE_URL", "https://g.example")
    monkeypatch.setenv("TIDEWALL_TOKEN", "t")

    from tidewall_otel._config import TidewallConfig
    from tidewall_otel._guard import TidewallGuard

    payload = TidewallGuard(TidewallConfig())._payload_for({"messages": []})
    assert payload["user_id"] == "service-account-7"


def test_the_TOKEN_is_never_in_the_config_repr(monkeypatch):
    """The bearer credential must not travel in a repr.

    `TidewallConfig` is public API, so its repr goes wherever the application
    puts it: `logger.info("config: %s", config)`, a crash handler dumping
    locals, or an error reporter capturing frame locals -- Sentry does that by
    default. The plain dataclass repr printed
    `token='...'` in cleartext.

    P0-3 protected this token IN TRANSIT: https enforced, redirects refused,
    no plaintext scheme. Nothing protected its REPRESENTATION, and they are
    the same asset -- an attacker reading it from a log has it just as
    completely as one reading it off the wire.
    """
    monkeypatch.setenv("TIDEWALL_BASE_URL", "https://guard.example")
    monkeypatch.setenv("TIDEWALL_TOKEN", "sk-SUPERSECRET-do-not-print")

    config = TidewallConfig()

    assert config.token == "sk-SUPERSECRET-do-not-print", "the value is still usable"
    assert "SUPERSECRET" not in repr(config), repr(config)
    assert "SUPERSECRET" not in str(config), str(config)
    assert "SUPERSECRET" not in f"{config}"
    assert "SUPERSECRET" not in "{}".format(config)          # noqa: UP032


def test_the_token_is_absent_from_EVERY_public_rendering(monkeypatch):
    """Not just repr: anything that stringifies the object by any route.

    Named separately because "not in repr" is a weaker claim than the one
    that matters, and this programme has repeatedly shipped the weaker one.
    """
    import pprint

    monkeypatch.setenv("TIDEWALL_BASE_URL", "https://guard.example")
    monkeypatch.setenv("TIDEWALL_TOKEN", "sk-SUPERSECRET-do-not-print")

    config = TidewallConfig()
    renderings = {
        "repr": repr(config),
        "str": str(config),
        "pprint": pprint.pformat(config),
        "list-repr": repr([config]),
        "dict-repr": repr({"config": config}),
        "exception": repr(ValueError(config)),
    }
    leaked = {name: text for name, text in renderings.items()
              if "SUPERSECRET" in text}
    assert not leaked, f"the token leaked through: {sorted(leaked)}"
