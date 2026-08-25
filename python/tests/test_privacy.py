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

    # ABSENT, not "absent or truthy". The previous form was
    #     "user_id" not in payload or payload["user_id"]
    # which passes for ANY non-empty value, so it accepted a leaked identity:
    # defaulting `user_id` to "leaked-default" left this test green. It guarded
    # only against sending an empty string, while its name and docstring
    # promise the field is not sent at all.
    assert "user_id" not in payload, f"user_id was sent: {payload.get('user_id')!r}"
    extra_info = payload.get("extra_info", {})
    assert "user_name" not in extra_info, (
        f"extra_info.user_name was sent: {extra_info.get('user_name')!r}")


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
    """Every route that RENDERS the config, not every route named `repr`.

    The first version of this test called itself "EVERY public rendering" and
    checked six repr-derived forms. `dataclasses.asdict`, `vars`,
    `copy(config).__dict__` and `pickle` all still carried the token --
    `field(repr=False)` protects exactly one route. `asdict` is how structured
    logging normally serialises a config object, so the redaction covered the
    careful case and missed the common one.

    The sixth quantifier defect of this session, in the test written to avoid
    quantifier defects. Redacting at the VALUE rather than at the field is
    what makes the general claim true.
    """
    import copy
    import dataclasses
    import pprint

    monkeypatch.setenv("TIDEWALL_BASE_URL", "https://guard.example")
    monkeypatch.setenv("TIDEWALL_TOKEN", "sk-SUPERSECRET-do-not-print")

    config = TidewallConfig()

    renderings = {
        "repr": repr(config),
        "str": str(config),
        "f-string": f"{config}",
        "format": "{}".format(config),                        # noqa: UP032
        "pprint": pprint.pformat(config),
        "list-repr": repr([config]),
        "dict-repr": repr({"config": config}),
        "exception": repr(ValueError(config)),
        "asdict": repr(dataclasses.asdict(config)),
        "vars": repr(vars(config)),
        "instance-dict": repr(config.__dict__),
        "copy-dict": repr(copy.copy(config).__dict__),
        "deepcopy-dict": repr(copy.deepcopy(config).__dict__),
        "token-repr": repr(config.token),
        "token-str": str(config.token),
        # NON-EMPTY format specs, on the field and on the container. The
        # previous version used only `f"{config}"` and `"{}".format(config)`,
        # both of which are the EMPTY spec -- so a `__format__` honouring a
        # spec by returning the raw value would have passed a test calling
        # itself EVERY rendering. Seventh quantifier defect of the session.
        "token-format-align": format(config.token, ">40"),
        "token-format-width": format(config.token, "50"),
        "token-format-fill": format(config.token, "*^60"),
        "token-fstring-spec": f"{config.token:>40}",
        "token-fstring-str": f"{config.token!s}",
        "token-fstring-repr": f"{config.token!r}",
    }
    leaked = sorted(name for name, text in renderings.items()
                    if "SUPERSECRET" in text)
    assert not leaked, f"the token leaked through: {leaked}"

    # Routes that RAISE are covered too, because an exception is a rendering:
    # its message and traceback are exactly what gets logged. `TidewallConfig`
    # has no `__format__`, so a non-empty spec raises -- standard behaviour for
    # any object, and not a leak, but only if the message stays clean.
    for spec in (">200", "50", "*^60"):
        for subject, label in ((config, "config"), (config.token, "token")):
            try:
                rendered = format(subject, spec)
            except Exception as exc:              # noqa: BLE001 -- the point
                rendered = f"{type(exc).__name__}: {exc}"
            assert "SUPERSECRET" not in rendered, (
                f"{label} leaked through format(..., {spec!r}): {rendered}")

    assert config.token.reveal() == "sk-SUPERSECRET-do-not-print", (
        "the value must still be retrievable by name")


def test_the_token_DOES_survive_pickle_and_that_is_documented(monkeypatch):
    """The honest exception, asserted so it cannot become a surprise.

    `pickle` must carry the real bytes -- a forked worker that loses its
    credential cannot call the guard. This is not a rendering; nothing
    displays it. Pinning it here means the boundary is stated rather than
    assumed, and a future change that silently starts redacting pickle (and
    so breaks multiprocessing) fails loudly instead.
    """
    import pickle

    monkeypatch.setenv("TIDEWALL_BASE_URL", "https://guard.example")
    monkeypatch.setenv("TIDEWALL_TOKEN", "sk-SUPERSECRET-do-not-print")

    config = TidewallConfig()
    restored = pickle.loads(pickle.dumps(config))

    assert restored.token.reveal() == "sk-SUPERSECRET-do-not-print"
    assert "SUPERSECRET" not in repr(restored), (
        "a round-tripped config must redact exactly as the original does")


def test_a_string_token_is_COERCED_so_the_obvious_call_still_works(monkeypatch):
    """`TidewallConfig(token="...")` is what anyone would write."""
    monkeypatch.delenv("TIDEWALL_TOKEN", raising=False)

    config = TidewallConfig(base_url="https://guard.example", token="plain-string")

    assert config.token.reveal() == "plain-string"
    assert "plain-string" not in repr(config)
    assert config.token, "a non-empty token must be truthy"
    assert not TidewallConfig(base_url="https://g.example", token="").token
