"""Spans, driven through real activated clients against a real tracer.

Seventh-pass finding: `gen_ai_span` and `record_response_in_span` were
defined, documented and unit-tested while NO production path called either,
so a fully guarded call emitted nothing at all. In a package named
`tidewall-otel`, whose instrumentor docstring promised spans "regardless of
mode".

Third instance of that shape in this codebase -- dispatch in round 1, the
late-import finder in round 13, and the spans here. Every span test called
the helper directly, so sixteen adversarial rounds went past it.
"""

import json

import httpx
import openai
import pytest

import tidewall_otel
import tidewall_otel._guard as guard_module

pytest.importorskip("opentelemetry.sdk")

from opentelemetry import trace                                   # noqa: E402
from opentelemetry.sdk.trace import TracerProvider                # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor    # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)

_OK = {"id": "x", "object": "chat.completion", "created": 0, "model": "gpt-4o",
       "choices": [{"index": 0, "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "ok"}}]}
_CLEAN = {"result": {"blocked": False, "transformed": False, "policy": "p"}}
_BLOCKED = {"result": {"blocked": True, "transformed": False, "policy": "p"}}


@pytest.fixture
def exporter(monkeypatch):
    """A private provider, injected past OTel's global resolution.

    The global provider is set exactly once per process and a `ProxyTracer`
    handed out before that binds to whatever was current -- so any earlier
    test that asked for a tracer (`test_privacy` calls `gen_ai_span`
    directly) pins the NoOp one, and every span here vanishes. That reads as
    "no span was emitted", which is the opposite of the truth and precisely
    the confusion this file exists to prevent.

    Patching `_get_tracer` is deliberately narrower than the real resolution
    path: what is under test is that DISPATCH calls the helper at every exit
    and records the right attributes, not OTel's own global lookup. The
    packaging tests cover the real entry point end to end.
    """
    from tidewall_otel import _span_helper

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("tidewall.test")

    monkeypatch.setattr(_span_helper, "_get_tracer", lambda: tracer)

    # Clean BEFORE as well as after. `activate()` returns early when the
    # lifecycle is already `installed`, so a test that inherits another's
    # activation never re-reads config -- it runs in whatever mode the
    # previous test left behind, and asserts against a world it did not set
    # up. Cleaning only on teardown fixed the next test and not this one.
    tidewall_otel.deactivate()
    tidewall_otel._residual_managers.clear()
    tidewall_otel._permanent_residuals.clear()
    tidewall_otel._instrumentor_instance = None

    yield exporter

    tidewall_otel.deactivate()
    tidewall_otel._residual_managers.clear()
    tidewall_otel._permanent_residuals.clear()
    tidewall_otel._instrumentor_instance = None


@pytest.fixture
def guard_says(monkeypatch):
    def install(response):
        monkeypatch.setattr(guard_module, "post_guard", lambda **kw: response)
    return install


def _client(reached=None):
    def handler(request):
        if reached is not None:
            reached.append(json.loads(request.content))
        return httpx.Response(200, json=_OK)

    return openai.OpenAI(api_key="t", http_client=httpx.Client(
        transport=httpx.MockTransport(handler)))


def _env(monkeypatch, mode="enforce"):
    monkeypatch.setenv("TIDEWALL_BASE_URL", "https://guard.example")
    monkeypatch.setenv("TIDEWALL_TOKEN", "t")
    monkeypatch.setenv("TIDEWALL_MODE", mode)


def test_a_CLEAN_call_emits_a_gen_ai_span(monkeypatch, exporter, guard_says):
    _env(monkeypatch)
    guard_says(_CLEAN)

    tidewall_otel.activate()
    _client().chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "hi"}])

    spans = exporter.get_finished_spans()
    assert len(spans) == 1, f"expected one span, got {[s.name for s in spans]}"
    span = spans[0]
    assert span.name == "gen_ai.chat"
    assert span.attributes["gen_ai.request.model"] == "gpt-4o"
    assert span.attributes["gen_ai.system"] == "openai"
    assert span.attributes["tidewall.guard.outcome"] == "clean"


def test_a_BLOCKED_call_still_emits_a_span_marked_as_blocked(
        monkeypatch, exporter, guard_says):
    """The exit an operator most needs a trace for. The provider is never
    called, so without this the event leaves no evidence at all."""
    _env(monkeypatch)
    guard_says(_BLOCKED)
    reached = []

    tidewall_otel.activate()
    with pytest.raises(tidewall_otel.TidewallBlockedError):
        _client(reached).chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "attack"}])

    assert reached == [], "precondition: the provider was not called"
    spans = exporter.get_finished_spans()
    assert len(spans) == 1, "a blocked call emitted no span"
    assert spans[0].attributes["tidewall.refused"] == "blocked"
    assert spans[0].attributes["gen_ai.response.finish_reasons"] == (
        "tidewall_content_filter",)


def test_a_REFUSED_lossy_call_emits_a_span(monkeypatch, exporter, guard_says):
    """`extra_body` is refused before the guard or the provider is contacted;
    that exit needs evidence too."""
    _env(monkeypatch)
    guard_says(_CLEAN)

    tidewall_otel.activate()
    with pytest.raises(tidewall_otel.LossyInputError):
        _client().chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "hi"}],
            extra_body={"messages": [{"role": "user", "content": "EVIL"}]})

    spans = exporter.get_finished_spans()
    assert len(spans) == 1, "a refused call emitted no span"
    assert spans[0].attributes["tidewall.refused"] == "lossy_input"


def test_DRY_RUN_emits_a_span_without_contacting_the_guard(
        monkeypatch, exporter, guard_says):
    """The docstring promises spans REGARDLESS of mode, and dry-run is the
    mode that never calls the guard."""
    _env(monkeypatch, mode="dry-run")
    asked = []
    monkeypatch.setattr(guard_module, "post_guard",
                        lambda **kw: asked.append(kw) or _CLEAN)

    tidewall_otel.activate()
    _client().chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "hi"}])

    assert asked == [], "dry-run contacted the guard"
    spans = exporter.get_finished_spans()
    assert len(spans) == 1, "dry-run emitted no span"
    assert "tidewall.guard.skipped" in spans[0].attributes


def test_the_ASYNC_path_emits_a_span_too(monkeypatch, exporter, guard_says):
    import asyncio

    _env(monkeypatch)
    guard_says(_CLEAN)

    tidewall_otel.activate()

    async def drive():
        client = openai.AsyncOpenAI(api_key="t", http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, json=_OK))))
        return await client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "hi"}])

    asyncio.run(drive())

    spans = exporter.get_finished_spans()
    assert len(spans) == 1, "the async path emitted no span"
    assert spans[0].name == "gen_ai.chat"


def test_prompt_CONTENT_is_absent_from_the_span_by_default(
        monkeypatch, exporter, guard_says):
    """P0-1, asserted where it can now actually be violated.

    Wiring the span helper is what makes this reachable at all: before it, no
    span existed to leak into. The helper serialised `messages`
    unconditionally, so wiring it as written would have reintroduced the very
    first P0 of this programme.
    """
    _env(monkeypatch)
    guard_says(_CLEAN)

    tidewall_otel.activate()
    _client().chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "sk-SUPERSECRET-in-a-prompt"}])

    span = exporter.get_finished_spans()[0]
    rendered = json.dumps(dict(span.attributes))
    assert "SUPERSECRET" not in rendered, (
        f"the prompt reached the span: {rendered}")
    assert "gen_ai.input.messages" not in span.attributes
    assert "gen_ai.output.messages" not in span.attributes
