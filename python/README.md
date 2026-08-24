# tidewall-otel (Python)

Python implementation of the Tidewall OTel instrumentation agent.

This package patches the OpenAI and Anthropic Python SDKs at runtime,
inserting a Tidewall guard call before each chat-completion request
reaches the provider. Patching is done with [`wrapt`][wrapt], the same
library used by Datadog, New Relic, Elastic, and OpenTelemetry's own
contrib instrumentations — so it interoperates cleanly with other
OTel agents.

[wrapt]: https://pypi.org/project/wrapt/

## Installation

```bash
pip install tidewall-otel
```

Optional extras pull in the SDKs you actually use:

```bash
# Just OTel + the Tidewall agent
pip install "tidewall-otel[otel]"

# Plus an AI SDK you want to instrument
pip install "tidewall-otel[otel,openai]"
pip install "tidewall-otel[otel,anthropic]"

# Everything
pip install "tidewall-otel[all]"
```

The base install only depends on `wrapt`. OpenTelemetry, OpenAI, and
Anthropic are optional — the agent detects what's installed at runtime
and patches only what it can find.

## Configuration

Configure via environment variables (recommended) or by passing a
`TidewallConfig` instance to `activate()`.

| Variable | Default | Description |
| --- | --- | --- |
| `TIDEWALL_BASE_URL` | (required) | Tidewall guard API base URL. **Must be `https://`** — the bearer token and the prompt travel in this request, so plaintext is refused before a connection is opened, with no loopback exception. |
| `TIDEWALL_TOKEN` | (required) | API token for the guard server |
| `TIDEWALL_MODE` | `enforce` | `enforce`, `monitor`, or `dry-run` |
| `TIDEWALL_APP_ID` | `tidewall-otel` | App identifier recorded with each guard event |
| `TIDEWALL_APP_NAME` | `Tidewall OTel Instrumentation` | Display name |
| `TIDEWALL_USER_ID` | `$USER` | User identifier |
| `TIDEWALL_LOG_LEVEL` | `info` | Log verbosity |
| `TIDEWALL_TIMEOUT` | `10` | Per-request timeout (seconds) |

## Activation

There are three ways to turn the agent on. Pick whichever fits your
deployment model — the runtime behaviour is identical.

### 1. CLI wrapper

The simplest option for ops-driven deployments. Wraps any Python
command with instrumentation pre-activated:

```bash
tidewall-instrument python my_app.py
tidewall-instrument python -m uvicorn main:app
tidewall-instrument gunicorn app:app
```

### 2. OpenTelemetry auto-instrumentation

If you already use `opentelemetry-instrument`, Tidewall is discovered
through the standard entry-point mechanism — no extra steps:

```bash
opentelemetry-instrument python my_app.py
```

### 3. Direct import (developer opts in)

```python
import tidewall_otel
tidewall_otel.activate()

# Now any openai / anthropic usage in this process is guarded.
from openai import OpenAI
client = OpenAI()
client.chat.completions.create(...)
```

## How It Works

For each instrumented call, the agent:

1. Reads the prompt and metadata from the SDK call's kwargs.
2. Normalizes them to OpenAI Chat Completions shape (handles Anthropic's
   separate `system` kwarg and content blocks).
3. POSTs to `{TIDEWALL_BASE_URL}/v1/guard_chat_completions` with a
   bearer token.
4. Applies the verdict:
   - `blocked` → raises `tidewall_otel.TidewallBlockedError`. The AI
     provider is never called.
   - `transformed` → swaps the messages with the guard's redacted
     version before passing them to the provider.
   - clean → no change.
5. Opens a `gen_ai.chat` OTel span with `gen_ai.*` attributes, runs
   the (possibly transformed) call inside the span, and records the
   response.

## Examples

The [`examples/`](./examples) directory contains runnable scripts:

- `plain_openai_app.py` — the canonical zero-code-change demo. The script
  has no Tidewall imports; protection is injected by `tidewall-instrument`.
- `openai_example.py` — explicit `activate()` from inside the application.
- `anthropic_example.py` — same as above, for the Anthropic SDK.

Each example sends three prompts (clean, PII, prompt-injection) so you
can see all three decision paths in a single run.

## Modes

| Mode | When to use |
| --- | --- |
| `enforce` | Production. The guard verdict drives blocking and redaction. |
| `monitor` | Onboarding. Detect what _would_ be blocked without affecting users. |
| `dry-run` | Smoke testing the integration without a live Tidewall backend. |

## Limitations

- Streaming responses are not currently inspected by the agent — input
  guarding still applies, but per-chunk output guarding is on the
  roadmap.
- Tool calls (function calling) are passed through unmodified for now.
- This is alpha-quality software; APIs may change before 1.0.

## License

Apache License 2.0. See [`../LICENSE`](../LICENSE) and [`../NOTICE`](../NOTICE).
