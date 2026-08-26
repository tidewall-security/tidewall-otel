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
| `TIDEWALL_USER_ID` | (none) | User identifier sent with each guard event. **No default** — the OS account name is not collected unless you set this explicitly. |
| `TIDEWALL_LOG_LEVEL` | `info` | Log verbosity |
| `TIDEWALL_SOCKET_TIMEOUT` | `10` | Per-connection read bound, seconds. Must not exceed the guard deadline. |
| `TIDEWALL_GUARD_DEADLINE` | `10` | Caller-latency bound, seconds — how long a caller waits for the guard before the mode contract applies. |
| `TIDEWALL_ON_ACTIVATION_FAILURE` | `exit` | What happens when activation fails: `exit` (raise, so the process does not continue believing it is guarded), `disable` (run unguarded, with the state saying so), or `block` (install refusers, so calls fail rather than pass unchecked). |


### Handling `TIDEWALL_TOKEN`

`TIDEWALL_TOKEN` is a bearer credential for your guard server. Anyone holding
it can submit prompts as your application and read the verdicts.

- Supply it through the environment or a secrets manager. Do not commit it,
  and do not pass it on a command line, where it is visible in the process
  table and shell history.
- `config.token` is a `Secret`, not a `str`. It renders as `Secret('***')`
  everywhere — `repr`, `str`, `dataclasses.asdict`, `vars`, copies — so
  logging or serialising the config object does not disclose it. Call
  `config.token.reveal()` to get the raw value, which is deliberately awkward
  so it cannot happen by accident.
- **`pickle` still carries the real value**, because a forked worker that
  loses its credential cannot call the guard. Do not persist a pickled config
  anywhere you would not persist the token itself.
- The redaction protects the config object, not a variable you assigned
  `reveal()` to. Error reporters that capture frame locals — Sentry and
  similar do this by default — will capture whatever you put in a local.
- The agent never writes the token to a log, and refuses to send it over a
  plaintext connection or follow a redirect that would carry it to another
  origin.

### What is refused in `enforce`

`enforce` declines any call it cannot show the guard faithfully, before
contacting either the guard or the provider. That is the point — the guard is
never asked about a body it was not shown — but it means some ordinary calls
are refused rather than guarded:

| Call shape | `enforce` | `monitor` / `dry-run` |
| --- | --- | --- |
| String message content | guarded | guarded / skipped |
| `tools` definitions | guarded | guarded / skipped |
| **Multi-part content blocks** (vision, Anthropic block lists) | **refused** | proceeds, recorded as a `lossy` skip |
| **Assistant `tool_calls`** | **refused** | proceeds, recorded as a `lossy` skip |
| `extra_body` | **refused** | proceeds, recorded as a `lossy` skip |

Multimodal calls are refused because the guard cannot read an image. Flattening
the blocks to their text and reporting the call covered would claim an
inspection that never happened.

`state()` reports every one of these: the surface is not `covered`, the skip
carries its reason, and `is_active()` is False. If your application makes these
calls and you need it running today, `monitor` observes without blocking — and
tells you plainly which calls it could not check.

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

   Two cases never reach step 3 at all, and both raise
   `tidewall_otel.TidewallRefusedError` in `enforce` mode before either the
   guard or the provider is contacted:

   - the call carries something the agent cannot represent losslessly for
     the guard — `extra_body`, or an argument shape outside the manifest.
     The guard is not asked about a body it was not shown.
   - the guard could not be reached, timed out, or answered with something
     that does not match its response schema.

   Every exception above is a `tidewall_otel.TidewallError`, so one
   `except tidewall_otel.TidewallError` covers every way the agent can
   decline a call.
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
