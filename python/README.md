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
| **A payload too deep or too large to verify** | **refused** (`unverifiable_payload`) | proceeds, recorded as `unverified` |
| **A request mutated during the guard call** | **refused** (`mutated_during_guard`) | proceeds, recorded as `unverified` |

Multimodal calls are refused because the guard cannot read an image. Flattening
the blocks to their text and reporting the call covered would claim an
inspection that never happened.

The last two rows are about time rather than shape. The guard is asked about a
snapshot and the provider is invoked with your own mutable arguments
afterwards, so anything running in between can swap the content. `enforce`
compares a fingerprint across the guard call and declines when the two no
longer match — and equally when the payload was too deeply nested or too large
to fingerprint completely, because a snapshot that omits content cannot testify
that the content did not change. Both raise `TidewallRefusedError`, whose
`outcome_kind` carries the reason above.

### What the mutation check does and does not claim

The guard is asked about a snapshot and the provider is invoked with your own
arguments afterwards, so Tidewall fingerprints every argument before
inspection and compares it after. That closes the cases that matter in a
cooperative application: a callback, a re-entrant guard, an OTel span
processor, or a container whose `__eq__` or `get()` reports something other
than what it stores. Between the final comparison and the SDK serialising the
request, Tidewall runs only its own code — no application callback is invoked
in that window.

It does not claim to defeat an attacker who is already executing arbitrary
code in your process. Another thread can mutate a shared object in that last
window, and nothing an in-process agent does can prevent it: code that can do
that can equally re-patch the SDK, patch Tidewall, or call the provider
directly. Closing it would mean sending the provider a rebuilt payload rather
than your own arguments, which would drop streaming and stream options,
sampling and token controls, `stop`, `seed`, `logprobs`, `response_format`,
`tool_choice`, parallel tool calls, metadata, service tier, per-request
timeouts and extra headers — and, for Anthropic, the required `max_tokens`
along with thinking, output config and cache controls. That is a worse
product for a threat this agent is not the right layer to address.

The threat Tidewall exists to address is content reaching the model that
should not — prompt injection, sensitive data, policy violations — in an
application that is not itself hostile.

### When the guard itself fails

The rows above are about calls the agent will not show the guard. This is the
other direction: the guard was asked and did not answer usefully. In `enforce`
each raises `TidewallRefusedError` with the `outcome_kind` named here, and the
provider is never contacted. In `monitor` and `dry-run` the call proceeds and
the reason is recorded instead.

| `outcome_kind` | Meaning |
| --- | --- |
| `blocked` | The guard answered, and its verdict was no. This is the product working, not failing — it raises `TidewallBlockedError`. |
| `unreachable` | The guard could not be contacted. |
| `timeout` | The guard did not answer within `TIDEWALL_GUARD_DEADLINE`. |
| `saturated` | Too many guard calls were already in flight; the bounded pool refused another rather than growing without limit. |
| `schema_invalid` | The guard answered with something this agent cannot parse, so it will not guess what the verdict was. |
| `invariant_violated` | An unmapped error inside the agent. Fail-closed by construction: an exception nothing anticipated refuses rather than passing the call through. |

Choosing `enforce` means accepting that an outage of the guard is an outage of
the calls it guards. `monitor` is the setting that does not make that trade.

The two tables report differently, because they are about different things.

A call the agent **cannot represent** is about a boundary: `state()` shows the
surface as something other than `covered`, the skip carries its reason, and
`is_active()` is False. If your application makes these calls and you need it
running today, `monitor` observes without blocking — and tells you plainly
which calls it could not check.

A **guard failure** is about the guard, not the boundary, and appears in
`state().guard_health` — `ok`, `degraded`, or the failing kind itself, so
`saturated` (this agent's own pool declining work) is not reported as
`unreachable` (the guard being unreachable). It deliberately does **not** flip
`is_active()`, which is a claim about whether every surface present is covered;
an unreachable guard is a runtime condition the mode contract handles per call,
and it does not retroactively mean the boundaries are unwrapped. Poll both:
`is_active()` for wiring, `guard_health` for the guard.

Health is a single current value rather than a log, so a sustained outage does
not grow the process's memory. Individual failures raise, and their spans carry
the per-call detail.

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

   Other cases raise `tidewall_otel.TidewallRefusedError` in `enforce` mode.
   In all of them the AI provider is never contacted, but they happen at
   different points, which matters when you are diagnosing one:

   - **Before the guard is asked**: the call carries something the agent
     cannot represent losslessly — `extra_body`, or an argument shape outside
     the manifest. The guard is not asked about a body it was not shown, so
     no guard request is made at all.
   - **While asking**: the guard could not be reached, or did not answer
     within `TIDEWALL_GUARD_DEADLINE`. A request was attempted, so this
     traffic is visible to the guard's own network.
   - **After it answers**: the guard replied with something that does not
     match its response schema. A response was received and rejected.
   - **After it answers, on the way to the provider**: the request changed
     while the guard was inspecting it, or was too deep or too large to
     fingerprint completely. See the refusal table above.

   Every exception above is a `tidewall_otel.TidewallError`, so one
   `except tidewall_otel.TidewallError` covers every way the agent can
   decline a call — and every one carries `outcome_kind`, so that single
   clause can branch on the reason without catching subclasses individually:

   ```python
   try:
       response = client.chat.completions.create(...)
   except tidewall_otel.TidewallError as declined:
       if declined.outcome_kind == "blocked":
           return "That request was blocked by policy."
       raise                      # a guard failure is not a policy decision
   ```

   The classes are `TidewallBlockedError` (`blocked`), `LossyInputError`
   (`lossy`, a subclass of `TidewallRefusedError`), `TidewallRefusedError`
   (every other refusal above), and `TidewallConfigError`
   (`config_invalid`), which is raised at ACTIVATION rather than per call —
   by default, since `TIDEWALL_ON_ACTIVATION_FAILURE` is `exit`.
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
- Function calling has two halves, and they are treated differently. **Tool
  definitions** — the `tools` you send — are shown to the guard and guarded
  like any other input. **Assistant `tool_calls`** — the model's replies
  carried back in a later request — are not representable to the guard, so
  `enforce` refuses them and `monitor` proceeds and records a `lossy` skip.
  An agent loop that feeds tool results back therefore needs `monitor` today.
  The guard also does not inspect tool RESULTS as a separate surface.
- This is alpha-quality software; APIs may change before 1.0.

## License

Apache License 2.0. See [`../LICENSE`](../LICENSE) and [`../NOTICE`](../NOTICE).
