# Tidewall OTel

Zero-code-change AI security instrumentation for AI SDK calls.

Tidewall OTel automatically intercepts calls to AI provider SDKs (OpenAI,
Anthropic, more to come) and routes them through a Tidewall guard server
for prompt-injection detection, PII redaction, and policy enforcement —
without any application code changes.

It plugs into the OpenTelemetry instrumentation framework, so a single
agent gives you both runtime security AND standard `gen_ai.*` observability
spans that flow into any OTel-compatible backend (Elastic APM, Tempo,
Jaeger, Datadog, ...).

## Languages

| Language | Status | Path |
| --- | --- | --- |
| Python | Available | [`python/`](./python) |
| Node.js | Planned | — |
| Java | Planned | — |
| .NET | Planned | — |
| Go | Planned | — |

This is a mono-repo: each language sits in its own subdirectory with
its own packaging, but the architecture, configuration model, and
behaviours are kept consistent across implementations.

## What It Does

For each instrumented AI SDK call, Tidewall OTel:

1. Extracts the prompt and metadata from the call arguments.
2. Sends a guard-evaluation request to the configured Tidewall server.
3. Applies the verdict inline:
   - `block` — raises a typed exception, the AI provider is never called.
   - `transform` — replaces the messages with a redacted version, then
     forwards the redacted call to the provider.
   - `allow` — passes through unchanged.
4. Emits an OpenTelemetry `gen_ai.chat` span with standard semantic
   convention attributes (`gen_ai.system`, `gen_ai.request.model`,
   `gen_ai.input.messages`, etc.).

The instrumentation degrades gracefully:

- If the guard server is unreachable, the original call proceeds (fail-open).
- If OpenTelemetry is not installed, span creation is a no-op.
- If neither OpenAI nor Anthropic is installed, the agent loads but patches nothing.

## Quick Start (Python)

```bash
cd python
pip install -e ".[all]"

export TIDEWALL_BASE_URL=https://guard.example.com
export TIDEWALL_TOKEN=your-tidewall-api-token
export TIDEWALL_MODE=enforce

# Run an unmodified OpenAI app — Tidewall is injected by the wrapper.
tidewall-instrument python examples/plain_openai_app.py
```

See [`python/README.md`](./python/README.md) for the full Python documentation.

## Activation Modes

| Mode | Guard called? | Decisions enforced? |
| --- | --- | --- |
| `enforce` (default) | Yes | Yes — block / transform applied inline |
| `monitor` | Yes | No — verdicts logged only |
| `dry-run` | No | No — for smoke-testing the agent without a backend |

## Project Structure

```
tidewall-otel/
├── python/                     # Python implementation (current)
│   ├── src/tidewall_otel/      # Package source
│   ├── examples/               # Runnable examples
│   ├── tests/                  # Unit tests
│   ├── pyproject.toml
│   └── README.md
├── LICENSE                     # Apache 2.0
├── NOTICE
└── README.md                   # This file
```

## License

Apache License 2.0. See [LICENSE](./LICENSE) and [NOTICE](./NOTICE).

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](./CONTRIBUTING.md) for
the contribution workflow and development setup. Security-related
findings should be reported per [SECURITY.md](./SECURITY.md).
