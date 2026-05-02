# Changelog

All notable changes to Tidewall OTel are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-05-02

### Added

- Initial release of `tidewall-otel` (Python).
- `wrapt`-based instrumentation for `openai.chat.completions.create`
  (sync + async) and `anthropic.messages.create` (sync + async).
- Three activation paths: CLI wrapper (`tidewall-instrument`), OpenTelemetry
  auto-discovery, and explicit `tidewall_otel.activate()`.
- Three runtime modes: `enforce`, `monitor`, `dry-run`.
- Inline minimal HTTP client (`urllib`-based) — no third-party SDK
  dependency for talking to the Tidewall guard API.
- `gen_ai.*` OpenTelemetry semantic-convention spans for observability.
- Examples: `plain_openai_app.py`, `openai_example.py`, `anthropic_example.py`.

[Unreleased]: https://github.com/tidewall-security/tidewall-otel/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/tidewall-security/tidewall-otel/releases/tag/v0.1.0
