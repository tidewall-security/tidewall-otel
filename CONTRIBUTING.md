# Contributing to Tidewall OTel

Thanks for your interest in contributing. Issues, pull requests, and
discussion of new language implementations are all welcome.

## Code of Conduct

Be respectful and constructive. Personal attacks, harassment, or
discriminatory behaviour are not tolerated in any project channel.
By participating, you agree to abide by the
[Contributor Covenant](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).

## Reporting Bugs

Please file an issue with:

- A short title that names the symptom.
- The exact command or code that triggers it.
- What you expected to happen.
- What actually happened, including any tracebacks or log output.
- Versions: `python --version`, `pip show tidewall-otel`, plus
  the OpenAI/Anthropic SDK versions if relevant.

## Suggesting Features

Open an issue describing the use case before opening a PR — features
are easier to discuss when there's a problem statement to anchor to.

## Pull Requests

1. Fork the repo and create a topic branch from `main`.
2. Make your change in the smallest reasonable diff. Multiple unrelated
   changes belong in separate PRs.
3. Add or update tests for any behavioural change. Tests live in
   `python/tests/` and run with `pytest`.
4. Update documentation if you change anything user-facing
   (env vars, CLI flags, config keys, error messages).
5. Make sure the test suite passes locally:
   ```bash
   cd python
   pip install -e ".[dev]"
   pytest
   ```
6. Open the PR with a clear description of the problem and the change.

## Development Setup (Python)

```bash
git clone https://github.com/tidewall-security/tidewall-otel
cd tidewall-otel/python
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Style

- Code should be type-annotated and pass `mypy` cleanly.
- Comments explain *why*, not *what* — the code already shows what it does.
- Keep public API surface area small. New symbols added to a module's
  top level should be added to `__all__`.

## Adding a New Language

If you'd like to add a Node.js / Java / Go / .NET implementation, open
an issue first so we can agree on the directory layout and the shared
configuration model. The principle is that all language packages
expose the same env-var contract and the same three modes
(`enforce` / `monitor` / `dry-run`), even if the SDK-specific
patching mechanism differs.

## Security Findings

Don't open a public issue for a security vulnerability — see
[SECURITY.md](./SECURITY.md) for the disclosure process.
