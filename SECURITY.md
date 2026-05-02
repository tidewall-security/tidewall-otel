# Security Policy

Thanks for helping keep Tidewall OTel and its users safe.

## Supported Versions

While the project is in alpha (0.x), only the latest minor release
receives security fixes. Once we ship 1.0 we'll publish a clear
support window.

## Reporting a Vulnerability

**Please do not open a public GitHub issue for security findings.**

Instead, use one of:

- GitHub Security Advisories on this repository
  (`Security` tab → `Report a vulnerability`).
- Email: `security@tidewall.ai`.

Please include:

- A description of the issue and where it lives in the code
  (file paths, function names, or commit hashes).
- Steps to reproduce, ideally with a minimal proof-of-concept.
- The impact: what an attacker could do if the issue were exploited.
- Any mitigations or workarounds you've already identified.

## What to Expect

- We aim to acknowledge new reports within **3 business days**.
- We'll work with you on a fix and a coordinated disclosure timeline.
- We're happy to credit you in the advisory once the fix is public,
  unless you prefer to remain anonymous.

## Out of Scope

The following are not considered vulnerabilities:

- Issues in third-party dependencies (please report them upstream).
- Theoretical attacks that require attacker control of the user's
  Python environment.
- Findings that depend on misconfiguration the documentation explicitly
  warns against (for example, leaking `TIDEWALL_TOKEN` in logs).

If in doubt, send the report anyway — we'll take a look.
