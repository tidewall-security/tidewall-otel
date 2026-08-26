"""Startup hook imported by ``tidewall_otel.pth``.

A ``.pth`` file at the site-directory root is executed by ``site`` at
interpreter startup, before any application code. Every ``.pth`` runs; they do
not shadow one another. That is the whole reason this replaces the previous
``sitecustomize.py`` approach -- ``import sitecustomize`` resolves ONE module
across the entire path, so shipping one either loses to another package's or
silently suppresses it.

Activation is gated on TIDEWALL_OTEL_ENABLED, and every failure is contained:
a startup hook that raises breaks every Python process on the machine, which
is a far worse outcome than not instrumenting.
"""

from __future__ import annotations

import os


def _activate() -> None:
    enabled = os.environ.get("TIDEWALL_OTEL_ENABLED", "").strip().lower()
    if enabled not in ("1", "true", "yes", "on"):
        return

    try:
        import tidewall_otel

        tidewall_otel.activate()
    except BaseException:                       # noqa: BLE001 -- see below
        # DELIBERATELY BROAD, and deliberately silent by default.
        #
        # `site` already contains exceptions from a .pth: it prints the
        # traceback, says "Remainder of file ignored", and the interpreter
        # continues -- verified. So this is NOT what keeps Python usable.
        # What it prevents is a TRACEBACK ON EVERY PROCESS START: this hook
        # runs in every interpreter on the machine, including pip's own, and
        # a misconfigured install would otherwise print a stack trace before
        # every command the user types.
        #
        # Diagnostics go to stderr only when explicitly requested.
        if os.environ.get("TIDEWALL_OTEL_DEBUG"):
            import traceback

            traceback.print_exc()


_activate()
