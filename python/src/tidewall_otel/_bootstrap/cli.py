"""CLI entry point for the ``tidewall-instrument`` wrapper.

Launches a user command with Tidewall instrumentation pre-activated.
This follows the same pattern used by ``ddtrace-run`` and Elastic's
``edot-run``: set an environment marker and prepend a bootstrap
directory to ``PYTHONPATH`` so :mod:`sitecustomize` runs before the
host application's code.

Usage:

.. code-block:: bash

    tidewall-instrument python my_app.py
    tidewall-instrument python -m my_module
    tidewall-instrument gunicorn app:app

The wrapper exits with the same status code as the wrapped command.
"""

from __future__ import annotations

import os
import subprocess
import sys


def main() -> None:
    """CLI entry point — fork the user's command with instrumentation enabled."""
    if len(sys.argv) < 2:
        print(
            "Usage: tidewall-instrument <command> [args...]\n"
            "\n"
            "Examples:\n"
            "  tidewall-instrument python my_app.py\n"
            "  tidewall-instrument python -m uvicorn main:app\n"
            "\n"
            "Environment variables:\n"
            "  TIDEWALL_BASE_URL   Tidewall guard API base URL\n"
            "  TIDEWALL_TOKEN      Tidewall API token\n"
            "  TIDEWALL_MODE       enforce | monitor | dry-run (default: enforce)\n"
            "  TIDEWALL_APP_ID     Application identifier\n"
            "  TIDEWALL_APP_NAME   Application display name\n"
            "  TIDEWALL_LOG_LEVEL  debug | info | warning | error (default: info)\n"
        )
        sys.exit(1)

    env = os.environ.copy()

    # Marker read by sitecustomize.py to decide whether to activate.
    # Anything non-empty enables — we use "1" by convention.
    env["TIDEWALL_OTEL_ENABLED"] = "1"

    # Prepend the _bootstrap directory to PYTHONPATH so our sitecustomize.py
    # is discovered before any other site-packages-resident sitecustomize.
    bootstrap_dir = os.path.dirname(os.path.abspath(__file__))
    existing_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        bootstrap_dir + os.pathsep + existing_path
        if existing_path
        else bootstrap_dir
    )

    result = subprocess.run(sys.argv[1:], env=env)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
