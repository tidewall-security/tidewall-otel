"""Auto-activate Tidewall instrumentation under sitecustomize.

Loaded automatically by Python during interpreter startup when this
file's directory is on ``PYTHONPATH``. The ``tidewall-instrument`` CLI
wrapper places the directory containing this file at the front of
``PYTHONPATH`` so this module is discovered before the user's code runs.

Activation is gated on ``TIDEWALL_OTEL_ENABLED=1`` so simply importing
the package or having it on the path doesn't activate the agent —
the explicit env-var opt-in prevents accidental activation in
unrelated processes.
"""

import os

if os.environ.get("TIDEWALL_OTEL_ENABLED") == "1":
    import tidewall_otel

    tidewall_otel.activate()
