"""Packaging acceptance for the P0 remediation programme.

Task 1 of the accepted implementation plan.
"""

import re
import subprocess
import sys
import tomllib
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def test_the_otel_extra_supplies_baseinstrumentor():
    """``BaseInstrumentor`` lives in opentelemetry-instrumentation, not in
    -api or -sdk. Without it ``pip install tidewall-otel[otel]`` silently uses
    the stub in ``_instrumentor.py`` and the OTel entry point never works."""
    extras = tomllib.loads(PYPROJECT.read_text())["project"]["optional-dependencies"]
    otel = " ".join(extras["otel"])
    assert "opentelemetry-instrumentation" in otel, extras["otel"]


def test_the_extra_installs_the_real_base_class_in_a_CLEAN_environment(tmp_path):
    """Builds a wheel and installs ONLY ``wheel[otel]`` into a fresh venv.

    Running the current interpreter proves nothing: the dev environment
    already contains opentelemetry-instrumentation, so the assertion would
    pass whether or not the extra declares it.
    """
    root = PYPROJECT.parent
    subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(tmp_path)],
        cwd=root, check=True, capture_output=True,
    )
    wheel = next(tmp_path.glob("*.whl"))

    venv = tmp_path / "venv"
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    python = venv / ("Scripts" if sys.platform == "win32" else "bin") / "python"

    subprocess.run(
        [str(python), "-m", "pip", "install", "-q", f"{wheel}[otel]"],
        check=True, capture_output=True,
    )

    result = subprocess.run(
        [str(python), "-c",
         "import tidewall_otel._instrumentor as m; print(m._HAS_OTEL_INSTRUMENTOR)"],
        capture_output=True, text=True,
    )
    assert result.stdout.strip() == "True", result.stdout + result.stderr


def test_the_CI_test_step_is_EXACTLY_the_accepted_command():
    """A WHITELIST, not a shell analyser.

    Earlier drafts matched ``|| true``, then ``|| :``, ``;`` and ``set +e``,
    and still missed ``set +e`` on a preceding line of a multiline ``run:``
    block while false-positiving on ``pytest -q -k "foo;bar"``. Pinning the
    exact command needs no shell parsing, cannot false-positive, and rejects
    swallow forms nobody has thought of.

    Widening this is a deliberate, reviewable act: add the new exact string.
    """
    workflow = PYPROJECT.resolve().parents[1] / ".github" / "workflows" / "ci.yml"
    assert workflow.exists(), f"workflow not found at {workflow}"

    ACCEPTED = {"pytest -q", "uv run pytest -q"}

    steps = re.findall(r"- name: (.+?)\n(.*?)(?=\n      - |\Z)",
                       workflow.read_text(), re.S)
    test_steps = [(name, body) for name, body in steps if "pytest" in body]
    assert test_steps, "no CI step runs pytest at all"

    for name, body in test_steps:
        run = re.search(r"run:\s*(?:\|\s*\n)?(.*?)(?=\n\s*[a-z-]+:|\Z)", body, re.S)
        command = " ".join(run.group(1).split()) if run else ""
        assert command in ACCEPTED, (
            f"CI step {name!r} runs {command!r}, which is not an accepted test "
            f"command. Accepted: {sorted(ACCEPTED)}. If this is a legitimate "
            f"change, widen ACCEPTED deliberately -- do not relax the check."
        )
