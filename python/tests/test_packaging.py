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


def test_the_DOCUMENTED_otel_command_guards_and_reports_itself(tmp_path):
    """`opentelemetry-instrument python app.py` -- README activation method 2.

    Both halves matter, and they failed independently:

    * The SDK must actually be patched and a real call must reach the guard.
    * `tidewall_otel.state()` must SAY so. The entry point never calls the
      public `activate()`; it constructs the instrumentor and calls its hook.
      That patched the SDK correctly while the module-level state stayed at
      its initial `uninstalled`, so an operator wiring `is_active()` into a
      health check under the documented zero-code workflow read False and
      would have concluded they were unprotected. The mirror image of
      reporting `active` while unguarded, and just as wrong.

    Only reproducible from a real wheel in a clean venv: an editable install
    has no `.pth`, and the dev environment resolves the entry point
    differently. Reading the source cannot find this at all.
    """
    root = PYPROJECT.parent
    subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(tmp_path)],
        cwd=root, check=True, capture_output=True,
    )
    wheel = next(tmp_path.glob("*.whl"))

    venv = tmp_path / "venv"
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    bindir = venv / ("Scripts" if sys.platform == "win32" else "bin")
    python = bindir / "python"

    subprocess.run(
        [str(python), "-m", "pip", "install", "-q", f"{wheel}[otel]",
         "openai", "httpx"],
        check=True, capture_output=True,
    )

    app = tmp_path / "app.py"
    app.write_text(
        "import inspect, httpx, openai, tidewall_otel\n"
        "import tidewall_otel._guard as G\n"
        "from openai.resources.chat.completions.completions import Completions\n"
        "asked = []\n"
        "G.post_guard = lambda **kw: (asked.append(kw['payload']), {'result': {\n"
        "    'blocked': False, 'transformed': False, 'policy': 'p'}})[1]\n"
        "def ok(request):\n"
        "    return httpx.Response(200, json={'id': 'x', 'object': 'chat.completion',\n"
        "        'created': 0, 'model': 'gpt-4o', 'choices': [{'index': 0,\n"
        "        'finish_reason': 'stop', 'message': {'role': 'assistant',\n"
        "        'content': 'ok'}}]})\n"
        "c = openai.OpenAI(api_key='t',\n"
        "                  http_client=httpx.Client(transport=httpx.MockTransport(ok)))\n"
        "c.chat.completions.create(model='gpt-4o',\n"
        "                          messages=[{'role': 'user', 'content': 'hi'}])\n"
        "patched = getattr(inspect.getattr_static(Completions, 'create'),\n"
        "                  '__tidewall_wrapper__', False)\n"
        "print(f'patched={patched} asked={len(asked)} "
        "lifecycle={tidewall_otel.state().lifecycle}')\n"
    )

    result = subprocess.run(
        [str(bindir / "opentelemetry-instrument"), str(python), str(app)],
        capture_output=True, text=True, cwd=tmp_path,
        env={"PATH": str(bindir) + ":/usr/bin:/bin",
             "TIDEWALL_BASE_URL": "https://guard.example", "TIDEWALL_TOKEN": "t"},
    )
    line = next((l for l in result.stdout.splitlines() if l.startswith("patched=")), "")
    assert line == "patched=True asked=1 lifecycle=installed", (
        f"stdout={result.stdout!r} stderr={result.stderr[-800:]!r}")
