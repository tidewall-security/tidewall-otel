"""Bootstrap .pth, replacing sitecustomize (O-5). Task 13."""

import base64
import csv
import hashlib
import subprocess
import sys
import sysconfig
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PTH = "tidewall_otel.pth"


def build_wheel_for(kind, tmp_path):
    """Build through the PEP 517 / PEP 660 hook and return the artifact.

    `python -m build` has no --editable flag: a PEP 660 wheel comes from
    calling build_editable directly. It is a DISTINCT artifact with different
    contents, so inspecting the ordinary wheel would not detect a missing or
    misplaced editable .pth.
    """
    out = tmp_path / f"dist-{kind}"
    out.mkdir(parents=True, exist_ok=True)

    if kind == "wheel":
        subprocess.run([sys.executable, "-m", "build", "--wheel",
                        "--outdir", str(out), str(ROOT)],
                       check=True, capture_output=True)
    else:
        subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.path.insert(0, %r); import _build_pth; "
             "print(_build_pth.build_editable(%r))" % (str(ROOT), str(out))],
            cwd=str(ROOT), check=True, capture_output=True)
    return next(out.glob("*.whl"))


def record_of(wheel):
    with zipfile.ZipFile(wheel) as archive:
        name = next(n for n in archive.namelist() if n.endswith(".dist-info/RECORD"))
        rows = [r for r in csv.reader(archive.read(name).decode().splitlines()) if r]
    return name, {r[0]: (r[1], r[2]) for r in rows}


@pytest.mark.parametrize("kind", ["wheel", "editable"])
def test_the_pth_is_listed_in_RECORD_with_a_correct_digest_and_size(kind, tmp_path):
    """Appending after setuptools writes RECORD leaves the file unrecorded:
    installable by a permissive installer, but the wheel is no longer
    internally consistent and uninstall does not know to remove it."""
    wheel = build_wheel_for(kind, tmp_path)
    _, record = record_of(wheel)

    assert PTH in record, "the .pth is not recorded"
    with zipfile.ZipFile(wheel) as archive:
        data = archive.read(PTH)
    expected = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
    digest, size = record[PTH]
    assert digest == f"sha256={expected}"
    assert int(size) == len(data)


@pytest.mark.parametrize("kind", ["wheel", "editable"])
def test_there_is_EXACTLY_ONE_pth_entry_and_ONE_RECORD(kind, tmp_path):
    """Rebuilding wrongly -- or appending a corrected RECORD -- yields
    duplicate archive members, which some tools read and others reject."""
    with zipfile.ZipFile(build_wheel_for(kind, tmp_path)) as archive:
        names = archive.namelist()
    assert names.count(PTH) == 1
    assert sum(n.endswith(".dist-info/RECORD") for n in names) == 1


@pytest.mark.parametrize("kind", ["wheel", "editable"])
def test_every_RECORD_entry_matches_the_archive(kind, tmp_path):
    """Not only our line: the rebuild must not have corrupted anyone else's."""
    wheel = build_wheel_for(kind, tmp_path)
    record_name, record = record_of(wheel)
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if name == record_name:
                assert record[name] == ("", ""), "RECORD must not hash itself"
                continue
            data = archive.read(name)
            expected = base64.urlsafe_b64encode(
                hashlib.sha256(data).digest()).rstrip(b"=").decode()
            assert record[name] == (f"sha256={expected}", str(len(data))), name
        assert set(record) == set(archive.namelist()), "RECORD and archive disagree"


def _venv(tmp_path):
    venv = tmp_path / "venv"
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    python = venv / ("Scripts" if sys.platform == "win32" else "bin") / "python"
    site = subprocess.run(
        [str(python), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
        capture_output=True, text=True, check=True).stdout.strip()
    return python, Path(site)


@pytest.mark.parametrize("kind", ["wheel", "editable"])
def test_uninstall_REMOVES_the_pth(kind, tmp_path):
    """The consequence of an unrecorded file, asserted directly."""
    wheel = build_wheel_for(kind, tmp_path)
    python, site = _venv(tmp_path)
    subprocess.run([str(python), "-m", "pip", "install", "-q", str(wheel)], check=True)

    assert (site / PTH).exists()
    subprocess.run([str(python), "-m", "pip", "uninstall", "-y", "-q", "tidewall-otel"],
                   check=True)
    assert not (site / PTH).exists(), "uninstall left the .pth -- it was unrecorded"


def test_a_sitecustomize_in_site_packages_is_SHADOWED(tmp_path):
    """Why the .pth exists at all.

    `import sitecustomize` resolves ONE module across the whole path, and the
    stdlib directory precedes site-packages -- so on an interpreter that ships
    one, a site-packages copy never runs. Skipped rather than silently passed
    where no stdlib-level sitecustomize exists.
    """
    python, site = _venv(tmp_path)
    (site / "sitecustomize.py").write_text(
        'import os\nos.environ["OURS_VIA_SITECUSTOMIZE"] = "yes"\n')

    probe = subprocess.run(
        [str(python), "-c", "import sitecustomize; print(sitecustomize.__file__)"],
        capture_output=True, text=True)
    if str(site) in probe.stdout:
        pytest.skip("no stdlib-level sitecustomize on this interpreter")

    out = subprocess.run(
        [str(python), "-c", "import os; print(os.environ.get('OURS_VIA_SITECUSTOMIZE'))"],
        capture_output=True, text=True, check=True).stdout.strip()
    assert out == "None", "expected the site-packages copy to be shadowed"


def test_the_pth_runs_ALONGSIDE_a_winning_sitecustomize(tmp_path):
    """The property that matters: a .pth is executed during addsitedir, so it
    cannot be shadowed by anyone's sitecustomize -- including one that beats
    the stdlib copy."""
    wheel = build_wheel_for("wheel", tmp_path)
    python, _site = _venv(tmp_path)
    subprocess.run([str(python), "-m", "pip", "install", "-q", str(wheel)], check=True)

    rival = tmp_path / "rival"
    rival.mkdir()
    (rival / "sitecustomize.py").write_text(
        'import os\nos.environ["RIVAL_RAN"] = "yes"\n')

    env = {"PATH": "/usr/bin:/bin", "PYTHONPATH": str(rival),
           "TIDEWALL_OTEL_ENABLED": "1", "TIDEWALL_BASE_URL": "https://g.example",
           "TIDEWALL_TOKEN": "t"}
    out = subprocess.run(
        [str(python), "-c",
         "import os, tidewall_otel; print(tidewall_otel.is_active(), "
         "os.environ.get('RIVAL_RAN'))"],
        capture_output=True, text=True, env=env).stdout.strip()

    assert "yes" in out, f"the rival sitecustomize did not run: {out!r}"


@pytest.mark.parametrize("enabled,expected", [("1", "installed"), ("", "uninstalled")])
def test_the_env_var_GATES_activation(tmp_path, enabled, expected):
    """An inert .pth and a correctly gated one look identical unless BOTH
    directions are checked.

    Reads `lifecycle`, NOT `is_active()`. This venv has the wheel and nothing
    else -- no `openai`, no `anthropic` -- so there are zero boundaries to
    guard and `is_active()` is correctly False in BOTH arms. It only read True
    while `activate()` fabricated `{surface: "covered"}` for every manifest
    entry without checking whether that SDK was even importable, which made
    this assertion pass for the wrong reason. Lifecycle is what the gate
    actually controls.
    """
    wheel = build_wheel_for("wheel", tmp_path)
    python, _site = _venv(tmp_path)
    subprocess.run([str(python), "-m", "pip", "install", "-q", str(wheel)], check=True)

    env = {"PATH": "/usr/bin:/bin", "TIDEWALL_BASE_URL": "https://g.example",
           "TIDEWALL_TOKEN": "t"}
    if enabled:
        env["TIDEWALL_OTEL_ENABLED"] = enabled

    out = subprocess.run(
        [str(python), "-c",
         "import tidewall_otel; print(tidewall_otel.state().lifecycle)"],
        capture_output=True, text=True, env=env).stdout.strip()
    assert out == expected, out


def test_a_failing_bootstrap_is_SILENT_unless_debugging(tmp_path):
    """The property that is actually true, and the one worth guarding.

    `site` already contains exceptions from a .pth -- it prints the traceback
    and continues -- so interpreter survival cannot distinguish a contained
    failure from an uncontained one; mutation-testing showed exactly that.
    What containment prevents is a TRACEBACK ON EVERY PROCESS START, in every
    interpreter on the machine including pip's own.
    """
    wheel = build_wheel_for("wheel", tmp_path)
    python, site = _venv(tmp_path)
    subprocess.run([str(python), "-m", "pip", "install", "-q", str(wheel)], check=True)

    # Break activation as thoroughly as possible.
    (site / "tidewall_otel" / "_config.py").write_text("raise RuntimeError('boom')\n")

    result = subprocess.run(
        [str(python), "-c", "print('interpreter still works')"],
        capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin", "TIDEWALL_OTEL_ENABLED": "1"})
    assert result.returncode == 0, result.stderr
    assert "interpreter still works" in result.stdout
    assert "Traceback" not in result.stderr, (
        f"a failing bootstrap printed a traceback on startup:\n{result.stderr}"
    )

    # ...and says so when asked.
    debug = subprocess.run(
        [str(python), "-c", "pass"], capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin", "TIDEWALL_OTEL_ENABLED": "1",
             "TIDEWALL_OTEL_DEBUG": "1"})
    assert "Traceback" in debug.stderr, "debug mode hid the failure too"
