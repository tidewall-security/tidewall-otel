"""The test suite's own integrity checks.

Task 1b. These guard the acceptance criteria the rest of the programme relies
on: a test that cannot run proves nothing, and neither does one with no body.
"""

import ast
import textwrap
from pathlib import Path

import pytest

from tests._fixtures import undefined_names_under

TESTS = Path(__file__).resolve().parent


def test_no_module_calls_an_UNDEFINED_name():
    """Delegated to pyflakes, over the WHOLE tests tree including _fixtures.py.

    A bare call to a helper the module has not imported is an undefined name,
    so the import discipline is checked rather than asserted.
    """
    problems = undefined_names_under(TESTS)
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("source,expected", [
    ("def test_x():\n    _missing()\n", "_missing"),
    ("def test_x():\n    obj.method()\n", "obj"),
    ("def helper(_missing):\n    pass\n\ndef test_x():\n    _missing()\n", "_missing"),
    ("def test_x():\n    return [_missing(x) for x in range(1)]\n", "_missing"),
])
def test_the_breadth_check_CATCHES_each_shape(source, expected, tmp_path):
    """The known-positive assertion.

    Rows two and three are cases a hand-rolled AST walk missed: an attribute
    call on an undefined root, and a function PARAMETER binding a name
    module-wide and thereby excusing undefined calls elsewhere in the file.
    They are here so a regression to bespoke analysis fails loudly.
    """
    (tmp_path / "test_planted.py").write_text(source)
    problems = undefined_names_under(tmp_path)
    assert any(expected in problem for problem in problems), problems


def test_the_breadth_check_does_NOT_fire_on_a_clean_module(tmp_path):
    """The other direction. A check that flags everything is not a check."""
    (tmp_path / "test_clean.py").write_text(
        "from pathlib import Path\n\n"
        "def test_x():\n    assert Path('.').exists()\n"
    )
    assert undefined_names_under(tmp_path) == []


def _test_functions():
    for path in sorted(TESTS.rglob("test_*.py")):
        tree = ast.parse(textwrap.dedent(path.read_text()))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("test_"):
                    yield path, node


def test_no_test_function_has_an_EMPTY_body():
    """A function whose body is only a docstring is reported by pytest as
    PASSED. Four such tests shipped in one draft of this plan, each the named
    detector for a defect it therefore never observed.
    """
    empty = []
    for path, node in _test_functions():
        statements = [
            statement for statement in node.body
            if not (isinstance(statement, ast.Expr)
                    and isinstance(statement.value, ast.Constant)
                    and isinstance(statement.value.value, str))
        ]
        if not statements:
            empty.append(f"{path.name}:{node.lineno} {node.name}")
    assert not empty, "test functions with no body:\n" + "\n".join(empty)


def _always_true_assertions(tree):
    """Assertions no input can fail.

    Three shapes, each of which has shipped somewhere in this programme:

    ``assert <anything> or True``   the `or True` makes the left side dead
    ``assert <truthy literal>``     `assert 1`, `assert "x"`, `assert (a, b)`
                                    -- a non-empty tuple is the classic typo
                                    for a two-argument assert
    ``assert not <x> or True``      the same as the first, seen in the wild

    Deliberately syntactic. Deciding whether an arbitrary expression can be
    false is undecidable; these are the shapes that actually get written.
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assert):
            continue
        test = node.test
        if (isinstance(test, ast.BoolOp) and isinstance(test.op, ast.Or)
                and any(isinstance(v, ast.Constant) and v.value is True
                        for v in test.values)):
            yield node.lineno, "`or True` makes the assertion unfailable"
        elif isinstance(test, ast.Constant) and test.value:
            yield node.lineno, f"constant {test.value!r} is always truthy"
        elif isinstance(test, ast.Tuple) and test.elts:
            yield node.lineno, "a non-empty tuple is always truthy (missing comma?)"


def test_no_test_function_has_an_ALWAYS_TRUE_assertion():
    """The vacuity check the empty-body detector could not see.

    `assert not isinstance(stored, type(Descriptored.create)) or True` shipped
    in test_manager.py and passed every review until an adversarial reviewer
    read the line. The empty-body detector cannot catch it: the body is not
    empty, it is merely inert. A test that cannot fail is the same defect as a
    test with no body, wearing an assertion.
    """
    offenders = []
    for path in sorted(Path(__file__).parent.glob("test_*.py")):
        tree = ast.parse(path.read_text())
        for lineno, why in _always_true_assertions(tree):
            offenders.append(f"{path.name}:{lineno} -- {why}")
    assert not offenders, "assertions that cannot fail:\n" + "\n".join(offenders)


def test_the_always_true_detector_CATCHES_the_shapes_it_claims_to():
    """The known-positive. A detector's green result is a claim like any other,
    and 'the scan found nothing' is not the same statement as 'there is
    nothing to find'. Plants the real defect that motivated it, first.
    """
    planted = ast.parse(
        "def test_a():\n"
        "    assert not isinstance(x, y) or True\n"      # the real one
        "def test_b():\n"
        "    assert 1\n"
        "def test_c():\n"
        "    assert ('a', 'b')\n"
    )
    found = list(_always_true_assertions(planted))
    assert len(found) == 3, found

    clean = ast.parse("def test_d():\n    assert x == y\n"
                      "def test_e():\n    assert not x, 'message'\n"
                      "def test_f():\n    assert x or y\n")
    assert list(_always_true_assertions(clean)) == []


def _plaintext_guard_urls(text):
    """`http://` offered as a guard base URL.

    Windowed, NOT line-based. The real defect in `_config.py` put the variable
    name and its example on DIFFERENT lines:

        TIDEWALL_BASE_URL   - Tidewall guard API base URL
                              (e.g. ``http://localhost:8080``)

    A single-line regex finds the README occurrences and misses that one --
    which is the shape the known-positive below plants, because a detector
    tested only against the easy case is how the hard case ships.

    The window stops at a blank line or the next ``TIDEWALL_`` variable, so an
    unrelated `http://` further down the file is not attributed to this one.
    Narrow on purpose: a link to an RFC is not a configuration example.
    """
    import re

    for match in re.finditer(r"TIDEWALL_BASE_URL", text):
        rest = text[match.start():]
        stop = len(rest)
        blank = re.search(r"\n\s*\n", rest)
        if blank:
            stop = min(stop, blank.start())
        nxt = re.search(r"TIDEWALL_(?!BASE_URL)", rest[1:])
        if nxt:
            stop = min(stop, nxt.start() + 1)
        window = rest[:stop]
        if "http://" in window:
            yield " ".join(window.split())


def test_no_document_offers_a_PLAINTEXT_guard_url():
    """The design called this out and the change was missed.

    HTTPS-only was accepted as a deliberate pre-release break: "HTTPS-only
    breaks every http://localhost:8080 quick start in the server docs.
    Deliberate pre-release break; the server documentation changes in the same
    release." The code shipped the enforcement; three documents kept telling a
    new user to configure exactly the URL that is now refused before a socket
    is opened -- including the config docstring they would read to fix it.

    Documentation that contradicts an enforced security control is not a
    cosmetic defect: it is a first-run failure that teaches the user the agent
    is broken.
    """
    repo = Path(__file__).resolve().parents[2]
    offenders = []
    for doc in (repo / "README.md", repo / "python" / "README.md",
                repo / "python" / "src" / "tidewall_otel" / "_config.py"):
        if not doc.exists():
            continue
        for line in _plaintext_guard_urls(doc.read_text()):
            offenders.append(f"{doc.relative_to(repo)}: {line}")
    assert not offenders, (
        "documents offering a plaintext guard URL:\n" + "\n".join(offenders))


def test_the_plaintext_url_detector_CATCHES_the_real_line():
    """Known-positive, planting the exact text that shipped -- including the
    CONTINUATION-LINE form, which a line-based detector would miss."""
    shell = "export TIDEWALL_BASE_URL=http://localhost:8080"
    assert len(list(_plaintext_guard_urls(shell))) == 1

    table = ("| `TIDEWALL_BASE_URL` | (required) | base URL, "
             "e.g. `http://localhost:8080` |")
    assert len(list(_plaintext_guard_urls(table))) == 1

    # The one that actually shipped in _config.py: name and example on
    # different lines.
    continuation = (
        "        TIDEWALL_BASE_URL   - Tidewall guard API base URL\n"
        "                              (e.g. ``http://localhost:8080``)\n"
        "        TIDEWALL_TOKEN      - API token\n"
    )
    found = list(_plaintext_guard_urls(continuation))
    assert len(found) == 1, found

    assert list(_plaintext_guard_urls(
        "export TIDEWALL_BASE_URL=https://guard.example.com")) == []
    # prose mentioning http elsewhere is not a configuration example
    assert list(_plaintext_guard_urls("see http://example.org for background")) == []
    # and an http:// beyond the window is not attributed to this variable
    assert list(_plaintext_guard_urls(
        "TIDEWALL_BASE_URL - the guard URL\n\nSee http://example.org too\n")) == []


def _env_var_sets():
    """(documented, accepted, removed) — read from the two sources of truth.

    `accepted` is every ``TIDEWALL_*`` literal in `_config.py` minus the
    removed set, which deliberately over-collects rather than under-collects:
    a variable this misses is a variable the drift test cannot police.
    """
    import re

    from tidewall_otel._config import _REMOVED_VARIABLES

    config = Path(__file__).resolve().parents[1] / "src" / "tidewall_otel" / "_config.py"
    readme = Path(__file__).resolve().parents[1] / "README.md"

    removed = set(_REMOVED_VARIABLES)
    accepted = set(re.findall(r'["\'](TIDEWALL_[A-Z_]+)["\']', config.read_text())) - removed
    documented = set(re.findall(r"\|\s*`(TIDEWALL_[A-Z_]+)`", readme.read_text()))
    return documented, accepted, removed


def test_no_documented_env_var_has_been_REMOVED():
    """Following the README must not raise.

    `TIDEWALL_TIMEOUT` was split into `TIDEWALL_SOCKET_TIMEOUT` and
    `TIDEWALL_GUARD_DEADLINE` and is now REFUSED rather than ignored -- quietly
    dropping a variable an operator set would pick a bound they did not choose.
    The README kept documenting it, so a user who followed the configuration
    table got a hard `ValueError` at activation. Refusing loudly is right; the
    documentation telling them to set it is not.
    """
    documented, _accepted, removed = _env_var_sets()
    offenders = sorted(documented & removed)
    assert not offenders, (
        "README documents variables the agent refuses: " + ", ".join(offenders))


def test_every_ACCEPTED_env_var_is_documented():
    """The other direction, and the one that hides security-relevant options.

    `TIDEWALL_ON_ACTIVATION_FAILURE` chooses between raising, running
    unguarded, and installing refusers when activation fails. It was
    undocumented, so the operator could not choose it -- they got the default
    without knowing there was a decision to make.
    """
    documented, accepted, _removed = _env_var_sets()
    missing = sorted(accepted - documented)
    assert not missing, "accepted but undocumented: " + ", ".join(missing)


def test_every_public_name_in_ALL_actually_resolves():
    """`from tidewall_otel import *` must not raise.

    An `__all__` entry with nothing behind it is the same defect as any other
    unevidenced claim, in one line: the package advertises a name it does not
    have. Caught here on the very commit that added the exception exports --
    `TidewallRefusedError` was listed before it was imported.
    """
    import tidewall_otel

    missing = [n for n in tidewall_otel.__all__ if not hasattr(tidewall_otel, n)]
    assert not missing, f"__all__ advertises names that do not exist: {missing}"


def test_the_README_exception_name_is_REACHABLE():
    """The README documents `tidewall_otel.TidewallBlockedError`.

    It resolved at no public name at all until 2026-08-25: an application
    following the README got an `AttributeError`, and the only way to catch a
    block was to import from the private `_exceptions` module. A library whose
    primary exception has no public name has no usable error contract.
    """
    import re

    import tidewall_otel

    root = Path(__file__).resolve().parents[2]
    documented = set()
    for readme in (root / "README.md", root / "python" / "README.md"):
        documented |= set(
            re.findall(r"`tidewall_otel\.(Tidewall\w+)`", readme.read_text()))
    assert documented, "no exception documented in either README -- did it move?"

    unreachable = sorted(n for n in documented if not hasattr(tidewall_otel, n))
    assert not unreachable, (
        f"README documents unreachable names: {unreachable}")


def test_every_public_exception_is_a_TidewallError():
    """One `except` clause must cover every way Tidewall declines a call.

    `TidewallActivationRefusedError` subclassed a bare `RuntimeError`. Under
    `TIDEWALL_ON_ACTIVATION_FAILURE=block` it is raised for EVERY call, so an
    application wrapping its AI calls in `except TidewallError` missed all of
    them and crashed on an exception it had no reason to expect.
    """
    import tidewall_otel

    outsiders = []
    for name in tidewall_otel.__all__:
        obj = getattr(tidewall_otel, name)
        if isinstance(obj, type) and issubclass(obj, BaseException):
            if not issubclass(obj, tidewall_otel.TidewallError):
                outsiders.append(name)
    assert not outsiders, (
        f"public exceptions outside the TidewallError hierarchy: {outsiders}")


def test_no_document_claims_the_OS_ACCOUNT_NAME_is_a_default():
    """P0-2's claim, asserted against every document that could restate it.

    The name-set drift checks compare only which variables appear, never what
    is said about them, so they passed while `_config.py`'s own reference still
    read "TIDEWALL_USER_ID - User identifier (defaults to ``$USER``)". The
    README had been corrected and the module docstring had not.

    A wrong default is worse than a missing one here: it tells an operator the
    OS account name is collected by default, which is precisely the privacy
    defect P0-2 removed. Anyone reading it would either wrongly avoid the
    library or wrongly file a privacy exception for it.
    """
    import re

    from tidewall_otel._config import TidewallConfig

    assert TidewallConfig(mode="dry-run").user_id == "", (
        "the implementation grew a default identity; this test is now wrong")

    root = Path(__file__).resolve().parents[2]
    docs = (root / "README.md", root / "python" / "README.md",
            root / "python" / "src" / "tidewall_otel" / "_config.py")

    offenders = []
    for doc in docs:
        for match in re.finditer(r"TIDEWALL_USER_ID", doc.read_text()):
            window = doc.read_text()[match.start():match.start() + 260]
            window = window.split("\n\n")[0]
            if re.search(r"\$USER|default[s]?\s+to\s+`*\$?USER", window):
                offenders.append(f"{doc.name}: {' '.join(window.split())[:110]}")
    assert not offenders, (
        "documents claiming an OS-account-name default:\n" + "\n".join(offenders))


def test_every_DOCUMENTED_default_matches_the_code():
    """The general form. Names matching is not the claim a table makes.

    Each row asserts a DEFAULT, and a table can name every variable correctly
    while getting every value wrong. Compares the README's Default column
    against a freshly constructed `TidewallConfig`.
    """
    import re

    from tidewall_otel._config import TidewallConfig

    config = TidewallConfig(base_url="https://guard.example", token="t")
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()

    mismatches = []
    for var, default in re.findall(
            r"\|\s*`(TIDEWALL_[A-Z_]+)`\s*\|\s*([^|]+?)\s*\|", readme):
        if default.startswith("("):          # (required) / (none)
            continue
        attribute = var[len("TIDEWALL_"):].lower()
        for candidate in (attribute, attribute + "_s"):
            if hasattr(config, candidate):
                actual = getattr(config, candidate)
                break
        else:
            mismatches.append(f"{var}: documented but no config field")
            continue
        documented = default.strip().strip("`")
        # Compare NUMERICALLY when both sides are numbers: the README writes
        # `10` and the field holds 10.0. An earlier version normalised by
        # stripping trailing zeros, which turned "10" into "1" and reported a
        # mismatch on two correct rows -- a detector wrong in the direction
        # that gets it deleted rather than trusted.
        try:
            same = float(documented) == float(actual)
        except (TypeError, ValueError):
            same = documented == str(actual)
        if not same:
            mismatches.append(f"{var}: README says {documented!r}, code gives {actual!r}")
    assert not mismatches, "\n".join(mismatches)
