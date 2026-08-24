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
