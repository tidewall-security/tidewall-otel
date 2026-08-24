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
