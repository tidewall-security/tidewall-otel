"""pytest fixtures.

Only things needing setup and teardown live here. Ordinary helpers are plain
functions in ``tests/_fixtures.py``, imported explicitly by the modules that
use them: pytest injects *fixtures* as test parameters and does not put
functions into a test module's globals, so a bare call to a helper defined
here would be a ``NameError``.
"""

import pytest

from tests._fixtures import LocalGuardServer


@pytest.fixture
def local_guard_server():
    """A real HTTP guard server, torn down after the test."""
    server = LocalGuardServer()
    yield server
    server.shutdown()
