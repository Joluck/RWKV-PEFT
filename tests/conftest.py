import os
import sys

import pytest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TESTS_DIR)
for path in (ROOT, TESTS_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)


@pytest.fixture
def restore_env():
    """Snapshot os.environ: the model layer reads config from env vars."""
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)
