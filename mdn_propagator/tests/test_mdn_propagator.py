"""
Unit and regression test for the mdn_propagator package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import mdn_propagator


def test_mdn_propagator_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "mdn_propagator" in sys.modules
