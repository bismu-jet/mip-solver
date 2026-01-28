"""
Shared pytest fixtures for MIP Solver tests.
"""
import pytest
import os
import sys

# Add the parent directory to the path so we can import solver modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def sample_lp_solution():
    """A sample LP solution with some fractional values."""
    return {
        'x1': 1.0,
        'x2': 0.5,
        'x3': 2.7,
        'x4': 0.0,
        'x5': 3.0,
    }


@pytest.fixture
def integer_solution():
    """A sample integer-feasible solution."""
    return {
        'x1': 1.0,
        'x2': 2.0,
        'x3': 3.0,
        'x4': 0.0,
        'x5': 5.0,
    }


@pytest.fixture
def data_dir():
    """Returns the path to the test data directory."""
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')


@pytest.fixture
def config_path():
    """Returns the path to the config file."""
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.yaml')
