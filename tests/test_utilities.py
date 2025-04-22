"""Utility functions for tests."""

from contextlib import contextmanager

import numpy as np
import pytest


@contextmanager
def not_raises(expected_exception):
    """Assert that an exception is not raised.

    Usage:
        with not_raises(SomeException):
            do_something()
    """
    try:
        yield
    except expected_exception as error:
        raise AssertionError(f"Raised {error} when it should not.")
    except Exception as error:
        raise AssertionError(f"Raised an unexpected exception {error}.")


def assert_model_parameters_equal(model1, model2, rtol=1e-5, equal_names=True):
    """Compare parameters between two models."""
    # Check parameter names if required
    if equal_names:
        assert set(model1.parameters.index) == set(model2.parameters.index)

    # Get common parameters
    common_params = set(model1.parameters.index) & set(model2.parameters.index)

    # Check initial values
    for param in common_params:
        assert np.isclose(
            model1.parameters.loc[param, "initial"],
            model2.parameters.loc[param, "initial"],
            rtol=rtol,
        ), f"Parameter {param} initial values differ"


def parametrize_with_cases(cases, ids=None):
    """Utility decorator for parameterizing with test cases.

    Example:
        test_cases = [
            {"name": "case1", "input": 1, "expected": 2},
            {"name": "case2", "input": 2, "expected": 4}
        ]

        @parametrize_with_cases(test_cases)
        def test_something(name, input, expected):
            assert input * 2 == expected
    """
    param_names = cases[0].keys()
    param_values = [tuple(case[name] for name in param_names) for case in cases]

    if ids is None and "name" in param_names:
        name_idx = list(param_names).index("name")
        ids = [case[name_idx] for case in param_values]

    return pytest.mark.parametrize(", ".join(param_names), param_values, ids=ids)
