from collections.abc import Callable
from typing import Any

import pandas as pd
import pytest

from pastas import Model, check


def test_response_memory(ml_noisemodel: Model) -> None:
    df = check.response_memory(ml_noisemodel, cutoff=0.95, factor_length_oseries=0.5)
    assert df["pass"].item()


def test_rsq_geq_threshold(ml_noisemodel: Model) -> None:
    df = check.rsq_geq_threshold(ml_noisemodel, threshold=0.7)
    assert df["pass"].item()


def test_acf_runs_test(ml_noisemodel: Model) -> None:
    df = check.acf_runs_test(ml_noisemodel)
    assert df["pass"].item()


def test_acf_stoffer_toloi(ml_noisemodel: Model) -> None:
    df = check.acf_stoffer_toloi_test(ml_noisemodel)
    assert df["pass"].item()


def test_parameter_bounds(ml_noisemodel: Model) -> None:
    df = check.parameter_bounds(ml_noisemodel)
    assert df["pass"].all().item()


def test_parameter_uncertainty(ml_noisemodel: Model) -> None:
    df = check.uncertainty_parameters(ml_noisemodel, n_std=2)
    assert df["pass"].all().item()


def test_uncertainty_gain(ml_noisemodel: Model) -> None:
    df = check.uncertainty_gain(ml_noisemodel, n_std=2)
    assert df["pass"].item()


def test_checklist(ml_noisemodel: Model) -> None:
    df = check.checklist(ml_noisemodel, check.checks_brakenhoff_2022)
    assert df["pass"].all().item()


# New tests to improve test coverage


class TestOperatorsAndUtilities:
    """Test the operator mapping and utility functions."""

    def test_operators_dict(self) -> None:
        """Test that the operators dictionary has correct mappings."""
        assert check.operators["greater_equal"] == ">="
        assert check.operators["less_equal"] == "<="  # Check the fixed operator
        assert check.operators["greater_than"] == ">"
        assert check.operators["less_than"] == "<"
        assert check.operators["equal"] == "=="
        assert check.operators["not_equal"] != "=="

    def test_guess_unit_or_dims(self) -> None:
        """Test the guess_unit_or_dims function."""
        # Test _A parameter
        result = check.guess_unit_or_dims("precip_A")
        assert "[L]" in result
        assert "precip" in result

        # Test noise_alpha
        result = check.guess_unit_or_dims("noise_alpha", return_dims=False)
        assert result == "days"

        # Test constant_d
        result = check.guess_unit_or_dims("constant_d")
        assert result == "[L]"

        # Test _f parameter
        result = check.guess_unit_or_dims("precip_f")
        assert result == "[-]"

        # Test unknown parameter
        result = check.guess_unit_or_dims("unknown_parameter")
        assert result == ""


class TestChecklistFunction:
    """Test the checklist function with different input types."""

    def test_checklist_with_string(self, ml_noisemodel: Model) -> None:
        """Test checklist with string function name."""
        checks: list[str] = ["rsq_geq_threshold"]
        result = check.checklist(ml_noisemodel, checks, report=False)

        # Check that result has the expected format
        assert isinstance(result, pd.DataFrame)
        assert "rsq>=0.7" in result.index

    def test_checklist_with_callable(self, ml_noisemodel: Model) -> None:
        """Test checklist with callable function."""
        checks: list[Callable[[Model], pd.DataFrame]] = [
            lambda ml: check.rsq_geq_threshold(ml, threshold=0.8)
        ]
        result = check.checklist(ml_noisemodel, checks, report=False)

        # Check that result has the expected format
        assert isinstance(result, pd.DataFrame)
        assert "rsq>=0.8" in result.index

    def test_checklist_with_dict(self, ml_noisemodel: Model) -> None:
        """Test checklist with dictionary."""
        checks: list[dict[str, str | float]] = [
            {"func": "rsq_geq_threshold", "threshold": 0.9}
        ]
        result = check.checklist(ml_noisemodel, checks, report=False)

        # Check that result has the expected format
        assert isinstance(result, pd.DataFrame)
        assert "rsq>=0.9" in result.index

    def test_checklist_with_invalid_type(self, ml_noisemodel: Model) -> None:
        """Test checklist with invalid type."""
        checks: list[int] = [123]  # Not a string, callable, or dict
        with pytest.raises(TypeError):
            check.checklist(ml_noisemodel, checks, report=False)


@pytest.mark.parametrize(
    "check_func,kwargs",
    [
        (check.rsq_geq_threshold, {"threshold": 0.5}),
        (check.acf_runs_test, {"p_threshold": 0.01}),
        (check.acf_stoffer_toloi_test, {"p_threshold": 0.01}),
    ],
)
def test_check_functions_parameterized(
    ml_noisemodel: Model,
    check_func: Callable[[Model, Any], pd.DataFrame],
    kwargs: dict[str, float],
) -> None:
    """Test various check functions with different parameters."""
    df = check_func(ml_noisemodel, **kwargs)
    assert isinstance(df, pd.DataFrame)
    assert "pass" in df.columns
    assert "statistic" in df.columns
    assert "threshold" in df.columns
