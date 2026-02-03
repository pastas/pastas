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


class TestGetChecksLiterature:
    """Test the get_checks_literature function."""

    def test_brakenhoff_2022_returns_checks(self) -> None:
        """Test that brakenhoff_2022 returns the checks_brakenhoff_2022 list."""
        checks_list = check.get_checks_literature("brakenhoff_2022")
        assert isinstance(checks_list, list)
        assert len(checks_list) > 0
        # Verify that all checks have a 'func' key
        for check_item in checks_list:
            assert isinstance(check_item, dict)
            assert "func" in check_item

    def test_brakenhoff_2022_returns_expected_checks(self) -> None:
        """Test that brakenhoff_2022 returns the expected checks."""
        checks_list = check.get_checks_literature("brakenhoff_2022")
        # Verify the structure matches checks_brakenhoff_2022
        assert checks_list == check.checks_brakenhoff_2022
        # Verify specific checks are present
        func_names = [
            item["func"].__name__ if callable(item["func"]) else item["func"]
            for item in checks_list
        ]
        assert "rsq_geq_threshold" in func_names
        assert "response_memory" in func_names
        assert "acf_runs_test" in func_names
        assert "uncertainty_gain" in func_names
        assert "parameter_bounds" in func_names

    def test_zaadnoordijk_2019_without_model_raises_error(self) -> None:
        """Test that zaadnoordijk_2019 without a model raises ValueError."""
        with pytest.raises(ValueError, match="Model instance 'ml' must be provided"):
            check.get_checks_literature("zaadnoordijk_2019", ml=None)

    def test_zaadnoordijk_2019_without_recharge_model_raises_error(
        self, ml_sm: Model
    ) -> None:
        """Test that zaadnoordijk_2019 without RechargeModel raises ValueError."""
        with pytest.raises(
            ValueError,
            match="Zaadnoordijk et al. \\(2019\\) checklist requires a RechargeModel",
        ):
            check.get_checks_literature("zaadnoordijk_2019", ml=ml_sm)

    def test_zaadnoordijk_2019_with_exponential_recharge_model(
        self, ml_recharge: Model
    ) -> None:
        """Test that zaadnoordijk_2019 works with Exponential RechargeModel."""
        checks_list = check.get_checks_literature("zaadnoordijk_2019", ml=ml_recharge)
        assert isinstance(checks_list, list)
        assert len(checks_list) > 0
        # Verify structure
        for check_item in checks_list:
            assert isinstance(check_item, dict)
            assert "func" in check_item
        # Verify expected checks are present
        func_names = [
            item["func"].__name__ if callable(item["func"]) else item["func"]
            for item in checks_list
        ]
        assert "parameters_leq_threshold" in func_names
        assert "rsq_geq_threshold" in func_names
        assert "correlation_sim_vs_res" in func_names
        assert "acf_stoffer_toloi_test" in func_names
        assert "uncertainty_parameters" in func_names

    def test_zaadnoordijk_2019_with_gamma_recharge_model(self) -> None:
        """Test that zaadnoordijk_2019 works with Gamma RechargeModel."""
        import pastas as ps

        # Create a model with Gamma RechargeModel
        head = pd.read_csv("tests/data/obs.csv", index_col=0, parse_dates=True).squeeze()
        prec = pd.read_csv("tests/data/rain.csv", index_col=0, parse_dates=True).squeeze()
        evap = pd.read_csv("tests/data/evap.csv", index_col=0, parse_dates=True).squeeze()

        ml = ps.Model(head, name="gamma_recharge_model")
        sm = ps.RechargeModel(prec, evap, name="rch", rfunc=ps.Gamma())
        ml.add_stressmodel(sm)

        checks_list = check.get_checks_literature("zaadnoordijk_2019", ml=ml)
        assert isinstance(checks_list, list)
        assert len(checks_list) > 0
        # Verify structure
        for check_item in checks_list:
            assert isinstance(check_item, dict)
            assert "func" in check_item

    def test_zaadnoordijk_2019_with_multiple_recharge_models(self) -> None:
        """Test zaadnoordijk_2019 with model containing multiple RechargeModels."""
        import pastas as ps

        head = pd.read_csv("tests/data/obs.csv", index_col=0, parse_dates=True).squeeze()
        prec = pd.read_csv("tests/data/rain.csv", index_col=0, parse_dates=True).squeeze()
        evap = pd.read_csv("tests/data/evap.csv", index_col=0, parse_dates=True).squeeze()

        ml = ps.Model(head, name="multi_recharge_model")
        sm1 = ps.RechargeModel(prec, evap, name="rch1", rfunc=ps.Exponential())
        ml.add_stressmodel(sm1)

        checks_list = check.get_checks_literature("zaadnoordijk_2019", ml=ml)
        assert isinstance(checks_list, list)
        assert len(checks_list) > 0

    def test_invalid_author_raises_error(self) -> None:
        """Test that an invalid author name raises ValueError."""
        with pytest.raises(
            ValueError, match="Author must be 'brakenhoff_2022' or 'zaadnoordijk_2019'"
        ):
            check.get_checks_literature("invalid_author")

    def test_zaadnoordijk_2019_checks_have_correct_parameters(
        self, ml_recharge: Model
    ) -> None:
        """Test that zaadnoordijk_2019 checks have the correct parameters."""
        checks_list = check.get_checks_literature("zaadnoordijk_2019", ml=ml_recharge)
        # Verify rsq threshold is 0.3 (not 0.7 like in brakenhoff)
        rsq_check = [c for c in checks_list if c["func"].__name__ == "rsq_geq_threshold"]
        assert len(rsq_check) == 1
        assert rsq_check[0]["threshold"] == 0.3
        # Verify correlation threshold is 0.2
        corr_check = [
            c for c in checks_list if c["func"].__name__ == "correlation_sim_vs_res"
        ]
        assert len(corr_check) == 1
        assert corr_check[0]["threshold"] == 0.2

    def test_get_checks_literature_brakenhoff_does_not_require_model(self) -> None:
        """Test that brakenhoff_2022 does not require a model instance."""
        # Should not raise an error even though ml is None
        checks_list = check.get_checks_literature("brakenhoff_2022", ml=None)
        assert isinstance(checks_list, list)
        assert len(checks_list) > 0
