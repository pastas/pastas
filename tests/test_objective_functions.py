"""Tests for the objective functions in pastas.objective_functions."""

from typing import Any, Type

import numpy as np
import pandas as pd
import pytest

from pastas.objective_functions import GaussianLikelihood, GaussianLikelihoodAr1


class TestGaussianLikelihood:
    """Test class for GaussianLikelihood."""

    def setup_method(self) -> None:
        """Setup for the tests."""
        self.gl = GaussianLikelihood()

    def test_init(self) -> None:
        """Test initialization."""
        assert self.gl.nparam == 1
        assert self.gl._name == "GaussianLikelihood"

    def test_get_init_parameters(self) -> None:
        """Test get_init_parameters method."""
        params = self.gl.get_init_parameters("test")

        assert isinstance(params, pd.DataFrame)
        assert "test_var" in params.index
        assert params.shape == (1, 7)
        assert params.loc["test_var", "initial"] == 0.05
        assert params.loc["test_var", "pmin"] == 1e-10
        assert params.loc["test_var", "pmax"] == 1
        assert params.loc["test_var", "vary"]
        assert params.loc["test_var", "name"] == "test"
        assert params.loc["test_var", "dist"] == "uniform"

    def test_compute(self) -> None:
        """Test compute method."""
        # Create a simple residual array
        rv = np.array([0.1, -0.2, 0.3, -0.1, 0.2])

        # Test with different variance values
        var_1 = 0.05
        ll_1 = self.gl.compute(rv, [var_1])

        var_2 = 0.1
        ll_2 = self.gl.compute(rv, [var_2])

        # Log-likelihood should be higher for variance closer to the true variance
        true_var = np.var(rv)
        var_closer = var_1 if abs(var_1 - true_var) < abs(var_2 - true_var) else var_2
        ll_closer = ll_1 if var_closer == var_1 else ll_2
        ll_farther = ll_2 if var_closer == var_1 else ll_1

        assert ll_closer > ll_farther

        # Test expected formula results
        N = len(rv)
        expected_ll = -0.5 * N * np.log(2 * np.pi * var_1) - np.sum(rv**2) / (2 * var_1)
        assert np.isclose(ll_1, expected_ll)


class TestGaussianLikelihoodAr1:
    """Test class for GaussianLikelihoodAr1."""

    def setup_method(self) -> None:
        """Setup for the tests."""
        self.glar1 = GaussianLikelihoodAr1()

    def test_init(self) -> None:
        """Test initialization."""
        assert self.glar1.nparam == 2
        assert self.glar1._name == "GaussianLikelihoodAr1"

    def test_get_init_parameters(self) -> None:
        """Test get_init_parameters method."""
        params = self.glar1.get_init_parameters("test")

        assert isinstance(params, pd.DataFrame)
        assert "test_var" in params.index
        assert "test_phi" in params.index
        assert params.shape == (2, 7)

        # Check var parameter
        assert params.loc["test_var", "initial"] == 0.05
        assert params.loc["test_var", "pmin"] == 1e-10
        assert params.loc["test_var", "pmax"] == 1

        # Check phi parameter
        assert params.loc["test_phi", "initial"] == 0.5
        assert params.loc["test_phi", "pmin"] == 1e-10
        assert params.loc["test_phi", "pmax"] == 0.99999

    def test_compute(self) -> None:
        """Test compute method."""
        # Create auto-correlated residuals
        np.random.seed(42)
        rv_base = np.random.normal(0, 1, 100)
        phi = 0.7
        rv = np.zeros_like(rv_base)
        rv[0] = rv_base[0]
        for i in range(1, len(rv)):
            rv[i] = phi * rv[i - 1] + rv_base[i]

        # Test with correct vs incorrect phi values
        var = np.var(rv)
        ll_correct = self.glar1.compute(rv, [var, phi])
        ll_wrong = self.glar1.compute(rv, [var, 0.3])

        # Log-likelihood should be higher for correct phi value
        assert ll_correct > ll_wrong

        # Test expected formula results
        N = len(rv)
        expected_ll = -(N - 1) / 2 * np.log(2 * np.pi * var) - np.sum(
            (rv[1:] - phi * rv[:-1]) ** 2
        ) / (2 * var)
        assert np.isclose(ll_correct, expected_ll)

    def test_edge_cases(self) -> None:
        """Test edge cases."""
        # Test with extreme values of phi
        rv = np.array([0.1, 0.08, 0.064, 0.0512])  # Simple AR(1) process with phi=0.8
        var = 0.001

        # Near-zero phi
        ll_zero = self.glar1.compute(rv, [var, 1e-10])

        # Near-one phi
        ll_one = self.glar1.compute(rv, [var, 0.99999])

        # These should both be valid log-likelihoods
        assert np.isfinite(ll_zero)
        assert np.isfinite(ll_one)


@pytest.mark.parametrize(
    "likelihood_class", [GaussianLikelihood, GaussianLikelihoodAr1]
)
def test_likelihood_parameter_types(likelihood_class: Type[Any]) -> None:
    """Test parameter types returned by get_init_parameters."""
    likelihood = likelihood_class()
    params = likelihood.get_init_parameters("test")
    # Check datatypes
    assert isinstance(params.loc[:, "initial"].values[0], float)
    assert isinstance(params.loc[:, "pmin"].values[0], float)
    assert isinstance(params.loc[:, "pmax"].values[0], float)
    assert isinstance(
        params.loc[:, "vary"].values[0], (bool, np.bool_)
    )  # Allow Python bool
    assert isinstance(params.loc[:, "stderr"].values[0], float)
    assert isinstance(params.loc[:, "name"].values[0], str)
    assert isinstance(params.loc[:, "dist"].values[0], str)
