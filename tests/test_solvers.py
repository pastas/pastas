"""Tests for the solver module in Pastas."""

import pandas as pd
import pytest

import pastas as ps
from pastas.solver import EmceeSolve, LmfitSolve


# Existing integration tests with real models
def test_least_squares(ml_recharge: ps.Model) -> None:
    ml_recharge.solve(solver=ps.LeastSquares())


def test_least_squares_lm(ml_recharge: ps.Model) -> None:
    ml_recharge.solve(solver=ps.LeastSquares(), method="lm")
    assert ml_recharge.parameters.loc[ml_recharge.parameters.vary, "pmin"].isna().all()


def test_fit_constant(ml_recharge: ps.Model) -> None:
    ml_recharge.solve(fit_constant=False)


def test_no_noise(ml_recharge: ps.Model) -> None:
    ml_recharge.del_noisemodel()
    ml_recharge.solve()


# Tests for confidence intervals and prediction intervals
def test_pred_interval(ml_recharge: ps.Model) -> None:
    ml_recharge.solve(solver=ps.LeastSquares())
    pi = ml_recharge.solver.prediction_interval(n=10)
    assert isinstance(pi, pd.DataFrame)
    assert pi.shape[1] == 2
    assert list(pi.columns) == [0.025, 0.975]


def test_ci_simulation(ml_recharge: ps.Model) -> None:
    ml_recharge.solve(solver=ps.LeastSquares())
    ci = ml_recharge.solver.ci_simulation(n=10)
    assert isinstance(ci, pd.DataFrame)
    assert ci.shape[1] == 2


def test_ci_block_response(ml_recharge: ps.Model) -> None:
    ml_recharge.solve(solver=ps.LeastSquares())
    ci = ml_recharge.solver.ci_block_response(name="rch", n=10)
    assert isinstance(ci, pd.DataFrame)
    assert ci.shape[1] == 2


def test_ci_step_response(ml_recharge: ps.Model) -> None:
    ml_recharge.solve(solver=ps.LeastSquares())
    ci = ml_recharge.solver.ci_step_response(name="rch", n=10)
    assert isinstance(ci, pd.DataFrame)
    assert ci.shape[1] == 2


def test_ci_contribution(ml_recharge: ps.Model) -> None:
    ml_recharge.solve(solver=ps.LeastSquares())
    ci = ml_recharge.solver.ci_contribution(name="rch", n=10)
    assert isinstance(ci, pd.DataFrame)
    assert ci.shape[1] == 2


# Test the EmceeSolver
def test_emcee(ml_recharge: ps.Model) -> None:
    try:
        ml_recharge.solve(solver=ps.LeastSquares())
        ml_recharge.del_noisemodel()
        ml_recharge.solve(
            solver=ps.EmceeSolve(nwalkers=10),
            initial=False,
            fit_constant=False,
            steps=2,
        )
    except ImportError:
        pytest.skip("emcee not installed, skipping test")


class TestOptionalSolvers:
    """Tests for solvers that depend on optional dependencies."""

    def test_lmfit_solve_init(self) -> None:
        """Test LmfitSolve initialization."""
        try:
            solver = LmfitSolve()
            assert solver._name == "LmfitSolve"
        except ImportError:
            pytest.skip("lmfit not installed")

    def test_emcee_solve_init(self) -> None:
        """Test EmceeSolve initialization."""
        try:
            solver = EmceeSolve()
            assert solver._name == "EmceeSolve"
            assert solver.nwalkers == 20
            assert solver.progress_bar is True
        except ImportError:
            pytest.skip("emcee not installed")

    def test_emcee_to_dict_raises(self) -> None:
        """Test that EmceeSolve.to_dict raises NotImplementedError."""
        try:
            solver = EmceeSolve()
            with pytest.raises(NotImplementedError):
                solver.to_dict()
        except ImportError:
            pytest.skip("emcee not installed")
