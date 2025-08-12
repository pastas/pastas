"""Tests for the solver module in Pastas."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame, Series

import pastas as ps
from pastas.solver import BaseSolver, EmceeSolve, LeastSquares, LmfitSolve


@pytest.fixture
def mock_model() -> MagicMock:
    """Create a mock model for solver testing."""
    model = MagicMock()

    # Setup parameters
    parameters = DataFrame(
        data={
            "initial": [1.0, 2.0, 3.0],
            "pmin": [0.1, 0.2, 0.3],
            "pmax": [10.0, 20.0, 30.0],
            "vary": [True, True, False],
            "name": ["param1", "param2", "param3"],
            "stderr": [0.1, 0.2, 0.3],
        },
        index=["p1", "p2", "p3"],
    )
    model.parameters = parameters

    # Setup time series data
    dates = pd.date_range(start="2000-01-01", periods=100, freq="D")
    residuals = Series(np.random.normal(0, 1, 100), index=dates)
    noise = Series(np.random.normal(0, 0.5, 100), index=dates)

    # Mock methods
    model.residuals.return_value = residuals
    model.noise.return_value = noise
    model.noise_weights.return_value = Series(np.ones(100), index=dates)
    model.get_parameters.return_value = np.array([1.0, 2.0])
    model.simulate.return_value = Series(
        np.random.normal(5, 1, 100), index=dates, name="Simulation"
    )
    model.get_contribution.return_value = Series(
        np.random.normal(2, 0.5, 100), index=dates, name="Contribution"
    )
    model.get_block_response.return_value = Series(
        np.exp(-np.arange(50) / 10), index=np.arange(50), name="Block Response"
    )
    model.get_step_response.return_value = Series(
        1 - np.exp(-np.arange(50) / 10), index=np.arange(50), name="Step Response"
    )

    return model


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


# Unit tests for the BaseSolver class using pytest style
class TestBaseSolver:
    """Test the BaseSolver class."""

    def test_init_empty(self) -> None:
        """Test empty initialization of BaseSolver."""
        solver = BaseSolver()
        assert solver._name == "BaseSolver"
        assert solver.ml is None
        assert solver.pcov is None
        assert solver.nfev is None

    def test_init_with_params(self) -> None:
        """Test initialization with parameters."""
        pcov_data = [[0.1, 0.02], [0.02, 0.2]]
        pcov = DataFrame(pcov_data, index=["p1", "p2"], columns=["p1", "p2"])
        solver = BaseSolver(pcov=pcov, nfev=10)

        assert solver.pcov is pcov
        assert solver.nfev == 10

        # Check correlation matrix calculation
        expected_corr = np.array(
            [
                [1.0, 0.02 / (np.sqrt(0.1) * np.sqrt(0.2))],
                [0.02 / (np.sqrt(0.1) * np.sqrt(0.2)), 1.0],
            ]
        )
        np.testing.assert_almost_equal(solver.pcor.values, expected_corr)

    def test_set_model(self, mock_model: MagicMock) -> None:
        """Test setting a model in the solver."""
        solver = BaseSolver()
        solver.set_model(mock_model)
        assert solver.ml is mock_model

    def test_set_model_raises_warning(self, mock_model: MagicMock) -> None:
        """Test that error is raised when setting a second model."""
        solver = BaseSolver()
        solver.set_model(mock_model)
        with pytest.raises(UserWarning):
            solver.set_model(MagicMock())

    @pytest.mark.parametrize(
        "noise,expected_calls",
        [
            (False, ["residuals"]),
            (True, ["noise", "noise_weights"]),
        ],
    )
    def test_misfit_basic(
        self, mock_model: MagicMock, noise: bool, expected_calls: list[str]
    ) -> None:
        """Test misfit function with different noise settings."""
        solver = BaseSolver()
        solver.ml = mock_model
        mock_model.reset_mock()

        result = solver.misfit(p=np.array([1.0, 2.0]), noise=noise)

        assert isinstance(result, np.ndarray)
        assert len(result) == 100

        # Verify the right methods were called
        for method in expected_calls:
            assert getattr(mock_model, method).called

    def test_misfit_with_weights(self, mock_model: MagicMock) -> None:
        """Test misfit function with weights."""
        solver = BaseSolver()
        solver.ml = mock_model
        weights = Series(
            np.random.uniform(0.5, 1.0, 100), index=mock_model.residuals().index
        )

        weighted_result = solver.misfit(
            p=np.array([1.0, 2.0]), noise=False, weights=weights
        )

        assert isinstance(weighted_result, np.ndarray)
        assert len(weighted_result) == 100

    def test_misfit_returnseparate(self, mock_model: MagicMock) -> None:
        """Test misfit function with returnseparate=True."""
        solver = BaseSolver()
        solver.ml = mock_model

        result = solver.misfit(p=np.array([1.0, 2.0]), noise=False, returnseparate=True)

        assert isinstance(result, tuple)
        assert len(result) == 3
        assert all(isinstance(arr, np.ndarray) for arr in result)

    def test_to_dict(self) -> None:
        """Test to_dict method."""
        solver = BaseSolver(nfev=100)
        data_dict = solver.to_dict()

        assert data_dict["class"] == "BaseSolver"
        assert data_dict["nfev"] == 100
        assert data_dict["pcov"] is None


class TestLeastSquares:
    """Test the LeastSquares solver class."""

    def test_init(self) -> None:
        """Test initialization of LeastSquares solver."""
        solver = LeastSquares()
        assert solver._name == "LeastSquares"

    @patch("pastas.solver.least_squares")
    def test_solve(self, mock_least_squares: MagicMock, mock_model: MagicMock) -> None:
        """Test solve method."""
        # Setup mock
        mock_result = MagicMock(
            success=True,
            x=np.array([1.5, 2.5]),
            cost=10.0,
            nfev=50,
            jac=np.array([[0.1, 0.05], [0.05, 0.2], [0.01, 0.02]]),
        )
        mock_least_squares.return_value = mock_result

        # Create solver and solve
        solver = LeastSquares()
        solver.ml = mock_model
        success, optimal, stderr = solver.solve(noise=True)

        # Assertions
        assert success is True
        assert len(optimal) == 3
        assert np.allclose(optimal[:2], [1.5, 2.5])
        assert isinstance(stderr, np.ndarray)
        assert solver.nfev == 50

    @pytest.mark.parametrize(
        "method,absolute_sigma",
        [("trf", False), ("trf", True), ("lm", False), ("dogbox", False)],
    )
    def test_get_covariances(self, method: str, absolute_sigma: bool) -> None:
        """Test get_covariances with different methods and parameters."""
        jacobian = np.array([[1.0, 0.5], [0.5, 1.0], [0.2, 0.3]])
        cost = 5.0

        pcov = LeastSquares.get_covariances(
            jacobian, cost, method=method, absolute_sigma=absolute_sigma
        )

        assert isinstance(pcov, np.ndarray)
        assert pcov.shape == (2, 2)
        assert not np.isnan(pcov).any()


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
