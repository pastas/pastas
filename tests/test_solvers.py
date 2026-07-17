"""Tests for the solver module in Pastas."""

import logging
from functools import partial

import numpy as np
import pandas as pd
import pytest
from scipy.optimize._numdiff import approx_derivative

import pastas as ps
from pastas.solver import EmceeSolve, LmfitSolve
from pastas.solver.objective_function import misfit


# Existing integration tests with real models
def test_least_squares(ml_recharge: ps.Model) -> None:
    ps.solver.LeastSquares(model=ml_recharge)
    ml_recharge.solve()


def test_least_squares_lm(ml_recharge: ps.Model) -> None:
    ps.solver.LeastSquares(model=ml_recharge)
    ml_recharge.solve(method="lm")
    assert ml_recharge.parameters.loc[ml_recharge.parameters.vary, "pmin"].isna().all()


def test_fit_constant(ml_recharge: ps.Model) -> None:
    ml_recharge.solve(fit_constant=False)


def test_no_noise(ml_recharge: ps.Model) -> None:
    ml_recharge.del_noisemodel()
    ml_recharge.solve()


def test_misfit_uses_sqrt_weights(ml_recharge: ps.Model) -> None:
    """Verify weighted least-squares uses weights on residual terms."""
    ml_recharge.del_noisemodel()
    ps.solver.LeastSquares(model=ml_recharge)
    ml_recharge.solve(report=False)

    p = ml_recharge.get_parameters()
    residuals = ml_recharge.residuals(p)

    weights = pd.Series(1.0, index=residuals.index)
    weights.iloc[::5] = 0.25

    misfit = ps.solver.misfit(ml=ml_recharge, p=p, noise=False, weights=weights)
    expected = (residuals * weights).values

    np.testing.assert_allclose(misfit, expected)


# Tests for confidence intervals and prediction intervals
def test_pred_interval(ml_solved: ps.Model) -> None:
    pi = ml_solved.solver.prediction_interval(n=10)
    assert isinstance(pi, pd.DataFrame)
    assert pi.shape[1] == 2
    assert list(pi.columns) == [0.025, 0.975]


def test_ci_simulation(ml_solved: ps.Model) -> None:
    ci = ml_solved.solver.ci_simulation(n=10)
    assert isinstance(ci, pd.DataFrame)
    assert ci.shape[1] == 2


def test_ci_block_response(ml_solved: ps.Model) -> None:
    ci = ml_solved.solver.ci_block_response(name="rch", n=10)
    assert isinstance(ci, pd.DataFrame)
    assert ci.shape[1] == 2


def test_ci_step_response(ml_solved: ps.Model) -> None:
    ci = ml_solved.solver.ci_step_response(name="rch", n=10)
    assert isinstance(ci, pd.DataFrame)
    assert ci.shape[1] == 2


def test_ci_contribution(ml_solved: ps.Model) -> None:
    ci = ml_solved.solver.ci_contribution(name="rch", n=10)
    assert isinstance(ci, pd.DataFrame)
    assert ci.shape[1] == 2


# Test the EmceeSolver
def test_emcee(ml_recharge: ps.Model) -> None:
    try:
        ps.solver.LeastSquares(model=ml_recharge)
        ml_recharge.solve()
        ml_recharge.del_noisemodel()

        ps.solver.Emcee(model=ml_recharge, nwalkers=10)

        ml_recharge.set_parameter("constant_d", pmin=26, pmax=29.0)

        for name in ml_recharge.parameters.index:
            ml_recharge.set_parameter(name, dist="uniform")

        ml_recharge.solve(
            initial=False,
            fit_constant=True,
            steps=2,
        )
    except ImportError:
        pytest.skip("emcee not installed, skipping test")


class TestOptionalSolvers:
    """Tests for solvers that depend on optional dependencies."""

    def test_lmfit_solve_init(self, ml_recharge: ps.Model) -> None:
        """Test LmfitSolve initialization."""
        try:
            solver = LmfitSolve(model=ml_recharge)
            assert solver._name == "Lmfit"
        except ImportError:
            pytest.skip("lmfit not installed")

    def test_emcee_solve_init(self, ml_recharge: ps.Model) -> None:
        """Test EmceeSolve initialization."""
        try:
            solver = EmceeSolve(model=ml_recharge)
            assert solver._name == "Emcee"
            assert solver.nwalkers == 20
            assert solver.progress_bar is True
        except ImportError:
            pytest.skip("emcee not installed")

    def test_emcee_to_dict_warning(self, ml_recharge: ps.Model, caplog) -> None:
        """Test that EmceeSolve.to_dict caplogs a logger.warning."""
        try:
            solver = EmceeSolve(model=ml_recharge)
            with caplog.at_level(logging.WARNING, logger="pastas.solver.mcmc"):
                solver.to_dict()
                assert (
                    "Note that the EmceeSolve class is not fully reproducible."
                    in caplog.text
                )
        except ImportError:
            pytest.skip("emcee not installed")


def test_leastsquares_covariance_scenarios(
    head: pd.Series, prec: pd.Series, evap: pd.Series
) -> None:
    """Test the covariance matrix calculation in LeastSquares solver under different scenarios.

    This test verifies that the internal SVD method for calculating the covariance matrix is
    consistent with manual calculations using the Jacobian and residuals, both in weighted and
    unweighted scenarios. It also checks the behavior when absolute_sigma=True.

    """
    # 1. Setup Data & Model
    # Using small subset for speed

    ml = ps.Model(head)
    ps.RechargeModel(model=ml, prec=prec, evap=evap, name="rch")

    weights_random = pd.Series(
        np.random.RandomState(seed=0).rand(len(head)), index=head.index
    )
    weights_random_root = weights_random.pow(0.5)  # For sqrt(weights) scenario

    # Solve
    jac_method = "2-point"
    ml.solve(weights=weights_random_root, report=False, jac=jac_method)
    p_opt = ml.parameters.optimal.values
    pcov_internal = ml.solver.pcov.values  # This uses the code you pasted

    # 2. Define Manual Inversion Function (The "Traditional" way)
    def manual_pcov(J, res, w, nobs, npar):
        sse = np.sum(w * res**2)
        s_sq = sse / (nobs - npar)
        # Standard WLS formula: (J.T @ W @ J)^-1 * s_sq
        return np.linalg.inv(J.T @ np.diag(w) @ J) * s_sq

    # --- SCENARIO A: Verify Internal vs Manual (Weighted) ---
    # We use the solver's Jacobian (which is weighted) and cost.
    # To use manual_pcov with a weighted Jacobian, we pass weights=ones.
    nobs, npar = ml.solver.result.jac.shape
    res_weighted = misfit(ml=ml, p=p_opt, noise=False, weights=weights_random_root)
    pcov_manual_weighted = manual_pcov(
        ml.solver.result.jac, res_weighted, np.ones(nobs), nobs, npar
    )

    assert np.allclose(pcov_internal, pcov_manual_weighted, rtol=1e-6), (
        "Internal SVD method differs from manual inversion using weighted Jacobian."
    )

    # --- SCENARIO B: Verify Pure Reconstruction ---
    # Get unweighted (pure) components
    res_pure = misfit(ml=ml, p=p_opt, noise=False, weights=None)

    fun_pure = partial(
        ml.solver.objfunction,
        initial=ml.parameters.initial.to_numpy(dtype=float, copy=True),
        vary=ml.parameters.vary.to_numpy(dtype=bool, copy=True),
        weights=None,
        noise=False,
    )
    # Using same 2-point precision to match scipy.least_squares default
    jac_pure = approx_derivative(fun_pure, x0=p_opt, method=jac_method)

    # Now we can use the pure Jacobian and pure residuals with the original
    # random weights to reconstruct pcov.
    pcov_pure_reconstruction = manual_pcov(
        jac_pure, res_pure, weights_random.values, nobs, npar
    )

    # Comparing numerical derivative result to analytical solver result
    # We allow a slightly larger tolerance (1e-4) due to finite difference approx.
    assert np.allclose(pcov_internal, pcov_pure_reconstruction, rtol=1e-4), (
        "Internal pcov differs from pure Jacobian reconstruction."
    )

    # --- SCENARIO C: Verify scaling (absolute_sigma=True) ---
    # If absolute_sigma=True, pcov should not be multiplied by s_sq.
    pcov_abs = ml.solver.get_covariances(
        ml.solver.result.jac, ml.solver.result.cost, absolute_sigma=True
    )
    # Recreate manually: (J_w.T @ J_w)^-1
    pcov_manual_abs = np.linalg.inv(ml.solver.result.jac.T @ ml.solver.result.jac)

    assert np.allclose(pcov_abs, pcov_manual_abs, rtol=1e-6), (
        "absolute_sigma=True calculation is incorrect."
    )
