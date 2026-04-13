"""Tests for the solver module in Pastas."""

from functools import partial

import numpy as np
import pandas as pd
import pytest
from scipy.optimize._numdiff import approx_derivative

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


def test_misfit_uses_sqrt_weights(ml_recharge: ps.Model) -> None:
    """Verify weighted least-squares uses sqrt(weights) on residual terms."""
    ml_recharge.del_noisemodel()
    ml_recharge.solve(solver=ps.LeastSquares(), report=False)

    p = ml_recharge.get_parameters()
    residuals = ml_recharge.residuals(p)

    weights = pd.Series(1.0, index=residuals.index)
    weights.iloc[::5] = 0.25

    misfit = ml_recharge.solver.misfit(p=p, noise=False, weights=weights)
    expected = (residuals * np.sqrt(weights)).values

    np.testing.assert_allclose(misfit, expected)


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


def test_leastsquares_covariance_scenarios(head, prec, evap):
    # 1. Setup Data & Model
    # Using small subset for speed

    ml = ps.Model(head)
    rm = ps.RechargeModel(prec, evap, name="rch")
    ml.add_stressmodel(rm)

    weights_random = pd.Series(
        np.random.RandomState(seed=0).rand(len(head)), index=head.index
    )

    # Solve
    jac_method = "2-point"
    ml.solve(weights=weights_random, report=False, jac=jac_method)
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
    res_weighted = ml.solver.misfit(p_opt, weights=weights_random, noise=False)
    pcov_manual_weighted = manual_pcov(
        ml.solver.result.jac, res_weighted, np.ones(nobs), nobs, npar
    )

    assert np.allclose(pcov_internal, pcov_manual_weighted, rtol=1e-6), (
        "Internal SVD method differs from manual inversion using weighted Jacobian."
    )

    # --- SCENARIO B: Verify Pure Reconstruction ---
    # Get unweighted (pure) components
    res_pure = ml.solver.misfit(p_opt, weights=None, noise=False)

    fun_pure = partial(
        ml.solver.objfunction,
        weights=None,
        noise=ml.settings["noise"],
        callback=None,
    )
    # Using 3-point for higher precision to match the solver
    jac_pure = approx_derivative(fun_pure, x0=p_opt, method=jac_method)

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
