import numpy as np
import pytest

from pastas.solver import jacobian
from pastas.typing import ArrayLike


def quad_lin_fun(x: ArrayLike) -> ArrayLike:
    # f: R^3 -> R^2
    # f1 = x0^2 + 3*x1 - x2
    # f2 = sin(x0) + x1*x2
    return np.array(
        [
            x[0] ** 2 + 3.0 * x[1] - x[2],
            np.sin(x[0]) + x[1] * x[2],
        ]
    )


def quad_lin_jac(x: ArrayLike) -> ArrayLike:
    # Analytical Jacobian (2 x 3)
    # df1/dx = [2*x0, 3, -1]
    # df2/dx = [cos(x0), x2, x1]
    return np.array(
        [
            [2.0 * x[0], 3.0, -1.0],
            [np.cos(x[0]), x[2], x[1]],
        ]
    )


def square_scalar(x: ArrayLike) -> float:
    # f: R^1 -> R^1
    return float(x[0] ** 2)


@pytest.mark.parametrize("method", ["2-point", "3-point"])
def test_jacobian_matches_analytic_vector_function(method):
    x0 = np.array([0.3, -1.2, 0.7])
    J_true = quad_lin_jac(x0)
    J_num = jacobian(quad_lin_fun, x0, method=method)
    # Finite-difference errors scale with step; be tolerant but strict
    assert J_num.shape == J_true.shape
    np.testing.assert_allclose(J_num, J_true, rtol=1e-5, atol=1e-7)


def test_invalid_method_raises():
    x0 = np.array([0.0])
    with pytest.raises(ValueError):
        jacobian(square_scalar, x0, method="5-point")  # not supported (yet)


@pytest.mark.parametrize("method", ["2-point", "3-point"])
def test_bounds_are_respected_during_evaluation(method):
    # Track every x at which fun is evaluated
    eval_points = []

    def tracked_fun(x):
        eval_points.append(np.array(x, copy=True))
        # simple linear function to keep Jacobian constant
        return np.array([2.0 * x[0] + 1.0])

    # Choose x0 at the lower bound; the implementation must not step below it
    lower, upper = 0.0, np.inf
    x0 = np.array([0.0])

    _ = jacobian(tracked_fun, x0, method=method, bounds=(lower, upper))

    # All evaluations must satisfy the bounds
    eval_points = np.array(eval_points, dtype=float).reshape(-1)
    assert np.all(eval_points >= lower - 1e-15)
    # upper is inf; no need to check


@pytest.mark.parametrize("method", ["2-point", "3-point"])
def test_abs_step_is_used_when_provided(method):
    # Track evaluation points to infer step magnitude
    eval_points = []

    def tracked_fun(x):
        eval_points.append(np.array(x, copy=True))
        return np.array([x[0] ** 3])  # smooth, non-linear

    x0 = np.array([0.5])
    abs_step = 1e-6
    _ = jacobian(
        tracked_fun, x0, method=method, abs_step=abs_step, bounds=(-np.inf, np.inf)
    )

    eval_points = np.array(eval_points, dtype=float).reshape(-1)
    # Compute distances from x0, ignore zero (exact x0 evaluation)
    deltas = np.abs(eval_points - x0[0])
    deltas = deltas[deltas > 0]

    assert deltas.size > 0
    # At least one of the steps should match the requested abs_step (within FD adjustments)
    assert np.any(np.isclose(deltas, abs_step, rtol=1e-3, atol=1e-12))


@pytest.mark.parametrize("method", ["2-point", "3-point"])
def test_rel_step_controls_error_scale(method):
    # Check that a smaller rel_step typically yields a smaller error (until rounding)
    x0 = np.array([0.7, -0.4, 1.1])
    J_true = quad_lin_jac(x0)

    J_big = jacobian(quad_lin_fun, x0, method=method, rel_step=1e-4)
    J_small = jacobian(quad_lin_fun, x0, method=method, rel_step=1e-6)

    err_big = np.linalg.norm(J_big - J_true)
    err_small = np.linalg.norm(J_small - J_true)

    # Allow rare rounding anomalies, but generally smaller step improves accuracy here
    assert err_small <= err_big * 1.1
