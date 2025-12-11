import numpy as np
import pytest

import pastas as ps


@pytest.mark.parametrize("rfunc_name", ps.rfunc.__all__)
@pytest.mark.parametrize("up", [True, False])
def test_rfunc(rfunc_name: str, up: bool) -> None:
    if rfunc_name not in []:
        rfunc = getattr(ps.rfunc, rfunc_name)()
        rfunc.update_rfunc_settings(up=up)
        if rfunc_name == "HantushWellModel":
            rfunc.set_distances(100.0)
        p = rfunc.get_init_parameters("test").initial.to_numpy()
        rfunc.block(p)
        rfunc.step(p)


@pytest.mark.parametrize("rfunc_name", ps.rfunc.__all__)
@pytest.mark.parametrize("up", [True, False])
def test_to_dict_rfuncs(rfunc_name: str, up: bool) -> None:
    rfunc1 = getattr(ps.rfunc, rfunc_name)(cutoff=0.5)
    rfunc1.update_rfunc_settings(up=up)

    # Create the exact same instance using to_dict
    data = rfunc1.to_dict()
    rfunc_class = data.pop("class")  # Determine response class
    rfunc_up = data.pop("up", None)
    rfunc_gsf = data.pop("gain_scale_factor", None)
    rfunc2 = getattr(ps.rfunc, rfunc_class)(**data)
    rfunc2.update_rfunc_settings(up=rfunc_up, gain_scale_factor=rfunc_gsf)
    rfunc2.update_rfunc_settings(up=rfunc_up, gain_scale_factor=rfunc_gsf)

    if rfunc_name == "HantushWellModel":
        rfunc1.set_distances(100.0)
        rfunc2.set_distances(100.0)

    p1 = rfunc1.get_init_parameters("test").initial.to_numpy()
    p2 = rfunc2.get_init_parameters("test").initial.to_numpy()

    assert (rfunc1.step(p1) - rfunc2.step(p2)).sum() == 0.0


@pytest.mark.parametrize("rfunc_name", ps.rfunc.__all__)
@pytest.mark.parametrize("up", [True, False, None])
def test_gain_methods(rfunc_name: str, up: bool) -> None:
    rfunc = getattr(ps.rfunc, rfunc_name)()
    rfunc.update_rfunc_settings(up=up)

    # Set distances for HantushWellModel
    if rfunc_name == "HantushWellModel":
        rfunc.set_distances(100.0)

    # Get parameters
    p = rfunc.get_init_parameters("test").initial.to_numpy()

    # Test gain method exists and returns expected type
    gain_value = rfunc.gain(p)
    assert isinstance(gain_value, (float, np.float64, np.ndarray))

    # Compare gain with final step value for steady-state response functions
    if rfunc_name not in ["FourParam"]:  # Some functions need special handling
        tmax = rfunc.get_tmax(p)
        if np.isfinite(tmax) and tmax > 0:
            step_response = rfunc.step(p)
            # Check if they're approximately equal at steady state
            if len(step_response) > 0:
                assert abs(gain_value - step_response[-1]) < 0.02


@pytest.mark.parametrize("rfunc_name", ["HantushWellModel"])
def test_gain_methods_with_distance(rfunc_name: str) -> None:
    """Test gain methods that require distance parameter."""
    rfunc = getattr(ps.rfunc, rfunc_name)()

    # Set distances
    distances = [50.0, 100.0, 200.0]
    rfunc.set_distances(distances[0])

    # Get parameters
    p = rfunc.get_init_parameters("test").initial.to_numpy()

    # Test gain method with distance parameter
    for distance in distances:
        gain_value = rfunc.gain(p, r=distance)
        assert isinstance(gain_value, (float, np.float64))

        # Test gain method with different distances
        rfunc.set_distances(distance)
        p2 = rfunc.get_init_parameters("test").initial.to_numpy()
        gain_value2 = rfunc.gain(p2)
        assert isinstance(gain_value2, (float, np.float64))


# Response functions that support both discrete and exact moment methods
# FourParam and Edelman have known issues with moment computation and are excluded
RFUNCS_WITH_EXACT_MOMENTS = [
    "Gamma",
    "Exponential",
    "Hantush",
    "Polder",
    "DoubleExponential",
    "FourParam"
]


@pytest.mark.parametrize("rfunc_name", RFUNCS_WITH_EXACT_MOMENTS)
@pytest.mark.parametrize("order", [0, 1, 2, 3, 4])
def test_moment_discrete_vs_exact(rfunc_name: str, order: int) -> None:
    """Test that discrete and exact moment methods produce similar results.

    Parameters
    ----------
    rfunc_name : str
        Name of the response function class to test.
    order : int
        Order of the moment to compute (0-4).
    """
    rfunc = getattr(ps.rfunc, rfunc_name)(cutoff=0.999999)
    p = rfunc.get_init_parameters("test").initial.to_numpy()

    # Compute moments using both methods
    # Use finer time step (dt=0.001) and high cutoff for better accuracy
    moment_discrete = rfunc.moment(p, order=order, method="discrete", dt=0.001)
    moment_exact = rfunc.moment(p, order=order, method="exact", dt=0.001)

    # Check that both are finite numbers
    assert np.isfinite(moment_discrete), (
        f"Discrete moment is not finite: {moment_discrete}"
    )
    assert np.isfinite(moment_exact), f"Exact moment is not finite: {moment_exact}"

    # With fine discretization (dt=0.001) and high cutoff, should get very close agreement
    # Hantush needs slightly higher tolerance (3%) due to numerical accuracy
    relative_error = abs(moment_discrete - moment_exact) / abs(moment_exact)
    tolerance = 0.03 if rfunc_name == "Hantush" else 0.01
    assert relative_error < tolerance, (
        f"{rfunc_name} order {order}: discrete={moment_discrete:.6f}, "
        f"exact={moment_exact:.6f}, relative_error={relative_error:.6f}"
    )


@pytest.mark.parametrize("rfunc_name", RFUNCS_WITH_EXACT_MOMENTS)
@pytest.mark.parametrize("dt", [0.01, 0.1, 0.5])
def test_moment_discrete_converges_with_dt(rfunc_name: str, dt: float) -> None:
    """Test that discrete moment converges as dt decreases.

    Parameters
    ----------
    rfunc_name : str
        Name of the response function class to test.
    dt : float
        Time step for discrete approximation.
    """
    rfunc = getattr(ps.rfunc, rfunc_name)()
    p = rfunc.get_init_parameters("test").initial.to_numpy()

    # Compute exact moments for comparison
    exact_moments = [rfunc.moment(p, order=order, method="exact") for order in range(5)]

    # Compute discrete moments with given dt
    discrete_moments = [
        rfunc.moment(p, order=order, method="discrete", dt=dt) for order in range(5)
    ]

    # All moments should be finite
    for moment_val in discrete_moments:
        assert np.isfinite(moment_val), f"Discrete moment is not finite: {moment_val}"

    # Check that discrete moments are reasonably close to exact
    # Tolerance increases with order and dt due to numerical integration challenges
    if dt == 0.01:
        tolerances = [0.02, 0.02, 0.05, 0.10, 0.20]
    elif dt == 0.1:
        tolerances = [0.05, 0.05, 0.10, 0.15, 0.20]
    else:  # dt == 0.5
        tolerances = [0.10, 0.10, 0.15, 0.25, 0.35]

    for order, (exact, discrete, tol) in enumerate(
        zip(exact_moments, discrete_moments, tolerances)
    ):
        relative_error = abs(discrete - exact) / abs(exact)
        assert relative_error < tol, (
            f"{rfunc_name} order {order} with dt={dt}: "
            f"discrete={discrete:.6f}, exact={exact:.6f}, "
            f"relative_error={relative_error:.4f}"
        )


@pytest.mark.parametrize("rfunc_name", RFUNCS_WITH_EXACT_MOMENTS)
@pytest.mark.parametrize("order", [0, 1, 2])
def test_moment_order_values(rfunc_name: str, order: int) -> None:
    """Test that moment values make physical sense.

    Parameters
    ----------
    rfunc_name : str
        Name of the response function class to test.
    order : int
        Order of the moment to compute (0-2).
    """
    rfunc = getattr(ps.rfunc, rfunc_name)()
    p = rfunc.get_init_parameters("test").initial.to_numpy()

    # Get moment using exact method
    m = rfunc.moment(p, order=order, method="exact")

    # Order 0 moment should be positive (integral of response function)
    if order == 0:
        assert m > 0, f"0-th order moment should be positive, got {m}"

    # For all orders, moment should be finite and non-negative
    assert np.isfinite(m), f"Moment should be finite, got {m}"
    assert m >= 0, f"Moment should be non-negative, got {m}"


@pytest.mark.parametrize("rfunc_name", RFUNCS_WITH_EXACT_MOMENTS)
def test_moment_invalid_method(rfunc_name: str) -> None:
    """Test that invalid method raises ValueError.

    Parameters
    ----------
    rfunc_name : str
        Name of the response function class to test.
    """
    rfunc = getattr(ps.rfunc, rfunc_name)()
    p = rfunc.get_init_parameters("test").initial.to_numpy()

    with pytest.raises(ValueError, match="Invalid method"):
        rfunc.moment(p, order=0, method="invalid")  # type: ignore
