import numpy as np
import pytest

import pastas as ps


@pytest.mark.parametrize("rfunc_name", ps.rfunc.__all__)
@pytest.mark.parametrize("up", [True, False])
def test_rfunc(rfunc_name: str, up: bool) -> None:
    if rfunc_name == "Edelman":
        with pytest.raises(DeprecationWarning):
            _ = getattr(ps.rfunc, rfunc_name)()
    else:
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
    if rfunc_name == "Edelman":
        with pytest.raises(DeprecationWarning):
            _ = getattr(ps.rfunc, rfunc_name)()
    else:
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
    if rfunc_name == "Edelman":
        with pytest.raises(DeprecationWarning):
            _ = getattr(ps.rfunc, rfunc_name)()
    else:
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
    "FourParam",
    "Kraijenhoff",
]
RFUNCS_WITHOUT_EXACT_MOMENTS = [
    r
    for r in ps.rfunc.__all__
    if r not in RFUNCS_WITH_EXACT_MOMENTS and r not in ("RfuncBase", "HantushWellModel")
]


@pytest.mark.parametrize(
    "rfunc_name", RFUNCS_WITHOUT_EXACT_MOMENTS + RFUNCS_WITH_EXACT_MOMENTS
)
def test_moment_discrete_works(rfunc_name: str) -> None:
    """Test that discrete moment method can be called for all response functions.

    All response functions should support calling the 'discrete' moment method,
    though it may not work correctly for all functions.

    Parameters
    ----------
    rfunc_name : str
        Name of the response function class to test.
    """
    if rfunc_name == "Edelman":
        with pytest.raises(DeprecationWarning):
            _ = getattr(ps.rfunc, rfunc_name)()
        return
    rfunc = getattr(ps.rfunc, rfunc_name)(cutoff=0.999)
    p = rfunc.get_init_parameters("test").initial.to_numpy()

    # Discrete method should be callable for all rfuncs
    # (though it may raise an error for some unsupported rfuncs)
    moment_val = rfunc.moment(p, order=0, method="discrete", dt=1.0)

    # If it returns a value, check it's valid
    assert isinstance(moment_val, (int, float, np.number)), (
        f"{rfunc_name}.moment() should return a number or None for discrete method, "
        f"got {type(moment_val)}"
    )
    if np.isfinite(moment_val):
        assert moment_val >= 0, (
            f"{rfunc_name}.moment() returned negative value: {moment_val}"
        )


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


@pytest.mark.parametrize("rfunc_name", RFUNCS_WITH_EXACT_MOMENTS)
@pytest.mark.parametrize("method", ["discrete", "exact"])
def test_moment_order_0_equals_gain(rfunc_name: str, method: str) -> None:
    """Test that the zero-th moment equals the gain of the response function.

    The zero-th moment (order=0) of a response function is the integral of the
    impulse response, which should equal the gain (amplitude) of the function.

    Parameters
    ----------
    rfunc_name : str
        Name of the response function class to test.
    method : str
        Method to compute moment ('discrete' or 'exact').
    """
    rfunc = getattr(ps.rfunc, rfunc_name)(cutoff=0.999999)
    p = rfunc.get_init_parameters("test").initial.to_numpy()

    # Get the zero-th moment using the specified method
    moment_0 = rfunc.moment(p, order=0, method=method, dt=0.001)

    # Get the gain
    gain = rfunc.gain(p)

    # The zero-th moment should approximately equal the gain
    # Allow for numerical integration errors (especially for discrete method)
    tolerance = 0.05 if method == "discrete" else 0.01  # 5% or 1% tolerance
    relative_error = abs(moment_0 - gain) / abs(gain) if gain != 0 else abs(moment_0)

    assert relative_error < tolerance, (
        f"{rfunc_name} zero-th moment ({moment_0:.6f}) != gain ({gain:.6f}), relative error: {relative_error:.4f}"
    )


@pytest.mark.parametrize("rfunc_name", RFUNCS_WITHOUT_EXACT_MOMENTS)
def test_moment_exact_not_implemented(rfunc_name: str) -> None:
    """Test that calling exact method on rfuncs without it raises ValueError.

    Response functions without an explicit exact moment implementation should
    raise ValueError when 'exact' method is used.

    Parameters
    ----------
    rfunc_name : str
        Name of the response function class to test.
    """
    if rfunc_name == "Edelman":
        with pytest.raises(DeprecationWarning):
            _ = getattr(ps.rfunc, rfunc_name)()
            return

    rfunc = getattr(ps.rfunc, rfunc_name)()
    p = rfunc.get_init_parameters("test").initial.to_numpy()

    # Call exact method - should raise ValueError for unimplemented methods
    with pytest.raises(ValueError, match="Invalid method"):
        rfunc.moment(p, order=0, method="exact")  # type: ignore
