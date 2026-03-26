import numpy as np
import pytest

import pastas as ps


@pytest.mark.parametrize("rfunc_name", ps.rfunc.__all__)
@pytest.mark.parametrize("up", [True, False])
def test_rfunc(rfunc_name: str, up: bool) -> None:
    if rfunc_name == "Edelman":
        with pytest.raises(AttributeError):
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
        with pytest.raises(AttributeError):
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
        with pytest.raises(AttributeError):
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
        with pytest.raises(AttributeError):
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
        with pytest.raises(AttributeError):
            _ = getattr(ps.rfunc, rfunc_name)()
        return

    rfunc = getattr(ps.rfunc, rfunc_name)()
    p = rfunc.get_init_parameters("test").initial.to_numpy()

    # Call exact method - should raise ValueError for unimplemented methods
    with pytest.raises(ValueError, match="Invalid method"):
        rfunc.moment(p, order=0, method="exact")  # type: ignore


# Tests for Hantush approximate_tmax option
@pytest.mark.parametrize("approximate_tmax", [True, False])
def test_hantush_approximate_tmax_parameter(approximate_tmax: bool) -> None:
    """Test that Hantush approximate_tmax parameter works and is preserved.

    Parameters
    ----------
    approximate_tmax : bool
        Whether to use approximate (True) or exact (False) tmax calculation.
    """
    rfunc = ps.Hantush(approximate_tmax=approximate_tmax)
    p = rfunc.get_init_parameters("test").initial.to_numpy()

    # Test that both modes can compute tmax without error
    tmax = rfunc.get_tmax(p)
    assert np.isfinite(tmax) and tmax > 0, f"Invalid tmax: {tmax}"

    # Test that to_dict preserves the parameter
    data = rfunc.to_dict()
    assert data["approximate_tmax"] == approximate_tmax


@pytest.mark.parametrize("quad", [True, False])
@pytest.mark.parametrize("approximate_tmax", [True, False])
def test_hantush_quad_and_approximate_tmax_combinations(
    quad: bool, approximate_tmax: bool
) -> None:
    """Test all combinations of quad and approximate_tmax parameters for Hantush.

    Parameters
    ----------
    quad : bool
        Whether to use numerical integration for step response.
    approximate_tmax : bool
        Whether to use approximate (True) or exact (False) tmax calculation.
    """
    rfunc = ps.Hantush(quad=quad, approximate_tmax=approximate_tmax)
    p = rfunc.get_init_parameters("test").initial.to_numpy()

    # Test that step response can be computed
    step = rfunc.step(p)
    assert len(step) > 0, "Step response should be non-empty"
    assert np.all(np.isfinite(step)), "Step response should have finite values"

    # Test that tmax can be computed
    tmax = rfunc.get_tmax(p)
    assert np.isfinite(tmax) and tmax > 0, f"Invalid tmax: {tmax}"

    # Test that to_dict preserves both parameters
    data = rfunc.to_dict()
    assert data["quad"] == quad, (
        f"quad parameter not preserved: {data['quad']} vs {quad}"
    )
    assert data["approximate_tmax"] == approximate_tmax, (
        f"approximate_tmax parameter not preserved: {data['approximate_tmax']} vs {approximate_tmax}"
    )


def test_hantush_exact_vs_approximate_tmax() -> None:
    """Test that exact tmax differs from approximate and hits target cutoff better.

    The exact method should find a tmax that achieves the target cutoff more
    accurately than the approximation, which is based on Lambert W asymptotic expansion.
    """
    rfunc_approx = ps.Hantush(approximate_tmax=True)
    rfunc_exact = ps.Hantush(approximate_tmax=False)

    p = rfunc_approx.get_init_parameters("test").initial.to_numpy()
    tmax_approx = rfunc_approx.get_tmax(p)
    tmax_exact = rfunc_exact.get_tmax(p)

    # Exact tmax should be different from approximate (smaller)
    assert tmax_exact < tmax_approx, (
        f"Exact tmax ({tmax_exact}) should be smaller than approximate ({tmax_approx})"
    )

    # Exact tmax should achieve cutoff more accurately
    A = p[0]
    cutoff = rfunc_exact.cutoff

    # Compute step response at each tmax
    step_approx = (
        rfunc_approx.step(p, dt=1.0, cutoff=cutoff, maxtmax=1)[int(tmax_approx)] / A
        if int(tmax_approx)
        < len(rfunc_approx.step(p, dt=1.0, cutoff=cutoff, maxtmax=1))
        else rfunc_approx.numpy_step(A, p[1], p[2], np.array([tmax_approx]))[0] / A
    )
    step_exact = rfunc_exact.numpy_step(A, p[1], p[2], np.array([tmax_exact]))[0] / A

    # The exact method should achieve cutoff more closely
    error_approx = abs(step_approx - cutoff)
    error_exact = abs(step_exact - cutoff)

    assert error_exact < error_approx, (
        f"Exact method should achieve cutoff more accurately. "
        f"Approximate error: {error_approx:.6f}, Exact error: {error_exact:.6f}"
    )


def test_hantush_approximate_tmax_to_dict_roundtrip() -> None:
    """Test that approximate_tmax parameter survives to_dict/from_dict roundtrip."""
    for use_exact in [True, False]:
        # Create original rfunc
        rfunc1 = ps.Hantush(approximate_tmax=not use_exact, cutoff=0.95)

        # Export to dict
        data = rfunc1.to_dict()

        # Create new rfunc from dict
        rfunc_class = data.pop("class")
        rfunc_up = data.pop("up", None)
        rfunc_gsf = data.pop("gain_scale_factor", None)
        rfunc2 = getattr(ps.rfunc, rfunc_class)(**data)
        rfunc2.update_rfunc_settings(up=rfunc_up, gain_scale_factor=rfunc_gsf)

        # Verify approximate_tmax is preserved
        assert rfunc2.approximate_tmax == rfunc1.approximate_tmax, (
            f"approximate_tmax not preserved: {rfunc1.approximate_tmax} vs {rfunc2.approximate_tmax}"
        )

        # Verify behavior is identical
        p = rfunc1.get_init_parameters("test").initial.to_numpy()
        tmax1 = rfunc1.get_tmax(p)
        tmax2 = rfunc2.get_tmax(p)

        assert tmax1 == tmax2, f"tmax values differ after roundtrip: {tmax1} vs {tmax2}"


# Tests for HantushWellModel approximate_tmax and log_b options
@pytest.mark.parametrize("log_b", [True, False])
@pytest.mark.parametrize("approximate_tmax", [True, False])
def test_hantush_well_model_parameters(log_b: bool, approximate_tmax: bool) -> None:
    """Test HantushWellModel parameters log_b and approximate_tmax."""
    rfunc = ps.HantushWellModel(log_b=log_b, approximate_tmax=approximate_tmax)
    rfunc.set_distances(100.0)
    p = rfunc.get_init_parameters("test").initial.to_numpy()

    # Step response works
    step = rfunc.step(p)
    assert len(step) > 0

    # tmax works
    tmax = rfunc.get_tmax(p)
    assert np.isfinite(tmax) and tmax > 0

    # to_dict works
    data = rfunc.to_dict()
    assert data["log_b"] == log_b
    assert data["approximate_tmax"] == approximate_tmax


@pytest.mark.parametrize("log_b", [True, False])
def test_hantush_well_model_variance_gain(log_b: bool) -> None:
    """Test variance_gain for HantushWellModel with log_b variations."""
    rfunc = ps.HantushWellModel(log_b=log_b)
    rfunc.set_distances(100.0)
    p = rfunc.get_init_parameters("test").initial.to_numpy()

    # Dummy covariance and variance values
    var_A = 0.1
    var_b = 0.2
    cov_Ab = 0.05

    # test variance_gain (calculates through internal method)
    vg = rfunc.variance_gain(
        A=p[0], b=p[2], var_A=var_A, var_b=var_b, cov_Ab=cov_Ab, r=100.0
    )
    assert np.isfinite(vg)
    assert vg >= 0.0
