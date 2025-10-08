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
