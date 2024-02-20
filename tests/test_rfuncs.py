import pytest

import pastas as ps


@pytest.mark.parametrize("rfunc_name", ps.rfunc.__all__)
def test_rfunc(rfunc_name) -> None:
    if rfunc_name not in []:
        rfunc = getattr(ps.rfunc, rfunc_name)()
        if rfunc_name == "HantushWellModel":
            rfunc.set_distances(100.0)
        p = rfunc.get_init_parameters("test").initial.to_numpy()
        rfunc.block(p)
        rfunc.step(p)


@pytest.mark.parametrize("rfunc_name", ps.rfunc.__all__)
def test_to_dict_rfuncs(rfunc_name) -> None:
    rfunc1 = getattr(ps.rfunc, rfunc_name)(cutoff=0.5)

    # Create the exact same instance using to_dict
    data = rfunc1.to_dict()
    rfunc_class = data.pop("class")  # Determine response class
    rfunc_up = data.pop("up", None)
    rfunc_gsf = data.pop("gain_scale_factor", None)
    rfunc2 = getattr(ps.rfunc, rfunc_class)(**data)
    rfunc2.update_rfunc_settings(up=rfunc_up, gain_scale_factor=rfunc_gsf)

    if rfunc_name == "HantushWellModel":
        rfunc1.set_distances(100.0)
        rfunc2.set_distances(100.0)

    p1 = rfunc1.get_init_parameters("test").initial.to_numpy()
    p2 = rfunc2.get_init_parameters("test").initial.to_numpy()

    assert (rfunc1.step(p1) - rfunc2.step(p2)).sum() == 0.0
