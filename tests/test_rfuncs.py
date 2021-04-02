import pytest

import pastas as ps


@pytest.mark.parametrize("rfunc_name", ps.rfunc.__all__)
def test_rfunc(rfunc_name):
    if rfunc_name not in []:
        rfunc = getattr(ps.rfunc, rfunc_name)()
        p = rfunc.get_init_parameters("test").initial
        rfunc.block(p)
        rfunc.step(p)
