import pytest
from pandas import read_csv

import pastas as ps


@pytest.mark.parametrize("rfunc_name", ps.rfunc.__all__)
def test_rfunc(rfunc_name):
    if rfunc_name not in []:
        obs = read_csv("tests/data/obs.csv", index_col=0, parse_dates=True,
                       squeeze=True)
        rain = read_csv("tests/data/rain.csv", index_col=0, parse_dates=True,
                        squeeze=True)
        evap = read_csv("tests/data/evap.csv", index_col=0, parse_dates=True,
                        squeeze=True)
        # Create the time series model
        ml = ps.Model(obs, name="Test_Model")

        ## Create stress
        rfunc = getattr(ps.rfunc, rfunc_name)
        sm = ps.StressModel2(stress=[rain, evap], rfunc=rfunc, name='test_sm')
        ml.add_stressmodel(sm)

        # Solve the time series model
        ml.solve()
