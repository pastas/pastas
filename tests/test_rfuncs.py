import pastas as ps
import pytest


@pytest.mark.parametrize("rfunc_name", ps.rfunc.__all__)
def test_rfunc(rfunc_name):
    if rfunc_name not in []:
        # Import and check the observed groundwater time series
        obs = ps.read_dino('tests/data/dino_gwl_data.csv')

        # read weather data
        rain = ps.read_knmi('tests/data/knmi_rain_data.txt',
                            variables='RD')
        evap = ps.read_knmi('tests/data/knmi_evap_data.txt', variables='EV24')

        # Create the time series model
        ml = ps.Model(obs, name="Test_Model")

        ## Create stress
        rfunc = getattr(ps.rfunc, rfunc_name)
        sm = ps.StressModel2(stress=[rain, evap], rfunc=rfunc, name='test_sm')
        ml.add_stressmodel(sm)

        # Solve the time series model
        ml.solve()
