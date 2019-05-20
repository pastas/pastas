import pastas as ps

def test_all_rfuncs():
    # Import and check the observed groundwater time series
    obs = ps.read_dino('tests/data/dino_gwl_data.csv')

    # read weather data
    rain = ps.read_knmi('tests/data/knmi_rain_data.txt',
                        variables='RD')
    evap = ps.read_knmi('tests/data/knmi_evap_data.txt', variables='EV24')

    for rfunc_name in ps.rfunc.__all__:
        # Create the time series model
        ml = ps.Model(obs, name="Test_Model")

        ## Create stress
        rfunc = getattr(ps.rfunc, rfunc_name)
        sm = ps.StressModel2(stress=[rain, evap], rfunc=rfunc, name='test_sm')
        ml.add_stressmodel(sm)

        # Solve the time series model
        ml.solve()