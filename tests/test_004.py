import pastas as ps

def test_create_model():
    # Import and check the observed groundwater time series
    obs = ps.read_dino('tests/data/dino_gwl_data.csv')

    # Create the time series model
    ml = ps.Model(obs, name="Test_Model")

    # read weather data
    rain = ps.read_knmi('tests/data/knmi_rain_data.txt', variables='RD')
    evap = ps.read_knmi('tests/data/knmi_evap_data.txt', variables='EV24')

    ## Create stress
    sm = ps.StressModel2(stress=[rain, evap], rfunc=ps.Exponential,
                         name='recharge')
    ml.add_stressmodel(sm)

    ## Solve
    ml.solve()

    return ml

def test_save_model():
    ml = test_create_model()
    ml.dump("test.pas")

    return

def test_load_model():
    test_save_model()
    ml = ps.io.load("test.pas")
    return
