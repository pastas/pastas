import pastas as ps

def create_model():
    # Import and check the observed groundwater time series
    obs = ps.read_dino('data/B58C0698001_1.csv')

    # Create the time series model
    ml = ps.Model(obs, name="Test_Model")

    # read weather data
    rain = ps.read_knmi('data/neerslaggeg_HEIBLOEM-L_967-2.txt',
                        variables='RD')
    evap = ps.read_knmi('data/etmgeg_380.txt', variables='EV24')

    ## Create stress
    sm = ps.StressModel2(stress=[rain, evap], rfunc=ps.Exponential,
                         name='recharge')
    ml.add_stressmodel(sm)

    ## Solve
    ml.solve()
    ml.plot()

    return ml

def save_model():
    ml = create_model()
    ml.dump("test.pas")

    return

def load_model():
    save_model()
    ml = ps.io.load("test.pas")
    return
