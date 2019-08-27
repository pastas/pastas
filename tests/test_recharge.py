import pastas as ps


def test_create_rechargemodel():
    rain = ps.read_knmi('tests/data/knmi_rain_data.txt', variables='RD')
    evap = ps.read_knmi('tests/data/knmi_evap_data.txt', variables='EV24')
    rm = ps.RechargeModel(prec=rain, evap=evap, name='recharge',
                          recharge="Linear")
    return rm


def test_create_model():
    obs = ps.read_dino('tests/data/dino_gwl_data.csv')
    ml = ps.Model(obs, name="Test_Model")
    sm = test_create_rechargemodel()
    ml.add_stressmodel(sm)
    return ml


def test_model_solve():
    ml = test_create_model()
    ml.solve()
    return


def test_model_copy():
    ml = test_create_model()
    ml.copy()
    return
