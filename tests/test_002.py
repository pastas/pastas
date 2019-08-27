import pastas as ps


def test_create_model():
    obs = ps.read_dino('tests/data/dino_gwl_data.csv')
    ml = ps.Model(obs, name="Test_Model")
    return ml


def test_add_stressmodel():
    ml = test_create_model()
    rain = ps.read_knmi('tests/data/knmi_rain_data.txt', variables='RD')
    evap = ps.read_knmi('tests/data/knmi_evap_data.txt', variables='EV24')
    sm = ps.StressModel2(stress=[rain, evap], rfunc=ps.Exponential,
                         name='recharge')
    ml.add_stressmodel(sm)
    return ml


def test_del_stressmodel():
    ml = test_add_stressmodel()
    ml.del_stressmodel("recharge")
    return


def test_solve_model():
    ml = test_add_stressmodel()
    ml.solve()
    return


def test_save_model():
    ml = test_create_model()
    ml.to_file("test.pas")
    return


def test_load_model():
    test_save_model()
    ml = ps.io.load("test.pas")
    return


def test_model_copy():
    ml = test_create_model()
    ml.copy()
    return
