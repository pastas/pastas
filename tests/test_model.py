from pandas import read_csv

import pastas as ps

rain = read_csv("tests/data/rain.csv", index_col=0, parse_dates=True,
                squeeze=True)
evap = read_csv("tests/data/evap.csv", index_col=0, parse_dates=True,
                squeeze=True)
obs = read_csv("tests/data/obs.csv", index_col=0, parse_dates=True,
               squeeze=True)


def test_create_model():
    ml = ps.Model(obs, name="Test_Model")
    return ml


def test_add_stressmodel():
    ml = test_create_model()
    sm = ps.RechargeModel(prec=rain, evap=evap, rfunc=ps.Gamma, name='rch')
    ml.add_stressmodel(sm)
    return ml


def test_add_stressmodels():
    ml = test_create_model()
    sm1 = ps.StressModel(rain, rfunc=ps.Exponential, name='prec')
    sm2 = ps.StressModel(evap, rfunc=ps.Exponential, name='evap')
    ml.add_stressmodel([sm1, sm2])
    return ml


def test_del_stressmodel():
    ml = test_add_stressmodel()
    ml.del_stressmodel("recharge")
    return


def test_add_constant():
    ml = test_create_model()
    ml.add_constant(ps.Constant())
    return


def test_del_constant():
    ml = test_create_model()
    ml.del_constant()
    return


def test_add_noisemodel():
    ml = test_create_model()
    ml.add_noisemodel(ps.NoiseModel())
    return


def test_armamodel():
    ml = test_create_model()
    ml.add_noisemodel(ps.ArmaModel())
    ml.solve()
    return


def test_del_noisemodel():
    ml = test_create_model()
    ml.del_noisemodel()
    return


def test_solve_model():
    ml = test_add_stressmodel()
    ml.solve()
    return


def test_solve_empty_model():
    ml = test_add_stressmodel()
    try:
        ml.solve(tmin="2016")
    except ValueError as e:
        if e.args[0].startswith("Calibration series "):
            return
        else:
            raise e


def test_save_model():
    ml = test_create_model()
    ml.to_file("test.pas")
    return


def test_load_model():
    test_save_model()
    _ = ps.io.load("test.pas")
    return


def test_model_copy():
    ml = test_create_model()
    ml.copy()
    return


def test_get_block():
    ml = test_add_stressmodel()
    ml.get_block_response("recharge")
    return


def test_get_step():
    ml = test_add_stressmodel()
    ml.get_step_response("recharge")
    return


def test_get_contribution():
    ml = test_add_stressmodel()
    ml.get_contribution("recharge")
    return


def test_get_stress():
    ml = test_add_stressmodel()
    ml.get_stress("recharge")
    return


def test_simulate():
    ml = test_add_stressmodel()
    ml.simulate()
    return


def test_residuals():
    ml = test_add_stressmodel()
    ml.residuals()
    return


def test_noise():
    ml = test_add_stressmodel()
    ml.noise()
    return


def test_observations():
    ml = test_add_stressmodel()
    ml.observations()
    return
