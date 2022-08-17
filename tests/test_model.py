import pastas as ps
from pandas import read_csv

rain = read_csv("tests/data/rain.csv", index_col=0,
                parse_dates=True).squeeze("columns")
evap = read_csv("tests/data/evap.csv", index_col=0,
                parse_dates=True).squeeze("columns")
obs = read_csv("tests/data/obs.csv", index_col=0,
               parse_dates=True).squeeze("columns")


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
    return ml


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
    ml = test_solve_model()
    # add some fictitious tiny value for testing float precision
    ml.parameters.loc["rch_f", "pmax"] = 1.23456789e-10
    ml.to_file("test.pas")
    ml2 = ps.io.load("test.pas")

    # dataframe dtypes don't match... make the same here
    # this is caused because the parameters df is loaded empty without
    # information on the datatype in each column
    for icol in ["initial", "optimal", "pmin", "pmax", "stderr"]:
        ml.parameters[icol] = ml.parameters[icol].astype(float)
    ml.parameters["vary"] = ml.parameters["vary"].astype(bool)

    # check if parameters and pcov dataframes are equal
    assert ml.parameters.equals(ml2.parameters)
    assert ml.fit.pcov.equals(ml2.fit.pcov)
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


def test_get_output_series():
    ml = test_add_stressmodel()
    ml.get_output_series()
    return


def test_get_output_series_arguments():
    ml = test_add_stressmodel()
    ml.get_output_series(split=False, add_contributions=False)
    return
