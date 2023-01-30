from pandas import read_csv
from pandas.testing import assert_series_equal

import pastas as ps

rain = read_csv("tests/data/rain.csv", index_col=0, parse_dates=True).squeeze("columns")
evap = read_csv("tests/data/evap.csv", index_col=0, parse_dates=True).squeeze("columns")
obs = (
    read_csv("tests/data/obs.csv", index_col=0, parse_dates=True)
    .squeeze("columns")
    .dropna()
)


def test_create_model():
    ml = ps.Model(obs, name="Test_Model")
    return None


def test_add_stressmodel():
    ml = ps.Model(obs, name="Test_Model")
    sm = ps.RechargeModel(prec=rain, evap=evap, rfunc=ps.Gamma(), name="rch")
    ml.add_stressmodel(sm)
    return None


def test_add_stressmodels():
    ml = ps.Model(obs, name="Test_Model")
    sm1 = ps.StressModel(rain, rfunc=ps.Exponential(), name="prec")
    sm2 = ps.StressModel(evap, rfunc=ps.Exponential(), name="evap")
    ml.add_stressmodel([sm1, sm2])
    return None


def test_del_stressmodel():
    ml = ps.Model(obs, name="Test_Model")
    sm = ps.RechargeModel(prec=rain, evap=evap, rfunc=ps.Gamma(), name="rch")
    ml.add_stressmodel(sm)
    ml.del_stressmodel("rch")
    return None


def test_add_constant():
    ml = ps.Model(obs, name="Test_Model")
    ml.add_constant(ps.Constant())
    return None


def test_del_constant():
    ml = ps.Model(obs, name="Test_Model")
    ml.del_constant()
    return None


def test_add_noisemodel():
    ml = ps.Model(obs, name="Test_Model")
    ml.add_noisemodel(ps.NoiseModel())
    return None


def test_armamodel():
    ml = ps.Model(obs, name="Test_Model", noisemodel=False)
    ml.add_noisemodel(ps.ArmaModel())
    ml.solve()
    return None


def test_del_noisemodel():
    ml = ps.Model(obs, name="Test_Model")
    ml.del_noisemodel()
    return None


def test_solve_model():
    ml = ps.Model(obs, name="Test_Model")
    sm = ps.RechargeModel(prec=rain, evap=evap, rfunc=ps.Gamma(), name="rch")
    ml.add_stressmodel(sm)
    ml.solve()
    return None


def test_solve_empty_model():
    ml = ps.Model(obs, name="Test_Model")
    sm = ps.RechargeModel(prec=rain, evap=evap, rfunc=ps.Gamma(), name="rch")
    ml.add_stressmodel(sm)
    try:
        ml.solve(tmin="2016")
    except ValueError as e:
        if e.args[0].startswith("Calibration series "):
            return None
        else:
            raise e


def test_save_model():
    ml = ps.Model(obs, name="Test_Model")
    sm = ps.RechargeModel(prec=rain, evap=evap, rfunc=ps.Gamma(), name="rch")
    ml.add_stressmodel(sm)
    ml.to_file("test.pas")
    return None


def test_load_model():
    ml = ps.Model(obs, name="Test_Model")
    sm = ps.RechargeModel(prec=rain, evap=evap, rfunc=ps.Gamma(), name="rch")
    ml.add_stressmodel(sm)
    ml.solve()
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
    return None


def test_model_copy():
    ml = ps.Model(obs, name="Test_Model")
    ml.copy()
    return None


def test_get_block():
    ml = ps.Model(obs, name="Test_Model")
    sm = ps.RechargeModel(prec=rain, evap=evap, rfunc=ps.Gamma(), name="rch")
    ml.add_stressmodel(sm)
    ml.get_block_response("rch")
    return None


def test_get_step():
    ml = ps.Model(obs, name="Test_Model")
    sm = ps.RechargeModel(prec=rain, evap=evap, rfunc=ps.Gamma(), name="rch")
    ml.add_stressmodel(sm)
    ml.get_step_response("rch")
    return None


def test_get_contribution():
    ml = ps.Model(obs, name="Test_Model")
    sm = ps.RechargeModel(prec=rain, evap=evap, rfunc=ps.Gamma(), name="rch")
    ml.add_stressmodel(sm)
    ml.get_contribution("rch")
    return None


def test_get_stress():
    ml = ps.Model(obs, name="Test_Model")
    sm = ps.RechargeModel(prec=rain, evap=evap, rfunc=ps.Gamma(), name="rch")
    ml.add_stressmodel(sm)
    ml.get_stress("rch")
    return None


def test_simulate():
    ml = ps.Model(obs, name="Test_Model")
    sm = ps.RechargeModel(prec=rain, evap=evap, rfunc=ps.Gamma(), name="rch")
    ml.add_stressmodel(sm)
    ml.simulate()
    return None


def test_residuals():
    ml = ps.Model(obs, name="Test_Model")
    sm = ps.RechargeModel(prec=rain, evap=evap, rfunc=ps.Gamma(), name="rch")
    ml.add_stressmodel(sm)
    ml.residuals()
    return None


def test_noise():
    ml = ps.Model(obs, name="Test_Model")
    sm = ps.RechargeModel(prec=rain, evap=evap, rfunc=ps.Gamma(), name="rch")
    ml.add_stressmodel(sm)
    ml.noise()
    return None


def test_observations():
    ml = ps.Model(obs, name="Test_Model")
    ml.observations()
    return None


def test_get_output_series():
    ml = ps.Model(obs, name="Test_Model")
    sm = ps.RechargeModel(prec=rain, evap=evap, rfunc=ps.Gamma(), name="rch")
    ml.add_stressmodel(sm)
    ml.get_output_series()
    return None


def test_get_output_series_arguments():
    ml = ps.Model(obs, name="Test_Model")
    sm = ps.RechargeModel(prec=rain, evap=evap, rfunc=ps.Gamma(), name="rch")
    ml.add_stressmodel(sm)
    ml.get_output_series(split=False, add_contributions=False)
    return None


def test_load_old_wellmodel():
    # model with new parameter b_new = np.log(b_old)
    ml_new = ps.io.load("./tests/data/wellmodel_new.pas")
    # model with old parameter b_old
    ml_old = ps.io.load("./tests/data/wellmodel_old.pas")
    assert_series_equal(ml_new.simulate(), ml_old.simulate())
    return
