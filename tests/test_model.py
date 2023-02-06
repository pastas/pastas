import pytest
from pandas import read_csv

import pastas as ps

from .fixtures import ml, ml_empty, ml_no_settings, sm_evap, sm_prec

obs = (
    read_csv("tests/data/obs.csv", index_col=0, parse_dates=True)
    .squeeze("columns")
    .dropna()
)


def test_create_model() -> None:
    ml = ps.Model(obs, name="Test_Model")


def test_add_stressmodel(ml_empty, sm_prec) -> None:
    ml_empty.add_stressmodel(sm_prec)


def test_add_stressmodels(ml_empty, sm_prec, sm_evap) -> None:
    ml_empty.add_stressmodel([sm_prec, sm_evap])


def test_del_stressmodel(ml) -> None:
    ml.del_stressmodel("rch")


def test_add_constant(ml_empty) -> None:
    ml_empty.add_constant(ps.Constant())


def test_del_constant(ml_empty) -> None:
    ml_empty.del_constant()


def test_add_noisemodel(ml_empty) -> None:
    ml_empty.add_noisemodel(ps.NoiseModel())


def test_armamodel() -> None:
    ml = ps.Model(obs, name="Test_Model", noisemodel=False)
    ml.add_noisemodel(ps.ArmaModel())
    ml.solve()


def test_del_noisemodel(ml_empty) -> None:
    ml_empty.del_noisemodel()


def test_solve_model(ml) -> None:
    ml.solve()


def test_solve_empty_model(ml) -> None:
    try:
        ml.solve(tmin="2016")
    except ValueError as e:
        if e.args[0].startswith("Calibration series"):
            return None
        else:
            raise e


def test_save_model(ml) -> None:
    ml.to_file("test.pas")


def test_load_model(ml) -> None:
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


def test_model_copy(ml_empty) -> None:
    ml_empty.copy()


def test_get_block(ml) -> None:
    ml.get_block_response("rch")


def test_get_step(ml) -> None:
    ml.get_step_response("rch")


def test_get_contribution(ml) -> None:
    ml.get_contribution("rch")


def test_get_stress(ml) -> None:
    ml.get_stress("rch")


def test_simulate(ml) -> None:
    ml.simulate()


def test_residuals(ml) -> None:
    ml.residuals()


def test_noise(ml) -> None:
    ml.noise()


def test_observations(ml_empty) -> None:
    ml_empty.observations()


def test_get_output_series(ml) -> None:
    ml.get_output_series()


def test_get_output_series_arguments(ml) -> None:
    ml.get_output_series(split=False, add_contributions=False)


def test_model_sim_w_nans_error(ml_no_settings):
    with pytest.raises(ValueError) as e_info:
        ml_no_settings.solve()
