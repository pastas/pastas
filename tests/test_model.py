import numpy as np
import pytest
from pandas import Series, Timedelta, date_range, read_csv

import pastas as ps

obs = (
    read_csv("tests/data/obs.csv", index_col=0, parse_dates=True)
    .squeeze("columns")
    .dropna()
)
prec = read_csv("tests/data/rain.csv", index_col=[0], parse_dates=True).squeeze() * 1e-3
evap = read_csv("tests/data/evap.csv", index_col=[0], parse_dates=True).squeeze() * 1e-3


def generate_synthetic_heads(input, rfunc, params, const=10.0, cutoff=0.999, dt=1.0):
    # Generate the head
    step = rfunc.block(params, cutoff=cutoff, dt=dt)

    h = const * np.ones(len(input) + step.size)

    for i in range(len(input)):
        h[i : i + step.size] += input.iat[i] * step

    head = Series(
        index=input.index,
        data=h[: len(input)],
    )
    return head


def test_create_model() -> None:
    ml = ps.Model(obs, name="Test_Model")
    ml.add_noisemodel(ps.ArNoiseModel())


def test_add_stressmodel(ml_empty: ps.Model, sm_prec) -> None:
    ml_empty.add_stressmodel(sm_prec)


def test_add_stressmodels(
    ml_empty: ps.Model, sm_prec: ps.StressModel, sm_evap: ps.StressModel
) -> None:
    ml_empty.add_stressmodel([sm_prec, sm_evap])


def test_del_stressmodel(ml: ps.Model) -> None:
    ml.del_stressmodel("rch")


def test_add_constant(ml_empty: ps.Model) -> None:
    ml_empty.add_constant(ps.Constant())


def test_del_constant(ml_empty: ps.Model) -> None:
    ml_empty.del_constant()


def test_add_noisemodel(ml_empty: ps.Model) -> None:
    ml_empty.add_noisemodel(ps.ArNoiseModel())


def test_armamodel() -> None:
    ml = ps.Model(obs, name="Test_Model")
    ml.add_noisemodel(ps.ArmaNoiseModel())
    ml.solve()
    ml.to_file("test.pas")


def test_del_noisemodel(ml_empty: ps.Model) -> None:
    ml_empty.del_noisemodel()


def test_solve_model(ml: ps.Model) -> None:
    ml.solve()


def test_solve_empty_model(ml: ps.Model) -> None:
    with pytest.raises(ValueError) as excinfo:
        ml.solve(tmin="2016")
    assert excinfo.value.args[0].startswith("Calibration series")


def test_save_model(ml: ps.Model) -> None:
    ml.to_file("test.pas")


def test_load_model(ml: ps.Model) -> None:
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
    assert ml.solver.pcov.equals(ml2.solver.pcov)


def test_model_copy(ml_empty: ps.Model) -> None:
    ml_empty.copy()


def test_get_block(ml: ps.Model) -> None:
    ml.get_block_response("rch")


def test_get_step(ml: ps.Model) -> None:
    ml.get_step_response("rch")


def test_get_contribution(ml: ps.Model) -> None:
    ml.get_contribution("rch")


def test_get_stress(ml: ps.Model) -> None:
    ml.get_stress("rch")


def test_simulate(ml: ps.Model) -> None:
    ml.simulate()


def test_residuals(ml: ps.Model) -> None:
    ml.residuals()


def test_noise(ml: ps.Model) -> None:
    ml.noise()


def test_observations(ml_empty: ps.Model) -> None:
    ml_empty.observations()


def test_get_output_series(ml: ps.Model) -> None:
    ml.get_output_series()


def test_get_output_series_arguments(ml: ps.Model) -> None:
    ml.get_output_series(split=False, add_contributions=False)


def test_model_sim_w_nans_error(ml_no_settings):
    with pytest.raises(ValueError) as _:
        ml_no_settings.solve()


def test_modelstats(ml: ps.Model) -> None:
    ml.solve()
    ml.stats.summary()


def test_fit_report(ml: ps.Model) -> None:
    ml.solve(report=False)
    ml.fit_report(corr=True, stderr=True)


def test_model_freq_geq_daily() -> None:
    rf_rch = ps.Exponential()
    A_rch = 800
    a_rch = 50
    f_rch = -1.3
    constant = 20

    stress = prec + f_rch * evap
    head = generate_synthetic_heads(stress, rf_rch, (A_rch, a_rch), const=constant)

    models = []
    freqs = ["1D", "7D", "14D", "28D"]
    for freq in freqs:
        iml = ps.Model(head, name=freq)
        rm = ps.RechargeModel(prec, evap, rfunc=rf_rch, name="recharge")
        iml.add_stressmodel(rm)
        iml.solve(freq=freq, report=False)
        models.append(iml)

    comparison = ps.CompareModels(models)
    assert (comparison.get_metrics(metric_selection=["rsq"]).squeeze() > 0.99).all()


def test_model_freq_h():
    rf_tide = ps.Exponential()
    A_tide = 1.0
    a_tide = 0.15

    # sine with period 12 hrs 25 minutes and amplitude 1.5 m
    tidx = date_range(obs.index[0], obs.index[-1] + Timedelta(hours=23), freq="h")
    tides = Series(
        index=tidx,
        data=1.5 * np.sin(2 * np.pi * np.arange(tidx.size) / (0.517375)),
    )

    ht = generate_synthetic_heads(tides, rf_tide, (A_tide, a_tide), dt=1 / 24.0)

    # model with hourly timestep
    ml_h = ps.Model(ht, name="tidal_model", freq="h")
    sm = ps.StressModel(
        tides,
        rfunc=ps.Exponential(),
        name="tide",
        settings="waterlevel",
    )
    ml_h.add_stressmodel(sm)
    ml_h.solve(report=False)

    assert ml_h.simulate().index.freq == "h"
    assert ml_h.stats.rsq() > 0.99999
