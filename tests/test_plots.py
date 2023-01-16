from pandas import read_csv

import pastas as ps
from pastas.plots import TrackSolve, compare

rain = read_csv("tests/data/rain.csv", index_col=0, parse_dates=True).squeeze("columns")
evap = read_csv("tests/data/evap.csv", index_col=0, parse_dates=True).squeeze("columns")
obs = (
    read_csv("tests/data/obs.csv", index_col=0, parse_dates=True)
    .squeeze("columns")
    .dropna()
)


def create_model():
    ml = ps.Model(obs, name="Test_Model")
    sm = ps.RechargeModel(prec=rain, evap=evap, rfunc=ps.Gamma(), name="rch")
    ml.add_stressmodel(sm)
    return ml


def test_plot():
    ml = create_model()
    ml.plot()


def test_decomposition():
    ml = create_model()
    ml.plots.decomposition(min_ylim_diff=0.1)


def test_results():
    ml = create_model()
    ml.plots.results()


def test_stacked_results():
    ml = create_model()
    ml.plots.stacked_results()


def test_block_response():
    ml = create_model()
    ml.plots.block_response()


def test_step_response():
    ml = create_model()
    ml.plots.step_response()


def test_diagnostics():
    ml = create_model()
    ml.plots.diagnostics(acf_options=dict(min_obs=10))


def test_stresses():
    ml = create_model()
    ml.plots.stresses()


def test_contributions_pie():
    ml = create_model()
    ml.plots.contributions_pie()


def test_compare():
    ml = create_model()
    ml2 = ml.copy()
    ml2.name = "Test_Model2"
    models = [ml, ml2]
    compare(models)


def test_tracksolve():
    ml = create_model()
    track = TrackSolve(ml)
    ml.solve(callback=track.plot_track_solve)


def test_summary_pdf():
    ml = create_model()
    ml.plots.summary_pdf()
