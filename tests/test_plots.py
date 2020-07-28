from test_model import test_add_stressmodel
from pastas.plots import TrackSolve, compare

def test_plot():
    ml = test_add_stressmodel()
    ml.plot()

def test_decomposition():
    ml = test_add_stressmodel()
    ml.plots.decomposition(min_ylim_diff=0.1)

def test_results():
    ml = test_add_stressmodel()
    ml.plots.results()

def test_stacked_results():
    ml = test_add_stressmodel()
    ml.plots.stacked_results()

def test_block_response():
    ml = test_add_stressmodel()
    ml.plots.block_response()

def test_step_response():
    ml = test_add_stressmodel()
    ml.plots.step_response()

def test_diagnostics():
    ml = test_add_stressmodel()
    ml.plots.diagnostics(acf_options=dict(min_obs=10))

def test_stresses():
    ml = test_add_stressmodel()
    ml.plots.stresses()

def test_contributions_pie():
    ml = test_add_stressmodel()
    ml.plots.contributions_pie()

def test_compare():
    ml = test_add_stressmodel()
    ml2 = ml.copy()
    ml2.name = "Test_Model2"
    models = [ml, ml2]
    compare(models)

def test_tracksolve():
    ml = test_add_stressmodel()
    track = TrackSolve(ml)
    track.initialize_figure()
    ml.solve(callback=track.update_figure)
