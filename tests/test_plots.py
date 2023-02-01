import matplotlib as mpl

from pastas.plots import TrackSolve, compare

from .fixtures import ml

mpl.use("Agg")  # prevent _tkinter.TclError: Can't find a usable tk.tcl error


def test_plot(ml) -> None:
    _ = ml.plot()


def test_decomposition(ml) -> None:
    _ = ml.plots.decomposition(min_ylim_diff=0.1)


def test_results(ml) -> None:
    _ = ml.plots.results()


def test_stacked_results(ml) -> None:
    _ = ml.plots.stacked_results()


def test_block_response(ml) -> None:
    _ = ml.plots.block_response()


def test_step_response(ml) -> None:
    _ = ml.plots.step_response()


def test_diagnostics(ml) -> None:
    _ = ml.plots.diagnostics(acf_options=dict(min_obs=10))


def test_stresses(ml) -> None:
    _ = ml.plots.stresses()


def test_contributions_pie(ml) -> None:
    _ = ml.plots.contributions_pie()


def test_compare(ml) -> None:
    ml2 = ml.copy()
    ml2.name = "Test_Model2"
    models = [ml, ml2]
    _ = compare(models)


def test_tracksolve(ml) -> None:
    track = TrackSolve(ml)
    _ = ml.solve(callback=track.plot_track_solve)


def test_summary_pdf(ml) -> None:
    _ = ml.plots.summary_pdf()
