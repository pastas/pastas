import matplotlib as mpl

from pastas import Model
from pastas.plots import TrackSolve, compare

mpl.use("Agg")  # prevent _tkinter.TclError: Can't find a usable tk.tcl error


def test_plot(ml: Model) -> None:
    _ = ml.plot()


def test_decomposition(ml: Model) -> None:
    _ = ml.plots.decomposition(min_ylim_diff=0.1)


def test_results(ml: Model) -> None:
    _ = ml.plots.results()


def test_stacked_results(ml: Model) -> None:
    _ = ml.plots.stacked_results()


def test_block_response(ml: Model) -> None:
    _ = ml.plots.block_response()


def test_step_response(ml: Model) -> None:
    _ = ml.plots.step_response()


def test_diagnostics(ml: Model) -> None:
    _ = ml.plots.diagnostics(acf_options=dict(min_obs=10))


def test_stresses(ml: Model) -> None:
    _ = ml.plots.stresses()


def test_contributions_pie(ml: Model) -> None:
    _ = ml.plots.contributions_pie()


def test_compare(ml: Model) -> None:
    ml2 = ml.copy()
    models = [ml, ml2]
    _ = compare(models)


def test_tracksolve(ml: Model) -> None:
    track = TrackSolve(ml)
    _ = ml.solve(callback=track.plot_track_solve)


def test_summary_pdf(ml: Model) -> None:
    _ = ml.plots.summary_pdf()
