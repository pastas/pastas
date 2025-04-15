import pytest
from pandas import Series

from pastas import Model
from pastas.plotting.plots import TrackSolve, compare, pairplot

# mpl.use("Agg")  # prevent _tkinter.TclError: Can't find a usable tk.tcl error


def test_plot(ml_solved: Model) -> None:
    _ = ml_solved.plot()


def test_decomposition(ml_solved: Model) -> None:
    _ = ml_solved.plots.decomposition(min_ylim_diff=0.1)


def test_results(ml_solved: Model) -> None:
    _ = ml_solved.plots.results()


def test_results_kwargs(ml_solved: Model) -> None:
    _ = ml_solved.plots.results(
        split=True,
        block_or_step="block",
        adjust_height=False,
        return_warmup=True,
    )


def test_results_mosaic(ml_solved: Model) -> None:
    _ = ml_solved.plots.results_mosaic(stderr=True)


def test_stacked_results(ml_solved: Model) -> None:
    _ = ml_solved.plots.stacked_results()


def test_block_response(ml_solved: Model) -> None:
    _ = ml_solved.plots.block_response()


def test_step_response(ml: Model) -> None:
    _ = ml.plots.step_response()


def test_diagnostics(ml_solved: Model) -> None:
    _ = ml_solved.plots.diagnostics(acf_options=dict(min_obs=10))


def test_stresses(ml_solved: Model) -> None:
    _ = ml_solved.plots.stresses()


def test_contributions_pie(ml_solved: Model) -> None:
    with pytest.raises(DeprecationWarning):
        _ = ml_solved.plots.contributions_pie()


def test_compare(ml_solved: Model) -> None:
    ml2 = ml_solved.copy()
    models = [ml_solved, ml2]
    _ = compare(models, names=["ml1", "ml2"], tmin="2011", tmax="2014")


def test_tracksolve(ml_solved: Model) -> None:
    track = TrackSolve(ml_solved)
    _ = ml_solved.solve(callback=track.plot_track_solve)


def test_summary_pdf(ml_solved: Model) -> None:
    _ = ml_solved.plots.summary_pdf()


def test_pairplot(prec: Series, evap: Series, head: Series) -> None:
    _ = pairplot([prec, evap, head])


def test_plot_contribution(ml_solved: Model) -> None:
    _ = ml_solved.plots.contribution(name="rch")
    _ = ml_solved.plots.contribution(
        name="rch", plot_stress=True, plot_response=True, block_or_step="step"
    )
    _ = ml_solved.plots.contribution(
        name="rch",
        plot_stress=True,
        plot_response=True,
        block_or_step="block",
        istress=1,
    )


def test_series(ml_solved: Model) -> None:
    """Test the series plot method."""
    # Basic series plot
    _ = ml_solved.plots.series()
    # With specific settings
    _ = ml_solved.plots.series(tmin="2011", tmax="2014")
    # Test with different styles and subplots
    _ = ml_solved.plots.series(figsize=(10, 8))


def test_cum_frequency(ml_solved: Model) -> None:
    """Test the cumulative frequency plot method."""
    # Basic cumulative frequency plot
    _ = ml_solved.plots.cum_frequency()
    # With specific settings
    _ = ml_solved.plots.cum_frequency(tmin="2011", tmax="2014")
    # Test with custom arguments
    _ = ml_solved.plots.cum_frequency(figsize=(8, 6))


def test_standalone_series(prec: Series, evap: Series, head: Series) -> None:
    """Test standalone series plotting function."""
    from pastas.plotting.plots import series

    # Basic usage with head only
    axes = series(head=head)
    assert axes is not None

    # With stresses
    stresses = [prec, evap]
    axes = series(head=head, stresses=stresses)
    assert axes is not None

    # Various options
    axes = series(
        head=head,
        stresses=stresses,
        hist=True,
        kde=True,
        table=True,
        titles=False,
        tmin="2011",
        tmax="2014",
        figsize=(8, 6),
    )
    assert axes is not None


def test_standalone_cum_frequency(head: Series) -> None:
    """Test standalone cumulative frequency function."""
    from pastas.plotting.plots import cum_frequency

    # Basic usage
    ax = cum_frequency(obs=head)
    assert ax is not None

    # With simulation series
    sim = head + 0.1  # Create a simple sim series
    ax = cum_frequency(obs=head, sim=sim)
    assert ax is not None

    # With custom figure size
    ax = cum_frequency(obs=head, figsize=(8, 4))
    assert ax is not None
