import matplotlib.pyplot as plt
import pytest
from pandas import Series

from pastas import Model
from pastas.plotting.plots import TrackSolve, compare, pairplot

# mpl.use("Agg")  # prevent _tkinter.TclError: Can't find a usable tk.tcl error


def test_plot(ml_noisemodel: Model) -> None:
    _ = ml_noisemodel.plot()
    plt.close()


def test_decomposition(ml_noisemodel: Model) -> None:
    _ = ml_noisemodel.plots.decomposition(min_ylim_diff=0.1)
    plt.close()


def test_results(ml_noisemodel: Model) -> None:
    _ = ml_noisemodel.plots.results()
    plt.close()


def test_results_kwargs(ml_noisemodel: Model) -> None:
    _ = ml_noisemodel.plots.results(
        split=True,
        block_or_step="block",
        adjust_height=False,
        return_warmup=True,
    )
    plt.close()


def test_results_mosaic(ml_noisemodel: Model) -> None:
    _ = ml_noisemodel.plots.results_mosaic(stderr=True)
    plt.close()


def test_stacked_results(ml_noisemodel: Model) -> None:
    _ = ml_noisemodel.plots.stacked_results()
    plt.close()


def test_block_response(ml_noisemodel: Model) -> None:
    _ = ml_noisemodel.plots.block_response()
    plt.close()


def test_step_response(ml_basic: Model) -> None:
    _ = ml_basic.plots.step_response()
    plt.close()


def test_diagnostics(ml_noisemodel: Model) -> None:
    _ = ml_noisemodel.plots.diagnostics(acf_options=dict(min_obs=10))
    plt.close()


def test_stresses(ml_noisemodel: Model) -> None:
    _ = ml_noisemodel.plots.stresses()
    plt.close()


def test_contributions_pie(ml_noisemodel: Model) -> None:
    with pytest.raises(DeprecationWarning):
        _ = ml_noisemodel.plots.contributions_pie()
        plt.close()


def test_compare(ml_noisemodel: Model) -> None:
    ml2 = ml_noisemodel.copy()
    models = [ml_noisemodel, ml2]
    _ = compare(models, names=["ml1", "ml2"], tmin="2011", tmax="2014")
    plt.close()


def test_tracksolve(ml_solved: Model) -> None:
    track = TrackSolve(ml_solved)
    _ = ml_solved.solve(callback=track.plot_track_solve)
    plt.close()


def test_summary_pdf(ml_noisemodel: Model) -> None:
    _ = ml_noisemodel.plots.summary_pdf()
    plt.close()


def test_pairplot(prec: Series, evap: Series, head: Series) -> None:
    _ = pairplot([prec, evap, head])
    plt.close()


def test_plot_contribution(ml_noisemodel: Model) -> None:
    _ = ml_noisemodel.plots.contribution(name="rch")
    plt.close()

    _ = ml_noisemodel.plots.contribution(
        name="rch", plot_stress=True, plot_response=True, block_or_step="step"
    )
    plt.close()

    _ = ml_noisemodel.plots.contribution(
        name="rch",
        plot_stress=True,
        plot_response=True,
        block_or_step="block",
        istress=1,
    )
    plt.close()


def test_series(ml_noisemodel: Model) -> None:
    """Test the series plot method."""
    # Basic series plot
    _ = ml_noisemodel.plots.series()
    plt.close()

    # With specific settings
    _ = ml_noisemodel.plots.series(tmin="2011", tmax="2014")
    plt.close()

    # Test with different styles and subplots
    _ = ml_noisemodel.plots.series(figsize=(10, 8))
    plt.close()


def test_cum_frequency(ml_noisemodel: Model) -> None:
    """Test the cumulative frequency plot method."""
    # Basic cumulative frequency plot
    _ = ml_noisemodel.plots.cum_frequency()
    plt.close()

    # With specific settings
    _ = ml_noisemodel.plots.cum_frequency(tmin="2011", tmax="2014")
    plt.close()

    # Test with custom arguments
    _ = ml_noisemodel.plots.cum_frequency(figsize=(8, 6))
    plt.close()


def test_standalone_series(prec: Series, evap: Series, head: Series) -> None:
    """Test standalone series plotting function."""
    from pastas.plotting.plots import series

    # Basic usage with head only
    axes = series(head=head)
    assert axes is not None
    plt.close()

    # With stresses
    stresses = [prec, evap]
    axes = series(head=head, stresses=stresses)
    assert axes is not None
    plt.close()

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
    plt.close()


def test_standalone_cum_frequency(head: Series) -> None:
    """Test standalone cumulative frequency function."""
    from pastas.plotting.plots import cum_frequency

    # Basic usage
    ax = cum_frequency(obs=head)
    assert ax is not None
    plt.close()

    # With simulation series
    sim = head + 0.1  # Create a simple sim series
    ax = cum_frequency(obs=head, sim=sim)
    assert ax is not None
    plt.close()

    # With custom figure size
    ax = cum_frequency(obs=head, figsize=(8, 4))
    assert ax is not None
    plt.close()


def test_standalone_acf(head: Series) -> None:
    """Test standalone acf plotting function."""
    from pastas.plotting.plots import acf

    # Basic usage
    ax = acf(series=head)
    assert ax is not None
    plt.close()

    # With custom parameters
    ax = acf(series=head, alpha=0.01, lags=100, smooth_conf=False, figsize=(10, 6))
    assert ax is not None
    plt.close()
