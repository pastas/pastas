import pytest

import pastas as ps

# Skip tests if plotly is not installed
pytest.importorskip("plotly")


def test_plotly_plots_available(ml_solved: ps.Model) -> None:
    """Test if all plotly plots are available from model interface."""
    # register plotly
    ml = ml_solved.copy()
    ps.extensions.register_plotly()

    # Check if key plotly plotting methods are available
    assert hasattr(ml.plotly, "plot")
    assert hasattr(ml.plotly, "results")
    assert hasattr(ml.plotly, "diagnostics")


def test_plotly_plot(ml_solved: ps.Model) -> None:
    """Test if the plotly plot can be generated without errors."""
    # register plotly
    ps.extensions.register_plotly()
    ml = ml_solved.copy()

    try:
        plot = ml.plotly.plot()
        assert plot is not None
    except Exception as e:
        pytest.fail(f"Generating plotly plot failed: {e}")


def test_plotly_results_plot(ml_solved: ps.Model) -> None:
    """Test if the plotly results plot can be generated without errors."""
    # register plotly
    ps.extensions.register_plotly()
    ml = ml_solved.copy()

    try:
        plot = ml.plotly.results()
        assert plot is not None
    except Exception as e:
        pytest.fail(f"Generating plotly results plot failed: {e}")


def test_plotly_diagnostics_plot(ml_solved: ps.Model) -> None:
    """Test if the plotly diagnostics plot can be generated without errors."""
    # register plotly
    ps.extensions.register_plotly()
    ml = ml_solved.copy()

    try:
        plot = ml.plotly.diagnostics()
        assert plot is not None
    except Exception as e:
        pytest.fail(f"Generating plotly diagnostics plot failed: {e}")
