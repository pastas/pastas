import pytest

import pastas as ps

# Skip tests if plotly is not installed
pytest.importorskip("plotly")


def test_plotly_plots_available(ml_rm: ps.Model):
    """Test if all plotly plots are available from model interface."""
    # register plotly
    ps.extensions.register_plotly()

    # Check if key plotly plotting methods are available
    assert hasattr(ml_rm.plotly, "plot")
    assert hasattr(ml_rm.plotly, "results")
    assert hasattr(ml_rm.plotly, "diagnostics")


def test_plotly_plot(ml_rm: ps.Model):
    """Test if the plotly plot can be generated without errors."""
    # register plotly
    ps.extensions.register_plotly()

    try:
        plot = ml_rm.plotly.plot()
        assert plot is not None
    except Exception as e:
        pytest.fail(f"Generating plotly plot failed: {e}")


def test_plotly_results_plot(ml_rm: ps.Model):
    """Test if the plotly results plot can be generated without errors."""
    # register plotly
    ps.extensions.register_plotly()

    try:
        plot = ml_rm.plotly.results()
        assert plot is not None
    except Exception as e:
        pytest.fail(f"Generating plotly results plot failed: {e}")


def test_plotly_diagnostics_plot(ml_rm: ps.Model):
    """Test if the plotly diagnostics plot can be generated without errors."""
    # register plotly
    ps.extensions.register_plotly()

    try:
        plot = ml_rm.plotly.diagnostics()
        assert plot is not None
    except Exception as e:
        pytest.fail(f"Generating plotly diagnostics plot failed: {e}")
