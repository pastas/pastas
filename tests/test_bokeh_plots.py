import pytest

import pastas as ps

# Skip tests if bokeh is not installed
pytest.importorskip("bokeh")


def test_bokeh_plots_available(ml_solved: ps.Model) -> None:
    """Test if all bokeh plots are available from model interface."""
    # Check if bokeh is available
    ml = ml_solved.copy()
    ps.extensions.register_bokeh()

    # Check if key bokeh plotting methods are available
    assert hasattr(ml.bokeh, "results")
    assert hasattr(ml.bokeh, "plot")


def test_bokeh_plot(ml_solved: ps.Model) -> None:
    """Test if the bokeh plot can be generated without errors."""
    ml = ml_solved.copy()
    ps.extensions.register_bokeh()
    try:
        plot = ml.bokeh.plot()
        assert plot is not None
    except Exception as e:
        pytest.fail(f"Generating bokeh plot failed: {e}")


def test_bokeh_results_plot(ml_solved: ps.Model) -> None:
    """Test if the bokeh results plot can be generated without errors."""
    ml = ml_solved.copy()
    ps.extensions.register_bokeh()
    try:
        plot = ml.bokeh.results()
        assert plot is not None
    except Exception as e:
        pytest.fail(f"Generating bokeh results plot failed: {e}")
