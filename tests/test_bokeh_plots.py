import pytest

import pastas as ps

# Skip tests if bokeh is not installed
pytest.importorskip("bokeh")


def test_bokeh_plots_available(ml_solved: ps.Model) -> None:
    """Test if all bokeh plots are available from model interface."""
    # Check if bokeh is available
    ps.extensions.register_bokeh()

    # Check if key bokeh plotting methods are available
    assert hasattr(ml_solved.bokeh, "results")
    assert hasattr(ml_solved.bokeh, "plot")


def generate_error_message(method_name: str) -> str:
    """Helper function to generate error messages for bokeh plots."""
    return f"The bokeh {method_name} method failed to generate a plot."


def test_bokeh_plot(ml_solved: ps.Model) -> None:
    """Test if the bokeh plot can be generated without errors."""
    ps.extensions.register_bokeh()
    plot = ml_solved.bokeh.plot()
    assert plot is not None, generate_error_message("plot")


def test_bokeh_results_plot(ml_solved: ps.Model) -> None:
    """Test if the bokeh results plot can be generated without errors."""
    ps.extensions.register_bokeh()
    plot = ml_solved.bokeh.results()
    assert plot is not None, generate_error_message("results")
