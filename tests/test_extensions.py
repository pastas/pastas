import pastas as ps


def test_register_plotly():
    """Test that register_plotly adds plotly attribute to Model class."""
    # Make sure plotly is not already registered
    if hasattr(ps.Model, "plotly"):
        delattr(ps.Model, "plotly")

    # Register plotly
    ps.extensions.register_plotly()

    # Check that plotly is registered
    assert hasattr(ps.Model, "plotly")

    # Clean up - remove plotly attribute
    if hasattr(ps.Model, "plotly"):
        delattr(ps.Model, "plotly")


def test_register_bokeh():
    """Test that register_bokeh adds bokeh attribute to Model class."""
    # Make sure bokeh is not already registered
    if hasattr(ps.Model, "bokeh"):
        delattr(ps.Model, "bokeh")

    # Register bokeh
    ps.extensions.register_bokeh()

    # Check that bokeh is registered
    assert hasattr(ps.Model, "bokeh")

    # Clean up - remove bokeh attribute
    if hasattr(ps.Model, "bokeh"):
        delattr(ps.Model, "bokeh")
