from bokeh.layouts import column, row
from bokeh.models import (
    ColumnDataSource,
    DataTable,
    RangeTool,
    ScientificFormatter,
    StringFormatter,
    TableColumn,
)
from bokeh.plotting import figure, show

from pastas.extensions import register_model_accessor


@register_model_accessor("bokeh")
class Bokeh:
    """Extension class for interactive bokeh figures for pastas Models.

    Usage
    -----
    >>> ps.extensions.register_bokeh()
    INFO: Registered bokeh plotting methods in Model class, e.g. `ml.bokeh.plot()`.
    >>> fig = ml.bokeh.results()
    >>> fig.write_html("results_figure.html")

    Methods
    -------
    plot
        plot oseries and model simulation, interactive version of `ml.plot()`
    results
        plot oseries, model simulation, contribution, step responses and parameters
        table,interactive version of `ml.plots.results()`

    Notes
    -----
    The `bokeh` extension is registered in the `Model` class by calling the
    `register_bokeh()` function. To work in Juptyer notebooks, the
    `bokeh.io.output_notebook()` function should be called before plotting. The `bokeh`
    extension is not registered by default, and should be called explicitly. Check the
    bokeh documentation for more information on how to interact with the plots.

    """

    def __init__(self, model):
        self._model = model

    def plot(self, tmin=None, tmax=None, height=300, width=600, show_plot=True):
        """Plot the observations and model simulation.

        Parameters
        ----------
        tmin : pd.Timestamp, optional
            start time for model simulation, by default None
        tmax : pd.Timestamp, optional
            end time for model simulation, by default None
        height : int, optional
             height of the plot, by default 500
        width : int, optional
           width of the plot, by default 800
        show_plot : bool, optional
            Show the plot (i.e., in Jupyter Notebooks), by default True

        Returns
        -------
        p : bokeh.plotting.figure
            Bokeh figure with the observations and model simulation.

        Examples
        --------
        >>> ps.extensions.register_bokeh()
        INFO: Registered bokeh plotting methods in Model class, e.g. `ml.bokeh.plot()`.
        >>>
        >>> fig = ml.bokeh.plot()

        """

        data = self._model.get_output_series(tmin=tmin, tmax=tmax, split=False)
        source = ColumnDataSource(data)
        rsq = self._model.stats.rsq(tmin=tmin, tmax=tmax)

        TOOLS = "zoom_in,zoom_out,reset,pan,xwheel_zoom,box_zoom,undo"

        p = figure(
            title="Pastas Model",
            y_axis_label="Head",
            x_axis_location=None,
            tools=TOOLS,
            width=width,
            height=height,
            x_axis_type="datetime",
            toolbar_location="above",
        )

        p.scatter(
            "index",
            "Head_Calibration",
            source=source,
            legend_label="Observations",
            color="black",
            alpha=0.7,
        )
        p.line(
            "index",
            "Simulation",
            source=source,
            legend_label=r"Simulation (R2 = {:.2f})".format(rsq),
            line_width=2,
        )
        p.legend.ncols = 2

        if show_plot:
            show(p)
        return p

    def results(self, tmin=None, tmax=None, height=500, width=800, show_plot=True):
        """Overview of the results of the pastas model.

        Parameters
        ----------
        tmin : pd.Timestamp, optional
            start time for model simulation, by default None
        tmax : pd.Timestamp, optional
            end time for model simulation, by default None
        height : int, optional
             height of the plot, by default 500
        width : int, optional
           width of the plot, by default 800
        show_plot : bool, optional
            Show the plot (i.e., in Jupyter Notebooks), by default True

        Returns
        -------
        grid : bokeh.layouts.column
            Bokeh layout with the results of the pastas model.

        Examples
        --------
        >>> ps.extensions.register_bokeh()
        INFO: Registered bokeh plotting methods in Model class, e.g. `ml.bokeh.plot()`.
        >>> fig = ml.bokeh.results()

        """
        data = self._model.get_output_series(tmin=tmin, tmax=tmax, split=False)
        ranges = data.max() - data.min()
        ranges = ranges.drop([ranges.iloc[:2].idxmin(), "Noise"])
        heights = (ranges / ranges.sum() * (height - 50)).values.astype(int)
        source = ColumnDataSource(data)
        rsq = self._model.stats.rsq(tmin=tmin, tmax=tmax)

        TOOLS = "zoom_in,zoom_out,reset,pan,xwheel_zoom,box_zoom,undo"

        p = figure(
            title="Pastas Model",
            y_axis_label="Head",
            x_axis_location=None,
            tools=TOOLS,
            width=int(0.75 * width),
            height=heights[0],
            x_axis_type="datetime",
            toolbar_location="above",
        )

        p.scatter(
            "index",
            "Head_Calibration",
            source=source,
            legend_label="Observations",
            color="black",
            alpha=0.7,
        )
        p.line(
            "index",
            "Simulation",
            source=source,
            legend_label=r"Simulation (R2 = {:.2f})".format(rsq),
        )
        p.legend.ncols = 2

        # Residuals
        res_plot = figure(
            y_axis_label="Residuals",
            toolbar_location=None,
            tools=TOOLS,
            x_range=p.x_range,
            width=int(0.75 * width),
            height=heights[2],
            x_axis_type="datetime",
            x_axis_location=None,
        )

        res_plot.scatter(
            "index",
            "Residuals",
            source=source,
            color="black",
            alpha=0.7,
            legend_label="Residuals",
        )
        res_plot.line(
            "index",
            "Residuals",
            source=source,
            color="black",
            alpha=0.7,
            legend_label="Residuals",
        )

        if self._model.settings["noise"]:
            res_plot.line("index", "Noise", source=source, legend_label="Noise")
            res_plot.scatter("index", "Noise", source=source, legend_label="Noise")

        res_plot.legend.ncols = 2

        # Parameter Table
        df = ColumnDataSource(self._model.parameters.loc[:, ["optimal"]])
        columns = [
            TableColumn(
                field="index",
                title="Name",
                formatter=StringFormatter(font_style="bold"),
            ),
            TableColumn(
                field="optimal",
                title="Optimal",
                formatter=ScientificFormatter(precision=2),
            ),
        ]
        table = DataTable(
            source=df,
            columns=columns,
            editable=False,
            index_position=None,
            width=int(0.25 * width),
            height=heights[0] + heights[2] - 10,
        )

        left_column = [p, res_plot]
        right_column = [table]

        # Contributions
        rfunc_plot = None

        for i, smname in enumerate(self._model.stressmodels.keys(), start=2):
            if i == int(len(self._model.stressmodels) + 1):
                x_axis_location = "below"
            else:
                x_axis_location = None

            contrib_plot = figure(
                y_axis_label="Rise",
                toolbar_location=None,
                tools=TOOLS,
                x_axis_location=x_axis_location,
                width=int(0.75 * width),
                height=heights[i],
                x_axis_type="datetime",
                x_range=p.x_range,
            )

            # xrange = False if rfunc_plot is None else rfunc_plot.x_range

            rfunc_plot = figure(
                x_axis_label=None,
                toolbar_location=None,
                x_axis_location=x_axis_location,
                width=int(0.25 * width),
                height=heights[i],
            )

            contrib_plot.line("index", smname, source=source)
            response = self._model.get_step_response(smname)
            rfunc_plot.line(response.index, response.values)

            left_column.append(contrib_plot)
            right_column.append(rfunc_plot)

        select = figure(
            height=50,
            width=int(0.75 * width),
            y_range=p.y_range,
            x_axis_type="datetime",
            y_axis_type=None,
            tools="",
            toolbar_location=None,
            background_fill_color="#ffffff",
        )

        range_tool = RangeTool(x_range=p.x_range)
        select.line("index", "Simulation", source=source)
        select.add_tools(range_tool)
        left_column.append(select)

        layout = row(column(left_column), column(right_column))
        grid = column(layout, width=width, height=height)

        if show_plot:
            show(grid)

        return grid
