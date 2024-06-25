"""This module contains interactive plots for Pastas models."""

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from scipy.stats import norm, probplot

from pastas.extensions import register_model_accessor
from pastas.plotting.plotutil import (
    _get_height_ratios,
    _table_formatter_params,
    _table_formatter_stderr,
)
from pastas.rfunc import HantushWellModel
from pastas.stats import acf


@register_model_accessor("plotly")
class Plotly:
    """Extension class for interactive plotly figures for pastas Models.

    Usage
    -----
    >>> ps.extensions.register_plotly_extension()
    INFO: Registered plotly plotting methods in Model class, e.g. `ml.plotly.plot()`.
    >>> fig = ml.plotly.results()
    >>> fig.write_html("results_figure.html")

    Methods
    -------
    plot
        plot oseries and model simulation, interactive version of `ml.plot()`
    results
        plot oseries, model simulation, contribution, step responses and parameters
        table,interactive version of `ml.plots.results()`
    diagnostics
        plot noise, autocorrelation, distribution of noise and heteroscedasticity,
        interactive verison of `ml.plots.diagnostics()`
    """

    def __init__(self, model):
        self._model = model

    def plot(self, tmin=None, tmax=None):
        """Plotly version of pastas.Model.plot().

        Parameters
        ----------
        tmin : pd.Timestamp, optional
            start time for model simulation, by default None
        tmax : pd.Timestamp, optional
            end time for model simulation, by default None

        Returns
        -------
        fig : plotly.Figure
            plotly Figure showing oseries and model simulation
        """

        traces = []

        o = self._model.observations()
        o_nu = self._model.oseries.series.drop(o.index)

        # add oseries
        if not o_nu.empty:
            trace_oseries_nu = go.Scattergl(
                x=o_nu.index,
                y=o_nu.values,
                mode="markers",
                marker={"color": "gray", "size": 3},
                name="(unused)",
                legendgroup="oseries",
                showlegend=False,
            )
            trace_oseries = go.Scattergl(
                x=o.index,
                y=o.values,
                mode="markers",
                marker={"color": "black", "size": 5},
                name=self._model.oseries.name,
                legendgroup="oseries",
            )
            traces.append(trace_oseries_nu)
            traces.append(trace_oseries)
        else:
            trace_oseries = go.Scattergl(
                x=o.index,
                y=o.values,
                mode="markers",
                marker_color="black",
                name=self._model.oseries.name,
                legendgroup="oseries",
            )
            traces.append(trace_oseries)

        sim = self._model.simulate(tmin=tmin, tmax=tmax)
        trace_sim = go.Scattergl(
            x=sim.index,
            y=sim.values,
            mode="lines",
            marker_color="#1F77B4",
            name=f"Sim (R<sup>2</sup> = {self._model.stats.rsq():.3f})",
            legendgroup="sim",
        )
        traces.append(trace_sim)

        layout = {
            "xaxis": {"range": [sim.index[0], sim.index[-1]]},
            "yaxis": {"title": "(m NAP)"},
            "legend": {
                "traceorder": "reversed+grouped",
                "orientation": "h",
                "xanchor": "left",
                "yanchor": "bottom",
                "x": 0.0,
                "y": 1.02,
            },
            "dragmode": "pan",
            "margin": dict(t=70, b=40, l=40, r=10),
        }

        return go.Figure(data=traces, layout=go.Layout(layout))

    def results(self, tmin=None, tmax=None):
        """Plotly version of pastas.Model.plots.results().

        Parameters
        ----------
        ml : pastas.Model
            model to plot results for
        tmin : pd.Timestamp, optional
            start time for model results, by default None
        tmax : pd.Timestamp, optional
            end time for model results, by default None

        Returns
        -------
        dict
            dictionary containing plotly subplots data and layout
        """
        if tmin is None:
            tmin = self._model.settings["tmin"]
        if tmax is None:
            tmax = self._model.settings["tmax"]

        # for collecting data
        traces = []

        # dimensions
        nsm = len(self._model.stressmodels)
        nrows = 2 + nsm
        naxes = 2 + 2 * nsm

        # oseries
        o = self._model.observations()
        o_nu = self._model.oseries.series.drop(o.index)

        trace_oseries_nu = go.Scattergl(
            x=o_nu.index,
            y=o_nu.values,
            mode="markers",
            marker={"color": "gray", "size": 3},
            name="(unused)",
            legendgroup="0",
            showlegend=False,
            xaxis="x",
            yaxis="y",
        )
        traces.append(trace_oseries_nu)

        trace_oseries = go.Scattergl(
            x=o.index,
            y=o.values,
            mode="markers",
            marker={"color": "black", "size": 5},
            name=self._model.oseries.name,
            legendgroup="1",
            xaxis="x",
            yaxis="y",
        )
        traces.append(trace_oseries)

        # simulation
        sim = self._model.simulate()
        trace_sim = go.Scattergl(
            x=sim.index,
            y=sim.values,
            mode="lines",
            marker_color="#1F77B4",
            name=f"simulation (R<sup>2</sup> = {self._model.stats.rsq():.3f})",
            legendgroup="2",
            xaxis="x",
            yaxis="y",
        )
        traces.append(trace_sim)

        # residuals
        res = self._model.residuals()
        trace_res = go.Scattergl(
            x=res.index,
            y=res.values,
            mode="lines",
            marker_color="black",
            name="residuals",
            # legendgroup="residuals",
            xaxis="x",
            yaxis="y2",
            showlegend=False,
        )
        traces.append(trace_res)

        # noise
        if self._model.settings["noise"]:
            noise = self._model.noise()
            trace_noise = go.Scattergl(
                x=noise.index,
                y=noise.values,
                mode="lines",
                marker_color="#1F77B4",
                name="noise",
                # legendgroup="residuals",
                xaxis="x",
                yaxis="y2",
                showlegend=False,
            )
            traces.append(trace_noise)

        # contributions
        contribs = self._model.get_contributions(
            split=False,  # tmin=tmin, tmax=tmax, return_warmup=return_warmup
        )
        for i, c in enumerate(contribs):
            iax_contrib = 3 + i
            iax_response = 3 + nsm + i

            trace_c = go.Scattergl(
                x=c.index,
                y=c.values,
                mode="lines",
                marker_color="#1F77B4",
                name=f"{c.name}",
                legendgroup=f"contrib{i}",
                xaxis="x",
                yaxis=f"y{iax_contrib}",
                showlegend=False,
            )
            traces.append(trace_c)

            # response
            rkwargs = {}
            p = None
            if self._model.stressmodels[c.name].rfunc is not None:
                if isinstance(self._model.stressmodels[c.name].rfunc, HantushWellModel):
                    rkwargs = {"warn": False}
                    p = self._model.stressmodels[c.name].get_parameters(
                        model=self._model, istress=0
                    )
            response = self._model._get_response(
                block_or_step="step", name=c.name, p=p, add_0=True, **rkwargs
            )
            trace_r = go.Scatter(
                x=response.index,
                y=response.values,
                mode="lines",
                marker_color="#1F77B4",
                name=f"{c.name}",
                # legendgroup=f"contrib{i}",
                xaxis=f"x{3+nsm}",
                yaxis=f"y{iax_response}",
                showlegend=False,
            )
            traces.append(trace_r)

        # calculate subplot
        ylims = [
            (
                min([sim.min(), o[tmin:tmax].min()]),
                max([sim.max(), o[tmin:tmax].max()]),
            ),
            (res.min(), res.max()),
        ]  # use residuals (b/c always bigger than noise)

        for contrib in contribs:
            hs = contrib.loc[tmin:tmax]
            if hs.empty:
                if contrib.empty:
                    ylims.append((0.0, 0.0))
                else:
                    ylims.append((contrib.min(), hs.max()))
            else:
                ylims.append((hs.min(), hs.max()))
        hrs = np.asarray(_get_height_ratios(ylims))
        # claim at least 1% figure height for zero contributions for visibility
        hrs = hrs.clip(min=0.01 * np.sum(hrs))
        # subplot positions and spacing
        dx = 0.015
        dy = 0.01
        # calculate whitespace
        wspace = (2 * len(hrs) - 2) * dy
        # scale plot heights
        hrs_frac = hrs / np.sum(hrs) * (1 - wspace)
        # add whitespace
        hrs_frac[[0, -1]] += dy
        hrs_frac[1:-1] += 2 * dy
        # calculate y-mid positions (between plots)
        y_pos = 1 - np.cumsum(np.concatenate([np.zeros(1), hrs_frac]))
        x_pos = 0.67
        # get tops and bottoms of axes from y-position array
        ytops = y_pos[:-1].copy()
        ytops[1:] -= dy  # half of whitespace by lowering tops by dy
        ybots = y_pos[1:].copy()
        ybots[:-1] += dy  # create other half of whitespace by raising bottoms by dy
        ybots[ybots < 0] = 0  # floating point issues, should alway be > 0

        # parameter table
        p = self._model.parameters.copy().loc[:, ["name", "optimal", "stderr"]]
        p.loc[:, "name"] = p.index
        stderr = p.loc[:, "stderr"] / p.loc[:, "optimal"]
        p.loc[:, "optimal"] = p.loc[:, "optimal"].apply(_table_formatter_params)
        p.loc[:, "stderr"] = stderr.abs().apply(_table_formatter_stderr)

        tab = go.Table(
            domain=dict(x=[x_pos + dx, 1.0], y=[y_pos[2] + dy, 1.0]),
            header=dict(
                values=[
                    "<b>Parameter</b>",
                    "<b>Optimal</b>",
                    "<b>Std. Err.</b>",
                ],
                font=dict(size=12),
                align=["left", "center", "center"],
                height=40,
            ),
            cells=dict(
                values=[p[k].tolist() for k in p.columns],
                align=["left", "right", "right"],
                height=30,
            ),
            columnwidth=[100, 70, 70],
        )

        traces.append(tab)

        layout_dict = dict(
            # top left (row 0, col 0) - oseries/simulation plot
            xaxis=dict(
                domain=[0.0, x_pos - dx],
                anchor=f"y{nrows}",
            ),
            yaxis=dict(
                domain=[ybots[0], ytops[0]],
                anchor="x",
            ),
            # row 1, col 0 - residuals/noise plot
            xaxis2=dict(
                domain=[0.0, x_pos - dx],
                anchor="y2",
            ),
            yaxis2=dict(
                domain=[ybots[1], ytops[1]],
                anchor="x2",
                scaleanchor="y",
            ),
        )
        # add layout for stressmodels
        # unfortunately I can only get it to work with numbering down the
        # columns, e.g.
        # ax1 table
        # ax2 table
        # ax3 ax5
        # ax4 ax6
        # also order is important, so first all contributions, then step responses

        # add contributions axes
        for ism in range(nsm):
            irow = 2 + ism  # 2, 3, 4, ..., (0-based)
            iax_contrib = 3 + ism  # 3, 4, 5, ..., (1-based)

            # contributions subplot
            layout_dict[f"xaxis{iax_contrib}"] = dict(
                domain=[0.0, x_pos - dx],
                anchor=f"y{iax_contrib}",
            )
            layout_dict[f"yaxis{iax_contrib}"] = dict(
                domain=[ybots[irow], ytops[irow]],
                anchor=f"x{iax_contrib}",
                scaleanchor="y",
            )
        # add response axes
        for ism in range(nsm):
            irow = 2 + ism  # 2, 3, 4, ..., (0-based)
            iax_response = 3 + nsm + ism

            # step response subplot
            layout_dict[f"xaxis{iax_response}"] = dict(
                domain=[x_pos + dx, 1],
                anchor=f"y{naxes}",
            )
            layout_dict[f"yaxis{iax_response}"] = dict(
                domain=[ybots[irow], ytops[irow]],
                anchor=f"x{3+nsm}",
            )

        layout = go.Layout(layout_dict)

        fig = go.Figure(data=traces, layout=layout)

        # add titles for subplots
        rnlabel = [
            "residuals / noise" if self._model.settings["noise"] else "residuals"
        ]
        labels = rnlabel + list(self._model.stressmodels.keys())
        for i, lbl in enumerate(labels):
            fig.add_annotation(
                xref="paper",
                yref="paper",
                x=0.015,
                y=y_pos[i + 1],
                xanchor="left",
                yanchor="middle",
                showarrow=False,
                text=lbl,
                font=dict(
                    size=12,
                ),
                align="left",
            )

        # add table title
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=x_pos + dx + 0.015,
            y=1.015,
            xanchor="left",
            yanchor="middle",
            showarrow=False,
            text="Model parameters (n<sub>c</sub>=7)",
            font=dict(
                size=12,
            ),
            align="left",
        )

        # add step response title
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=x_pos + dx + 0.015,
            y=y_pos[2],
            xanchor="left",
            yanchor="middle",
            showarrow=False,
            text="Step response",
            font=dict(
                size=12,
            ),
            align="left",
        )

        # set size, axes labels and legend position
        fig.update_layout(
            xaxis={"range": [tmin, tmax]},
            yaxis={"title": "[m NAP]"},  # oseries, simulation
            yaxis2={"title": "[m]"},  # residuals
            legend={
                "traceorder": "reversed+grouped",
                "orientation": "h",
                "xanchor": "left",
                "yanchor": "bottom",
                "x": 0.0,
                "y": 1.00,
            },
            # height=1000,
            # width=1000,
            margin=dict(t=60, b=20, l=10, r=25),
            dragmode="pan",
        )
        update_labels = {f"yaxis{i}": {"title": "[m]"} for i in range(3, nrows + 1)}
        update_labels[f"xaxis{nrows+1}"] = {"title": "time [d]"}
        fig.update_layout(**update_labels)

        return fig

    def diagnostics(self):
        """Plotly version of pastas.Model.plots.diagnostics().

        Parameters
        ----------
        ml : pastas.Model
            model to plot results for

        Returns
        -------
        fig : plotly.Figure
            plotly figure with model diagnostics
        """
        # prepare data
        sim = self._model.simulate()
        if self._model.settings["noise"]:
            series = self._model.noise()
            resnoisename = "noise"
        else:
            series = self._model.residuals()
            resnoisename = "residuals"

        df_acf = acf(series, full_output=True)
        x = df_acf.index.total_seconds() / (24 * 60 * 60)
        conf = df_acf["conf"].rolling(10, min_periods=1).mean().values

        if self._model.interpolate_simulation:
            sim_interpolated = np.interp(series.index.asi8, sim.index.asi8, sim.values)
            sim = pd.Series(index=series.index, data=sim_interpolated)

        sim = sim.loc[series.index]

        # let the plotting begin
        fig = make_subplots(
            rows=4,
            cols=2,
            specs=[
                [{"colspan": 2}, None],
                [{"colspan": 2}, None],
                [{}, {}],
                [{}, {}],
            ],
            subplot_titles=[
                f"{resnoisename} (&#956;={series.mean():.1f})",
                "Autocorrelation",
                "Histogram",
                "Probability plot",
                "Heteroscedasticity",
                "",
            ],
            horizontal_spacing=0.1,
            vertical_spacing=0.075,
        )

        # Noise
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                line_color="rgba(31,119,180,1)",
                showlegend=False,
                name=resnoisename,
            ),
            row=1,
            col=1,
        )

        # ACF
        fig.add_trace(
            go.Scatter(
                x=x,
                y=conf,
                mode="lines",
                fillcolor="rgba(32,146,230,0.3)",
                line_color="rgba(255,255,255,0)",
                showlegend=False,
                name="CI",
                hoverinfo="skip",
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=-conf,
                mode="lines",
                fillcolor="rgba(32,146,230,0.3)",
                fill="tonexty",
                line_color="rgba(255,255,255,0)",
                legendgroup="1",
                showlegend=False,
                name="CI",
                hoverinfo="skip",
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=x,
                y=df_acf.iloc[:, 0],
                marker={"color": ["black"] * df_acf.index.size},
                legendgroup="0",
                showlegend=False,
                width=0.9,
                name="ACF",
            ),
            row=2,
            col=1,
        )

        # normality
        _, bins = np.histogram(series.values, bins=50)
        bins = 0.5 * (bins[:-1] + bins[1:])
        fig.add_trace(
            go.Histogram(
                x=series.values,
                ybins={"size": 50},
                name="Histogram",
                showlegend=False,
                histnorm="probability density",
                marker=dict(color="rgba(31,119,180,1)"),
            ),
            row=3,
            col=1,
        )
        pdf = norm.pdf(bins, series.mean(), series.std())
        fig.add_trace(
            go.Scatter(
                x=bins,
                y=pdf,
                mode="lines",
                name="PDF",
                line=dict(
                    dash="dash",
                    color="black",
                ),
                showlegend=False,
            ),
            row=3,
            col=1,
        )

        (osm, osr), (slope, intercept, r) = probplot(series, dist="norm", rvalue=False)
        fig.add_trace(
            go.Scatter(
                x=osm,
                y=osr,
                mode="markers",
                marker={"color": "rgba(31,119,180,1)"},
                showlegend=False,
                name="probplot",
            ),
            row=3,
            col=2,
        )

        fig.add_trace(
            go.Scatter(
                x=osm,
                y=slope * osm + intercept,
                mode="lines",
                line={"color": "black"},
                showlegend=False,
                name="fit",
            ),
            row=3,
            col=2,
        )
        fig.add_annotation(
            xref="x4 domain",
            yref="y4 domain",
            x=0.05,
            y=0.95,
            xanchor="left",
            yanchor="top",
            showarrow=False,
            text=f"R<sup>2</sup>={r:.3f}",
            font=dict(
                size=12,
            ),
            align="left",
        )

        # heteroscedasticity
        fig.add_trace(
            go.Scatter(
                name="heteroscedasticity",
                x=sim,
                y=series,
                mode="markers",
                showlegend=False,
                marker=dict(
                    size=5,
                    color="rgba(31,119,180,0.5)",
                    line=dict(
                        color="black",
                        width=0.25,
                    ),
                ),
            ),
            row=4,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                name="heteroscedasticity",
                x=sim,
                y=np.sqrt(series.abs()),
                mode="markers",
                showlegend=False,
                marker=dict(
                    size=5,
                    color="rgba(31,119,180,0.5)",
                    line=dict(
                        color="black",
                        width=0.25,
                    ),
                ),
            ),
            row=4,
            col=2,
        )

        # update x-axes, y-axes
        fig.update_yaxes(title_text=resnoisename, title_standoff=0, row=1, col=1)

        fig.update_xaxes(title_text="Lags [d]", title_standoff=0, row=2, col=1)
        fig.update_yaxes(title_text="ACF [-]", title_standoff=0, row=2, col=1)

        fig.update_yaxes(
            title_text="Probability density", title_standoff=0, row=3, col=1
        )

        fig.update_xaxes(
            title_text="Theoretical quantiles", title_standoff=0, row=3, col=2
        )
        fig.update_yaxes(title_text="Ordered values", title_standoff=0, row=3, col=2)

        fig.update_xaxes(title_text="Simulated values", title_standoff=0, row=4, col=1)
        fig.update_yaxes(title_text="Residuals", title_standoff=0, row=4, col=1)

        fig.update_yaxes(
            title_text=r"&#8730; Residuals", title_standoff=0, row=4, col=2
        )
        fig.update_xaxes(title_text="Simulated values", title_standoff=0, row=4, col=2)

        # update titles

        fig.layout.annotations[0].update(x=0.05, font={"size": 13})
        fig.layout.annotations[1].update(x=0.05, font={"size": 13})
        fig.layout.annotations[2].update(x=0.05, font={"size": 13})
        fig.layout.annotations[3].update(x=0.60, font={"size": 13})
        fig.layout.annotations[4].update(x=0.05, font={"size": 13})

        #
        fig.update_layout(
            legend={
                "traceorder": "grouped",
                "orientation": "h",
                "xanchor": "left",
                "yanchor": "bottom",
                "x": 0.0,
                "y": 1.00,
            },
            # width=750,
            # height=1000,
            margin=dict(t=60, b=20, l=25, r=10),
        )
        return fig
