from itertools import combinations
from typing import List, Optional, Tuple
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame, concat

import pastas as ps
from pastas.typing import Axes, Model


class CompareModels:
    """Class for visually comparing pastas Models.

    This is a versatile class for constructing visual model comparison plots. The
    default `CompareModels.plot()` method mimics `ml.plots.results()` but allows
    multiple models to be included in the figure. Instead of parameter uncertainties,
    by default only optimal values are shown for each model and a table containing
    fit metrics is included in the top right of the figure.

    The visualization of each component (i.e. time series, a table) is controlled by
    separate functions allowing users to easily customize their figure. The layout of
    the figure is controlled by a so-called "mosaic", which is essentially a 2D array
    (in the form of nested lists) containing labels that refer to specific axes::

        mosaic = [
            ["sim", "sim", "met"],   # oseries+simulation, and metrics
            ["sim", "sim", "tab"],   # oseries+simulation and parameters
            ["res", "res", "tab"],   # residuals+noise and parameters
            ["con0", "con0", "rf0"]  # contributions and step response
        ]

    In this example, the "sim" axis will be 2x2 in the top left portion of the figure
    (with total dimensions (4x3)), while the "met" axis will be 1x1 in the top right.
    Users can either use the default mosaic or provide their own.

    Additional logic is available to control plotting of multiple contributions of
    stresses on the same set of axes. Additionally some helper methods are defined to
    obtain relevant information from the models passed to CompareModels.

    Example usage::

        mc = ps.CompareModels([ml1, ml2])
        mc.plot()

        # obtain axes handles
        sim_ax = mc.axes["sim"]
        sim_ax.grid(False)

        # save figure
        mc.figure.savefig("modelcomparison.png")
    """

    def __init__(self, models: Optional[List[Model]] = None) -> None:
        """Initialize model compare class.

        Parameters
        ----------
        models : list of ps.Model, optional
            list of models to compare.
        """
        self.models = models
        # attributes that are set and used later
        self.figure = None
        self.axes = None
        self.mosaic = None
        self.cmap = None
        self.adjust_height = False
        self.smdict = None

    def initialize_figure(
        self,
        mosaic: Optional[List[List[str]]] = None,
        figsize: Tuple[int, int] = (10, 8),
        cmap: str = "tab10",
    ) -> None:
        """initialize a custom figure based on a mosaic.

        Parameters
        ----------
        mosaic : list, optional
            subplot mosaic, by default None which uses the default mosaic.
        figsize : tuple, optional
            figure size, by default (10, 8).
        cmap : str, optional
            colormap, by default "tab10".
        """
        if mosaic is None:
            mosaic = self.get_default_mosaic()

        self.mosaic = mosaic
        figure, axes = plt.subplot_mosaic(self.mosaic, figsize=figsize)
        self.figure = figure
        self.axes = axes
        self.cmap = plt.get_cmap(cmap)

    def initialize_adjust_height_figure(
        self,
        mosaic: Optional[List[List[str]]] = None,
        figsize: Tuple[int] = (10, 8),
        cmap: str = "tab10",
        smdict: Optional[dict] = None,
    ) -> None:
        """initialize subplots based on a mosaic with equal vertical scales.

        The height of each subplot is calculated based on the y-data limits in each
        subplot. This is calculation is performed on the first column of axes in the
        mosaic.

        Parameters
        ----------
        mosaic : list, optional
            subplot mosaic, by default None which uses the default mosaic.
        figsize : tuple, optional
            figure size, by default (10, 8)
        cmap : str, optional
            colormap, by default "tab10"
        smdict : dict, optional
            Dictionary with integers (index) as keys and list of stressmodel names as
            values that have to be in each subplot. For example, `{0: ['prec',
            'evap'], 1: ['rech']}` where stressmodels 'prec' and 'evap' are plotted
            in the first respons function window and 'rech' in the second. By
            default, None, which creates a separate subplot for each stressmodel.
        """
        self.adjust_height = True

        if mosaic is None:
            mosaic = self.get_default_mosaic()

        if smdict is None and self.smdict is None:
            self.smdict = {
                i: [smn] for i, smn in enumerate(self.get_unique_stressmodels())
            }
        elif smdict is not None and self.smdict is None:
            self.smdict = smdict

        # loop through models to get min and max
        sim_minmax = [1e30, -1e30]
        res_minmax = [1e30, -1e30]
        contrib_minmax = DataFrame(
            index=self.get_unique_stressmodels(), columns=["min", "max"]
        )
        for ml in self.models:
            # get sim min/max
            sim = ml.simulate()
            o = ml.observations()
            sim_minmax[0] = np.nanmin([np.nanmin([sim.min(), o.min()]), sim_minmax[0]])
            sim_minmax[1] = np.nanmax([np.nanmax([sim.max(), o.max()]), sim_minmax[1]])

            # get res min/max
            res = ml.residuals()
            res_minmax[0] = np.nanmin([res.min(), res_minmax[0]])
            res_minmax[1] = np.nanmax([res.max(), res_minmax[1]])

            # get contrib min/max
            smnames = ml.get_stressmodel_names()
            for smname in smnames:
                contribution = ml.get_contribution(smname)
                contrib_minmax.loc[smname, "min"] = np.nanmin(
                    [contrib_minmax.loc[smname, "min"], np.min(contribution)]
                )
                contrib_minmax.loc[smname, "max"] = np.nanmax(
                    [contrib_minmax.loc[smname, "max"], np.max(contribution)]
                )

        # get maximum dy for each subplot
        heights = {}
        # convert mosaic to dataframe and take first column
        dfmos = DataFrame(mosaic).iloc[:, 0]
        # get original ratio of each string in first column of mosaic
        mosfrac = (dfmos.value_counts() / len(dfmos)).to_dict()
        for ky in mosfrac:
            if ky == "sim":
                heights[ky] = sim_minmax[1] - sim_minmax[0]
            elif ky == "res":
                heights[ky] = res_minmax[1] - res_minmax[0]
            elif "con" in ky:  # if entry is contribution
                # loop through stressmodelnames provided for subplot
                smnames = self.smdict[int(ky[3:])]  # index is after 'con'
                # fmt: off
                heights[ky] = (
                        np.nanmax(contrib_minmax.loc[smnames, "max"]) -
                        np.nanmin(contrib_minmax.loc[smnames, "min"])
                )
                # fmt: on

        # sum of scaled dy
        hsum = np.sum(list(heights.values()))
        # total height ratio of scaled dy subplots
        hratio = 1.0 - np.sum([mosfrac[ky] for ky in mosfrac if ky not in heights])
        heights_list = []  # collect heights
        for ky in mosfrac:
            nrows = dfmos.value_counts().loc[ky]
            if ky in heights:
                # add entry if axes spans multiple tiles in mosaic
                heights_list += [heights[ky] / hsum * hratio / nrows] * nrows
            else:  # use the ratio of mosaic
                heights_list += [mosfrac[ky]]

        self.mosaic = mosaic
        fig, axes = plt.subplot_mosaic(
            self.mosaic,
            figsize=figsize,
            gridspec_kw=dict(height_ratios=heights_list),
        )

        self.figure = fig
        self.axes = axes
        self.cmap = plt.get_cmap(cmap)

        # set ylimits to data limits for scaling properly
        for axlbl in self.axes:
            if axlbl in ["sim", "res"] or axlbl.startswith("con"):
                self.axes[axlbl].autoscale(enable=None, axis="y", tight=True)

    def get_unique_stressmodels(self, models: List[Model] = None) -> List[str]:
        """Get all unique stressmodel names.

        Parameters
        ----------
        models : list of ps.Model, optional
            list of models, by default None
        """
        if models is None:
            models = self.models

        sm_unique = []
        for ml in models:
            sm_unique += [x for x in ml.get_stressmodel_names() if x not in sm_unique]

        return sm_unique

    def get_default_mosaic(
        self, n_stressmodels: Optional[int] = None
    ) -> List[List[str]]:
        """Get default mosaic for matplotlib.subplot_mosaic().

        Parameters
        ----------
        n_stressmodels : None, optional
            number of stressmodel plots to include in mosaic by default None which
            uses the number of unique stressmodels in all models.

        Returns
        -------
        mosaic : list
            list of lists containing axes labels.
        """
        if n_stressmodels is None:
            n_stressmodels = len(self.get_unique_stressmodels(models=self.models))

        mosaic = [
            ["sim", "sim", "met"],
            ["sim", "sim", "tab"],
            ["res", "res", "tab"],
        ]
        for i in range(n_stressmodels):
            mosaic.append([f"con{i}", f"con{i}", f"rf{i}"])

        return mosaic

    def get_tmin_tmax(self, models: List[Model] = None) -> DataFrame:
        """get tmin and tmax of all models.

        Parameters
        ----------
        models : list of ps.Model, optional
            list of models to get tmin/tmax for, by default None.
        """
        if models is None:
            models = self.models

        tmintmax = DataFrame(columns=["tmin", "tmax"], dtype="datetime64[ns]")
        for ml in models:
            tmintmax.loc[ml.name, ["tmin", "tmax"]] = [
                ml.get_tmin(),
                ml.get_tmax(),
            ]

        return tmintmax

    def get_metrics(
        self,
        models: Optional[List[Model]] = None,
        metric_selection: Optional[List[str]] = None,
    ) -> DataFrame:
        """get metrics of all models in a DataFrame.

        Parameters
        ----------
        models : list of ps.Model, optional
            list of models to calculate metrics for, by default None.
        metric_selection : list of str, optional
            names of metrics to calculate, by default None.

        Returns
        -------
        metrics : pd.DataFrame
            DataFrame containing calculated metrics.
        """
        if models is None:
            models = self.models

        metrics = concat(
            [ml.stats.summary(stats=metric_selection) for ml in models],
            axis=1,
            sort=False,
        )
        metrics.columns = [ml.name for ml in models]
        metrics.index.name = None

        return metrics

    def get_parameters(
        self,
        models: Optional[List[Model]] = None,
        param_col: str = "optimal",
        param_selection: Optional[List[str]] = None,
    ) -> DataFrame:
        """get parameter values of all models in a DataFrame.

        Parameters
        ----------
        models : list of ps.Model, optional
            list of models to get parameters for, by default None.
        param_col : str, optional
            name of parameter column to obtain, by default "optimal".
        param_selection : list of str, optional
            string to filter parameter selection, by default None.

        Returns
        -------
        params : pd.DataFrame
            parameter DataFrame containing parameters for each model.
        """
        if models is None:
            models = self.models

        params = concat([ml.parameters[param_col] for ml in models], axis=1, sort=False)
        params.columns = [x.name for x in models]

        if param_selection:
            sel = np.array([])
            for sub in param_selection:
                sel = np.append(sel, [idx for idx in params.index if sub in idx])
            return params.loc[sel].sort_index()
        else:
            return params

    def get_diagnostics(
        self, models: Optional[List[Model]] = None, diag_col: str = "P-value"
    ) -> DataFrame:
        """Get p-values of statistical tests in a DataFrame.

        Parameters
        ----------
        models : list of ps.Model, optional
            list of models to calculate diagnostics for.
        diag_col : str, optional
            name of diagnostics column to obtain, by default "P-value".
        """
        if models is None:
            models = self.models

        diags = DataFrame(index=[x.name for x in models])
        for ml in models:
            mldiag = ml.stats.diagnostics()
            diags.loc[f"{ml.name}", mldiag.index] = mldiag[diag_col].values

        return diags.transpose()

    def plot_oseries(self, axn: str = "sim") -> None:
        """Plot all oseries, unless all oseries are the same.

        Parameters
        ----------
        axn : str, optional
            name of labeled axes to plot oseries on, by default "sim".
        """
        if self.axes is None:
            self.initialize_figure(mosaic=[[axn]], figsize=(10, 3))

        oseries = [ml.oseries.series for ml in self.models]
        equals = np.array([])
        for pair in combinations(oseries, 2):
            equals = np.append(equals, np.array_equal(pair[0], pair[1]))
        if equals.all():
            self.axes[axn].plot(
                oseries[0].index,
                oseries[0].values,
                label=oseries[0].name,
                linestyle="",
                marker="o",
                color="k",
                markersize=3,
            )
        else:
            for i, oseries in enumerate(oseries):
                self.axes[axn].scatter(
                    oseries.index,
                    oseries.values,
                    label=oseries.name,
                    color=self.cmap(i),
                    s=15,
                    edgecolor="k",
                    linewidth=0.5,
                )

    def plot_simulation(self, axn: str = "sim") -> None:
        """plot model simulation.

        Parameters
        ----------
        axn : str, optional
            name of labeled axes to plot model simulations on, by default "sim".
        """
        if self.axes is None:
            self.initialize_figure(mosaic=[[axn]], figsize=(10, 3))

        for i, ml in enumerate(self.models):
            simulation = ml.simulate()
            self.axes[axn].plot(
                simulation.index,
                simulation.values,
                label=ml.name,
                linestyle="-",
                color=self.cmap(i),
            )

    def plot_residuals(self, axn: str = "res") -> None:
        """plot residuals.

        Parameters
        ----------
        axn : str, optional
            name of labeled axes to plot residuals on, by default "res".
        """
        if self.axes is None:
            self.initialize_figure(mosaic=[[axn]], figsize=(10, 3))

        for i, ml in enumerate(self.models):
            residuals = ml.residuals()
            self.axes[axn].plot(
                residuals.index,
                residuals.values,
                label="Residuals",
                color=self.cmap(i),
            )

    def plot_noise(self, axn: str = "res") -> None:
        """plot noise.

        Parameters
        ----------
        axn : str, optional
            name of labeled axes to plot noise on, by default "res".
        """
        if self.axes is None:
            self.initialize_figure(mosaic=[[axn]], figsize=(10, 3))

        for i, ml in enumerate(self.models):
            noise = ml.noise()
            if noise is not None:
                self.axes[axn].plot(
                    noise.index,
                    noise.values,
                    label="Noise",
                    linestyle="--",
                    color=f"C{i}",
                )

    def plot_response(
        self, smdict: Optional[dict] = None, axn: str = "rf{i}", response: str = "step"
    ) -> None:
        """plot step or block responses.

        Parameters
        ----------
        smdict : dict, optional
            Dictionary with integers (index) as keys and list of stressmodel names as
            values that have to be in each subplot. For example, `{0: ['prec',
            'evap'], 1: ['rech']}` where stressmodels 'prec' and 'evap' are plotted
            in the first respons function window and 'rech' in the second. By
            default, None, which creates a separate subplot for each stressmodel.
        axn : str, optional
            name of labeled axes to plot response functions on, by default "rf{i}".
            If smdict is not None, keys of that dictionary are used to fill in axes
            label, e.g. key 0 indicates the response functions will be plotted on
            axes with label 'rf0'. Otherwise, each response function will be plotted
            on a separate subplot (i.e. 'rf0', 'rf1', ...).
        response : str, optional
            type of response to plot, either "step" or "block", by default "step".
        """
        if self.axes is None:
            self.initialize_figure(mosaic=[[axn]], figsize=(5, 3))

        if smdict is None and self.smdict is None:
            self.smdict = {
                i: [smn] for i, smn in enumerate(self.get_unique_stressmodels())
            }
        elif smdict is not None and self.smdict is None:
            self.smdict = smdict

        for i, ml in enumerate(self.models):
            for j, namlist in self.smdict.items():
                for smn in namlist:
                    # skip if contribution not in model
                    if smn not in ml.stressmodels:
                        continue
                    if response == "step":
                        step = ml.get_step_response(smn, add_0=True)
                        self.axes[axn.format(i=j)].plot(
                            step.index,
                            step.values,
                            label=f"{smn}",
                            color=self.cmap(i),
                        )
                    elif response == "block":
                        block = ml.get_block_response(smn)
                        self.axes[axn.format(i=j)].semilogx(
                            block.index,
                            block.values,
                            label=f"{smn}",
                            color=self.cmap(i),
                        )

    def plot_contribution(
        self,
        smdict: Optional[dict] = None,
        axn: str = "con{i}",
        normalized: bool = False,
    ) -> None:
        """plot stressmodel contributions.

        Parameters
        ----------
        smdict : dict, optional
            Dictionary with integers (index) as keys and list of stressmodel names as
            values that have to be in each subplot. For example, `{0: ['prec',
            'evap'], 1: ['rech']}` where stressmodels 'prec' and 'evap' are plotted
            in the first contribution window and 'rech' in the second. By default,
            None, which creates a separate subplot for each stressmodel.
        axn : str, optional
            name of labeled axes to plot the contributions on, by default "con{i}". If
            smdict is not None, keys of that dictionary are used to fill in axes
            label, e.g. key 0 indicates the contributions will be plotted on axes
            with label 'con0'. Otherwise, each contribution will be plotted on a
            separate subplot (i.e. 'con0', 'con1', ...).
        normalized : bool, optional
            normalize contributions with min/max depending on mean value, by default
            False.
        """
        if self.axes is None:
            self.initialize_figure(mosaic=[[axn]], figsize=(10, 3))

        if smdict is None and self.smdict is None:
            if self.adjust_height:
                warn(
                    "When combining stressmodels into one subplot in combination "
                    "with 'adjust_height', provide 'smdict' to "
                    "`initialize_adjust_height_figure()` for best results."
                )
            self.smdict = {
                i: [smn] for i, smn in enumerate(self.get_unique_stressmodels())
            }
        elif smdict is not None and self.smdict is None:
            self.smdict = smdict

        for i, ml in enumerate(self.models):
            for j, namlist in self.smdict.items():
                for smn in namlist:
                    if smn not in ml.stressmodels:
                        continue
                    for con in ml.get_contributions(split=False):
                        if smn in con.name:
                            label = f"{con.name}"
                            if normalized:
                                label += " (normalized)"
                                if con.mean() < 0:
                                    con -= con.max()
                                else:
                                    con -= con.min()

                            self.axes[axn.format(i=j)].plot(
                                con.index,
                                con.values,
                                label=label,
                                color=self.cmap(i),
                            )

    def plot_stress(
        self, axn: str = "stress", names: Optional[List[str]] = None
    ) -> None:
        """plot stresses time series.

        Parameters
        ----------
        axn : str, optional
            name of labeled axes to plot stresses on, by default "stress".
        names : list of str, optional
            names of stresses to plot, by default None.
        """
        if self.axes is None:
            self.initialize_figure(mosaic=[[axn]], figsize=(10, 3))

        if names is None:
            names = self.get_unique_stressmodels()

        for i, ml in enumerate(self.models):
            for smn in names:
                if smn in ml.get_stressmodel_names():
                    stress = ml.get_stress(smn)
                    self.axes[axn].plot(
                        stress.index,
                        stress.values,
                        label=f"{smn}",
                        color=self.cmap(i),
                    )

    def plot_acf(self, axn: str = "acf") -> None:
        """plot autocorrelation plot.

        Parameters
        ----------
        axn : str, optional
            name of labeled axes to plot ACF on, by default "acf".
        """
        if self.axes is None:
            self.initialize_figure(mosaic=[[axn]], figsize=(10, 3))

        for i, ml in enumerate(self.models):
            if ml.noise() is not None:
                r = ps.stats.core.acf(ml.noise(), full_output=True)
                label = "Autocorrelation Noise"
            else:
                r = ps.stats.core.acf(ml.residuals(), full_output=True)
                label = "Autocorrelation Residuals"
            conf = r.conf.rolling(10, min_periods=1).mean().values

            self.axes[axn].fill_between(
                r.index.days, conf, -conf, alpha=0.3, color=self.cmap(i)
            )
            self.axes[axn].vlines(
                r.index.days,
                [0],
                r.loc[:, "acf"].values,
                color=self.cmap(i),
                label=label,
            )

    def plot_table(self, axn: str = "table", df: Optional[DataFrame] = None) -> None:
        """Plot dataframe as table.

        Parameters
        ----------
        axn : str, optional
            name of labeled axes to plot table on, by default "table".
        df : pandas.DataFrame, optional
            The Pandas.Dataframe to plot. Note that the first column is the index
            column that is shown.
        """
        if self.axes is None:
            self.initialize_figure(mosaic=[[axn]], figsize=(6, 4))

        if df is None:
            df = DataFrame(["empty"])

        self.axes[axn].table(
            df.values.tolist(),
            colLabels=df.columns,
            colColours=[(1.0, 1.0, 1.0, 1.0)]
            + [self.cmap(i, alpha=0.75) for i in range(len(df.columns) - 1)],
            bbox=(0.0, 0.0, 1.0, 1.0),
        )
        self.axes[axn].set_xticks([])
        self.axes[axn].set_yticks([])

    def plot_table_params(
        self,
        axn: str = "tab",
        param_col: str = "optimal",
        param_selection: Optional[List[str]] = None,
    ) -> None:
        """plot model parameters table.

        Parameters
        ----------
        axn : str, optional
            name of labeled axes to plot table on, by default "tab"
        param_col : str, optional
            name of parameter column to include, by default "optimal"
        param_selection : list of str, optional
            string to filter parameter names that are included in table, by default
            None.
        """
        if self.axes is None:
            self.initialize_figure(mosaic=[[axn]], figsize=(6, 4))

        params = self.get_parameters(
            self.models,
            param_selection=param_selection,
            param_col=param_col,
        ).applymap(ps.plots._table_formatter_params)

        # add seperate column with parameter names
        params.loc[:, "Parameters"] = params.index
        cols = params.columns.to_list()[-1:] + params.columns.to_list()[:-1]
        self.plot_table(axn=axn, df=params[cols])

    def plot_table_metrics(
        self, axn: str = "met", metric_selection: Optional[List[str]] = None
    ) -> None:
        """plot metrics table.

        Parameters
        ----------
        axn : str, optional
            name of labeled axes to plot table on, by default "met"
        metric_selection : list, optional
            list of str describing which metrics to include, by default None which
            uses ["rsq", "aic"].
        """
        if self.axes is None:
            self.initialize_figure(mosaic=[[axn]], figsize=(6, 4))

        if metric_selection is None:
            metric_selection = ["rsq", "aic"]

        metrics = self.get_metrics(self.models, metric_selection=metric_selection)
        for met in ["aic", "bic"]:
            if met in metrics.index:
                metrics.loc[met] -= metrics.loc[met].min()
                metrics = metrics.rename(
                    index={met: f"\N{GREEK CAPITAL LETTER DELTA}{met.upper()}"}
                )
        if "rsq" in metrics.index:
            metrics = metrics.rename(index={"rsq": f"R\N{SUPERSCRIPT TWO}"})

        # add seperate column with parameter names
        metrics.loc[:, "Metrics"] = metrics.index
        cols = metrics.columns.to_list()[-1:] + metrics.columns.to_list()[:-1]
        self.plot_table(axn=axn, df=metrics[cols].round(2))

    def plot_table_diagnostics(
        self, axn: str = "diag", diag_col: str = "P-value"
    ) -> None:
        """plot diagnostics table.

        Parameters
        ----------
        axn : str, optional
            name of labeled axis to plot table on, by default "diag".
        diag_col : str, optional
            name of diagnostics column to obtain, by default "P-value".
        """
        if self.axes is None:
            self.initialize_figure(mosaic=[[axn]], figsize=(6, 4))

        # add seperate column with parameter names
        diags = self.get_diagnostics(self.models, diag_col=diag_col)
        diags.loc[:, f"Test\n{diag_col}"] = diags.index
        cols = diags.columns.to_list()[-1:] + diags.columns.to_list()[:-1]
        self.plot_table(axn=axn, df=diags[cols])

    def share_xaxes(self, axes: List[Axes]) -> None:
        """share x-axes.

        Parameters
        ----------
        axes : list of matplotlib.Axes
            list of axes objects.
        """
        for i, iax in enumerate(axes):
            if i < (len(axes) - 1):
                iax.sharex(axes[-1])
                for t in iax.get_xticklabels():
                    t.set_visible(False)

    def share_yaxes(self, axes: List[Axes]) -> None:
        """share y-axes.

        Parameters
        ----------
        axes : list of matplotlib.Axes
            list of axes objects.
        """
        for iax in axes[1:]:
            iax.sharey(axes[0])

    def plot(
        self,
        smdict: Optional[dict] = None,
        normalized: bool = False,
        param_selection: Optional[list] = None,
        figsize: Optional[tuple] = (10, 8),
        grid: bool = True,
        legend: bool = True,
        adjust_height: bool = False,
        legend_kwargs: Optional[dict] = None,
    ) -> None:
        """plot the models in a comparison plot.

        The resulting plot is similar to `ml.plots.results()`.

        Parameters
        ----------
        smdict : dict, optional
            dictionary with integers (index) as keys and list of stressmodel names as
            values that have to be in each subplot. For example, `{0: ['prec',
            'evap'], 1: ['rech']}` where stressmodels 'prec' and 'evap' are plotted
            in the first contribution/response function window and 'rech' in the
            second. By default, None, which creates a separate subplot for each
            stressmodel.
        normalized : bool, optional
            normalize contributions such that minimum or maximum value is equal to
            zero, by default False.
        param_selection : list, optional
            list of (sub)strings of which parameters to show in table, by default None.
        figsize : tuple, optional
            figure size, by default (10, 8).
        grid : bool, optional
            grid in each subplots, by default True.
        legend : bool, optional
            add legend in each subplot, by default True.
        adjust_height : bool, optional
            adjust the height of the graphs, so that the vertical scale of all the
            subplots on the left is equal. Default is False. When combining stress
            contributions in one subplot, please also provide smdict for best results.
        legend_kwargs : dict, optional
            pass legend keyword arguments to plots.
        """
        self.adjust_height = adjust_height
        if self.axes is None and not self.adjust_height:
            self.initialize_figure(figsize=figsize)
        if self.axes is None and self.adjust_height:
            self.initialize_adjust_height_figure(smdict=smdict, figsize=figsize)

        # sim
        self.plot_oseries()
        self.plot_simulation()

        # res
        self.plot_residuals()
        self.plot_noise()

        # smn, rfn
        self.plot_contribution(smdict=smdict, normalized=normalized)
        self.plot_response(smdict=smdict)

        # share x-axes
        xshare_left = []
        xshare_right = []
        for axn in self.axes.keys():
            if axn not in ("tab", "met", "dia"):
                self.axes[axn].grid(grid)
                if legend and not axn.startswith("rf"):
                    if legend_kwargs is None:
                        legend_kwargs = {}
                    _, l = self.axes[axn].get_legend_handles_labels()
                    self.axes[axn].legend(
                        ncol=legend_kwargs.pop("ncol", max([int(np.ceil(len(l))), 4])),
                        loc=legend_kwargs.pop("loc", (0, 1)),
                        frameon=legend_kwargs.pop("frameon", False),
                        markerscale=legend_kwargs.pop("markerscale", 1.0),
                        numpoints=legend_kwargs.pop("numpoints", 3),
                        **legend_kwargs,
                    )
            if axn in ("sim", "res") or axn.startswith("con"):
                xshare_left.append(self.axes[axn])
            if axn.startswith("rf"):
                xshare_right.append(self.axes[axn])

        if len(xshare_left) > 1:
            self.share_xaxes(xshare_left)
        if len(xshare_right) > 1:
            self.share_xaxes(xshare_right)

        # met
        self.plot_table_metrics()

        # tab
        self.plot_table_params(param_selection=param_selection)

        self.figure.tight_layout(pad=0.0)
