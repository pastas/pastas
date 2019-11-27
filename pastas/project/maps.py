"""This module contains the mapping methods for Pastas Projects.

Raoul Collenteur, 2018 - Artesia Water

"""

import matplotlib.pyplot as plt


class Map:
    def __init__(self, mls):
        """

        Parameters
        ----------
        mls: pastas.Project
            Pastas project

        """
        self.mls = mls

    def parameter(self, parameter, models=None, param_value="optimal", s=30,
                  show_nan=True, label=False, **kwargs):
        """Plot the value of a parameter.

        Parameters
        ----------
        parameter: str
            String with the name of the parameter to plot.
        models: list, optional
            List of the models top plot the parameter for. By default,
            all models are plotted.
        param_value: str, optional
             String with the parameter kind to be plotted. Any column name
             of the Model parameter DataFrame (e.g. optimal, stderr)
        s: int, optional
            Size of the marker.
        show_nan: bool, optional
            Show nan-values (default is True), which occur when the
            parameter is not in the model.
        label: bool, optional
            Label the points by the model name, default is False.
        kwargs: dict, optional
            Any arguments that are passed on the the "values" method".

        Returns
        -------
        sc: matplotlib.axes
            The axes are returned.

        """
        values = self.mls.get_parameters(parameters=[parameter], models=models,
                                         param_value=param_value)

        sc = self.values(values, models, show_nan, label, s, **kwargs)

        return sc

    def statistic(self, statistic, models=None, s=30, show_nan=True,
                  label=False, **kwargs):
        """Plot the value of a parameter.

        Parameters
        ----------
        statistic: str
            String with the name of the statistic to plot. Must be exactly
            similar to the methods name in the pastas.stats module.
        models: list, optional
            List of the models top plot the parameter for. By default,
            all models are plotted.
        s: int, optional
            Size of the marker.
        show_nan: bool, optional
            Show nan-values (default is True), which occur when the
            parameter is not in the model.
        label: bool, optional
            Label the points by the model name, default is False.
        kwargs: dict, optional
            Any arguments that are passed on the the "values" method".

        Returns
        -------
        sc: matplotlib.axes
            The axes are returned.

        """
        values = self.mls.get_statistics(statistics=[statistic], models=models)

        sc = self.values(values, models, show_nan, label, s, **kwargs)

        return sc

    def values(self, values, models=None, show_nan=True, label=False, s=30,
               **kwargs):
        """Plot the value of a parameter.

        Parameters
        ----------
        values: pandas.Series
            Series with the values to plot, the index should be the model
            names as are used in mls.models.keys()
        models: list, optional
            List of the models top plot the parameter for. By default,
            all models are plotted.
        param_value: str, optional
             String with the parameter kind to be plotted. Any column name
             of the Model parameter DataFrame (e.g. optimal, stderr)
        s: int, optional
            Size of the marker.
        show_nan: bool, optional
            Show nan-values (default is True), which occur when the
            parameter is not in the model.
        label: bool, optional
            Label the points by the model name, default is False.
        kwargs: dict, optional
            Any arguments that are passed on the the "values" method".

        Returns
        -------
        sc: matplotlib.axes
            The axes are returned.

        """
        if models is None:
            models = values.index
            models = self.mls.oseries.loc[models, "z"].sort_values(
                ascending=False).index
        else:
            values = values.loc[models]

        x = self.mls.oseries.loc[models, "x"].astype(float)
        y = self.mls.oseries.loc[models, "y"].astype(float)
        s = self._normalize(self.mls.oseries.loc[models, "z"].astype(float), s)

        if show_nan:
            nan = values[values.isnull()].fillna(-999)
            plt.scatter(x[nan.index], y[nan.index], c="none", edgecolors="k",
                        s=s)

        sc = plt.scatter(x, y, c=values, s=s, edgecolors="k", **kwargs)

        if label:
            not_nan = values[~values.isnull()].index
            labels = values[not_nan].astype(str)
            for name, xy in zip(labels, zip(x[not_nan], y[not_nan])):
                plt.annotate(s=name, xy=xy,
                             bbox=dict(facecolor='w', edgecolor='k'),
                             textcoords="offset points", xytext=(10, 10))

        return sc

    def series(self, kind="stresses", label=False, **kwargs):
        """Plot the location of the oseries or the stresses on a map.

        Parameters
        ----------
        kind: str
            kind of series to plot. Possible values are the oseries,
            stresses or a specific type of stress (e.g. prec, evap or well).
        label: bool, optional
            Display a label next to the point with the name of the series.
        kwargs: dict, optional
            Any keyword arguments are passed on to plt.scatter.

        Returns
        -------
        sc: matplotlib.axes
            Return the axes.

        """
        if kind == "oseries":
            series = self.mls.oseries
        elif kind == "stresses":
            series = self.mls.stresses
        else:
            series = self.mls.stresses.loc[self.mls.stresses.kind == kind]

        x = series.loc[:, "x"].astype(float)
        y = series.loc[:, "y"].astype(float)

        sc = plt.scatter(x, y, **kwargs)

        if label:
            for name, xy in zip(x.index, zip(x, y)):
                plt.annotate(s=name, xy=xy, fontsize=10,
                             bbox=dict(facecolor='w', edgecolor='k'),
                             textcoords="offset points", xytext=(10, 10))

        return sc

    @staticmethod
    def _normalize(series, s=30):
        """Internal method to normalize the series for the size op the scatterplot.

        """
        mu = series.mean()
        if mu == 0.0:  # Prevent sizes of zero to be calculates
            mu = 1
        series = (series.subtract(series.min()) / mu + 1) * s
        return series
