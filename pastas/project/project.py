"""This files contains the Project class that deals with multiple models at
once.

Notes
-----
This module is created at Artesia Water by Raoul Collenteur.

Usage
-----

>>> mls = Project()

"""

import logging
import os

import numpy as np
import pandas as pd
import pastas as ps

logger = logging.getLogger(__name__)


class Project:
    """The Project class is a placeholder when multiple time series models are
    analyzed in a batch.

    """

    def __init__(self, name, metadata=None):
        self.models = {}
        self.name = name
        # Store the data in Pandas dataframes
        self.data = pd.DataFrame()

        # DataFrames to store the data of the oseries and stresses
        self.stresses = pd.DataFrame(index=[],
                                     columns=["name", "series", "kind", "x",
                                             "y", "z", "metadata"])
        self.oseries = pd.DataFrame(index=[],
                                    columns=["name", "series", "kind", "x",
                                             "y", "z", "metadata"])

        self.distances = pd.DataFrame(index=self.oseries.index,
                                      columns=self.stresses.index)

        # Project metadata and file information
        self.metadata = self.get_metadata(metadata)
        self.file_info = self._get_file_info()

    def add_series(self, series, name, kind, metadata=None, settings=None):
        """Method to add series to the oseries or tseries database.

        Parameters
        ----------
        series: pandas.Series or pastas.TimeSeries
            Series object.
        name: str
            String with the name of the series that will be maintained in
            the database.
        kind: str
            The kind of series that is added. When oseries are added it is
            necessary to state "oseries" here.
        metadata: dict
            Dictionary with any metadata that will be passed to the
            TimeSeries object that is created internally.
        settings: dict
            Dictionary with any settings that will be passed to the
            TimeSeries object that is created internally.

        Returns
        -------

        """
        try:
            ts = ps.TimeSeries(series=series, name=name, kind=kind,
                               settings=settings, metadata=metadata)
        except:
            logger.warning("Time series %s is ommitted from the database."
                           % name)
            return

        if kind == "oseries":
            data = self.oseries
        else:
            data = self.stresses

        data.set_value(name, "name", name)
        data.set_value(name, "series", ts)
        data.set_value(name, "kind", kind)

        # Transfer x, y and z to dataframe as well to increase speed.
        for i in ["x", "y", "z"]:
            value = ts.metadata[i]
            data.set_value(name, i, value)

    def del_oseries(self, oseries):
        """Method that removes oseries from the project.

        Parameters
        ----------
        oseries: list or str
            list with multiple or string with a single oseries name.

        Returns
        -------

        """
        self.oseries.drop(oseries, inplace=True)
        self.update_distances()

    def del_tseries(self, tseries):
        """Method that removes oseries from the project.

        Parameters
        ----------
        tseries: list or str
            list with multiple or string with a single oseries name.

        Returns
        -------

        """
        self.stresses.drop(tseries, inplace=True)
        self.update_distances()

    def add_model(self, oseries, ml_name=None, **kwargs):
        """Method to add a Pastas Model instance based on one of the oseries.

        Parameters
        ----------
        oseries: str
            string with the exact names of one of the oseries indices.
        ml_name: str
            Name of the model
        kwargs: dict
            any arguments that are taken by the Pastas Model instance can be
            provided.

        Returns
        -------
        ml: pastas.Model
            Pastas Model generated with the oseries and arguments provided.

        """
        if ml_name is None:
            ml_name = oseries

        # Validate name and ml_name before continuing
        if ml_name in self.models.keys():
            logger.warning("Model name is not unique, provide a new ml_name.")
        if oseries not in self.oseries.index:
            logger.warning("Oseries name is not present in the database. "
                           "Make sure to provide a valid oseries name.")

        oseries = self.oseries.loc[oseries, "series"]
        ml = ps.Model(oseries, name=ml_name, **kwargs)

        # Add new model to the models dictionary
        self.models[ml_name] = ml

        return ml

    def del_model(self, ml_name):
        """Method to safe-delete a model from the project.

        Parameters
        ----------
        model_name: str
            String with the model name.

        """
        self.models.pop(ml_name)

    def set_stats(self, stats=None):
        if not stats:
            stats = ["evp", "rmse", "rmsi", "durbin_watson"]
        data = pd.DataFrame(columns=stats)
        for name, ml in self.models.items():
            p = ml.stats.many(stats=stats).loc[0].values
            data.loc[name] = p

        self.data = pd.concat([self.data, data], axis=1)

    def set_parameters(self, parameters=None):
        data = pd.DataFrame(columns=parameters)

        for name, ml in self.models.items():
            p = ml.parameters.loc[parameters].optimal
            data.loc[name] = p

        self.data = pd.concat([self.data, data], axis=1)

    def update_distances(self):
        """
        Calculate the distances between the observed series and the stresses.

        Returns
        -------
        distances: pandas.DataFrame
            pandas dataframe with the distances between the oseries (index)
            and the stresses (columns).

        """
        # Make sure these are values, even when actually objects.
        xo = pd.to_numeric(self.oseries.x)
        xt = pd.to_numeric(self.stresses.x)
        yo = pd.to_numeric(self.oseries.y)
        yt = pd.to_numeric(self.stresses.y)

        xh, xi = np.meshgrid(xt, xo)
        yh, yi = np.meshgrid(yt, yo)

        self.distances = pd.DataFrame(np.sqrt((xh - xi) ** 2 + (yh - yi) ** 2),
                                      index=self.oseries.index,
                                      columns=self.stresses.index)

    def add_recharge(self, ml, **kwargs):
        """Adds a recharge element to the time series model. The
        selection of the precipitation and evaporation time series is based
        on the shortest distance to the a tseries in the tserieslist.

        Returns
        -------

        """
        key = ml.name
        prec_name = self.distances.loc[key, self.stresses.kind ==
                                       "prec"].argmin()
        prec = self.stresses.loc[prec_name, "series"]
        evap_name = self.distances.loc[key, self.stresses.kind ==
                                       "evap"].argmin()
        evap = self.stresses.loc[evap_name, "series"]

        recharge = ps.StressModel2([prec, evap], ps.Gamma, name="recharge",
                                   **kwargs)

        ml.add_stressmodel(recharge)

    def get_metadata(self, meta):
        metadata = dict(
            projection=None
        )

        return metadata

    def _get_file_info(self):
        file_info = dict()
        file_info["date_created"] = pd.Timestamp.now()
        file_info["date_modified"] = pd.Timestamp.now()
        file_info["pastas_version"] = ps.__version__
        try:
            file_info["owner"] = os.getlogin()
        except:
            file_info["owner"] = "Unknown"
        return file_info

    def dump(self, fname=None, **kwargs):
        """Method to write a Pastas project to a file.

        Parameters
        ----------
        fname

        Returns
        -------

        """
        data = self.dump_data(**kwargs)
        return ps.io.base.dump(fname, data)

    def dump_data(self, series=False, metadata=True, sim_series=False):
        """Method to export a Pastas Project and return a dictionary with
        the data to be exported.

        Parameters
        ----------
        fname: string
            string with the name and optionally the file-extension.

        Returns
        -------
        message: str
            Returns a message if export was successful or not.

        """
        data = dict(
            name=self.name,
            models=dict(),
            metadata=self.metadata,
            file_info=self.file_info
        )

        # Series DataFrame
        data["oseries"] = self._series_to_dict(self.oseries)
        data["tseries"] = self._series_to_dict(self.stresses)

        # Models
        data["models"] = dict()
        for name, ml in self.models.items():
            data["models"][name] = ml.dump_data(series=series,
                                                metadata=metadata,
                                                sim_series=sim_series,
                                                file_info=False)

        return data

    def _series_to_dict(self, series):
        series = series.to_dict(orient="index")

        for name in series.keys():
            ts = series[name]["series"]
            series[name]["series"] = ts.dump(series=True)

        return series

    def get_nearest_tseries(self, oseries, kind, n=1):
        """Method to obtain the nearest (n) tseries of a specific kind.

        Parameters
        ----------
        oseries: str
            String with the name of the oseries
        kind:
            String with the name of the tseries
        n: int
            Number of tseries to obtain

        Returns
        -------
        tseries:
            List with the names of the tseries.

        """
        if isinstance(oseries, str):
            oseries = [oseries]

        tseries = self.stresses[self.stresses.kind == kind].index

        distances = self.get_distances(oseries, tseries)

        sorted = pd.DataFrame(columns=np.arange(n))

        for series in oseries:
            series = pd.Series(distances.loc[series].sort_values().index[:n],
                               name=series)
            sorted = sorted.append(series)

        return sorted

    def get_distances(self, oseries=None, tseries=None):
        """Method to obtain the distances in meters between the tseries and
        oseries.

        Parameters
        ----------
        oseries: str or list of str
        tseries: str or list of str

        Returns
        -------
        distances: pandas.DataFrame
            Pandas DataFrame with the distances between the oseries (index)
            and the tseries (columns).

        """
        if oseries is None:
            oseries = self.oseries.index
        if tseries is None:
            tseries = self.stresses.index

        xo = pd.to_numeric(self.oseries.loc[oseries, "x"])
        xt = pd.to_numeric(self.stresses.loc[tseries, "x"])
        yo = pd.to_numeric(self.oseries.loc[oseries, "y"])
        yt = pd.to_numeric(self.stresses.loc[tseries, "y"])

        xh, xi = np.meshgrid(xt, xo)
        yh, yi = np.meshgrid(yt, yo)

        distances = pd.DataFrame(np.sqrt((xh - xi) ** 2 + (yh - yi) ** 2),
                                 index=oseries, columns=tseries)

        return distances
