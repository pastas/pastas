"""This files contains the Project class that deals with multiple models at
once.

Notes
-----
This module is created at Artesia Water by Raoul Collenteur.

Usage
-----

>>> mls = Project()

"""

import os
from warnings import warn

import numpy as np
import pandas as pd
import pastas as ps


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
        self.tseries = pd.DataFrame(index=[],
                                    columns=["name", "series", "kind", "x",
                                             "y", "z", "metadata"])
        self.oseries = pd.DataFrame(index=[],
                                    columns=["name", "series", "kind", "x",
                                             "y", "z", "metadata"])

        self.distances = pd.DataFrame(index=self.oseries.index,
                                      columns=self.tseries.index)

        # Project metadata and file information
        self.metadata = self.get_metadata(metadata)
        self.file_info = self.get_file_info()

    def add_oseries(self, series, name, kind="oseries", metadata=None,
                    settings=None):
        """Method to add a oseries to the database.

        """
        ts = ps.TimeSeries(series=series, name=name, kind=kind,
                           settings=settings)
        self.oseries.set_value(name, "name", name)
        self.oseries.set_value(name, "series", ts)
        self.oseries.set_value(name, "metadata", metadata)
        self.oseries.set_value(name, "kind", kind)

        # Transfer x, y and z to dataframe as well to increase speed.
        for i in ["x", "y", "z"]:
            if i in metadata.keys():
                value = metadata[i]
            else:
                value = 0.0
            self.oseries.set_value(name, i, value)

    def add_tseries(self, series, name, kind=None, metadata=None,
                    settings=None):
        """Method to add a tseries to the database.

        """
        ts = ps.TimeSeries(series=series, name=name, kind=kind,
                           settings=settings)
        self.tseries.set_value(name, "name", name)
        self.tseries.set_value(name, "series", ts)
        self.tseries.set_value(name, "metadata", metadata)
        self.tseries.set_value(name, "kind", kind)

        # Transfer x, y and z to dataframe as well to increase speed.
        for i in ["x", "y", "z"]:
            if i in metadata.keys():
                value = metadata[i]
            else:
                value = 0.0
            self.tseries.set_value(name, i, value)

    def del_tseries(self, tseries):
        """Method that removes tseries from the project.

        Parameters
        ----------
        tseries: list or str
            list with multiple or string with a single tseries name.

        Returns
        -------

        """
        self.tseries.drop(tseries, inplace=True)
        self.update_distances()

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

    def add_model(self, name, ml_name=None, **kwargs):
        if ml_name is None:
            ml_name = name

        # Validate name and ml_name before continuing
        if ml_name in self.models.keys():
            return warn("Model name is not unique, provide a new name.")
        if name not in self.oseries.index:
            return warn(
                "Oseries name is not present in the database. Make sure to provide a valid oseries name.")

        oseries = self.oseries.loc[name, "series"]
        metadata = self.oseries.loc[name, "metadata"]
        ml = ps.Model(oseries, name=ml_name, metadata=metadata, **kwargs)

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
        xt = pd.to_numeric(self.tseries.x)
        yo = pd.to_numeric(self.oseries.y)
        yt = pd.to_numeric(self.tseries.y)

        xh, xi = np.meshgrid(xt, xo)
        yh, yi = np.meshgrid(yt, yo)

        self.distances = pd.DataFrame(np.sqrt((xh - xi) ** 2 + (yh - yi) ** 2),
                                      index=self.oseries.index,
                                      columns=self.tseries.index)

    def add_recharge(self, ml, **kwargs):
        """Adds a recharge element to the time series model. The
        selection of the precipitation and evaporation time series is based
        on the shortest distance to the a tseries in the tserieslist.

        Returns
        -------

        """
        key = ml.name
        prec_name = self.distances.loc[key, self.tseries.type ==
                                       "prec"].argmin()
        prec = self.tseries.loc[prec_name, "series"]
        evap_name = self.distances.loc[key, self.tseries.type ==
                                       "evap"].argmin()
        evap = self.tseries.loc[evap_name, "series"]

        prec.index = prec.index.round("D")
        evap.index = evap.index.round("D")

        recharge = ps.Tseries2(prec, evap, ps.Gamma, name="recharge", **kwargs)

        ml.add_tseries(recharge)

    def get_metadata(self, meta):
        metadata = dict(
            projection=None
        )

        return metadata

    def get_file_info(self):
        file_info = dict()
        file_info["date_created"] = pd.Timestamp.now()
        file_info["date_modified"] = pd.Timestamp.now()
        file_info["pastas_version"] = ps.__version__
        try:
            file_info["owner"] = os.getlogin()
        except:
            file_info["owner"] = "Unknown"
        return file_info

    def dump_data(self, series=True, sim_series=False, metadata=True):
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
            oseries=self.oseries,
            tseries=self.tseries,
            models=dict(),
            metadata=self.metadata,
            file_info=self.file_info
        )
        # Models
        for name, ml in self.models.items():
            data["models"][name] = ml.dump_data(series=series,
                                                sim_series=sim_series,
                                                metadata=metadata,
                                                file_info=False)

        return data

    def dump(self, fname=None):
        """Method to write a Pastas project to a file.

        Parameters
        ----------
        fname

        Returns
        -------

        """
        data = self.dump_data()
        return ps.io.base.dump(fname, data)
