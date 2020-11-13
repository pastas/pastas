"""This module contains the Project class (DEPRECATED).

Warning
-------
This class will soon be deprecated and replaced by a separate Python package
Pastastore that deals with large number of Pastas models. It is strongly
recommended to switch to `Pastastore <https://github.com/pastas/pastastore>`_

Notes
-----
This module is created at Artesia Water by Raoul Collenteur.

Example
-------

>>> mls = Project()

"""

from logging import getLogger
from os import getlogin

import numpy as np
import pandas as pd

from .maps import Map
from .plots import Plot
from ..io.base import dump
from ..model import Model
from ..rfunc import Gamma
from ..stressmodels import StressModel2
from ..timeseries import TimeSeries
from ..version import __version__

logger = getLogger(__name__)


class Project:
    """The Project class is a placeholder when multiple time series models
    are analyzed in a batch.

    """

    def __init__(self, name, metadata=None):
        """Initialize a Project instance.

        Parameters
        ----------
        name: str
            Name of the project
        metadata: dict
            Dictionary with any metadata information on the project.

        """
        logger.warning("DeprecationWarning: The Project class will be "
                       "deprecated in a future version of Pastas (end of "
                       "2020)! Consider switching to pastastore: "
                       "https://github.com/pastas/pastastore.")
        self.models = {}
        self.name = name

        # DataFrames to store the data of the oseries and stresses
        columns = ["name", "series", "kind", "x", "y", "z"]
        self.stresses = pd.DataFrame(columns=columns)
        self.oseries = pd.DataFrame(columns=columns)

        # Project metadata and file information
        self.metadata = self.get_metadata(metadata)
        self.file_info = self.get_file_info()

        # Load other modules
        self.plots = Plot(self)
        self.maps = Map(self)

    def add_series(self, series, name=None, kind=None, metadata=None,
                   settings=None, **kwargs):
        """Internal method to add series to the oseries or stresses database.

        Parameters
        ----------
        series: pandas.Series / pastas.TimeSeries
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
        settings: dict or str
            Dictionary with any settings that will be passed to the
            TimeSeries object that is created internally.

        """
        if name is None:
            name = series.name

        if not isinstance(name, str):
            name = str(name)

        series.name = name

        if kind == "oseries":
            data = self.oseries
            if settings is None:
                settings = "oseries"
        else:
            data = self.stresses

        if name in data.index:
            warning = ("Time series with name {} is already present in the "
                       "database. Existing series is overwitten.").format(name)
            logger.warning(warning)
        try:
            ts = TimeSeries(series=series, name=name, settings=settings,
                            metadata=metadata, **kwargs)
        except:
            logger.error("An error occurred. Time series {} is omitted "
                         "from the database.".format(name))
            return

        data.at[name, "name"] = name
        data.at[name, "series"] = ts  # Do not add as first!
        data.at[name, "kind"] = kind

        # Transfer the metadata (x, y and z) to dataframe as well to increase speed.
        for i in ts.metadata.keys():
            value = ts.metadata[i]
            data.loc[name, i] = value

    def add_oseries(self, series, name=None, metadata=None, settings="oseries",
                    **kwargs):
        """Convenience method to add oseries to project

        Parameters
        ----------
        series: pandas.Series / pastas.TimeSeries
            Series object.
        name: str
            String with the name of the series that will be maintained in
            the database.
        metadata: dict
            Dictionary with any metadata that will be passed to the
            TimeSeries object that is created internally.
        settings: dict or str
            Dictionary with any settings that will be passed to the
            TimeSeries object that is created internally.

        Returns
        -------

        """
        self.add_series(series, name=name, metadata=metadata,
                        settings=settings, kind="oseries", **kwargs)

    def add_stress(self, series, name=None, kind=None, metadata=None,
                   settings=None, **kwargs):
        """Convenience method to add stress series to project

        Parameters
        ----------
        series: pandas.Series / pastas.TimeSeries
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
        settings: dict or str
            Dictionary with any settings that will be passed to the
            TimeSeries object that is created internally.

        """
        self.add_series(series, name=name, metadata=metadata,
                        settings=settings, kind=kind, **kwargs)

    def add_model(self, oseries, model_name=None, **kwargs):
        """Method to add a Pastas Model instance based on one of the oseries.

        Parameters
        ----------
        oseries: str
            string with the exact names of one of the oseries indices.
        model_name: str
            Name of the model
        kwargs: dict
            any arguments that are taken by the Pastas Model instance can be
            provided.

        Returns
        -------
        ml: pastas.Model
            Pastas Model generated with the oseries and arguments provided.

        """
        if model_name is None:
            model_name = oseries

        # Validate name and ml_name before continuing
        if model_name in self.models.keys():
            warning = ("Model name {} is not unique, existing model is "
                       "overwritten.").format(model_name)
            logger.warning(warning)
        if oseries not in self.oseries.index:
            error = ("Oseries name {} is not present in the database. Make "
                     "sure to provide a valid name.").format(model_name)
            logger.error(error)
            return

        oseries = self.oseries.loc[oseries, "series"]
        ml = Model(oseries, name=model_name, **kwargs)

        # Add new model to the models dictionary
        self.models[model_name] = ml

        return ml

    def add_models(self, oseries="all", model_name_prefix="",
                   model_name_suffix="", **kwargs):
        """Method to add multiple Pastas Model instances based on one
        or more of the oseries.

        Parameters
        ----------
        oseries: str or list, optional
            names of the oseries, if oseries is "all" all series in self.series
            are used
        model_name_prefix: str, optional
            prefix to use for model names
        model_name_suffix: str, optional
            suffix to use for model names
        kwargs: dict
            any arguments that are taken by the Pastas Model instance can be
            provided.

        Returns
        -------
        mls: list of str
            list of modelnames corresponding to the keys in the self.models
            dictionary

        """

        if oseries == "all":
            oseries_list = self.oseries.index
        elif isinstance(oseries, str):
            oseries_list = [oseries]
        elif isinstance(oseries, list):
            oseries_list = oseries

        mls = []
        for oseries_name in oseries_list:
            model_name = model_name_prefix + oseries_name + model_name_suffix

            # Add new model
            ml = self.add_model(oseries_name, model_name, **kwargs)
            mls.append(ml.name)

        return mls

    def add_recharge(self, mls=None, rfunc=Gamma, name="recharge", **kwargs):
        """Add a StressModel2 to the time series models. The
        selection of the precipitation and evaporation time series is based
        on the shortest distance to the a stresses in the stresses list.

        Parameters
        ----------
        mls: list of str, optional
            list of model names, if None all models in project are
            used.
        rfunc: pastas.rfunc, optional
            response function, default is the Gamma function.
        name: str, optional
            name of the stress, default is "recharge".
        **kwargs:
            arguments are pass to the StressModel2 function

        Notes
        -----
        The behaviour of this method will change in the future to add a
        RechargeModel instead of StressModel2.


        """
        if mls is None:
            mls = self.models.keys()
        elif isinstance(mls, Model):
            mls = [mls.name]

        for mlname in mls:
            ml = self.models[mlname]
            oseries = ml.oseries.name
            prec_name = \
                self.get_nearest_stresses(oseries, kind="prec").iloc[0][0]
            prec = self.stresses.loc[prec_name, "series"]
            evap_name = \
                self.get_nearest_stresses(oseries, kind="evap").iloc[0][0]
            evap = self.stresses.loc[evap_name, "series"]

            recharge = StressModel2([prec, evap], rfunc, name=name, **kwargs)

            ml.add_stressmodel(recharge)

    def del_oseries(self, name):
        """Method that safely removes oseries from the project. It validates
        that the oseries is not used in any model.

        Parameters
        ----------
        name: str
            string with a single oseries name.

        """
        if name not in self.oseries.index:
            error = ("Time series with name {} is not present in the database."
                     " Please provide a different name.").format(name)
            logger.error(error)
        else:
            self.oseries.drop(name, inplace=True)
            logger.info(
                "Oseries with name {} deleted from the database.".format(name))

    def del_stress(self, name):
        """Method that removes oseries from the project.

        Parameters
        ----------
        name: list or str
            list with multiple or string with a single stress name.

        """
        if name not in self.stresses.index:
            error = ("Stress with name {} is not present in the database."
                     " Please provide a different name.").format(name)
            logger.error(error)
        else:
            self.stresses.drop(name, inplace=True)
            logger.info(
                "Stress with name {} deleted from the database.".format(name))

    def del_model(self, ml_name):
        """Method to safe-delete a model from the project.

        Parameters
        ----------
        ml_name: str
            String with the model name.

        """
        name = self.models.pop(ml_name, None)
        info = "Model with name {} deleted from the database.".format(name)
        logger.info(info)

    def update_model_series(self):
        """Update all the Model series by their originals in self.oseries and
        self.stresses. This can for example be useful when new data is
        added to any of the series in mls.oseries and mls.stresses

        """
        for ml in self.models.values():
            oname = ml.oseries.name
            ml.oseries.series_original = self.oseries.loc[
                oname, "series"].series_original
            for sm in ml.stressmodels:
                for st in ml.stressmodels[sm].stress:
                    st.series_original = self.stresses.loc[
                        st.name, "series"].series_original
            # set oseries_calib empty, so it is determined again the next time
            ml.oseries_calib = None

    def solve_models(self, mls=None, report=False, ignore_solve_errors=False,
                     verbose=False, **kwargs):
        """Solves the models in mls

        Parameters
        ----------
        mls: list of str, optional
            list of model names, if None all models in the project are solved.
        report: boolean, optional
            determines if a report is printed when the model is solved.
        ignore_solve_errors: boolean, optional
            if True errors emerging from the solve method are ignored.
        **kwargs:
            arguments are passed to the solve method.

        """
        if mls is None:
            mls = self.models.keys()
        elif isinstance(mls, Model):
            mls = [mls.name]

        for ml_name in mls:
            if verbose:
                print("solving model -> {}".format(ml_name))

            ml = self.models[ml_name]

            m_kwargs = {}
            for key, value in kwargs.items():
                if isinstance(value, pd.Series):
                    m_kwargs[key] = value.loc[ml_name]
                else:
                    m_kwargs[key] = value
            # Convert timestamps
            for tstamp in ["tmin", "tmax"]:
                if tstamp in m_kwargs:
                    m_kwargs[tstamp] = pd.Timestamp(m_kwargs[tstamp])

            try:
                ml.solve(report=report, **m_kwargs)
            except Exception as e:
                if ignore_solve_errors:
                    warning = "solve error ignored for -> {}".format(ml.name)
                    logger.warning(warning)
                else:
                    raise e

    def get_nearest_stresses(self, oseries=None, stresses=None, kind=None,
                             n=1):
        """Method to obtain the nearest (n) stresses of a specific kind.

        Parameters
        ----------
        oseries: str
            String with the name of the oseries
        stresses: str or list of str
            String with the name of the stresses
        kind:
            String with the name of the stresses
        n: int
            Number of stresses to obtain

        Returns
        -------
        stresses:
            List with the names of the stresses.

        """

        distances = self.get_distances(oseries, stresses, kind)

        data = pd.DataFrame(columns=np.arange(n))

        for series in distances.index:
            series = pd.Series(distances.loc[series].sort_values().index[:n],
                               name=series)
            data = data.append(series)

        return data

    def get_distances(self, oseries=None, stresses=None, kind=None):
        """Method to obtain the distances in meters between the stresses and
        oseries.

        Parameters
        ----------
        oseries: str or list of str
        stresses: str or list of str
        kind: str

        Returns
        -------
        distances: pandas.DataFrame
            Pandas DataFrame with the distances between the oseries (index)
            and the stresses (columns).

        """
        if isinstance(oseries, str):
            oseries = [oseries]
        elif oseries is None:
            oseries = self.oseries.index

        if stresses is None and kind is None:
            stresses = self.stresses.index
        elif stresses is None:
            stresses = self.stresses[self.stresses.kind == kind].index
        elif stresses is not None and kind is not None:
            mask = self.stresses.kind == kind
            stresses = self.stresses.loc[stresses].loc[mask].index

        xo = pd.to_numeric(self.oseries.loc[oseries, "x"])
        xt = pd.to_numeric(self.stresses.loc[stresses, "x"])
        yo = pd.to_numeric(self.oseries.loc[oseries, "y"])
        yt = pd.to_numeric(self.stresses.loc[stresses, "y"])

        xh, xi = np.meshgrid(xt, xo)
        yh, yi = np.meshgrid(yt, yo)

        distances = pd.DataFrame(np.sqrt((xh - xi) ** 2 + (yh - yi) ** 2),
                                 index=oseries, columns=stresses)

        return distances

    def get_parameters(self, parameters, models=None, param_value="optimal"):
        """Method to get the parameters from each model. NaN-values are
        returned when the parameters is not present in the model.

        Parameters
        ----------
        parameters: list
            List with the names of the parameters. The parameter does not
            have to be used in all models.
        models: list
            List with the names of the models. These have to be in the
            Project models dictionary.
        param_value: str
            String with the parameter value that needs to be collected:
            Options are: initial, optimal (default), pmax, pmin and vary.

        Returns
        -------
        data: pandas.DataFrame or pandas.Series
            Returns a pandas DataFrame with the models name as the index and
            the parameters as columns. A pandas Series is returned when only
            one parameter values is collected.

        """
        if models is None:
            models = self.models.keys()

        data = pd.DataFrame(index=models, columns=parameters)

        for ml_name in models:
            ml = self.models[ml_name]
            for parameter in parameters:
                if parameter in ml.parameters.index:
                    value = ml.parameters.loc[parameter, param_value]
                    data.loc[ml_name, parameter] = value

        data = data.squeeze()
        return data.astype(float)

    def get_statistics(self, statistics, models=None, **kwargs):
        """Method to get the statistics for each model.

        Parameters
        ----------
        statistics: list
            List with the names of the statistics to calculate for each model.
        models: list
            List with the names of the models. These have to be in the
            Project models dictionary.

        Returns
        -------
        data: pandas.DataFrame or pandas.Series

        """
        if models is None:
            models = self.models.keys()

        data = pd.DataFrame(index=models, columns=statistics)

        for ml_name in models:
            ml = self.models[ml_name]
            for statistic in statistics:
                value = ml.stats.__getattribute__(statistic)(**kwargs)
                data.loc[ml_name, statistic] = value

        data = data.squeeze()
        return data.astype(float)

    def get_locations_geodataframe(self, models=None, **kwargs):
        import geopandas as gpd
        from shapely.geometry import Point

        if models is None:
            models = self.models.keys()

        data = pd.DataFrame(index=models)

        data = data.join(self.oseries.loc[models, ["x", "y", "z"]])
        data["geometry"] = [Point(xy[0], xy[1]) for xy in
                            self.oseries.loc[models, ["x", "y"]].values]
        data = gpd.GeoDataFrame(data, geometry="geometry", **kwargs)
        return data

    def get_oseries_metadata(self, oseries, metadata):
        """Method to get the metadata for all oseries.

        Parameters
        ----------
        oseries: list
            list with the oseries.
        metadata: list
            list with the metadata keywords to obtain.

        Returns
        -------
        data: pandas.DataFrame

        """
        data = pd.DataFrame(data=None, index=oseries, columns=metadata)

        for oseries in data.index:
            meta = self.oseries.loc[oseries, "series"].metadata
            for key in metadata:
                data.loc[oseries, key] = meta[key]

        return data

    def get_oseries_settings(self, oseries, settings):
        """Method to obtain the settings from each oseries TimeSeries object.

        Parameters
        ----------
        oseries
        settings

        Returns
        -------
        data: pandas.DataFrame
            Pandas DataFrame with the oseries as index, settings as columns
            and the values as data.

        """
        data = pd.DataFrame(data=None, index=oseries, columns=settings)

        for oseries in data.index:
            sets = self.oseries.loc[oseries, "series"].settings
            for key in settings:
                data.loc[oseries, key] = sets[key]

        return data

    @staticmethod
    def get_metadata(meta=None):
        metadata = {"projection": None}
        if meta:
            metadata.update(meta)

        return metadata

    @staticmethod
    def get_file_info():
        file_info = {
            "date_created": pd.Timestamp.now(),
            "date_modified": pd.Timestamp.now(),
            "pastas_version": __version__,
        }
        try:
            file_info["owner"] = getlogin()
        except:
            file_info["owner"] = "Unknown"
        return file_info

    def to_file(self, fname, **kwargs):
        """Method to write a Pastas project to a file.

        Parameters
        ----------
        fname: str

        """
        data = self.to_dict(**kwargs)
        return dump(fname, data)

    def to_dict(self, series=False):
        """Internal method to export a Pastas Project as a dictionary.

        Parameters
        ----------
        series: bool, optional
            export model input-series when True. Only export the name of
            the model input_series when False

        Returns
        -------
        data: dict
            A dictionary with all the project data

        """
        data = {
            "name": self.name,
            "metadata": self.metadata,
            "file_info": self.file_info,
            "oseries": self.series_to_dict(self.oseries),
            "stresses": self.series_to_dict(self.stresses),
            "models": {}
        }

        # Add Models
        for name, ml in self.models.items():
            data["models"][name] = ml.to_dict(series=series, file_info=False)

        return data

    @staticmethod
    def series_to_dict(series):
        """Internal method used to export the time series."""
        series = series.to_dict(orient="index")

        for name in series.keys():
            ts = series[name]["series"]
            series[name]["series"] = ts.to_dict(series=True)

        return series
