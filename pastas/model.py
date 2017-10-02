from __future__ import print_function, division

import json
import logging
import logging.config
import os
from collections import OrderedDict
from warnings import warn

import numpy as np
import pandas as pd
from scipy import interpolate

from .decorators import get_tseries
from .io.base import dump
from .plots import Plotting
from .solver import LmfitSolve
from .stats import Statistics
from .timeseries import TimeSeries
from .tseries import Constant
from .utils import get_dt, get_time_offset
from .version import __version__


class Model:
    """Initiates a time series model.

    Parameters
    ----------
    oseries: pandas.Series
        pandas Series object containing the dependent time series. The
        observation can be non-equidistant.
    name: str, optional
        String with the name of the model, used in plotting and saving.
    metadata: dict, optional
        Dictionary containing metadata of the model.
    warmup: float, optional
        Number of days used for warmup.
    fillnan: str or float, optional
        Methods or float number to fill nan-values. Default values is
        'drop'. Currently supported options are: 'interpolate', float,
        'mean' and, 'drop'. Interpolation is performed with a standard
        linear interpolation.
    constant: bool, optional
        Add a constant to the model (Default=True).

    Examples
    --------

    >>> oseries = pd.Series([1,2,1], index=pd.to_datetime(range(3), unit="D"))
    >>> ml = Model(oseries)

    """

    def __init__(self, oseries, constant=True, name="Observations",
                 metadata=None, settings=None, log_level=""):
        self.logger = self.get_logger()
        # Construct the different model components
        self.oseries = TimeSeries(oseries, name=name, kind="oseries")
        self.odelt = self.oseries.index.to_series().diff() / \
                     pd.Timedelta(1, "D")
        self.oseries_calib = None
        self.name = name

        self.parameters = pd.DataFrame(
            columns=['initial', 'name', 'optimal', 'pmin', 'pmax', 'vary'])
        self.tseriesdict = OrderedDict()

        self.noisemodel = None

        if constant:
            self.add_constant()
        else:
            self.constant = None

        # Store the simulation settings
        self.settings = dict()
        self.settings["tmin"] = None
        self.settings["tmax"] = None
        self.settings["freq"] = "D"
        self.settings["warmup"] = 3650
        self.settings["time_offset"] = pd.Timedelta(0)
        self.settings["noise"] = False
        self.settings["solver"] = None
        if settings:
            self.settings.update(settings)

        # Metadata & File Information
        self.metadata = self.get_metadata(metadata)
        self.file_info = self.get_file_info()

        # initialize some attributes for solving and simulation
        self.sim_index = None
        self.interpolate_simulation = None
        self.fit = None
        self.report = "Model has not been solved yet. "

        # Load other modules
        self.stats = Statistics(self)
        self.plots = Plotting(self)
        self.plot = self.plots.plot  # because we are lazy

    def add_tseries(self, tseries):
        """Adds a time series component to the model.

        Parameters
        ----------
        tseries: pastas.tseries.TseriesBase
            pastas.tseries object.

        Notes
        -----
        To obtain a list of the tseries names type:

        >>> ml.tseriesdict.keys()

        """
        if tseries.name in self.tseriesdict.keys():
            self.logger.warning('The name for the series you are trying to '
                                'add already exists for this model. Select '
                                'another name.')
        else:
            self.tseriesdict[tseries.name] = tseries
            self.parameters = self.get_init_parameters()

            # Call these methods to set tmin, tmax and freq
            tseries.update_stress(freq=self.settings["freq"])
            self.set_time_offset()
            self.settings["tmin"], self.settings["tmax"] = self.get_tmin_tmax()

    def add_noisemodel(self, noisemodel):
        """Adds a noise model to the time series Model.

        """
        self.noisemodel = noisemodel
        self.parameters = self.get_init_parameters()

    def add_constant(self):
        """Adds a Constant to the time series Model.

        """
        self.constant = Constant(value=self.oseries.mean(), name='constant')
        self.parameters = self.get_init_parameters()

    @get_tseries
    def del_tseries(self, name):
        """ Save deletion of a tseries from the tseriesdict.

        Parameters
        ----------
        name: str
            string with the name of the tseries object.

        Notes
        -----
        To obtain a list of the tseries names type:

        >>> ml.tseriesdict.keys()

        """
        self.parameters = self.parameters.ix[self.parameters.name != name]
        self.tseriesdict.pop(name)

    def del_constant(self):
        """ Save deletion of the constant from a Model.

        """
        if self.constant is None:
            self.logger.warning("No constant is present in this model.")
        else:
            self.parameters = self.parameters.ix[self.parameters.name !=
                                                 'constant']
            self.constant = None

    def del_noisemodel(self):
        """Save deletion of the noisemodel from the Model.

        """
        if self.noisemodel is None:
            self.logger.warning("No noisemodel is present in this model.")
        else:
            self.parameters = self.parameters.ix[self.parameters.name !=
                                                 self.noisemodel.name]
            self.noisemodel = None

    def simulate(self, parameters=None, tmin=None, tmax=None, freq=None):
        """Simulate the time series model.

        Parameters
        ----------
        parameters: list, optional
            Array of the parameters used in the time series model.
        tmin: str, optional
        tmax: str, optional
        freq: str, optional
            frequency at which the time series are simulated.

        Returns
        -------
        h: pandas.Series
            pandas.Series object containing the simulated time series

        Notes
        -----
        This method can be used without any parameters. When the model is
        solved, the optimal parameters values are used and if not,
        the initial parameter values are used. This allows the user to
        obtain an idea of how the simulation looks with only the initial
        parameters and no calibration.

        """

        # Default option when freq is not provided.
        if freq is None:
            freq = self.settings["freq"]

        if self.sim_index is None:
            tmin, tmax = self.get_tmin_tmax(tmin, tmax, freq,
                                            use_oseries=False)
            sim_index = self.get_sim_index(tmin, tmax, freq,
                                           self.settings["warmup"])
        else:
            sim_index = self.sim_index

        dt = get_dt(freq)

        # Get parameters if none are provided
        if parameters is None:
            parameters = self.get_parameters()

        h = pd.Series(data=0, index=sim_index)
        istart = 0  # Track parameters index to pass to ts object
        for ts in self.tseriesdict.values():
            c = ts.simulate(parameters[istart: istart + ts.nparam], sim_index,
                            dt)
            h = h.add(c, fill_value=0.0)
            istart += ts.nparam
        if self.constant:
            h += self.constant.simulate(parameters[istart])

        h.name = "Simulation"

        return h

    def residuals(self, parameters=None, tmin=None, tmax=None, freq=None):
        """Calculate the residual series.

        Parameters
        ----------
        parameters: list, optional
            Array of the parameters used in the time series model.
        tmin: str, optional
        tmax: str, optional
        freq: str, optional
            frequency at which the time series are simulated.

        Returns
        -------
        res: pandas.Series
            pandas.Series with the residuals series.

        """
        if freq is None:
            freq = self.settings["freq"]

        tmin, tmax = self.get_tmin_tmax(tmin, tmax, freq, use_oseries=True)

        # simulate model
        simulation = self.simulate(parameters, tmin, tmax, freq)

        if self.oseries_calib is None:
            oseries_calib = self.get_oseries_calib(tmin, tmax,
                                                   simulation.index)
        else:
            oseries_calib = self.oseries_calib

        obs_index = oseries_calib.index  # times used for calibration

        # Get h_simulated at the correct indices
        interpolate_simulation = self.interpolate_simulation
        if interpolate_simulation is None:
            interpolate_simulation = obs_index.difference(
                simulation.index).size != 0
        if interpolate_simulation:
            # interpolate simulation to measurement-times
            h_simulated = np.interp(oseries_calib.index.asi8,
                                    simulation.index.asi8, simulation)
        else:
            # all of the observation indexes are in the simulation
            h_simulated = simulation[obs_index]
        res = oseries_calib - h_simulated

        if np.isnan(sum(res ** 2)):
            self.logger.warning(
                'nan problem in residuals')  # quick and dirty check
        return res

    def get_oseries_calib(self, tmin, tmax, sim_index):
        """Method to get the oseries to use for calibration.

        This method is for performance improvements only.

        """
        oseries_calib = self.oseries.loc[tmin: tmax]
        # sample measurements, so that frequency is not higher than model
        # keep the original timestamps, as they will be used during
        # interpolation of the simulation
        oseries_calib = self.sample(oseries_calib, sim_index)
        return oseries_calib

    def innovations(self, parameters=None, tmin=None, tmax=None, freq=None):
        """Method to simulate the innovations when a noisemodel is present.

        Parameters
        ----------
        parameters: list, optional
            Array of the parameters used in the time series model.
        tmin: str, optional
        tmax: str, optional
        freq: str, optional
            frequency at which the time series are simulated.

        Returns
        -------
        v : pandas.Series
            Pandas series of the innovations.

        Notes
        -----
        The innovations are the time series that result when applying a noise
        model.

        """
        if self.noisemodel is None:
            warn("Innovations can not be calculated as there is no noisemodel")
            return None

        tmin, tmax = self.get_tmin_tmax(tmin, tmax, freq, use_oseries=True)

        # Get parameters if none are provided
        if parameters is None:
            parameters = self.get_parameters()

        # Calculate the residuals
        res = self.residuals(parameters, tmin, tmax, freq)

        # Calculate the innovations
        v = self.noisemodel.simulate(res, self.odelt[res.index],
                                     parameters[-self.noisemodel.nparam:],
                                     res.index)
        return v

    def observations(self, tmin=None, tmax=None):
        """Method that returns the observations series.

        """
        tmin, tmax = self.get_tmin_tmax(tmin, tmax, use_oseries=True)

        return self.oseries.loc[tmin: tmax]

    def initialize(self, tmin=None, tmax=None, freq=None, warmup=None,
                   noise=True, weights=None, initial=True):
        """Initialize the model. This method is called by "solve" but can
        also be triggered manually.

        Parameters
        ----------
        tmin
        tmax
        freq
        warmup
        noise
        initial

        Returns
        -------

        """

        if noise and (self.noisemodel is None):
            self.logger.warning('Warning, solution with noise model while '
                                'noise model is not defined. No noise model is'
                                'used.')
            noise = False
        self.settings["noise"] = noise
        self.settings["weights"] = weights

        # Set tmin and tmax
        self.settings["tmin"], self.settings["tmax"] = self.get_tmin_tmax(tmin,
                                                                          tmax)

        # Set the frequency & warmup
        if freq:
            self.settings["freq"] = freq
        if warmup:
            self.settings["warmup"] = warmup

        # make sure calibration data is renewed
        self.sim_index = self.get_sim_index(self.settings["tmin"],
                                            self.settings["tmax"],
                                            self.settings["freq"],
                                            self.settings["warmup"])
        self.oseries_calib = self.get_oseries_calib(self.settings["tmin"],
                                                    self.settings["tmax"],
                                                    self.sim_index)

        # Prepare tseries stresses
        for ts in self.tseriesdict.values():
            ts.update_stress(freq=self.settings["freq"],
                             tmin=self.sim_index.min(),
                             tmax=self.sim_index.max())

        self.interpolate_simulation = self.oseries_calib.index.difference(
            self.sim_index).size != 0
        if self.interpolate_simulation:
            self.logger.info('There are observations between the simulation'
                             'timesteps. Linear interpolation is used')

        # Initialize parameters
        self.parameters = self.get_init_parameters(noise, initial)

    def get_sim_index(self, tmin, tmax, freq, warmup):
        """Method to get the indices for the simulation, including the warmup
        period.

        Parameters
        ----------
        tmin
        tmax
        freq
        warmup

        Returns
        -------

        """
        sim_index = pd.date_range(tmin - pd.DateOffset(days=warmup), tmax,
                                  freq=freq)
        return sim_index

    def solve(self, tmin=None, tmax=None, solver=LmfitSolve, report=True,
              noise=True, initial=True, weights=None, freq=None, warmup=None,
              **kwargs):
        """Method to solve the time series model.

        Parameters
        ----------
        tmin: str, optional
            String with a start date for the simulation period (E.g. '1980')
        tmax: str, optional
            String with an end date for the simulation period (E.g. '2010')
        solver: pastas.solver.BaseSolver, optional
            Class used to solve the model. Default is lmfit (LmfitSolve)
        report: bool, optional
            Print a report to the screen after optimization finished.
        noise: bool, optional
            Use the noise model (True) or not (False).
        initial: bool, optional
            Reset initial parameters.

        """

        # Initialize the model
        self.initialize(tmin, tmax, freq, warmup, noise, weights, initial)

        # Solve model
        self.settings["solver"] = solver._name
        fit = solver(self, tmin=self.settings["tmin"],
                     tmax=self.settings["tmax"], noise=self.settings["noise"],
                     freq=self.settings["freq"],
                     weights=self.settings["weights"], **kwargs)

        # make calibration data empty again (was set in initialize)
        self.sim_index = None
        self.oseries_calib = None
        self.interpolate_simulation = None

        self.fit = fit.fit
        self.parameters.optimal = fit.optimal_params

        self.report = fit.report
        if report:
            print(self.report)

    def get_tmin_tmax(self, tmin=None, tmax=None, freq=None, use_oseries=True):
        """Method that checks and returns valid values for tmin and tmax.

        Parameters
        ----------
        tmin: str, optional
            string with a year or date that can be turned into a pandas
            Timestamp (e.g. pd.Timestamp(tmin)).
        tmax: str, optional
            string with a year or date that can be turned into a pandas
            Timestamp (e.g. pd.Timestamp(tmax)).
        freq: str, optional
            string with the frequency.
        use_oseries: bool, optional
            boolean to check the tmin and tmax against the oseries.

        Returns
        -------
        tmin, tmax: pandas.Timestamp
            returns a pandas timestamp for tmin and tmax.

        Notes
        -----
        The tmin and tmax are checked and returned according to the
        following rules:

        A. If no value for tmin/tmax is provided:
            1. if use_oseries is false, tmin is set to minimum of the tseries.
            2. if use_series is true tmin is set to minimum of the oseries.

        B. If a values for tmin/tmax is provided:
            1. A pandas timestamp is made from the string
            2. if use_oseries is True, tmin is checked against oseries.
            3. tmin is checked against the tseries.

        C. In all cases an offset for the tmin and tmax is added.

        A detailed description of dealing with tmin and tmax and timesteps
        in general can be found in the developers section of the docs.

        """
        # Get tmin and tmax from the tseries
        if self.tseriesdict:
            ts_tmin = pd.Timestamp.max
            ts_tmax = pd.Timestamp.min
            for tseries in self.tseriesdict.values():
                if tseries.tmin < ts_tmin:
                    ts_tmin = tseries.tmin
                if tseries.tmax > ts_tmax:
                    ts_tmax = tseries.tmax
        else:
            # When there are no tseries use the oseries, regardless of
            # use_oseries:
            ts_tmin = self.oseries.index.min()
            ts_tmax = self.oseries.index.max()

        # Set tmin properly
        if tmin is None and use_oseries is None:
            tmin = ts_tmin
        elif tmin is None:
            tmin = max(ts_tmin, self.oseries.index.min())
        else:
            tmin = pd.Timestamp(tmin)
            # Check if tmin > oseries.tmin (Needs to be True)
            if tmin < self.oseries.index.min() and use_oseries:
                warn("Specified tmin is before the first observation. tmin"
                     " automatically set to %s" % self.oseries.index.min())
                tmin = self.oseries.index.min()
            # Check if tmin > tseries.tmin (Needs to be True)
            if tmin < ts_tmin:
                warn("Specified tmin is before any of the tseries tmin. tmin"
                     " automatically set to tseries tmin %s" % ts_tmin)
                tmin = ts_tmin

        # Set tmax properly
        if tmax is None and use_oseries is None:
            tmax = ts_tmax
        elif tmax is None:
            tmax = min(ts_tmax, self.oseries.index.max())
        else:
            tmax = pd.Timestamp(tmax)
            # Check if tmax < oseries.tmax (Needs to be True)
            if tmax > self.oseries.index.max() and use_oseries:
                warn("Specified tmax is after the last observation. tmax"
                     " automatically set to %s" % self.oseries.index.max())
                tmax = self.oseries.index.max()
            # Check if tmax < tseries.tmax (Needs to be True)
            if tmax > ts_tmax:
                warn("Specified tmax is after any of the tseries tmax. tmax"
                     " automatically set to tseries tmax %s" % ts_tmax)
                tmax = ts_tmax

        # adjust tmin and tmax so that the time-offset is equal to the tseries.
        if freq is None:
            freq = self.settings["freq"]
        tmin = tmin - get_time_offset(tmin, freq) + self.settings[
            "time_offset"]
        tmax = tmax - get_time_offset(tmax, freq) + self.settings[
            "time_offset"]

        assert tmax > tmin, \
            self.logger.error('Error: Specified tmax not larger than '
                              'specified tmin')
        assert self.oseries.loc[tmin: tmax].size > 0, \
            self.logger.error('Error: no observations between tmin and tmax')

        return tmin, tmax

    def set_time_offset(self):
        """Set the time offset for the model class.

        Notes
        -----
        Method to check if the Tseries timestamps match (e.g. similar hours)

        """
        time_offsets = set()
        for tseries in self.tseriesdict.values():
            if tseries.stress:
                # calculate the offset from the default frequency
                time_offset = get_time_offset(
                    tseries.stress[0].index.min(),
                    self.settings["freq"])
                time_offsets.add(time_offset)

        assert len(
            time_offsets) <= 1, self.logger.error('The time-differences with '
                                                  'the default frequency is '
                                                  'not the same for all '
                                                  'stresses.')
        if len(time_offsets) == 1:
            self.settings["time_offset"] = next(iter(time_offsets))
        else:
            self.settings["time_offset"] = pd.Timedelta(0)

    def get_init_parameters(self, noise=True, initial=True):
        """Method to get all initial parameters from the individual objects.

        Parameters
        ----------
        noise: bool, optional
            Add the parameters for the noisemodel to the parameters
            Dataframe or not.
        initial: Boolean
            True to get initial parameters, False to get optimized parameters.

        Returns
        -------
        parameters: pandas.DataFrame
            pandas.Dataframe with the parameters.

        """

        # Store optimized values in case they are needed
        if not initial:
            optimal = self.parameters.optimal

        parameters = pd.DataFrame(columns=['initial', 'pmin', 'pmax',
                                           'vary', 'optimal', 'name'])
        for ts in self.tseriesdict.values():
            parameters = parameters.append(ts.parameters)
        if self.constant:
            parameters = parameters.append(self.constant.parameters)
        if self.noisemodel and noise:
            parameters = parameters.append(self.noisemodel.parameters)

        # Set initial parameters to optimal parameters
        if not initial:
            parameters.initial = optimal

        return parameters

    def get_parameters(self, name=None):
        """Helper method to obtain the parameters needed for calculation if
        none are provided. This method is used by the simulation, residuals
        and the innovations methods.

        Parameters
        ----------
        name: str, optional
            string with the name of the pastas.tseries object.

        Returns
        -------
        p: list, optional
            Array of the parameters used in the time series model.

        """
        if name:
            p = self.parameters[self.parameters.name == name]
        else:
            p = self.parameters

        if p.optimal.hasnans:
            self.logger.warning(
                "Model is not optimized yet, initial parameters are used.")
            parameters = p.initial
        else:
            parameters = p.optimal

        return parameters.values

    @get_tseries
    def get_contribution(self, name, tindex=None):
        p = self.get_parameters(name)
        dt = get_dt(self.settings["freq"])
        return self.tseriesdict[name].simulate(p, tindex=tindex, dt=dt)

    @get_tseries
    def get_block_response(self, name):
        p = self.get_parameters(name)
        dt = get_dt(self.settings["freq"])
        b = self.tseriesdict[name].rfunc.block(p, dt)
        t = np.arange(dt, (len(b) + 1) * dt, dt)
        return pd.Series(b, index=t, name=name)

    @get_tseries
    def get_step_response(self, name):
        p = self.get_parameters(name)
        dt = get_dt(self.settings["freq"])
        s = self.tseriesdict[name].rfunc.step(p, dt)
        t = np.arange(dt, (len(s) + 1) * dt, dt)
        return pd.Series(s, index=t, name=name)

    @get_tseries
    def get_stress(self, name):
        p = self.get_parameters(name)
        self.tseriesdict[name].get_stress(p)

    def sample(self, series, tindex):
        """Sample the series so that the frequency is not higher that tindex.

        Parameters
        ----------
        series: pandas.Series
            pandas series object.
        tindex: pandas.index
            Pandas index object

        Returns
        -------
        series: pandas.Series


        Notes
        -----
        Find the index closest to the tindex, and then return a selection
        of series.

        """
        f = interpolate.interp1d(series.index.asi8,
                                 np.arange(0, series.index.size),
                                 kind='nearest', bounds_error=False,
                                 fill_value='extrapolate')
        ind = np.unique(f(tindex.asi8).astype(int))
        return series[ind]

    def get_metadata(self, meta=None):
        """Method that returns a metadata dict with the basic information.

        Parameters
        ----------
        meta: dict, optional
            dictionary containing user defined metadata

        Returns
        -------
        metadata: dict
            dictionary containing the basic information.

        """
        metadata = dict()
        metadata["y"] = None
        metadata["x"] = None
        metadata["z"] = None
        metadata["projection"] = None
        metadata["name"] = None

        if meta:  # Update metadata with user-provided metadata if possible
            metadata.update(meta)

        return metadata

    def get_file_info(self):
        """Method to get the file information, mainly used for saving files.

        Returns
        -------
        file_info: dict
            dictionary with file information.

        """
        file_info = dict()
        file_info["date_created"] = pd.Timestamp.now()
        file_info["date_modified"] = pd.Timestamp.now()
        file_info["pastas_version"] = __version__
        try:
            file_info["owner"] = os.getlogin()
        except:
            file_info["owner"] = "Unknown"

        return file_info

    def get_logger(self, default_path='log_config.json',
                   default_level=logging.WARNING,
                   env_key='LOG_CFG'):
        """This file creates a logger instance to log program output.

        Returns
        -------
        logger: logging.Logger
            Logging instance that handles all logging throughout pastas,
            including all sub modules and packages.


        """
        dir_path = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(dir_path, default_path)
        value = os.getenv(env_key, None)
        if value:
            path = value
        if os.path.exists(path):
            with open(path, 'rt') as f:
                config = json.load(f)
            logging.config.dictConfig(config)
        else:
            logging.basicConfig(level=default_level)

        logger = logging.getLogger(__name__)

        return logger

    def dump_data(self, series=True, sim_series=False, metadata=True,
                  file_info=True):
        """Method to export a PASTAS model to the json export format. Helper
         function for the self.export method.

         Parameters
         ----------
         series: Boolean
            True if the original series are to be stored.
         sim_series: Boolean
            True if the simulated series are to be stored.
         metadata: Boolean
            True if the model metadata is to be stored.

         The following attributes are stored:

         - oseries
         - tseriesdict
         - noisemodel
         - constant
         - parameters
         - metadata
         - settings
         - ..... future attributes?

         Notes
         -----
         To increase backward compatibility most attributes are stored in
         dictionaries that can be updated when a model is created.

        """

        # Create a dictionary to store all data
        data = dict()
        data["name"] = self.name
        data["oseries"] = self.oseries.dump(series=series)

        # Tseriesdict
        data["tseriesdict"] = dict()
        for name, ts in self.tseriesdict.items():
            data["tseriesdict"][name] = ts.dump(series=series)

        # Constant
        if self.constant:
            data["constant"] = True
        if self.noisemodel:
            data["noisemodel"] = self.noisemodel.dump()

        # Parameters
        data["parameters"] = self.parameters

        # Metadata
        if metadata:
            data["metadata"] = self.metadata

        # Simulation Settings
        data["settings"] = self.settings

        # Update and save file information
        if file_info:
            self.file_info["date_modified"] = pd.Timestamp.now()
            data["file_info"] = self.file_info

        # Export simulated series if necessary
        if sim_series:
            # TODO dump the simulation, residuals and innovation series.
            NotImplementedError()

        return data

    def dump(self, fname, series=True):

        # Get dicts for all data sources
        data = self.dump_data(series)

        # Write the dicts to a file
        return dump(fname, data)
