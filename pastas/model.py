from __future__ import print_function, division

import json
import logging
import logging.config
import os
from collections import OrderedDict

import numpy as np
import pandas as pd
from scipy import interpolate

from .decorators import get_stressmodel
from .io.base import dump
from .noisemodels import NoiseModel
from .plots import Plotting
from .solver import LmfitSolve
from .stats import Statistics
from .stressmodels import Constant
from .timeseries import TimeSeries
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
    constant: bool, optional
        Add a constant to the model (Default=True).

    Examples
    --------

    >>> oseries = pd.Series([1,2,1], index=pd.to_datetime(range(3), unit="D"))
    >>> ml = Model(oseries)

    """

    def __init__(self, oseries, constant=True, noisemodel=True,
                 name="Observations", metadata=None, settings=None,
                 log_level="ERROR"):
        self.logger = self.get_logger(log_level=log_level)

        # Construct the different model components
        self.oseries = TimeSeries(oseries, kind="oseries")
        self.odelt = self.oseries.index.to_series().diff() / \
                     pd.Timedelta(1, "D")
        self.name = name

        self.parameters = pd.DataFrame(
            columns=["initial", "name", "optimal", "pmin", "pmax", "vary",
                     "stderr"])
        self.stressmodels = OrderedDict()
        self.constant = None
        self.transform = None
        self.noisemodel = None

        if constant:
            constant = Constant(value=self.oseries.mean(), name="constant")
            self.add_constant(constant)
        if noisemodel:
            self.add_noisemodel(NoiseModel())

        # Store the simulation settings
        self.settings = {}
        self.settings["tmin"] = None
        self.settings["tmax"] = None
        self.settings["freq"] = "D"
        self.settings["warmup"] = 3650
        self.settings["time_offset"] = pd.Timedelta(0)
        self.settings["noise"] = noisemodel
        self.settings["solver"] = None
        if settings:
            self.settings.update(settings)

        # Metadata & File Information
        self.metadata = self.get_metadata(metadata)
        self.file_info = self.get_file_info()

        # initialize some attributes for solving and simulation
        self.sim_index = None
        self.oseries_calib = None
        self.interpolate_simulation = None
        self.fit = None
        self.report = "Model has not been solved yet. "

        # Load other modules
        self.stats = Statistics(self)
        self.plots = Plotting(self)
        self.plot = self.plots.plot  # because we are lazy

    def add_stressmodel(self, stressmodel, replace=False):
        """Adds a stressmodel to the main model.

        Parameters
        ----------
        stressmodel: pastas.stressmodel.stressmodelBase
            instance of a pastas.stressmodel object.
        replace: bool
            replace the stressmodel if a stressmodel with the same name
            already exists. Not recommended but useful at times. Default is
            False.

        Notes
        -----
        To obtain a list of the stressmodel names type:

        >>> ml.stressmodels.keys()

        """
        if (stressmodel.name in self.stressmodels.keys()) and not replace:
            self.logger.error("""The name for the series you are trying to
                                add already exists for this model. Select
                                another name.""")
        else:
            self.stressmodels[stressmodel.name] = stressmodel
            self.parameters = self.get_init_parameters()
            if self.settings["freq"] is None:
                self.set_freq()
            stressmodel.update_stress(freq=self.settings["freq"])
            # Call these methods to set time offset
            self.set_time_offset()

    def add_constant(self, constant):
        """Adds a Constant to the time series Model.

        Parameters
        ----------
        constant: pastas.Constant
            Pastas constant instance, possibly more things in the future.

        """
        self.constant = constant
        self.parameters = self.get_init_parameters()

    def add_transform(self, transform):
        self.transform = transform
        self.parameters = self.get_init_parameters()

    def add_noisemodel(self, noisemodel):
        """Adds a noise model to the time series Model.

        Parameters
        ----------
        noisemodel: pastas.noisemodels.NoiseModelBase
            Instance of NoiseModelBase

        """
        self.noisemodel = noisemodel
        self.parameters = self.get_init_parameters()

    @get_stressmodel
    def del_stressmodel(self, name):
        """ Save deletion of a stressmodel from the stressmodels dict.

        Parameters
        ----------
        name: str
            string with the name of the stressmodel object.

        Notes
        -----
        To obtain a list of the stressmodel names type:

        >>> ml.stressmodels.keys()

        """
        self.parameters = self.parameters.ix[self.parameters.name != name]
        self.stressmodels.pop(name)

    def del_constant(self):
        """ Save deletion of the constant from a Model.

        """
        if self.constant is None:
            self.logger.warning("No constant is present in this model.")
        else:
            self.parameters = self.parameters.ix[self.parameters.name !=
                                                 self.constant.name]
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
                                            use_oseries=False,
                                            use_stresses=True)
            sim_index = self.get_sim_index(tmin, tmax, freq, self.settings[
                "warmup"])
            self.update_stresses(tmin=sim_index.min(), tmax=sim_index.max(),
                                 freq=freq)

        else:
            sim_index = self.sim_index

        dt = get_dt(freq)

        # Get parameters if none are provided
        if parameters is None:
            parameters = self.get_parameters()

        h = pd.Series(data=0, index=sim_index)
        istart = 0  # Track parameters index to pass to ts object
        for ts in self.stressmodels.values():
            c = ts.simulate(parameters[istart: istart + ts.nparam], sim_index,
                            dt)
            h = h.add(c, fill_value=0.0)
            istart += ts.nparam
        if self.constant:
            h = h + self.constant.simulate(parameters[istart])
            istart += 1
        if self.transform:
            h = self.transform.simulate(h, parameters[istart:istart + self.transform.nparam])
        h.name = "Simulation"
        h.index.name = "Date"
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

        # simulate model
        simulation = self.simulate(parameters, tmin, tmax, freq)
        if self.oseries_calib is None:
            tmin, tmax = self.get_tmin_tmax(tmin, tmax, freq, use_oseries=True)
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

        if np.isnan(sum(res ** 2)):  # quick and dirty check
            self.logger.warning('nan problem in residuals')
        res.name = "Residuals"
        res.index.name = "Date"
        return res

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
            self.logger.error("Innovations can not be calculated if there is "
                              "no noisemodel.")
            return None

        # Get parameters if none are provided
        if parameters is None:
            parameters = self.get_parameters()

        # Calculate the residuals
        res = self.residuals(parameters, tmin, tmax, freq)

        # Calculate the innovations
        v = self.noisemodel.simulate(res, self.odelt[res.index],
                                     parameters[-self.noisemodel.nparam:])
        v.name = "Innovations"
        v.index.name = "Date"
        return v

    def observations(self, tmin=None, tmax=None):
        """Method that returns the observations series.

        """
        tmin, tmax = self.get_tmin_tmax(tmin, tmax, use_oseries=True)

        return self.oseries.loc[tmin: tmax]

    def initialize(self, tmin=None, tmax=None, freq=None, warmup=None,
                   noise=None, weights=None, initial=True):
        """Initialize the model. This method is called by the solve
        method but can also be triggered manually.

        """

        if noise is None and self.noisemodel:
            noise = True
        elif noise is True and self.noisemodel is None:
            self.logger.error('Warning, solution with noise model while '
                              'noise model is not defined. No noise model is '
                              'used.')
            noise = False

        self.settings["noise"] = noise
        self.settings["weights"] = weights

        # Set tmin and tmax
        self.settings["tmin"], self.settings["tmax"] = self.get_tmin_tmax(tmin,
                                                                          tmax,
                                                                          use_stresses=True)

        # Set the frequency & warmup
        if freq:
            self.settings["freq"] = freq
        if warmup is not None:
            self.settings["warmup"] = warmup

        # make sure calibration data is renewed
        if all(self.stressmodels[key]._name == "NoConvModel" for key in
               self.stressmodels.keys()):
            self.sim_index = self.oseries.index
        else:
            self.sim_index = self.get_sim_index(self.settings["tmin"],
                                                self.settings["tmax"],
                                                self.settings["freq"],
                                                self.settings["warmup"])

        self.oseries_calib = self.get_oseries_calib(self.settings["tmin"],
                                                    self.settings["tmax"],
                                                    self.sim_index)

        # Prepare the stressmodels
        self.update_stresses(freq=self.settings["freq"],
                             tmin=self.sim_index.min(),
                             tmax=self.sim_index.max())

        self.interpolate_simulation = self.oseries_calib.index.difference(
            self.sim_index).size != 0
        if self.interpolate_simulation:
            self.logger.info('There are observations between the simulation'
                             'timesteps. Linear interpolation is used')

        # Initialize parameters
        self.parameters = self.get_init_parameters(noise, initial)

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
            Class used to solve the model. Default is LmfitSolve
        report: bool, optional
            Print a report to the screen after optimization finished.
        noise: bool, optional
            Use the noise model (True) or not (False). The default is
            noise=True
        weights: pandas.Series
            Pandas Series with values by which the residuals are multiplied,
             index-based.
        initial: bool, optional
            Reset initial parameters. Default is True.
        freq: str
            String with the frequency the stressmodels are simulated.

        """

        # Initialize the model
        self.initialize(tmin, tmax, freq, warmup, noise, weights, initial)
        self.settings["solver"] = solver._name

        # Solve model
        self.fit = solver(self, tmin=self.settings["tmin"],
                          tmax=self.settings["tmax"],
                          noise=self.settings["noise"],
                          freq=self.settings["freq"],
                          weights=self.settings["weights"], **kwargs)

        # make calibration data empty again (was set in initialize)
        self.sim_index = None
        self.oseries_calib = None
        self.interpolate_simulation = None

        self.parameters.optimal = self.fit.optimal_params
        self.parameters.stderr = self.fit.stderr

        if report:
            print(self.fit_report())

    def get_sample(self, series, tindex):
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

    def get_oseries_calib(self, tmin, tmax, sim_index):
        """Method to get the oseries to use for calibration.

        This method is for performance improvements only.

        """
        oseries_calib = self.oseries.loc[tmin: tmax]
        # sample measurements, so that frequency is not higher than model
        # keep the original timestamps, as they will be used during
        # interpolation of the simulation
        oseries_calib = self.get_sample(oseries_calib, sim_index)
        return oseries_calib

    def get_tmin_tmax(self, tmin=None, tmax=None, freq=None, use_oseries=True,
                      use_stresses=False):
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
            Obtain the tmin and tmax from the oseries. Default is True.
        use_stresses: bool, optional
            Obtain the tmin and tmax from the stresses. The minimum/maximum
            time from all stresses is taken.

        Returns
        -------
        tmin, tmax: pandas.Timestamp
            returns pandas timestamps for tmin and tmax.

        Notes
        -----
        The parameters tmin and tmax are leading, unless use_oseries is
        True, then these are checked against the oseries index. The tmin and
        tmax are checked and returned according to the following rules:

        A. If no value for tmin/tmax is provided:
            1. If use_oseries is True, tmin and/or tmax is based on the
            oseries.
            2. If use_stresses is True, tmin and/or tmax is based on the
            stressmodels.

        B. If a values for tmin/tmax is provided:
            1. A pandas timestamp is made from the string
            2. if use_oseries is True, tmin is checked against oseries.

        C. In all cases an offset for the tmin and tmax is added.

        A detailed description of dealing with tmin and tmax and timesteps
        in general can be found in the developers section of the docs.

        """
        # Get tmin and tmax from the oseries
        if use_oseries:
            ts_tmin = self.oseries.index.min()
            ts_tmax = self.oseries.index.max()
        # Get tmin and tmax from the stressmodels
        elif use_stresses:
            ts_tmin = pd.Timestamp.max
            ts_tmax = pd.Timestamp.min
            for stressmodel in self.stressmodels.values():
                if stressmodel.tmin < ts_tmin:
                    ts_tmin = stressmodel.tmin
                if stressmodel.tmax > ts_tmax:
                    ts_tmax = stressmodel.tmax
        # Get tmin and tmax from user provided values
        else:
            ts_tmin = pd.Timestamp(tmin)
            ts_tmax = pd.Timestamp(tmax)

        # Set tmin properly
        if tmin is not None and use_oseries:
            tmin = max(pd.Timestamp(tmin), ts_tmin)
        elif tmin is not None:
            tmin = pd.Timestamp(tmin)
        else:
            tmin = ts_tmin

        # Set tmax properly
        if tmax is not None and use_oseries:
            tmax = min(pd.Timestamp(tmax), ts_tmax)
        elif tmax is not None:
            tmax = pd.Timestamp(tmax)
        else:
            tmax = ts_tmax

        # adjust tmin and tmax so that the time-offset is equal to the stressmodels.
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

    def set_freq(self):
        freqs = set()
        if self.oseries.freq:
            # when the oseries has a constant frequency, us this
            freqs.add(self.oseries.freq)
        else:
            # otherwise determine frequency from the stressmodels
            for stressmodel in self.stressmodels.values():
                if stressmodel.stress:
                    for stress in stressmodel.stress:
                        if stress.settings['freq']:
                            # first check the frequency, and use this
                            freqs.add(stress.settings['freq'])
                        elif stress.freq_original:
                            # if this is not available, and the original frequency is, take the original frequency
                            freqs.add(stress.freq_original)

        if len(freqs) == 1:
            # if there is only one frequency, use this frequency
            self.settings["freq"] = next(iter(freqs))
        elif len(freqs) > 1:
            # if there are more frequencies, take the highest frequency (lowest dt)
            freqs = list(freqs)
            dt = np.array([get_dt(f) for f in freqs])
            self.settings["freq"] = freqs[np.argmin(dt)]
        else:
            self.logger.info("Frequency of model cannot be determined. "
                             "Frequency is set to daily")
            self.settings["freq"] = "D"

    def set_time_offset(self):
        """Set the time offset for the model class.

        Notes
        -----
        Method to check if the StressModel timestamps match (e.g. similar hours)

        """
        time_offsets = set()
        for stressmodel in self.stressmodels.values():
            if stressmodel.stress:
                # calculate the offset from the default frequency
                time_offset = get_time_offset(
                    stressmodel.stress[0].index.min(),
                    self.settings["freq"])
                time_offsets.add(time_offset)

        assert len(
            time_offsets) <= 1, self.logger.error("""The time-differences with
                                                  the default frequency is
                                                  not the same for all
                                                  stresses.""")
        if len(time_offsets) == 1:
            self.settings["time_offset"] = next(iter(time_offsets))
        else:
            self.settings["time_offset"] = pd.Timedelta(0)

    def update_stresses(self, tmin, tmax, freq, **kwargs):
        """

        Parameters
        ----------
        tmin
        tmax
        freq
        kwargs

        Returns
        -------

        """
        for ts in self.stressmodels.values():
            ts.update_stress(freq=freq, tmin=tmin, tmax=tmax, **kwargs)

    def update_parameters(self, parameters):
        """internal method to add parameters while maintaining the original
        parameters dataframe.

        Returns
        -------

        """
        self.parameters = self.parameters.append(parameters)

        pass

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
            paramold = self.parameters

        parameters = pd.DataFrame(columns=['initial', 'pmin', 'pmax', 'vary',
                                           'optimal', 'name', 'stderr'])
        for ts in self.stressmodels.values():
            parameters = parameters.append(ts.parameters)
        if self.constant:
            parameters = parameters.append(self.constant.parameters)
        if self.transform:
            parameters = parameters.append(self.transform.parameters)
        if self.noisemodel and noise:
            parameters = parameters.append(self.noisemodel.parameters)

        # Set initial parameters to optimal parameters
        if not initial:
            parameters.loc[paramold.index, 'initial'] = paramold.optimal

        return parameters

    def get_parameters(self, name=None):
        """Internal method to obtain the parameters needed for calculation if
        none are provided. This method is used by the simulation, residuals
        and the innovations methods as well as other methods that need
        parameters values as arrays.

        Parameters
        ----------
        name: str, optional
            string with the name of the pastas.stressmodel object.

        Returns
        -------
        p: numpy.ndarray
            Numpy array with the parameters used in the time series model.

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

    @get_stressmodel
    def get_contribution(self, name, tmin=None, tmax=None, tindex=None, istress=None):
        p = self.get_parameters(name)
        dt = get_dt(self.settings["freq"])
        if istress is None:
            contrib = self.stressmodels[name].simulate(p, tindex=tindex, dt=dt)
        else:
            contrib = self.stressmodels[name].simulate(p, tindex=tindex, dt=dt, istress=istress)
        return contrib.loc[tmin:tmax]

    def get_transform_contribution(self, simulation):
        p = self.get_parameters(self.transform.name)
        return self.transform.simulate(simulation, p) - simulation

    @get_stressmodel
    def get_block_response(self, name):
        p = self.get_parameters(name)
        dt = get_dt(self.settings["freq"])
        b = self.stressmodels[name].rfunc.block(p, dt)
        t = np.arange(dt, (len(b) + 1) * dt, dt)
        return pd.Series(b, index=t, name=name)

    @get_stressmodel
    def get_step_response(self, name):
        p = self.get_parameters(name)
        dt = get_dt(self.settings["freq"])
        s = self.stressmodels[name].rfunc.step(p, dt)
        t = np.arange(dt, (len(s) + 1) * dt, dt)
        return pd.Series(s, index=t, name=name)

    @get_stressmodel
    def get_stress(self, name):
        p = self.get_parameters(name)
        stress = self.stressmodels[name].get_stress(p)
        return stress

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
        metadata["projection"] = None

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
                   log_level=logging.INFO, env_key='LOG_CFG'):
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
            logging.basicConfig(level=log_level)

        logger = logging.getLogger(__name__)

        return logger

    def fit_report(self, output="full"):
        """Method that reports on the fit after a model is optimized. May be
        called independently as well.

        """
        model = {
            "nfev": self.fit.nfev,
            "nobs": self.oseries.index.size,
            "noise": self.noisemodel._name if self.noisemodel else None,
            "tmin": str(self.settings["tmin"]),
            "tmax": str(self.settings["tmax"]),
            "freq": self.settings["freq"],
            "warmup": self.settings["warmup"],
            "solver": self.settings["solver"]
        }

        fit = {
            "EVP": format("%.2f" % self.stats.evp()),
            "RMSE": format("%.2f" % self.stats.rmse()),
            "Pearson R2": format("%.2f" % self.stats.rsq()),
            "AIC": format("%.2f" % self.stats.aic()),
            "BIC": format("%.2f" % self.stats.bic()),
            "_": "",
            "__": "",
            "___": ""
        }

        basic = str()
        for item, item2 in zip(model.items(), fit.items()):
            val1, val2 = item
            val3, val4 = item2
            basic = basic + (
                "{:<8} {:<22} {:<10} {:>17}\n".format(val1, val2, val3, val4))

        parameters = self.parameters.loc[:,
                     ["optimal", "stderr", "initial", "vary"]]
        n_param = parameters.vary.sum()

        w = ["Standard Errors assume that the covariance matrix of the errors "
             "is correctly specified."]
        pmin, pmax = self.check_parameters_bounds()
        if any(pmin):
            w.append("Parameter values of %s are close to their minimum "
                     "values." % self.parameters[pmin].index.tolist())
        if any(pmax):
            w.append("Parameter values of %s are close to their maximum "
                     "values." % self.parameters[pmax].index.tolist())

        warnings = str("Warnings\n============================================"
                       "================\n")
        for n, warn in enumerate(w, start=1):
            warnings = warnings + "[{}] {}\n".format(n, warn)

        if output == "basic":
            output = ["model", "parameters"]
        else:
            output = ["model", "parameters", "correlations", "warnings",
                      "tests"]

        report = """
Model Results %s                Fit Statistics
============================    ============================
%s
Parameters (%s were optimized)
============================================================
%s

%s
        """ % (self.name, basic, n_param, parameters, warnings)

        return report

    def check_parameters_bounds(self):
        """Check if the optimal parameters are close to pmin or pmax

        Returns
        -------

        """
        prange = self.parameters.pmax - self.parameters.pmin
        pnorm = (self.parameters.optimal - self.parameters.pmin) / prange
        pmax = pnorm > 0.99
        pmin = pnorm < 0.01
        return pmin, pmax

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
         - stressmodeldict
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

        # stressmodels
        data["stressmodels"] = dict()
        for name, ts in self.stressmodels.items():
            data["stressmodels"][name] = ts.dump(series=series)

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
