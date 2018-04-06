from __future__ import print_function, division

import copy
import inspect
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
from .solver import LeastSquares
from .stats import Statistics
from .stressmodels import Constant
from .timeseries import TimeSeries
from .utils import get_dt, get_time_offset
from .version import __version__


class Model:
    """Initiates a time series model.

    Parameters
    ----------
    oseries: pandas.Series or pastas.TimeSeries
        pandas Series object containing the dependent time series. The
        observation can be non-equidistant.
    constant: bool, optional
        Add a constant to the model (Default=True).
    noisemodel: bool, optional
        Add the default noisemodel to the model. A custom noisemodel can be
        added later in the modelling process as well.
    name: str, optional
        String with the name of the model, used in plotting and saving.
    metadata: dict, optional
        Dictionary containing metadata of the model. By default,
        a dictionary containing basic metadata info is created.
    settings: dict, optional
        Dictionary containing the model settings used by the different
        methods of the model instance. These values will be forwarded to the
        model settings (see ml.settings).
    log_level: str, optional
        String to set the level of the log-messages that is forwarded to the
        Python console. Options are: ERROR (default), WARNING and INFO.

    Returns
    -------
    ml: pastas.Model
        Pastas Model instance, the base object in Pastas.

    Examples
    --------

    >>> oseries = pd.Series([1,2,1], index=pd.to_datetime(range(3), unit="D"))
    >>> ml = Model(oseries)

    """

    def __init__(self, oseries, constant=True, noisemodel=True,
                 name=None, metadata=None, settings=None, log_level=None):
        self.logger = self.get_logger(log_level=log_level)

        # Construct the different model components
        self.oseries = TimeSeries(oseries, settings="oseries")
        self.odelt = self.get_odelt()
        if name is None:
            name = self.oseries.name
            if name is None:
                name = 'Observations'
        self.name = name

        self.parameters = pd.DataFrame(
            columns=["initial", "name", "optimal", "pmin", "pmax", "vary",
                     "stderr"])
        self.stressmodels = OrderedDict()
        self.constant = None
        self.transform = None
        self.noisemodel = None

        # Store the simulation settings
        self.settings = {"tmin": None,
                         "tmax": None,
                         "freq": "D",
                         "warmup": 3650,
                         "time_offset": pd.Timedelta(0),
                         "noise": noisemodel,
                         "solver": None,
                         "fit_constant": True}
        if settings:
            self.settings.update(settings)

        if constant:
            constant = Constant(value=self.oseries.mean(), name="constant")
            self.add_constant(constant)
        if noisemodel:
            self.add_noisemodel(NoiseModel())

        # Metadata & File Information
        self.metadata = self.get_metadata(metadata)
        self.file_info = self.get_file_info()

        # initialize some attributes for solving and simulation
        self.sim_index = None
        self.oseries_calib = None
        self.interpolate_simulation = None
        self.normalize_residuals = False
        self.fit = None

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
        replace: bool, optional
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
        if inspect.isclass(transform):
            transform = transform(self)
        self.transform = transform
        self.parameters = self.get_init_parameters()

    def add_noisemodel(self, noisemodel):
        """Adds a noisemodel to the time series Model.

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
        self.stressmodels.pop(name, None)
        self.parameters = self.get_init_parameters(initial=False)

    def del_constant(self):
        """ Save deletion of the constant from a Model.

        """
        if self.constant is None:
            self.logger.warning("No constant is present in this model.")
        else:
            self.constant = None
            self.parameters = self.get_init_parameters(initial=False)

    def del_transform(self):
        if self.transform is None:
            self.logger.warning("No transform is present in this model.")
        else:
            self.transform = None
            self.parameters = self.get_init_parameters(initial=False)

    def del_noisemodel(self):
        """Save deletion of the noisemodel from the Model.

        """
        if self.noisemodel is None:
            self.logger.warning("No noisemodel is present in this model.")
        else:
            self.noisemodel = None
            self.parameters = self.get_init_parameters(initial=False)

    def simulate(self, parameters=None, tmin=None, tmax=None, freq=None):
        """Method to simulate the time series model.

        Parameters
        ----------
        parameters: array-like, optional
            Array with the parameters used in the time series model. See
            Model.get_parameters() for more info if parameters is None.
        tmin: str, optional
        tmax: str, optional
        freq: str, optional
            Frequency at which the time series are simulated.

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

        h = pd.Series(data=np.zeros(sim_index.size, dtype=float),
                      index=sim_index, fastpath=True)

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
            h = self.transform.simulate(h, parameters[
                                           istart:istart + self.transform.nparam])
        h.name = 'Simulation'
        return h

    def residuals(self, parameters=None, tmin=None, tmax=None, freq=None):
        """Method to calculate the residual series.

        Parameters
        ----------
        parameters: list, optional
            Array of the parameters used in the time series model. See
            Model.get_parameters() for more info if parameters is None.
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
            h_simulated = simulation.loc[obs_index]
        res = oseries_calib.subtract(h_simulated)

        if self.normalize_residuals:
            res = res - res.mean()

        if np.isnan(sum(res ** 2)):  # quick and dirty check
            self.logger.warning('nan problem in residuals')

        res.name = "Residuals"
        return res

    def innovations(self, parameters=None, tmin=None, tmax=None, freq=None):
        """Method to simulate the innovations when a noisemodel is present.

        Parameters
        ----------
        parameters: list, optional
            Array of the parameters used in the time series model. See
            Model.get_parameters() for more info if parameters is None.
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
        v = self.noisemodel.simulate(res, self.odelt.loc[res.index],
                                     parameters[-self.noisemodel.nparam:])
        return v

    def observations(self, tmin=None, tmax=None):
        """Method that returns the observations series.

        """
        tmin, tmax = self.get_tmin_tmax(tmin, tmax, use_oseries=True)

        return self.oseries.loc[tmin: tmax]

    def initialize(self, tmin=None, tmax=None, freq=None, warmup=None,
                   noise=None, weights=None, initial=True, fit_constant=None):
        """Method to initialize the model.

        This method is called by the solve-method, but can also be triggered
        manually. See the solve-method for a description of the arguments.

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
        self.settings["tmin"], self.settings["tmax"] = \
            self.get_tmin_tmax(tmin, tmax, use_stresses=True)

        # Set the frequency & warmup
        if freq:
            self.settings["freq"] = freq
        if warmup is not None:
            self.settings["warmup"] = warmup

        # set fit_constant
        if fit_constant is not None:
            self.settings["fit_constant"] = fit_constant

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

        self.odelt = self.get_odelt()

        # Prepare the stressmodels
        self.update_stresses(freq=self.settings["freq"],
                             tmin=self.sim_index.min(),
                             tmax=self.sim_index.max())

        if self.oseries_calib.index.difference(self.sim_index).size is not 0:
            self.interpolate_simulation = True
            self.logger.info('There are observations between the simulation '
                             'timesteps. Linear interpolation is used.')
        else:
            self.interpolate_simulation = False

        # Initialize parameters
        self.parameters = self.get_init_parameters(noise, initial)

        # Prepare model if not fitting the constant as a parameter
        if not self.settings["fit_constant"]:
            self.parameters.loc["constant_d", "vary"] = 0
            self.parameters.loc["constant_d", "initial"] = 0.0
            self.normalize_residuals = True

    def solve(self, tmin=None, tmax=None, solver=LeastSquares, report=True,
              noise=True, initial=True, weights=None, freq=None, warmup=None,
              fit_constant=True, **kwargs):
        """Method to solve the time series model.

        Parameters
        ----------
        tmin: str, optional
            String with a start date for the simulation period (E.g. '1980').
            If none is provided, the tmin from the oseries is used.
        tmax: str, optional
            String with an end date for the simulation period (E.g. '2010').
            If none is provided, the tmax from the oseries is used.
        solver: pastas.solver.BaseSolver, optional
            Class used to solve the model. Options are: ps.LeastSquares
            (default) or ps.LmfitSolve. A class is needed, not an instance
            of the class!
        report: bool, optional
            Print a report to the screen after optimization finished. This
            can also be manually triggered after optimization by calling
            print(ml.fit_report()) on the Pastas model instance.
        noise: bool, optional
            Argument that determines if a noisemodel is used (only if
            present). The default is noise=True
        initial: bool, optional
            Reset initial parameters from the individual stressmodels.
            Default is True. If False, the optimal values from an earlier
            optimization are used.
        weights: pandas.Series, optional
            Pandas Series with values by which the residuals are multiplied,
            index-based.
        freq: str, optional
            String with the frequency the stressmodels are simulated.
        warmup: float/int, optinal
            Warmup period (in Days) for which the simulation is calculated,
            but not used for the calibration period.
        fit_constant: bool, optional
            Argument that determines if the constant is fitted as a parameter.
            If it is set to False, the constant is set equal to the mean of
            the residuals.
        **kwargs: dict, optional
            All keyword arguments will be passed onto the solver. It depends
            on the solver used which

        """

        # Initialize the model
        self.initialize(tmin, tmax, freq, warmup, noise, weights, initial,
                        fit_constant)
        self.settings["solver"] = solver._name

        # Solve model
        self.fit = solver(self, tmin=self.settings["tmin"],
                          tmax=self.settings["tmax"],
                          noise=self.settings["noise"],
                          freq=self.settings["freq"],
                          weights=self.settings["weights"], **kwargs)

        if not self.settings['fit_constant']:
            # do this before setting oseries_calib to None
            self.normalize_residuals = False
            res = self.residuals(self.fit.optimal_params,
                                 self.settings['tmin'], self.settings['tmax'],
                                 self.settings["freq"])
            # set the constant to the mean of the residuals
            mask = self.parameters.name == self.constant.name
            self.fit.optimal_params[mask] = res.mean()

        # make calibration data empty again (was set in initialize)
        self.sim_index = None
        self.oseries_calib = None
        self.interpolate_simulation = None

        self.parameters.optimal = self.fit.optimal_params
        self.parameters.stderr = self.fit.stderr

        if report:
            print(self.fit_report())

    def set_initial(self, name, value):
        """Method to set the initial value of any parameter.

        Parameters
        ----------
        name: str
            name of the parameter to update.
        value: float
            parameters value to use as initial estimate.

        """
        self.set_parameter(name, value, "initial")

    def set_vary(self, name, value):
        """Method to set if the parameter is allowed to vary.

        Parameters
        ----------
        name: str
            name of the parameter to update.
        value: bool
            boolean (True, False, 0 or 1) to vary a parameter or not.

        """
        self.set_parameter(name, value, "vary")

    def set_min(self, name, value):
        """Method to set the minimum value of a parameter.

        Parameters
        ----------
        name: str
            name of the parameter to update.
        value: float
            minimum value for the parameter.

        """
        self.set_parameter(name, value, "min")

    def set_max(self, name, value):
        """Method to set the maximum values of a parameter.

        Parameters
        ----------
        name: str
            name of the parameter to update.
        value: float
            maximum value for the parameter.


        """
        self.set_parameter(name, value, "max")

    def set_parameter(self, name, value, kind):
        """Internal method to set the parameter value for some kind.

        """
        cat, param = name.rsplit("_", maxsplit=1)
        if cat in self.stressmodels.keys():
            self.stressmodels[cat].__getattribute__("set_" + kind)(name, value)
            self.parameters.loc[name, kind] = value
        elif self.noisemodel:
            if cat == self.noisemodel.name:
                self.noisemodel.__getattribute__("set_" + kind)(name, value)
                self.parameters.loc[name, kind] = value
        elif self.constant:
            if cat == self.constant.name:
                self.constant.__getattribute__("set_" + kind)(name, value)
                self.parameters.loc[name, kind] = value
        else:
            self.logger.warning("parameters with name %s is not present in"
                                "the model" % name)

    def set_freq(self):
        """Internal method to set the frequency in the settings. This is
        method is not yet applied and is for future development.

        """
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
        """Internal method to set the time offset for the model class.

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

    def set_log_level(self, log_level):
        """Method to set the log_level for which messages are printen to the
        Python console. This can be usefull for when more or less info is
        desirable.

        Parameters
        ----------
        log_level: str
            String with the level, options are: ERROR, WARNING and INFO.

        """
        self.logger.parent.handlers[0].setLevel(log_level)

    def get_sim_index(self, tmin, tmax, freq, warmup):
        """Internal method to get the indices for the simulation, including
        the warmup period.

        Parameters
        ----------
        tmin: str or pd.TimeStamp
        tmax: str or pd.TimeStamp
        freq: str
        warmup: int

        Returns
        -------
        sim_index: pd.DatetimeIndex
            Pandas DatetimeIndex instance with the datetimes values for
            which the model is simulated.

        """
        if isinstance(tmin, str):
            tmin = pd.Timestamp(tmin)
        if isinstance(tmax, str):
            tmax = pd.Timestamp(tmax)

        sim_index = pd.date_range(tmin - pd.DateOffset(days=warmup), tmax,
                                  freq=freq, name="Date")
        return sim_index

    def get_oseries_calib(self, tmin, tmax, sim_index):
        """Internal method to get the oseries to use for calibration.

        Parameters
        ----------
        tmin: str
        tmax: str
        sim_index: pd.DatetimeIndex


        Returns
        -------
        oseries_calib: pd.Series
            pandas series of the oseries used for calibration of the model

        Notes
        -----
        This method is for performance improvements only. Find the index
        closest to the tindex, and then return a selection of series.

        """
        oseries_calib = self.oseries.loc[tmin:tmax]
        # sample measurements, so that frequency is not higher than model
        # keep the original timestamps, as they will be used during
        # interpolation of the simulation
        f = interpolate.interp1d(oseries_calib.index.asi8,
                                 np.arange(0, oseries_calib.index.size),
                                 kind='nearest', bounds_error=False,
                                 fill_value='extrapolate')
        ind = np.unique(f(sim_index.asi8).astype(int))
        oseries_calib = oseries_calib.iloc[ind]

        return oseries_calib

    def get_odelt(self, freq="D"):
        """Internal method to get the timesteps between the observations.

        Parameters
        ----------
        freq: str
            Frequency string.

        Returns
        -------
        odelt: pandas.Series
            Pandas Series object

        """
        odelt = self.oseries.index.to_series().diff() / pd.Timedelta(1, freq)
        return odelt

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

    def get_init_parameters(self, noise=None, initial=True):
        """Method to get all initial parameters from the individual objects.

        Parameters
        ----------
        noise: bool, optional
            Add the parameters for the noisemodel to the parameters
            Dataframe or not.
        initial: bool, optional
            True to get initial parameters, False to get optimized parameters.

        Returns
        -------
        parameters: pandas.DataFrame
            pandas.Dataframe with the parameters.

        """
        if noise is None:
            noise = self.settings['noise']

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

        # Set initial parameters to optimal parameters from model
        if not initial:
            paramold = self.parameters.optimal
            parameters.initial.update(paramold)

        return parameters

    def get_parameters(self, name=None):
        """Internal method to obtain the parameters needed for calculation.

        This method is used by the simulation, residuals and the innovations
        methods as well as other methods that need parameters values as arrays.

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
    def get_contribution(self, name, tmin=None, tmax=None, tindex=None,
                         istress=None):
        """Method to get the contribution of a stressmodel.

        The optimal parameters are used when available, initial otherwise.

        Parameters
        ----------
        name: str
            String with the name of the stressmodel.
        tmin: str or pandas.TimeStamp
        tmax: str or pandas.TimeStamp
        tindex: pandas.DatetimeIndex
            Pandas datetimeindex to simulate the contribution for.
        istress: int
            When multiple stresses are present in a stressmodel,
            this keyword can be used to obtain the contribution of an
            individual stress.

        Returns
        -------
        contrib: pandas.Series
            Pandas Series with the contribution.

        """
        p = self.get_parameters(name)
        dt = get_dt(self.settings["freq"])
        if istress is None:
            contrib = self.stressmodels[name].simulate(p, tindex=tindex, dt=dt)
        else:
            contrib = self.stressmodels[name].simulate(p, tindex=tindex, dt=dt,
                                                       istress=istress)
        return contrib.loc[tmin:tmax]

    def get_transform_contribution(self, tmin=None, tmax=None):
        sim = self.simulate(tmin=tmin, tmax=tmax)
        # calculate what the simulation without the transform is
        ml = copy.copy(self)
        ml.del_transform()
        sim_org = ml.simulate(tmin=tmin, tmax=tmax)
        return sim - sim_org

    @get_stressmodel
    def get_block_response(self, name):
        """Method to obtain the block response for a stressmodel.

        The optimal parameters are used when available, initial otherwise.

        Parameters
        ----------
        name: str
            String with the name of the stressmodel.

        Returns
        -------
        pandas.Series
            Pandas Series with the block response. The index is based on the
            frequency that is present in the model.settings.

        TODO
        ----
        - Make sure an error is thrown when no rfunc is present.

        """
        p = self.get_parameters(name)
        dt = get_dt(self.settings["freq"])
        b = self.stressmodels[name].rfunc.block(p, dt)
        t = np.arange(dt, (len(b) + 1) * dt, dt)
        return pd.Series(b, index=t, name=name)

    @get_stressmodel
    def get_step_response(self, name):
        """Method to obtain the step response for a stressmodel.

        The optimal parameters are used when available, initial otherwise.

        Parameters
        ----------
        name: str
            String with the name of the stressmodel.

        Returns
        -------
        pandas.Series
            Pandas Series with the step response. The index is based on the
            frequency that is present in the model.settings.

        TODO
        ----
        - Make sure an error is thrown when no rfunc is present.

        """
        p = self.get_parameters(name)
        dt = get_dt(self.settings["freq"])
        s = self.stressmodels[name].rfunc.step(p, dt)
        t = np.arange(dt, (len(s) + 1) * dt, dt)
        return pd.Series(s, index=t, name=name)

    @get_stressmodel
    def get_stress(self, name):
        """Method to obtain the stress(es) from the stressmodel.

        Parameters
        ----------
        name: str
            String with the name of the stressmodel.

        Returns
        -------
        stress: pandas.Series/list
            If one stress is present, a pandas Series is returned. If more
            are present, a list of pandas Series is returned.

        """
        p = self.get_parameters(name)
        stress = self.stressmodels[name].get_stress(p)
        return stress

    def get_metadata(self, meta=None):
        """Internal method that returns a metadata dictionary.

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
        """Internal method to get the file information.

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

    def get_logger(self, log_level=None, config_file='log_config.json',
                   env_key='LOG_CFG'):
        """Internal method to create a logger instance to log program output.

        Returns
        -------
        logger: logging.Logger
            Logging instance that handles all logging throughout pastas,
            including all sub modules and packages.

        Notes
        -----

        """
        fname = os.getenv(env_key, None)
        if not fname or not os.path.exists(fname):
            dir_path = os.path.dirname(os.path.realpath(__file__))
            fname = os.path.join(dir_path, config_file)
        if os.path.exists(fname):
            with open(fname, 'rt') as f:
                config = json.load(f)
            logging.config.dictConfig(config)
        else:
            logging.basicConfig(level=logging.INFO)

        logger = logging.getLogger(__name__)

        # Set log_level for console to user-defined value
        if log_level is not None:
            logger.parent.handlers[0].setLevel(log_level)

        return logger

    def update_stresses(self, tmin=None, tmax=None, freq=None, **kwargs):
        """Method to update the settings of all stresses simultaneously.

        This method is used in the initialize method of the model.

        Parameters
        ----------
        tmin
        tmax
        freq
        kwargs

        """
        for ts in self.stressmodels.values():
            ts.update_stress(freq=freq, tmin=tmin, tmax=tmax, **kwargs)

    def update_oseries(self, tmin=None, tmax=None, freq=None, **kwargs):
        """Method to update the oseries.

        This will change the values used by the model when calibrating.

        Parameters
        ----------
        tmin
        tmax
        freq
        kwargs

        """
        self.oseries.update_series(tmin=tmin, tmax=tmax, freq=freq, **kwargs)

    def fit_report(self, output="full"):
        """Method that reports on the fit after a model is optimized.

        Parameters
        ----------
        output: str
            NotImplementedYet


        Returns
        -------
        report: str
            String with the report.

        Usage
        -----
        This method is called by the solve method if report=True, but can
        also be called on its own:

        >>> print(ml.fit_report)

        """
        if output != "full":
            raise NotImplementedError

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
            "NS": format("%.2f" % self.stats.nash_sutcliffe()),
            "Pearson R2": format("%.2f" % self.stats.rsq()),
            "RMSE": format("%.2f" % self.stats.rmse()),
            "AIC": format("%.2f" % self.stats.aic()),
            "BIC": format("%.2f" % self.stats.bic()),
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

        for name, vals in parameters.loc[:, ["optimal", "stderr"]].iterrows():
            popt, stderr = vals
            val = np.abs(np.divide(stderr, popt) * 100)
            parameters.loc[name, "stderr"] = \
                "{:} {:.5e} ({:.2f}{:})".format("\u00B1", stderr, val,
                                                "\u0025")

        n_param = parameters.vary.sum()

        w = []
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

    def check_parameters_bounds(self, alpha=0.01):
        """Check if the optimal parameters are close to pmin or pmax.

        Returns
        -------
        pmin: pandas.Series
            pandas series with boolean values of the parameters that are
            close to the minimum values.
        pmax: pandas.Series
            pandas series with boolean values of the parameters that are
            close to the maximum values.

        Notes
        -----
        The criteria to determine if the parameters is close to the maximum or
        minimum is determined as the percentage of the parameter range.

        """
        prange = self.parameters.pmax - self.parameters.pmin
        pnorm = (self.parameters.optimal - self.parameters.pmin) / prange
        pmax = pnorm > 1 - alpha
        pmin = pnorm < alpha
        return pmin, pmax

    def dump_data(self, series=True, sim_series=False, metadata=True,
                  file_info=True, transformed_series=False):
        """Internal method to export a PASTAS model to the json export format.

        Helper function for the self.export method.

         Parameters
         ----------
         transformed_series
         file_info
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
        data["oseries"] = self.oseries.dump(series=series,
                                            transformed_series=transformed_series)

        # Stressmodels
        data["stressmodels"] = dict()
        for name, ts in self.stressmodels.items():
            data["stressmodels"][name] = ts.dump(series=series,
                                                 transformed_series=transformed_series)

        # Constant
        if self.constant:
            data["constant"] = True

        # Transform
        if self.transform:
            data["transform"] = self.transform.dump()

        # Noisemodel
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

    def dump(self, fname, series=True, transformed_series=False, **kwargs):
        """Method to dump the Pastas model to a file.

        Parameters
        ----------
        fname: str
            String with the name and the extension of the file. File
            extension has to be supported by Pastas. E.g. "model.pas"
        series: bool
            Export the simulated series or not. Default is False.
        transformed_series: bool
            Export transformed series.
        kwargs: any argument that is passed to the Model.dump_data() method.

        """

        # Get dicts for all data sources
        data = self.dump_data(series, transformed_series=transformed_series)

        # Write the dicts to a file
        return dump(fname, data, **kwargs)
