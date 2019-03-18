"""The Model class is the main object for creating model in Pastas.

Examples
--------

>>> oseries = pd.Series([1,2,1], index=pd.to_datetime(range(3), unit="D"))
>>> ml = Model(oseries)


"""

import json
from collections import OrderedDict
from copy import copy
from inspect import isclass
from logging import basicConfig, getLogger, config
from os import path, getlogin, getenv

import numpy as np
import pandas as pd

from .decorators import get_stressmodel
from .io.base import dump, load_model
from .noisemodels import NoiseModel
from .plots import Plotting
from .solver import LeastSquares
from .modelstats import Statistics
from .stressmodels import Constant
from .timeseries import TimeSeries
from .utils import get_dt, get_time_offset, get_sample, frequency_is_supported
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
        Dictionary containing metadata of the oseries, passed on the to
        oseries when creating a pastas TimeSeries object. hence,
        ml.oseries.metadata will give you the metadata.
    log_level: str, optional
        String to set the level of the log-messages that is forwarded to the
        Python console. Options are: ERROR, WARNING and INFO (default).

    Returns
    -------
    ml: pastas.Model
        Pastas Model instance, the base object in Pastas.

    Examples
    --------

    >>> oseries = pd.Series([1,2,1], index=pd.to_datetime(range(3), unit="D"))
    >>> ml = Model(oseries)

    """

    def __init__(self, oseries, constant=True, noisemodel=True, name=None,
                 metadata=None, log_level="INFO"):

        self.logger = self.get_logger(log_level=log_level)

        # Construct the different model components
        self.oseries = TimeSeries(oseries, settings="oseries",
                                  metadata=metadata)

        if name is None:
            name = self.oseries.name
            if name is None:
                name = 'Observations'
        self.name = str(name)

        self.parameters = pd.DataFrame(
            columns=["initial", "name", "optimal", "pmin", "pmax", "vary",
                     "stderr"])

        # Define the model components
        self.stressmodels = OrderedDict()
        self.constant = None
        self.transform = None
        self.noisemodel = None

        # Default solve/simulation settings
        self.settings = {
            "tmin": None,
            "tmax": None,
            "freq": "D",
            "warmup": 3650,
            "time_offset": pd.Timedelta(0),
            "noise": noisemodel,
            "solver": None,
            "fit_constant": True,
        }

        if constant:
            constant = Constant(value=self.oseries.series.mean(),
                                name="constant")
            self.add_constant(constant)
        if noisemodel:
            self.add_noisemodel(NoiseModel())

        # File Information
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

    def __repr__(self):
        """Prints a simple string representation of the model.
        """
        template = ('{cls}(oseries={os}, name={name}, constant={const}, '
                    'noisemodel={noise})')
        return template.format(cls=self.__class__.__name__,
                               os=self.oseries.name,
                               name=self.name,
                               const=not self.constant is None,
                               noise=not self.noisemodel is None)

    def add_stressmodel(self, stressmodel, *args, replace=False, **kwargs):
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
        To obtain a list of the stressmodel names, type:

        >>> ml.stressmodels.keys()

        Examples
        --------
        >>> sm = ps.StressModel(stress, rfunc=ps.Gamma, name="stress")
        >>> ml.add_stressmodel(sm)

        """
        # Method can take multiple stressmodels at once through args
        if args:
            for arg in args:
                self.add_stressmodel(arg)
        if (stressmodel.name in self.stressmodels.keys()) and not replace:
            self.logger.error("The name for the stressmodel you are trying "
                              "to add already exists for this model. Select "
                              "another name.")
        else:
            self.stressmodels[stressmodel.name] = stressmodel
            self.parameters = self.get_init_parameters(initial=False)
            if self.settings["freq"] is None:
                self._set_freq()
            stressmodel.update_stress(freq=self.settings["freq"])

    def add_constant(self, constant):
        """Adds a Constant to the time series Model.

        Parameters
        ----------
        constant: pastas.Constant
            Pastas constant instance, possibly more things in the future.

        Examples
        --------
        >>> d = ps.Constant()
        >>> ml.add_constant(d)

        """
        self.constant = constant
        self.parameters = self.get_init_parameters(initial=False)

    def add_transform(self, transform):
        """Adds a Transform to the time series Model.

        Parameters
        ----------
        transform: pastas.transform
            instance of a pastas.transform object.

        Examples
        --------
        >>> tt = ps.ThresholdTransform()
        >>> ml.add_transform(tt)

        """
        if isclass(transform):
            # keep this line for backwards compatibilty for now
            transform = transform()
        transform.set_model(self)
        self.transform = transform
        self.parameters = self.get_init_parameters(initial=False)

    def add_noisemodel(self, noisemodel):
        """Adds a noisemodel to the time series Model.

        Parameters
        ----------
        noisemodel: pastas.noisemodels.NoiseModelBase
            Instance of NoiseModelBase

        Examples
        --------
        >>> n = ps.NoiseModel()
        >>> ml.add_noisemodel(n)

        """
        self.noisemodel = noisemodel
        self.parameters = self.get_init_parameters(initial=False)

    @get_stressmodel
    def del_stressmodel(self, name):
        """ Safely delete a stressmodel from the stressmodels dict.

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
        """ Safely delete the constant from the Model.

        """
        if self.constant is None:
            self.logger.warning("No constant is present in this model.")
        else:
            self.constant = None
            self.parameters = self.get_init_parameters(initial=False)

    def del_transform(self):
        """Safely delete the transform from the Model.

        """
        if self.transform is None:
            self.logger.warning("No transform is present in this model.")
        else:
            self.transform = None
            self.parameters = self.get_init_parameters(initial=False)

    def del_noisemodel(self):
        """Safely delete the noisemodel from the Model.

        """
        if self.noisemodel is None:
            self.logger.warning("No noisemodel is present in this model.")
        else:
            self.noisemodel = None
            self.parameters = self.get_init_parameters(initial=False)

    def simulate(self, parameters=None, tmin=None, tmax=None, freq=None,
                 warmup=None, return_warmup=False):
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
        warmup: int, optional
            Length of the warmup period in days
        return_warmup: bool, optional
            Return the simulation including the the warmup period or not,
            default is False.

        Returns
        -------
        sim: pandas.Series
            pandas.Series containing the simulated time series

        Notes
        -----
        This method can be used without any parameters. When the model is
        solved, the optimal parameters values are used and if not,
        the initial parameter values are used. This allows the user to
        get an idea of how the simulation looks with only the initial
        parameters and no calibration.

        """
        # Default options when tmin, tmax, freq and warmup are not provided.
        if tmin is None and self.settings['tmin']:
            tmin = self.settings['tmin']
        else:
            tmin = self.get_tmin(tmin, freq, use_oseries=False,
                                 use_stresses=True)
        if tmax is None and self.settings['tmax']:
            tmax = self.settings['tmax']
        else:
            tmax = self.get_tmax(tmax, freq, use_oseries=False,
                                 use_stresses=True)
        if freq is None:
            freq = self.settings["freq"]
        if warmup is None:
            warmup = self.settings["warmup"]
        elif isinstance(warmup, str):
            warmup = get_dt(warmup)

        # Get the simulation index and the time step
        sim_index = self.get_sim_index(tmin, tmax, freq, warmup)
        dt = get_dt(freq)

        # Get parameters if none are provided
        if parameters is None:
            parameters = self.get_parameters()

        sim = pd.Series(data=np.zeros(sim_index.size, dtype=float),
                        index=sim_index, fastpath=True)

        istart = 0  # Track parameters index to pass to stressmodel object
        for sm in self.stressmodels.values():
            contrib = sm.simulate(parameters[istart: istart + sm.nparam],
                                  sim_index.min(), sim_index.max(), freq, dt)
            sim = sim + contrib
            istart += sm.nparam
        if self.constant:
            sim = sim + self.constant.simulate(parameters[istart])
            istart += 1
        if self.transform:
            sim = self.transform.simulate(sim, parameters[
                                               istart:istart + self.transform.nparam])

        # Respect provided tmin/tmax at this point, since warmup matters for
        # simulation but should not be returned, unless return_warmup=True.
        if not return_warmup:
            sim = sim.loc[tmin:tmax]

        if sim.hasnans:
            sim = sim.dropna()
            self.logger.warning('Nan-values were removed from the simulation.')

        sim.name = 'Simulation'
        return sim

    def residuals(self, parameters=None, tmin=None, tmax=None, freq=None,
                  warmup=None):
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
        warmup: int, optional
            length of the warmup period in days

        Returns
        -------
        res: pandas.Series
            pandas.Series with the residuals series.

        """
        # Default options when tmin, tmax, freq and warmup are not provided.
        if tmin is None:
            tmin = self.settings['tmin']
        if tmax is None:
            tmax = self.settings['tmax']
        if freq is None:
            freq = self.settings["freq"]
        if warmup is None:
            warmup = self.settings["warmup"]
        elif isinstance(warmup, str):
            warmup = get_dt(warmup)

        # simulate model
        sim = self.simulate(parameters, tmin, tmax, freq, warmup,
                            return_warmup=False)

        # Get the oseries calibration series
        oseries_calib = self.observations(tmin, tmax, freq)

        # Get simulation at the correct indices
        if self.interpolate_simulation is None:
            if oseries_calib.index.difference(sim.index).size is not 0:
                self.interpolate_simulation = True
                self.logger.info('There are observations between the '
                                 'simulation timesteps. Linear interpolation '
                                 'between simulated values is used.')
        if self.interpolate_simulation:
            # interpolate simulation to measurement-times
            # TODO RC: Somehow switch to pandas methods with maximum gap (gap_limit?)
            sim_interpolated = np.interp(oseries_calib.index.asi8,
                                         sim.index.asi8, sim)
        else:
            # all of the observation indexes are in the simulation
            sim_interpolated = sim.loc[oseries_calib.index]

        # Calculate the actual residuals here
        res = oseries_calib.subtract(sim_interpolated)

        if res.hasnans:
            res = res.dropna()
            self.logger.warning('Nan-values were removed from the residuals.')

        if self.normalize_residuals:
            res = res - res.values.mean()

        res.name = "Residuals"
        return res

    def noise(self, parameters=None, tmin=None, tmax=None, freq=None,
              warmup=None):
        """Method to simulate the noise when a noisemodel is present.

        Parameters
        ----------
        parameters: list, optional
            Array of the parameters used in the time series model. See
            Model.get_parameters() for more info if parameters is None.
        tmin: str, optional
        tmax: str, optional
        freq: str, optional
            frequency at which the time series are simulated.
        warmup: int, optional
            length of the warmup period in days

        Returns
        -------
        noise : pandas.Series
            Pandas series of the noise.

        Notes
        -----
        The noise are the time series that result when applying a noise
        model.

        """
        if self.noisemodel is None:
            self.logger.error("Noise cannot be calculated if there is "
                              "no noisemodel.")
            return None

        if freq is None:
            freq = self.settings["freq"]

        # Get parameters if none are provided
        if parameters is None:
            parameters = self.get_parameters()

        # Calculate the residuals
        res = self.residuals(parameters, tmin, tmax, freq, warmup)

        # Calculate the noise
        noise = self.noisemodel.simulate(res,
                                         parameters[-self.noisemodel.nparam:],
                                         freq=freq)
        return noise

    def observations(self, tmin=None, tmax=None, freq=None):
        """Method that returns the observations series used for calibration.

        Parameters
        ----------
        tmin: str or pandas.TimeStamp, optional
        tmax: str or pandas.TimeStamp, optional
        freq: str, optional

        Returns
        -------
        oseries_calib: pandas.Series
            pandas series of the oseries used for calibration of the model

        Notes
        -----
        This method makes sure the simulation is compared to the nearest
        observation. It finds the index closest to sim_index, and then returns
        a selection of the oseries. in the residuals method, the simulation is
        interpolated to the observation-timestamps.

        """
        if tmin is None and self.settings['tmin']:
            tmin = self.settings['tmin']
        else:
            tmin = self.get_tmin(tmin, freq, use_oseries=False,
                                 use_stresses=True)
        if tmax is None and self.settings['tmax']:
            tmax = self.settings['tmax']
        else:
            tmax = self.get_tmax(tmax, freq, use_oseries=False,
                                 use_stresses=True)
        if freq is None:
            freq = self.settings["freq"]

        update_observations = False
        for key, setting in zip([tmin, tmax, freq], ["tmin", "tmax", "freq"]):
            if key != self.settings[setting]:
                update_observations = True

        if self.oseries_calib is None or update_observations:
            oseries_calib = self.oseries.series.loc[tmin:tmax]

            # sample measurements, so that frequency is not higher than model
            # keep the original timestamps, as they will be used during
            # interpolation of the simulation
            sim_index = self.get_sim_index(tmin, tmax, freq,
                                           self.settings["warmup"])
            if not oseries_calib.empty:
                index = get_sample(oseries_calib.index, sim_index)
                oseries_calib = oseries_calib.loc[index]

            if not update_observations:
                # tmin, tmax and freq are equal to the settings
                # so we can set self.oseries_calib to improve speed of next run
                self.oseries_calib = oseries_calib
        else:
            oseries_calib = self.oseries_calib
        return oseries_calib

    def initialize(self, tmin=None, tmax=None, freq=None, warmup=None,
                   noise=None, weights=None, initial=True, fit_constant=None):
        """Method to initialize the model.

        This method is called by the solve-method, but can also be triggered
        manually. See the solve-method for a description of the arguments.

        """
        if noise is None and self.noisemodel:
            noise = True
        elif noise is True and self.noisemodel is None:
            self.logger.warning("""Warning, solving with noisemodel while no
                              noisemodel is defined. No noisemodel is used.""")
            noise = False

        self.settings["noise"] = noise
        self.settings["weights"] = weights

        # Set the frequency & warmup
        if freq:
            self.settings["freq"] = frequency_is_supported(freq)

        if warmup is not None:
            if isinstance(warmup, str):
                warmup = get_dt(warmup)
            self.settings["warmup"] = warmup

        # Set the time offset from the frequency (this does not work as expected yet)
        # self._set_time_offset()

        # Set tmin and tmax
        self.settings["tmin"] = self.get_tmin(tmin)
        self.settings["tmax"] = self.get_tmax(tmax)

        # set fit_constant
        if fit_constant is not None:
            self.settings["fit_constant"] = fit_constant

        # make sure calibration data is renewed
        self.sim_index = None
        self.oseries_calib = None
        self.interpolate_simulation = None

        # Initialize parameters
        self.parameters = self.get_init_parameters(noise, initial)

        # Prepare model if not fitting the constant as a parameter
        if not self.settings["fit_constant"]:
            self.parameters.loc["constant_d", "vary"] = 0
            self.parameters.loc["constant_d", "initial"] = 0.0
            self.normalize_residuals = True

    def solve(self, tmin=None, tmax=None, freq=None, warmup=None, noise=None,
              solver=LeastSquares, report=True, initial=True, weights=None,
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
        solver: pastas.solver.BaseSolver class, optional
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
        freq: str, optional
            String with the frequency the stressmodels are simulated. Must
            be one of the following: (D,h,m,s,ms,us,ns) or a multiple of
            that e.g. "7D".
        warmup: float/int, optinal
            Warmup period (in Days) for which the simulation is calculated,
            but not used for the calibration period.
        weights: pandas.Series, optional
            Pandas Series with values by which the residuals are multiplied,
            index-based.
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
        self.fit = solver(self, noise=self.settings["noise"],
                          weights=self.settings["weights"], **kwargs)

        if not self.settings['fit_constant']:
            # Determine the residuals
            self.normalize_residuals = False
            res = self.residuals(self.fit.optimal_params)
            # set the constant to the mean of the residuals
            mask = self.parameters.name == self.constant.name
            self.fit.optimal_params[mask] = res.mean()

        self.parameters.optimal = self.fit.optimal_params
        self.parameters.stderr = self.fit.stderr

        if report:
            print(self.fit_report())

    def set_initial(self, name, value, move_bounds=False):
        """Method to set the initial value of any parameter.

        Parameters
        ----------
        name: str
            name of the parameter to update.
        value: float
            parameters value to use as initial estimate.
        move_bounds: bool, optional
            Reset pmin/pmax based on new initial value.

        """
        if move_bounds:
            factor = value / self.parameters.loc[name, 'initial']
            min_new = self.parameters.loc[name, 'pmin'] * factor
            self.set_parameter(name, min_new, 'pmin')
            max_new = self.parameters.loc[name, 'pmax'] * factor
            self.set_parameter(name, max_new, 'pmax')

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

    def set_pmin(self, name, value):
        """Method to set the minimum value of a parameter.

        Parameters
        ----------
        name: str
            name of the parameter to update.
        value: float
            minimum value for the parameter.

        """
        self.set_parameter(name, value, "pmin")

    def set_pmax(self, name, value):
        """Method to set the maximum values of a parameter.

        Parameters
        ----------
        name: str
            name of the parameter to update.
        value: float
            maximum value for the parameter.


        """
        self.set_parameter(name, value, "pmax")

    def set_parameter(self, name, value, kind):
        """Internal method to set the parameter value for some kind.

        """
        if name not in self.parameters.index:
            msg = "parameters with name {} is not present in the " \
                  "model".format(name)
            self.logger.error(msg)
            raise Exception(msg)

        cat = self.parameters.loc[name, "name"]

        # Because either of the following is not necessarily present
        noisemodel = self.noisemodel.name if self.noisemodel else "NotPresent"
        constant = self.constant.name if self.constant else "NotPresent"

        if cat in self.stressmodels.keys():
            self.stressmodels[cat].__getattribute__("set_" + kind)(name, value)
            self.parameters.loc[name, kind] = value
        elif cat == noisemodel:
            self.noisemodel.__getattribute__("set_" + kind)(name, value)
            self.parameters.loc[name, kind] = value
        elif cat == constant:
            self.constant.__getattribute__("set_" + kind)(name, value)
            self.parameters.loc[name, kind] = value

    def _set_freq(self):
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

    def _set_time_offset(self):
        """Internal method to set the time offset for the model class.

        Notes
        -----
        Method to check if the StressModel timestamps match (e.g. similar hours)

        """
        time_offsets = set()
        for stressmodel in self.stressmodels.values():
            for st in stressmodel.stress:
                if st.freq_original:
                    # calculate the offset from the default frequency
                    time_offset = get_time_offset(
                        st.series_original.index.min(),
                        self.settings["freq"])
                    time_offsets.add(time_offset)
        if len(time_offsets) > 1:
            msg = (
                "The time-differences with the default frequency is not the "
                "same for all stresses.")
            self.logger.error(msg)
            raise (Exception(msg))
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

    def get_stressmodel_names(self):
        """Returns list of stressmodel names"""
        return list(self.stressmodels.keys())

    def get_sim_index(self, tmin, tmax, freq, warmup):
        """Internal method to get the simulation index, including the warmup
        period.

        Parameters
        ----------
        tmin: pandas.TimeStamp
        tmax: pandas.TimeStamp
        freq: str
        warmup: int

        Returns
        -------
        sim_index: pandas.DatetimeIndex
            Pandas DatetimeIndex instance with the datetimes values for
            which the model is simulated.

        """
        # Check if any of the settings are updated
        update_sim_index = False
        for key, setting in zip([tmin, tmax, freq, warmup],
                                ["tmin", "tmax", "freq", "warmup"]):
            if key != self.settings[setting]:
                update_sim_index = True

        if self.sim_index is None or update_sim_index:
            tmin = (tmin - pd.DateOffset(days=warmup)).floor(freq) + \
                   self.settings["time_offset"]
            sim_index = pd.date_range(tmin, tmax, freq=freq)
            if not update_sim_index:
                # tmin, tmax, freq and warmup are equal to the settings
                # so we can set self.sim_index to improve speed of next run
                self.sim_index = sim_index
        else:
            sim_index = self.sim_index
        return sim_index

    def get_tmin(self, tmin=None, freq=None, use_oseries=True,
                 use_stresses=False):
        """Method that checks and returns valid values for tmin.

        Parameters
        ----------
        tmin: str, optional
            string with a year or date that can be turned into a pandas
            Timestamp (e.g. pd.Timestamp(tmin)).
        freq: str, optional
            string with the frequency.
        use_oseries: bool, optional
            Obtain the tmin and tmax from the oseries. Default is True.
        use_stresses: bool, optional
            Obtain the tmin and tmax from the stresses. The minimum/maximum
            time from all stresses is taken.

        Returns
        -------
        tmin: pandas.Timestamp
            returns pandas timestamps for tmin.

        Notes
        -----
        The parameters tmin and tmax are leading, unless use_oseries is
        True, then these are checked against the oseries index. The tmin and
        tmax are checked and returned according to the following rules:

        A. If no value for tmin is provided:
            1. If use_oseries is True, tmin is based on the oseries.
            2. If use_stresses is True, tmin is based on the stressmodels.

        B. If a values for tmin is provided:
            1. A pandas timestamp is made from the string
            2. if use_oseries is True, tmin is checked against oseries.

        C. In all cases an offset for the tmin is added.

        A detailed description of dealing with tmin and timesteps in general
        can be found in the developers section of the docs.

        """
        # Get tmin from the oseries
        if use_oseries:
            ts_tmin = self.oseries.series.index.min()
        # Get tmin from the stressmodels
        elif use_stresses:
            ts_tmin = pd.Timestamp.max
            for stressmodel in self.stressmodels.values():
                if stressmodel.tmin < ts_tmin:
                    ts_tmin = stressmodel.tmin
        # Get tmin and tmax from user provided values
        else:
            ts_tmin = pd.Timestamp(tmin)

        # Set tmin properly
        if tmin is not None and use_oseries:
            tmin = max(pd.Timestamp(tmin), ts_tmin)
        elif tmin is not None:
            tmin = pd.Timestamp(tmin)
        else:
            tmin = ts_tmin

        # adjust tmin and tmax so that the time-offset is equal to the stressmodels.
        if freq is None:
            freq = self.settings["freq"]
        tmin = tmin.floor(freq) + self.settings["time_offset"]

        # assert tmax > tmin, \
        #     self.logger.error('Error: Specified tmax not larger than '
        #                       'specified tmin')
        # if use_oseries:
        #     assert self.oseries.series.loc[tmin: tmax].size > 0, \
        #         self.logger.error(
        #             'Error: no observations between tmin and tmax')

        return tmin

    def get_tmax(self, tmax=None, freq=None, use_oseries=True,
                 use_stresses=False):
        """Method that checks and returns valid values for tmin and tmax.

        Parameters
        ----------
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
        tmax: pandas.Timestamp
            returns pandas timestamps for tmax.

        Notes
        -----
        The parameters tmin and tmax are leading, unless use_oseries is
        True, then these are checked against the oseries index. The tmin and
        tmax are checked and returned according to the following rules:

        A. If no value for tmax is provided:
            1. If use_oseries is True, tmax is based on the
            oseries.
            2. If use_stresses is True, tmax is based on the
            stressmodels.

        B. If a values for tmax is provided:
            1. A pandas timestamp is made from the string
            2. if use_oseries is True, tmax is checked against oseries.

        C. In all cases an offset for the tmax is added.

        A detailed description of dealing with tmax and timesteps
        in general can be found in the developers section of the docs.

        """
        # Get tmax from the oseries
        if use_oseries:
            ts_tmax = self.oseries.series.index.max()
        # Get tmax from the stressmodels
        elif use_stresses:
            ts_tmax = pd.Timestamp.min
            for stressmodel in self.stressmodels.values():
                if stressmodel.tmax > ts_tmax:
                    ts_tmax = stressmodel.tmax
        # Get tmax from user provided values
        else:
            ts_tmax = pd.Timestamp(tmax)

        # Set tmax properly
        if tmax is not None and use_oseries:
            tmax = min(pd.Timestamp(tmax), ts_tmax)
        elif tmax is not None:
            tmax = pd.Timestamp(tmax)
        else:
            tmax = ts_tmax

        # adjust tmax so that the time-offset is equal to the stressmodels.
        if freq is None:
            freq = self.settings["freq"]
        tmax = tmax.floor(freq) + self.settings["time_offset"]

        # assert tmax > tmin, \
        #     self.logger.error('Error: Specified tmax not larger than '
        #                       'specified tmin')
        # if use_oseries:
        #     assert self.oseries.series.loc[tmin: tmax].size > 0, \
        #         self.logger.error(
        #             'Error: no observations between tmin and tmax')

        return tmax

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
        for sm in self.stressmodels.values():
            parameters = parameters.append(sm.parameters, sort=False)
        if self.constant:
            parameters = parameters.append(self.constant.parameters,
                                           sort=False)
        if self.transform:
            parameters = parameters.append(self.transform.parameters,
                                           sort=False)
        if self.noisemodel and noise:
            parameters = parameters.append(self.noisemodel.parameters,
                                           sort=False)

        # Set initial parameters to optimal parameters from model
        if not initial:
            paramold = self.parameters.optimal
            parameters.initial.update(paramold)
            parameters.optimal.update(paramold)

        return parameters

    def get_parameters(self, name=None):
        """Internal method to obtain the parameters needed for calculation.

        This method is used by the simulation, residuals and the noise
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
    def get_contribution(self, name, tmin=None, tmax=None, freq=None,
                         warmup=None, istress=None, return_warmup=False,
                         parameters=None):
        """Method to get the contribution of a stressmodel.

        The optimal parameters are used when available, initial otherwise.

        Parameters
        ----------
        name: str
            String with the name of the stressmodel.
        tmin: str or pandas.TimeStamp, optional
        tmax: str or pandas.TimeStamp, optional
        freq: str, optional
        istress: int
            When multiple stresses are present in a stressmodel,
            this keyword can be used to obtain the contribution of an
            individual stress.

        Returns
        -------
        contrib: pandas.Series
            Pandas Series with the contribution.

        """
        if parameters is None:
            parameters = self.get_parameters(name)

        if tmin is None:
            tmin = self.settings['tmin']
        if tmax is None:
            tmax = self.settings['tmax']
        if freq is None:
            freq = self.settings["freq"]
        if warmup is None:
            warmup = self.settings["warmup"]
        elif isinstance(warmup, str):
            warmup = get_dt(warmup)

        # use warmup
        if tmin:
            tmin_warm = pd.Timestamp(tmin) - pd.DateOffset(days=warmup)
        else:
            tmin_warm = None

        dt = get_dt(freq)

        if istress is None:
            contrib = self.stressmodels[name].simulate(parameters,
                                                       tmin=tmin_warm,
                                                       tmax=tmax,
                                                       freq=freq, dt=dt)
        else:
            contrib = self.stressmodels[name].simulate(parameters,
                                                       tmin=tmin_warm,
                                                       tmax=tmax,
                                                       dt=dt, freq=freq,
                                                       istress=istress)
        # Respect provided tmin/tmax at this point, since warmup matters for
        # simulation but should not be returned, unless return_warmup=True.
        if not return_warmup:
            contrib = contrib.loc[tmin:tmax]

        return contrib

    def get_transform_contribution(self, tmin=None, tmax=None):
        """Method to get the contribution of a transform.

        Parameters
        ----------
        tmin: str or pandas.TimeStamp, optional
        tmax: str or pandas.TimeStamp, optional

        Returns
        -------
        contrib: pandas.Series
            Pandas Series with the contribution.

        """
        sim = self.simulate(tmin=tmin, tmax=tmax)
        # calculate what the simulation without the transform is
        ml = copy(self)
        ml.del_transform()
        sim_org = ml.simulate(tmin=tmin, tmax=tmax)
        return sim - sim_org

    @get_stressmodel
    def get_block_response(self, name, parameters=None, **kwargs):
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

        TODO: Make sure an error is thrown when no rfunc is present.

        """
        if parameters is None:
            parameters = self.get_parameters(name)
        dt = get_dt(self.settings["freq"])
        b = self.stressmodels[name].rfunc.block(parameters, dt, **kwargs)
        t = np.linspace(dt, len(b) * dt, len(b))
        b = pd.Series(b, index=t, name=name)
        b.index.name = "Time [days]"
        return b

    @get_stressmodel
    def get_step_response(self, name, parameters=None, **kwargs):
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

        TODO: Make sure an error is thrown when no rfunc is present.

        """
        if parameters is None:
            parameters = self.get_parameters(name)
        dt = get_dt(self.settings["freq"])
        s = self.stressmodels[name].rfunc.step(parameters, dt, **kwargs)
        t = np.linspace(dt, len(s) * dt, len(s))
        s = pd.Series(s, index=t, name=name)
        s.index.name = "Time [days]"
        return s

    @get_stressmodel
    def get_stress(self, name, istress=None):
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
        stress = self.stressmodels[name].get_stress(p=p, istress=istress)
        return stress

    def get_file_info(self):
        """Internal method to get the file information.

        Returns
        -------
        file_info: dict
            dictionary with file information.

        """
        # Check if file_info already exists
        if hasattr(self, "file_info"):
            file_info = self.file_info
        else:
            file_info = dict()
            file_info["date_created"] = pd.Timestamp.now()

        file_info["date_modified"] = pd.Timestamp.now()
        file_info["pastas_version"] = __version__

        try:
            file_info["owner"] = getlogin()
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
        fname = getenv(env_key, None)
        if not fname or not path.exists(fname):
            dir_path = path.dirname(path.realpath(__file__))
            fname = path.join(dir_path, config_file)
        if path.exists(fname):
            with open(fname, 'rt') as f:
                config_dict = json.load(f)
            config.dictConfig(config_dict)
        else:
            basicConfig(level="INFO")

        logger = getLogger(__name__)
        # Set log_level for console to user-defined value
        if log_level is not None:
            logger.parent.handlers[0].setLevel(log_level)

        return logger

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
        
        if self.fit is None:
            return 'Model is not optimized or read from file. Solve first.'
        
        model = {
            "nfev": self.fit.nfev,
            "nobs": self.oseries_calib.index.size,
            "noise": self.settings["noise"],
            "tmin": str(self.settings["tmin"]),
            "tmax": str(self.settings["tmax"]),
            "freq": self.settings["freq"],
            "warmup": self.settings["warmup"],
            "solver": self.settings["solver"]
        }

        fit = {
            "EVP": "{:.2f}".format(self.stats.evp()),
            "R2": "{:.2f}".format(self.stats.rsq()),
            "RMSE": "{:.2f}".format(self.stats.rmse()),
            "AIC": "{:.2f}".format(self.stats.aic() if
                                   self.settings["noise"] else np.nan),
            "BIC": "{:.2f}".format(self.stats.bic() if
                                   self.settings["noise"] else np.nan),
            "___": "",
            "___ ": "",
            "___  ": ""
        }

        parameters = self.parameters.loc[:,
                     ["optimal", "stderr", "initial", "vary"]]
        parameters.loc[:, "stderr"] = \
            (parameters.loc[:, "stderr"] / parameters.loc[:, "optimal"]) \
                .abs() \
                .apply("\u00B1{:.2%}".format)

        # Determine the width of the fit_report based on the parameters
        width = len(parameters.__str__().split("\n")[1])
        string = "{:{fill}{align}{width}}"

        # Create the first header with model information and stats
        header = "Model Results {name:<16}{string}Fit Statistics\n" \
                 "{line}\n".format(
            name=self.name[:14],
            string=string.format("", fill=' ', align='>', width=width - 44),
            line=string.format("", fill='=', align='>', width=width)
        )

        basic = str()
        for item, item2 in zip(model.items(), fit.items()):
            val1, val2 = item
            val3, val4 = item2
            val4 = string.format(val4, fill=' ', align='>', width=width - 38)
            basic = basic + (
                "{:<8} {:<22} {:<5} {}\n".format(val1, val2, val3, val4))

        # Create the parameters block
        parameters = "\nParameters ({n_param} were optimized)\n{line}\n" \
                     "{parameters}".format(
            n_param=parameters.vary.sum(),
            line=string.format("", fill='=', align='>', width=width),
            parameters=parameters
        )

        # w = []
        #
        # warnings = "Warnings\n{line}\n".format(
        #     "",
        #     line=string.format("", fill='=', align='>', width=width)
        # )
        #
        # for n, warn in enumerate(w, start=1):
        #     warnings = warnings + "[{}] {}\n".format(n, warn)
        #
        # if len(w) == 0:
        #     warnings = ""

        report = "{header}{basic}{parameters}".format(
            header=header, basic=basic, parameters=parameters
        )

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

    def dump_data(self, series=True, sim_series=False, file_info=True):
        """Internal method to export a model to a dictionary.

        Helper function for the self.export method.

        Notes
        -----
        To increase backward compatibility most attributes are stored in
        dictionaries that can be updated when a model is created.

        The following attributes are exported:

        - oseries
        - stressmodeldict
        - noisemodel
        - constant
        - parameters
        - metadata
        - settings
        - ..... future attributes?

        """

        # Create a dictionary to store all data
        data = dict()
        data["name"] = self.name
        data["oseries"] = self.oseries.dump(series=series)

        # Stressmodels
        data["stressmodels"] = dict()
        for name, sm in self.stressmodels.items():
            data["stressmodels"][name] = sm.dump(series=series)

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

        # Simulation Settings
        data["settings"] = self.settings

        # Update and save file information
        if file_info:
            data["file_info"] = self.get_file_info()

        # Export simulated series if necessary
        if sim_series:
            # TODO to_file the simulation, residuals and noise series.
            NotImplementedError()

        return data

    def to_file(self, fname, series=True, **kwargs):
        """Method to to_file the Pastas model to a file.

        Parameters
        ----------
        fname: str
            String with the name and the extension of the file. File
            extension has to be supported by Pastas. E.g. "model.pas"
        series: bool or str, optional
            Export the simulated series or not. If series is "original", the
            original series are exported, if series is "modified",
            the series are exported after being changed with the timeseries
            settings. Default is True.
        kwargs: any argument that is passed to the Model.dump_data() method.

        """

        # Get dicts for all data sources
        data = self.dump_data(series)

        # Write the dicts to a file
        return dump(fname, data, **kwargs)

    def copy(self):
        """Method to copy a model

        Returns
        -------
        ml: pastas.Model instance
            Copy of the original model with no references to the old model.

        """
        data = self.dump_data()
        ml = load_model(data)
        return ml
