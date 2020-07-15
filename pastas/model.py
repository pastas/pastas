"""The Model class is the main object to create a model in Pastas.

.. currentmodule:: pastas.model

.. autoclass:: Model

Examples
--------
A minimal working example of the Model class is shown below:

>>> oseries = pd.Series([1,2,1], index=pd.to_datetime(range(3), unit="D"))
>>> ml = Model(oseries)

.. currentmodule:: pastas.model.Model

General Methods
---------------
.. autosummary::
    :nosignatures:
    :toctree: ./generated

    observations
    residuals
    noise
    simulate
    initialize
    solve
    fit_report
    to_dict
    to_file
    copy
    check_parameters_bounds

Add and Delete Methods
----------------------
The following methods can be used to add model components.

.. autosummary::
    :nosignatures:
    :toctree: ./generated

    add_constant
    add_noisemodel
    add_stressmodel
    add_transform
    del_constant
    del_noisemodel
    del_stressmodel
    del_transform

.. seealso::
    :mod:`pastas.stressmodels`, :mod:`pastas.noisemodels`,
    :mod:`pastas.transform`

Get Methods
-----------
.. autosummary::
    :nosignatures:
    :toctree: ./generated

    get_block_response
    get_step_response
    get_response_tmax
    get_contribution
    get_contributions
    get_transform_contribution
    get_stress
    get_stressmodel_names
    get_init_parameters
    get_parameters
    get_tmax
    get_tmin


Set Methods
-----------
.. autosummary::
    :nosignatures:
    :toctree: ./generated

    set_parameter

"""

from collections import OrderedDict
from logging import getLogger
from os import getlogin

import numpy as np
from pandas import date_range, Series, Timedelta, DataFrame, Timestamp

from .decorators import get_stressmodel
from .io.base import dump, _load_model
from .modelstats import Statistics
from .noisemodels import NoiseModel
from .plots import Plotting
from .solver import LeastSquares
from .stressmodels import Constant
from .timeseries import TimeSeries
from .utils import _get_dt, _get_time_offset, get_sample, \
    frequency_is_supported, validate_name
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

    Returns
    -------
    ml: pastas.model.Model
        Pastas Model instance, the base object in Pastas.

    """

    def __init__(self, oseries, constant=True, noisemodel=True, name=None,
                 metadata=None):

        self.logger = getLogger(__name__)

        # Construct the different model components
        self.oseries = TimeSeries(oseries, settings="oseries",
                                  metadata=metadata)

        if name is None:
            name = self.oseries.name
            if name is None:
                name = 'Observations'
        self.name = validate_name(name)

        self.parameters = DataFrame(
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
            "warmup": Timedelta(3650, "D"),
            "time_offset": Timedelta(0),
            "noise": noisemodel,
            "solver": None,
            "fit_constant": True,
        }

        if constant:
            constant = Constant(initial=self.oseries.series.mean(),
                                name="constant")
            self.add_constant(constant)
        if noisemodel:
            self.add_noisemodel(NoiseModel())

        # File Information
        self.file_info = self._get_file_info()

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

    def add_stressmodel(self, stressmodel, replace=False):
        """Add a stressmodel to the main model.

        Parameters
        ----------
        stressmodel: pastas.stressmodel or list of pastas.stressmodel
            instance of a pastas.stressmodel class. Multiple stress models
            can be provided (e.g., ml.add_stressmodel([sm1, sm2]) in one call.
        replace: bool, optional
            force replace the stressmodel if a stressmodel with the same name
            already exists. Not recommended but useful at times. Default is
            False.

        Notes
        -----
        To obtain a list of the stressmodel names, type:

        >>> ml.get_stressmodel_names()

        Examples
        --------
        >>> sm = ps.StressModel(stress, rfunc=ps.Gamma, name="stress")
        >>> ml.add_stressmodel(sm)

        To add multiple stress models at once you can do the following:

        >>> sm1 = ps.StressModel(stress, rfunc=ps.Gamma, name="stress1")
        >>> sm1 = ps.StressModel(stress, rfunc=ps.Gamma, name="stress2")
        >>> ml.add_stressmodel([sm1, sm2])

        See Also
        --------
        pastas.stressmodels

        """
        # Method can take multiple stressmodels at once through args
        if isinstance(stressmodel, list):
            for sm in stressmodel:
                self.add_stressmodel(sm)
        elif (stressmodel.name in self.stressmodels.keys()) and not replace:
            self.logger.error("The name for the stressmodel you are trying "
                              "to add already exists for this model. Select "
                              "another name.")
        else:
            self.stressmodels[stressmodel.name] = stressmodel
            self.parameters = self.get_init_parameters(initial=False)
            if self.settings["freq"] is None:
                self._set_freq()
            stressmodel.update_stress(freq=self.settings["freq"])

            # Check if stress overlaps with oseries, if not give a warning
            if (stressmodel.tmin > self.oseries.series.index.max()) or \
                    (stressmodel.tmax < self.oseries.series.index.min()):
                self.logger.warning("The stress of the stressmodel has no "
                                    "overlap with ml.oseries.")
        self._check_stressmodel_compatibility()

    def add_constant(self, constant):
        """Add a Constant to the time series Model.

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
        self._check_stressmodel_compatibility()

    def add_transform(self, transform):
        """Add a Transform to the time series Model.

        Parameters
        ----------
        transform: pastas.transform
            instance of a pastas.transform object.

        Examples
        --------
        >>> tt = ps.ThresholdTransform()
        >>> ml.add_transform(tt)

        See Also
        --------
        pastas.transform

        """
        transform.set_model(self)
        self.transform = transform
        self.parameters = self.get_init_parameters(initial=False)
        self._check_stressmodel_compatibility()

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
        self.noisemodel.set_init_parameters(oseries=self.oseries.series)

        # check whether noise_alpha is not smaller than ml.settings["freq"]
        freq_in_days = _get_dt(self.settings["freq"])
        noise_alpha = self.noisemodel.parameters.initial.iloc[0]
        if freq_in_days > noise_alpha:
            self.noisemodel.set_initial("noise_alpha", freq_in_days)

        self.parameters = self.get_init_parameters(initial=False)

    @get_stressmodel
    def del_stressmodel(self, name):
        """Method to safely delete a stress model from the Model.

        Parameters
        ----------
        name: str
            string with the name of the stressmodel object.

        Notes
        -----
        To obtain a list of the stressmodel names type:

        >>> ml.get_stressmodel_names()

        """
        self.stressmodels.pop(name, None)
        self.parameters = self.get_init_parameters(initial=False)

    def del_constant(self):
        """Method to safely delete the Constant from the Model.

        """
        if self.constant is None:
            self.logger.warning("No constant is present in this model.")
        else:
            self.constant = None
            self.parameters = self.get_init_parameters(initial=False)

    def del_transform(self):
        """Method to safely delete the transform from the Model.

        """
        if self.transform is None:
            self.logger.warning("No transform is present in this model.")
        else:
            self.transform = None
            self.parameters = self.get_init_parameters(initial=False)

    def del_noisemodel(self):
        """Method to safely delete the noise model from the Model.

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
            String with a start date for the simulation period (E.g. '1980').
            If none is provided, the tmin from the oseries is used.
        tmax: str, optional
            String with an end date for the simulation period (E.g. '2010').
            If none is provided, the tmax from the oseries is used.
        freq: str, optional
            String with the frequency the stressmodels are simulated. Must
            be one of the following: (D, h, m, s, ms, us, ns) or a multiple of
            that e.g. "7D".
        warmup: float or int, optional
            Warmup period (in Days).
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
        elif not isinstance(warmup, Timedelta):
            warmup = Timedelta(warmup, "D")

        # Get the simulation index and the time step
        sim_index = self._get_sim_index(tmin, tmax, freq, warmup)
        dt = _get_dt(freq)

        # Get parameters if none are provided
        if parameters is None:
            parameters = self.get_parameters()

        sim = Series(data=np.zeros(sim_index.size, dtype=float),
                     index=sim_index, fastpath=True)

        istart = 0  # Track parameters index to pass to stressmodel object
        for sm in self.stressmodels.values():
            contrib = sm.simulate(parameters[istart: istart + sm.nparam],
                                  sim_index.min(), sim_index.max(), freq, dt)
            sim = sim.add(contrib)
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
            String with a start date for the simulation period (E.g. '1980').
            If none is provided, the tmin from the oseries is used.
        tmax: str, optional
            String with an end date for the simulation period (E.g. '2010').
            If none is provided, the tmax from the oseries is used.
        freq: str, optional
            String with the frequency the stressmodels are simulated. Must
            be one of the following: (D, h, m, s, ms, us, ns) or a multiple of
            that e.g. "7D".
        warmup: float or int, optional
            Warmup period (in Days).

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
        else:
            warmup = Timedelta(warmup, "D")

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
            # interpolate simulation to times of observations
            sim_interpolated = np.interp(oseries_calib.index.asi8,
                                         sim.index.asi8, sim.values)
        else:
            # all of the observation indexes are in the simulation
            sim_interpolated = sim.reindex(oseries_calib.index)

        # Calculate the actual residuals here
        res = oseries_calib.subtract(sim_interpolated)

        if res.hasnans:
            res = res.dropna()
            self.logger.warning('Nan-values were removed from the residuals.')

        if self.normalize_residuals:
            res = res.subtract(res.values.mean())

        res.name = "Residuals"
        return res

    def noise(self, parameters=None, tmin=None, tmax=None, freq=None,
              warmup=None, weights=True):
        """Method to simulate the noise when a noisemodel is present.

        Parameters
        ----------
        parameters: list, optional
            Array of the parameters used in the time series model. See
            Model.get_parameters() for more info if parameters is None.
        tmin: str, optional
            String with a start date for the simulation period (E.g. '1980').
            If none is provided, the tmin from the oseries is used.
        tmax: str, optional
            String with an end date for the simulation period (E.g. '2010').
            If none is provided, the tmax from the oseries is used.
        freq: str, optional
            String with the frequency the stressmodels are simulated. Must
            be one of the following: (D, h, m, s, ms, us, ns) or a multiple of
            that e.g. "7D".
        warmup: float or int, optional
            Warmup period (in Days).

        Returns
        -------
        noise : pandas.Series
            Pandas series of the noise.

        Notes
        -----
        The noise are the time series that result when applying a noise
        model.

        .. Note::
            The noise is sometimes also referred to as the innovations.

        Warnings
        --------
        This method returns None is no noise model is added to the model.

        """
        if (self.noisemodel is None) or (self.settings["noise"] is False):
            self.logger.error("Noise cannot be calculated if there is no "
                              "noisemodel present or is not used during "
                              "parameter estimation.")
            return None

        # Get parameters if none are provided
        if parameters is None:
            parameters = self.get_parameters()

        # Calculate the residuals
        res = self.residuals(parameters, tmin, tmax, freq, warmup)
        p = parameters[-self.noisemodel.nparam:]

        # Calculate the noise
        noise = self.noisemodel.simulate(res, p)
        # Calculate the weights
        if weights:
            self.logger.warning("The default argument for weights will be "
                                "changed to weights=False in a future "
                                "version of Pastas.")
            weights = self.noisemodel.weights(res, p)
            noise = noise * weights
            noise.name = "Weighted Noise"

        return noise

    def noise_weights(self, parameters=None, tmin=None, tmax=None, freq=None,
                      warmup=None):
        """ Internal method to calculate the noise weights."""
        # Get parameters if none are provided
        if parameters is None:
            parameters = self.get_parameters()

        # Calculate the residuals
        res = self.residuals(parameters, tmin, tmax, freq, warmup)

        # Calculate the weights
        weights = self.noisemodel.weights(res,
                                          parameters[-self.noisemodel.nparam:])

        return weights

    def observations(self, tmin=None, tmax=None, freq=None,
                     update_observations=False):
        """Method that returns the observations series used for calibration.

        Parameters
        ----------
        tmin: str, optional
            String with a start date for the simulation period (E.g. '1980').
            If none is provided, the tmin from the oseries is used.
        tmax: str, optional
            String with an end date for the simulation period (E.g. '2010').
            If none is provided, the tmax from the oseries is used.
        freq: str, optional
            String with the frequency the stressmodels are simulated. Must
            be one of the following: (D, h, m, s, ms, us, ns) or a multiple of
            that e.g. "7D".
        update_observations: bool, optional
            if True, force recalculation of the observations series, default
            is False.

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

        for key, setting in zip([tmin, tmax, freq], ["tmin", "tmax", "freq"]):
            if key != self.settings[setting]:
                update_observations = True

        if self.oseries_calib is None or update_observations:
            oseries_calib = self.oseries.series.loc[tmin:tmax]

            # sample measurements, so that frequency is not higher than model
            # keep the original timestamps, as they will be used during
            # interpolation of the simulation
            sim_index = self._get_sim_index(tmin, tmax, freq,
                                            self.settings["warmup"])
            if not oseries_calib.empty:
                index = get_sample(oseries_calib.index, sim_index)
                oseries_calib = oseries_calib.loc[index]
        else:
            oseries_calib = self.oseries_calib
        return oseries_calib

    def initialize(self, tmin=None, tmax=None, freq=None, warmup=None,
                   noise=None, weights=None, initial=True, fit_constant=True):
        """Method to initialize the model.

        This method is called by the solve-method, but can also be triggered
        manually. See the solve-method for a description of the arguments.

        """
        if noise is None and self.noisemodel:
            noise = True
        elif noise is True and self.noisemodel is None:
            self.logger.warning("Warning, solving with noise=True while no "
                                "noisemodel is present. noise set to False")
            noise = False

        self.settings["noise"] = noise
        self.settings["weights"] = weights
        self.settings["fit_constant"] = fit_constant

        # Set the frequency & warmup
        if freq:
            self.settings["freq"] = frequency_is_supported(freq)

        if warmup is not None:
            self.settings["warmup"] = Timedelta(warmup, "D")

        # Set time offset from the frequency (does not work as expected yet)
        # self._set_time_offset()

        # Set tmin and tmax
        self.settings["tmin"] = self.get_tmin(tmin)
        self.settings["tmax"] = self.get_tmax(tmax)

        # make sure calibration data is renewed
        self.sim_index = self._get_sim_index(self.settings["tmin"],
                                             self.settings["tmax"],
                                             self.settings["freq"],
                                             self.settings["warmup"],
                                             update_sim_index=True)
        self.oseries_calib = self.observations(tmin=self.settings["tmin"],
                                               tmax=self.settings["tmax"],
                                               freq=self.settings["freq"],
                                               update_observations=True)
        self.interpolate_simulation = None

        # Initialize parameters
        self.parameters = self.get_init_parameters(noise, initial)

        # Prepare model if not fitting the constant as a parameter
        if self.settings["fit_constant"] is False:
            self.parameters.loc["constant_d", "vary"] = False
            self.parameters.loc["constant_d", "initial"] = 0.0
            self.normalize_residuals = True

    def solve(self, tmin=None, tmax=None, freq=None, warmup=None, noise=True,
              solver=None, report=True, initial=True, weights=None,
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
        freq: str, optional
            String with the frequency the stressmodels are simulated. Must
            be one of the following (D, h, m, s, ms, us, ns) or a multiple of
            that e.g. "7D".
        warmup: float or int, optional
            Warmup period (in Days) for which the simulation is calculated,
            but not used for the calibration period.
        noise: bool, optional
            Argument that determines if a noisemodel is used (only if
            present). The default is noise=True.
        solver: pastas.solver.BaseSolver class, optional
            Class used to solve the model. Options are: ps.LeastSquares
            (default) or ps.LmfitSolve. A class is needed, not an instance
            of the class!
        report: bool, optional
            Print a report to the screen after optimization finished. This
            can also be manually triggered after optimization by calling
            print(ml.fit_report()) on the Pastas model instance.
        initial: bool, optional
            Reset initial parameters from the individual stress models.
            Default is True. If False, the optimal values from an earlier
            optimization are used.
        weights: pandas.Series, optional
            Pandas Series with values by which the residuals are multiplied,
            index-based. Must have the same indices as the oseries.
        fit_constant: bool, optional
            Argument that determines if the constant is fitted as a parameter.
            If it is set to False, the constant is set equal to the mean of
            the residuals.
        **kwargs: dict, optional
            All keyword arguments will be passed onto minimization method
            from the solver. It depends on the solver used which arguments
            can be used.

        Notes
        -----
        - The solver object including some results are stored as ml.fit.
          From here one can access the covariance (ml.fit.pcov) and
          correlation matrix (ml.fit.pcor).
        - Each solver return a number of results after optimization. These
          solver specific results are stored in ml.fit.result and can be
          accessed from there.

        See Also
        --------
        pastas.solver
            Different solver objects are available to estimate parameters.

        """

        # Initialize the model
        self.initialize(tmin, tmax, freq, warmup, noise, weights, initial,
                        fit_constant)

        if self.oseries_calib.empty:
            raise ValueError("Calibration series 'oseries_calib' is empty! "
                             "Check 'tmin' or 'tmax'.")

        # Store the solve instance
        if solver is None:
            if self.fit is None:
                self.fit = LeastSquares(ml=self)
        elif not issubclass(solver, self.fit.__class__):
            self.fit = solver(ml=self)

        self.settings["solver"] = self.fit._name

        # Solve model
        success, optimal, stderr = self.fit.solve(noise=noise, weights=weights,
                                                  **kwargs)
        if not success:
            self.logger.warning("Model parameters could not be estimated "
                                "well.")

        if self.settings['fit_constant'] is False:
            # Determine the residuals and set the constant to their mean
            self.normalize_residuals = False
            res = self.residuals(optimal).mean()
            optimal[self.parameters.name == self.constant.name] = res

        self.parameters.optimal = optimal
        self.parameters.stderr = stderr

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

        Examples
        --------
        >>> ml.set_initial("constant_d", 10)

        """
        msg = "Deprecation warning: method is deprecated and will be removed" \
              " in version 0.16.0. Use ml.set_parameter instead."
        self.logger.warning(msg)
        self.set_parameter(name, initial=value, move_bounds=move_bounds)

    def set_vary(self, name, value):
        """Method to set if the parameter is allowed to vary.

        Parameters
        ----------
        name: str
            name of the parameter to update.
        value: bool
            boolean to vary a parameter (True) or not (False).

        Examples
        --------
        >>> ml.set_vary("constant_d", False)

        """
        msg = "Deprecation warning: method is deprecated and will be removed" \
              " in version 0.16.0. Use ml.set_parameter instead."
        self.logger.warning(msg)
        self.set_parameter(name, vary=bool(value))

    def set_pmin(self, name, value):
        """Method to set the minimum value of a parameter.

        Parameters
        ----------
        name: str
            name of the parameter to update.
        value: float
            minimum value for the parameter.

        Examples
        --------
        >>> ml.set_pmin("constant_d", -10)

        """
        msg = "Deprecation warning: method is deprecated and will be removed" \
              " in version 0.16.0. Use ml.set_parameter instead."
        self.logger.warning(msg)
        self.set_parameter(name, pmin=value)

    def set_pmax(self, name, value):
        """Method to set the maximum values of a parameter.

        Parameters
        ----------
        name: str
            name of the parameter to update.
        value: float
            maximum value for the parameter.

        Examples
        --------
        >>> ml.set_pmax("constant_d", 10)

        """
        msg = "Deprecation warning: method is deprecated and will be removed" \
              " in version 0.16.0. Use ml.set_parameter instead."
        self.logger.warning(msg)
        self.set_parameter(name, pmax=value)

    def set_parameter(self, name, initial=None, vary=None, pmin=None,
                      pmax=None, move_bounds=False):
        """
        Method to change the parameter properties.

        Parameters
        ----------
        name: str
            name of the parameter to update. This has to be a single variable.
        initial: float, optional
            parameters value to use as initial estimate.
        vary: bool, optional
            boolean to vary a parameter (True) or not (False).
        pmin: float, optional
            minimum value for the parameter.
        pmax: float, optional
            maximum value for the parameter.
        move_bounds: bool, optional
            Reset pmin/pmax based on new initial value. Of move_bounds=True,
            pmin and pmax must be None.

        Examples
        --------
        >>> ml.set_parameter(name="constant_d", initial=10, vary=True,
        >>>                  pmin=-10, pmax=20)

        Note
        ----
        It is highly recommended to use this method to set parameter
        properties. Changing the parameter properties directly in the
        parameter `DataFrame` may not work as expected.

        """
        if name not in self.parameters.index:
            msg = "parameter {} is not present in the model".format(name)
            self.logger.error(msg)
            raise KeyError(msg)

        # Because either of the following is not necessarily present
        noisemodel = self.noisemodel.name if self.noisemodel else "NotPresent"
        constant = self.constant.name if self.constant else "NotPresent"
        transform = self.transform.name if self.transform else "NotPresent"

        # Get the model component for the parameter
        cat = self.parameters.loc[name, "name"]

        if cat in self.stressmodels.keys():
            obj = self.stressmodels[cat]
        elif cat == noisemodel:
            obj = self.noisemodel
        elif cat == constant:
            obj = self.constant
        elif cat == transform:
            obj = self.transform

        # Move pmin and pmax based on the initial
        if move_bounds and initial:
            if pmin or pmax:
                raise KeyError("Either pmin/pmax or move_bounds must "
                               "be provided, but not both.")
            factor = initial / self.parameters.loc[name, 'initial']
            pmin = self.parameters.loc[name, 'pmin'] * factor
            pmax = self.parameters.loc[name, 'pmax'] * factor

        # Set the parameter properties
        if initial is not None:
            obj.set_initial(name, initial)
            self.parameters.loc[name, "initial"] = initial
        if vary is not None:
            obj.set_vary(name, vary)
            self.parameters.loc[name, "vary"] = bool(vary)
        if pmin is not None:
            obj.set_pmin(name, pmin)
            self.parameters.loc[name, "pmin"] = pmin
        if pmax is not None:
            obj.set_pmax(name, pmax)
            self.parameters.loc[name, "pmax"] = pmax

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
                            # if this is not available, and the original
                            # frequency is, take the original frequency
                            freqs.add(stress.freq_original)

        if len(freqs) == 1:
            # if there is only one frequency, use this frequency
            self.settings["freq"] = next(iter(freqs))
        elif len(freqs) > 1:
            # if there are more frequencies, take the highest (lowest dt)
            freqs = list(freqs)
            dt = np.array([_get_dt(f) for f in freqs])
            self.settings["freq"] = freqs[np.argmin(dt)]
        else:
            self.logger.info("Frequency of model cannot be determined. "
                             "Frequency is set to daily")
            self.settings["freq"] = "D"

    def _set_time_offset(self):
        """Internal method to set the time offset for the model class.

        Notes
        -----
        Method to check if the StressModel timestamps match
        (e.g. similar hours)

        """
        time_offsets = set()
        for stressmodel in self.stressmodels.values():
            for st in stressmodel.stress:
                if st.freq_original:
                    # calculate the offset from the default frequency
                    time_offset = _get_time_offset(
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
            self.settings["time_offset"] = Timedelta(0)

    def get_stressmodel_names(self):
        """Returns list of stressmodel names"""
        return list(self.stressmodels.keys())

    def _get_sim_index(self, tmin, tmax, freq, warmup, update_sim_index=False):
        """Internal method to get the simulation index, including the warmup.

        Parameters
        ----------
        tmin: pandas.Timestamp
            String with a start date for the simulation period (E.g. '1980').
            If none is provided, the tmin from the oseries is used.
        tmax: pandas.Timestamp
            String with an end date for the simulation period (E.g. '2010').
            If none is provided, the tmax from the oseries is used.
        freq: str
            String with the frequency the stressmodels are simulated. Must
            be one of the following: (D, h, m, s, ms, us, ns) or a multiple of
            that e.g. "7D".
        warmup: pandas.Timedelta
            Warmup period (in Days).
        update_sim_index : bool, optional
            if True, force recalculation of sim_index, default is False

        Returns
        -------
        sim_index: pandas.DatetimeIndex
            Pandas DatetimeIndex instance with the datetimes values for
            which the model is simulated.

        """
        # Check if any of the settings are updated
        for key, setting in zip([tmin, tmax, freq, warmup],
                                ["tmin", "tmax", "freq", "warmup"]):
            if key != self.settings[setting]:
                update_sim_index = True
                break

        if self.sim_index is None or update_sim_index:
            tmin = (tmin - warmup).floor(freq) + self.settings["time_offset"]
            sim_index = date_range(tmin, tmax, freq=freq)
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

            1. If use_oseries is True, tmin is based on the oseries
            2. If use_stresses is True, tmin is based on the stressmodels.

        B. If a values for tmin is provided:

            1. A pandas timestamp is made from the string
            2. if use_oseries is True, tmin is checked against oseries.

        C. In all cases an offset for the tmin is added.

        """
        # Get tmin from the oseries
        if use_oseries:
            ts_tmin = self.oseries.series.index.min()
        # Get tmin from the stressmodels
        elif use_stresses:
            ts_tmin = Timestamp.max
            for stressmodel in self.stressmodels.values():
                if stressmodel.tmin < ts_tmin:
                    ts_tmin = stressmodel.tmin
        # Get tmin and tmax from user provided values
        else:
            ts_tmin = Timestamp(tmin)

        # Set tmin properly
        if tmin is not None and use_oseries:
            tmin = max(Timestamp(tmin), ts_tmin)
        elif tmin is not None:
            tmin = Timestamp(tmin)
        else:
            tmin = ts_tmin

        # adjust tmin and tmax so that the time-offset is equal to the stressmodels.
        if freq is None:
            freq = self.settings["freq"]
        tmin = tmin.floor(freq) + self.settings["time_offset"]

        return tmin

    def get_tmax(self, tmax=None, freq=None, use_oseries=True,
                 use_stresses=False):
        """Method that checks and returns valid values for tmax.

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

            1. If use_oseries is True, tmax is based on the oseries.
            2. If use_stresses is True, tmax is based on the stressmodels.

        B. If a values for tmax is provided:

            1. A pandas timestamp is made from the string.
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
            ts_tmax = Timestamp.min
            for stressmodel in self.stressmodels.values():
                if stressmodel.tmax > ts_tmax:
                    ts_tmax = stressmodel.tmax
        # Get tmax from user provided values
        else:
            ts_tmax = Timestamp(tmax)

        # Set tmax properly
        if tmax is not None and use_oseries:
            tmax = min(Timestamp(tmax), ts_tmax)
        elif tmax is not None:
            tmax = Timestamp(tmax)
        else:
            tmax = ts_tmax

        # adjust tmax so that the time-offset is equal to the stressmodels.
        if freq is None:
            freq = self.settings["freq"]
        tmax = tmax.floor(freq) + self.settings["time_offset"]

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

        parameters = DataFrame(columns=self.parameters.columns)
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
        """Method to obtain the parameters needed for calculation.

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
            p = self.parameters.loc[self.parameters.name == name]
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

        Parameters
        ----------
        name: str
            String with the name of the stressmodel.
        tmin: str, optional
            String with a start date for the simulation period (E.g. '1980').
            If none is provided, the tmin from the oseries is used.
        tmax: str, optional
            String with an end date for the simulation period (E.g. '2010').
            If none is provided, the tmax from the oseries is used.
        freq: str, optional
            String with the frequency the stressmodels are simulated. Must
            be one of the following: (D, h, m, s, ms, us, ns) or a multiple of
            that e.g. "7D".
        warmup: float or int, optional
            Warmup period (in Days).
        istress: int, optional
            When multiple stresses are present in a stressmodel, this keyword
            can be used to obtain the contribution of an individual stress.
        return_warmup: bool, optional
            Include warmup in contribution calculation or not.
        parameters: list or numpy.ndarray
            iterable with the parameters. If none, the optimal parameters are
            used when available, initial otherwise.

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
        else:
            warmup = Timedelta(warmup, "D")

        # use warmup
        if tmin:
            tmin_warm = Timestamp(tmin) - warmup
        else:
            tmin_warm = None

        dt = _get_dt(freq)

        kwargs = {'tmin': tmin_warm, 'tmax': tmax, 'freq': freq, 'dt': dt}
        if istress is not None:
            kwargs['istress'] = istress
        contrib = self.stressmodels[name].simulate(parameters, **kwargs)

        # Respect provided tmin/tmax at this point, since warmup matters for
        # simulation but should not be returned, unless return_warmup=True.
        if not return_warmup:
            contrib = contrib.loc[tmin:tmax]

        return contrib

    def get_contributions(self, split=True, **kwargs):
        """Method to get contributions of all stressmodels.

        Parameters
        ----------
        split: bool, optional
            Split the stresses in multiple stresses when possible.
        kwargs: any other arguments are passed to get_contribution

        Returns
        -------
        contribs: list
            a list of Pandas Series of the contributions.

        See Also
        --------
        pastas.model.Model.get_contribution
            This method is called to get the individual contributions,
            kwargs are passed on to this method.

        """
        contribs = []
        for name in self.stressmodels:
            nsplit = self.stressmodels[name].get_nsplit()
            if split and nsplit > 1:
                for istress in range(nsplit):
                    contrib = self.get_contribution(name, istress=istress,
                                                    **kwargs)
                    contribs.append(contrib)
            else:
                contrib = self.get_contribution(name, **kwargs)
                contribs.append(contrib)
        return contribs

    def get_transform_contribution(self, tmin=None, tmax=None):
        """Method to get the contribution of a transform.

        Parameters
        ----------
        tmin: str, optional
            String with a start date for the simulation period (E.g. '1980').
            If none is provided, the tmin from the oseries is used.
        tmax: str, optional
            String with an end date for the simulation period (E.g. '2010').
            If none is provided, the tmax from the oseries is used.

        Returns
        -------
        contrib: pandas.Series
            Pandas Series with the contribution.

        """
        sim = self.simulate(tmin=tmin, tmax=tmax)
        # calculate what the simulation without the transform is
        ml = self.copy()
        ml.del_transform()
        sim_org = ml.simulate(tmin=tmin, tmax=tmax)
        return sim - sim_org

    def _get_response(self, block_or_step, name, parameters=None, dt=None,
                      add_0=False, **kwargs):
        """Internal method to compute the block and step response.

        Parameters
        ----------
        block_or_step: str
            String with "step" or "block"
        name: str
            string with the name of the stressmodel
        parameters: ndarray, optional
            array with the parameters
        dt: float, optional
            timestep for the response function.
        add_0: bool, optional
            Add a zero at t=0.
        kwargs

        Returns
        -------
        response: pandas.Series

        """
        if self.stressmodels[name].rfunc is None:
            self.logger.warning("Stressmodel {} has no rfunc".format(name))
            return None
        else:
            block_or_step = getattr(self.stressmodels[name].rfunc,
                                    block_or_step)

        if parameters is None:
            parameters = self.get_parameters(name)

        if dt is None:
            dt = _get_dt(self.settings["freq"])
        response = block_or_step(parameters, dt, **kwargs)

        if add_0:
            response = np.insert(response, 0, 0.0)

        if isinstance(dt, np.ndarray):
            t = dt
        else:
            t = np.linspace(dt, response.size * dt, response.size)
        response = Series(response, index=t, name=name)
        response.index.name = "Time [days]"

        return response

    @get_stressmodel
    def get_block_response(self, name, parameters=None, add_0=False, dt=None,
                           **kwargs):
        """Method to obtain the block response for a stressmodel.

        The optimal parameters are used when available, initial otherwise.

        Parameters
        ----------
        name: str
            String with the name of the stressmodel.
        parameters: list or numpy.ndarray
            iterable with the parameters. If none, the optimal parameters are
            used when available, initial otherwise.
        add_0: bool, optional
            Adds 0 at t=0 at the start of the response, defaults to False.
        dt: float, optional
            timestep for the response function.

        Returns
        -------
        b: pandas.Series
            Pandas.Series with the block response. The index is based on the
            frequency that is present in the model.settings.

        """
        return self._get_response(block_or_step="block", name=name, dt=dt,
                                  parameters=parameters, add_0=add_0, **kwargs)

    @get_stressmodel
    def get_step_response(self, name, parameters=None, add_0=False, dt=None,
                          **kwargs):
        """Method to obtain the step response for a stressmodel.

        The optimal parameters are used when available, initial otherwise.

        Parameters
        ----------
        name: str
            String with the name of the stressmodel.
        parameters: list or numpy.ndarray, optional
            iterable with the parameters. If none, the optimal parameters are
            used when available, initial otherwise.
        add_0: bool, optional
            Adds 0 at t=0 at the start of the response, defaults to False.
        dt: float, optional
            timestep for the response function.

        Returns
        -------
        s: pandas.Series
            Pandas.Series with the step response. The index is based on the
            frequency that is present in the model.settings.

        """
        return self._get_response(block_or_step="step", name=name, dt=dt,
                                  parameters=parameters, add_0=add_0, **kwargs)

    def get_response_tmax(self, name, parameters=None, cutoff=0.999):
        """Method to get the tmax used for the response function.

        Parameters
        ----------
        name: str
            String with the name of the stressmodel.
        parameters: list or numpy.ndarray, optional
            iterable with the parameters. If none, the optimal parameters are
            used when available, initial otherwise.
        cutoff: float, optional
            float between 0 and 1. Default is 0.999 or 99.9% of the response.

        Returns
        -------
        tmax: float
            Float with the number of days.

        Example
        -------
        >>> ml.get_response_tmax("recharge", cutoff=0.99)
        >>> 703

        This means that after 1053 days, 99% of the response of the
        groundwater levels to a recharge pulse has taken place.

        """
        if self.stressmodels[name].rfunc is None:
            self.logger.warning("Stressmodel {} has no rfunc".format(name))
            return None
        else:
            if parameters is None:
                parameters = self.get_parameters(name)
            tmax = self.stressmodels[name].rfunc.get_tmax(p=parameters,
                                                          cutoff=cutoff)
            return tmax

    @get_stressmodel
    def get_stress(self, name, tmin=None, tmax=None, freq=None, warmup=None,
                   istress=None, return_warmup=False, parameters=None):
        """Method to obtain the stress(es) from the stressmodel.

        Parameters
        ----------
        name: str
            String with the name of the stressmodel.
        istress: int, optional
            When multiple stresses are present in a stressmodel, this keyword
            can be used to obtain the contribution of an individual stress.

        Returns
        -------
        stress: pandas.Series or list of pandas.Series
            If one stress is present, a pandas Series is returned. If more
            are present, a list of pandas Series is returned.

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
        else:
            warmup = Timedelta(warmup, "D")

        # use warmup
        if tmin:
            tmin_warm = Timestamp(tmin) - warmup
        else:
            tmin_warm = None

        kwargs = {"tmin": tmin_warm, "tmax": tmax, "freq": freq}
        if istress is not None:
            kwargs["istress"] = istress

        stress = self.stressmodels[name].get_stress(p=parameters, **kwargs)
        if not return_warmup:
            stress = stress.loc[tmin:tmax]

        return stress

    def _get_file_info(self):
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
            file_info = {"date_created": Timestamp.now()}

        file_info["date_modified"] = Timestamp.now()
        file_info["pastas_version"] = __version__

        try:
            file_info["owner"] = getlogin()
        except:
            file_info["owner"] = "Unknown"

        return file_info

    def fit_report(self, output="full"):
        """Method that reports on the fit after a model is optimized.

        Parameters
        ----------
        output: str, optional
            If any other value than "full" is provided, the parameter
            correlations will be removed from the output.

        Returns
        -------
        report: str
            String with the report.

        Examples
        --------
        This method is called by the solve method if report=True, but can
        also be called on its own::

        >>> print(ml.fit_report)

        """
        model = {
            "nfev": self.fit.nfev,
            "nobs": self.observations().index.size,
            "noise": str(self.settings["noise"]),
            "tmin": str(self.settings["tmin"]),
            "tmax": str(self.settings["tmax"]),
            "freq": self.settings["freq"],
            "warmup": str(self.settings["warmup"]),
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
            "Obj": "{:.2f}".format(self.fit.obj_func),
            "___": "", "___ ": "",  # Make columns equal
        }

        parameters = self.parameters.loc[:, ["optimal", "stderr",
                                             "initial", "vary"]]
        stderr = parameters.loc[:, "stderr"] / parameters.loc[:, "optimal"]
        parameters.loc[:, "stderr"] = stderr.abs().apply("\u00B1{:.2%}".format)

        # Determine the width of the fit_report based on the parameters
        width = len(parameters.__str__().split("\n")[1])
        string = "{:{fill}{align}{width}}"

        # Create the first header with model information and stats
        w = max(width - 44, 0)
        header = "Model Results {name:<16}{string}Fit Statistics\n" \
                 "{line}\n".format(
            name=self.name[:14],
            string=string.format("", fill=' ', align='>', width=w),
            line=string.format("", fill='=', align='>', width=width)
        )

        basic = ""
        for (val1, val2), (val3, val4) in zip(model.items(), fit.items()):
            w = max(width - 38, 0)
            val4 = string.format(val4, fill=' ', align='>', width=w)
            basic = basic + "{:<8} {:<22} {:<5} {}\n".format(val1, val2,
                                                             val3, val4)

        # Create the parameters block
        parameters = "\nParameters ({n_param} were optimized)\n{line}\n" \
                     "{parameters}".format(
            n_param=parameters.vary.sum(),
            line=string.format(
                "", fill='=', align='>', width=width),
            parameters=parameters)

        if output == "full":
            cor = {}
            pcor = self.fit.pcor
            for idx in pcor:
                for col in pcor:
                    if (np.abs(pcor.loc[idx, col]) > 0.5) and (idx != col) \
                            and ((col, idx) not in cor.keys()):
                        cor[(idx, col)] = pcor.loc[idx, col].round(2)

            cor = DataFrame(data=cor.values(), index=cor.keys(),
                            columns=["rho"])
            correlations = "\n\nParameter correlations |rho| > 0.5\n{}" \
                           "\n{}".format(string.format("", fill='=', align='>',
                                                       width=width),
                                         cor.to_string(header=False))
        else:
            correlations = ""

        report = "{header}{basic}{parameters}{correlations}".format(
            header=header, basic=basic, parameters=parameters,
            correlations=correlations)

        return report

    def _check_parameters_bounds(self, alpha=0.01):
        """Internal method toCheck if the optimal parameters are close to
        pmin or pmax.

        Parameters
        ----------
        alpha: float, optional
            value between 0 and 1 to determine if the parameters is close to
            the maximum or minimum is determined as the percentage of the
            parameter range.

        Returns
        -------
        pmin: pandas.Series
            pandas series with boolean values of the parameters that are
            close to the minimum values.
        pmax: pandas.Series
            pandas series with boolean values of the parameters that are
            close to the maximum values.

        """
        prange = self.parameters.pmax - self.parameters.pmin
        pnorm = (self.parameters.optimal - self.parameters.pmin) / prange
        pmax = pnorm > 1 - alpha
        pmin = pnorm < alpha
        return pmin, pmax

    def to_dict(self, series=True, file_info=True):
        """Method to export a model to a dictionary.

        Parameters
        ----------
        series: bool, optional
            True to export the time series (default), False to not export them.
        file_info: bool, optional
            Export file_info or not. See method Model.get_file_info

        Notes
        -----
        Helper function for the self.to_file method. To increase backward
        compatibility most attributes are stored in dictionaries that can be
        updated when a model is created.

        """

        # Create a dictionary to store all data
        data = {"name": self.name,
                "oseries": self.oseries.to_dict(series=series),
                "parameters": self.parameters,
                "settings": self.settings,
                "stressmodels": dict()}

        # Stressmodels
        for name, sm in self.stressmodels.items():
            data["stressmodels"][name] = sm.to_dict(series=series)

        # Constant
        if self.constant:
            data["constant"] = True

        # Transform
        if self.transform:
            data["transform"] = self.transform.to_dict()

        # Noisemodel
        if self.noisemodel:
            data["noisemodel"] = self.noisemodel.to_dict()

        # Solver object
        if self.fit:
            data["fit"] = self.fit.to_dict()

        # Update and save file information
        if file_info:
            data["file_info"] = self._get_file_info()

        return data

    def to_file(self, fname, series=True, **kwargs):
        """Method to save the Pastas model to a file.

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
        **kwargs:
            any argument that is passed to :mod:`pastas.io.dump`.

        See Also
        --------
        :mod:`pastas.io.dump`

        """

        # Get dicts for all data sources
        data = self.to_dict(series=series)

        # Write the dicts to a file
        return dump(fname, data, **kwargs)

    def copy(self, name=None):
        """Method to copy a model.

        Parameters
        ----------
        name: str, optional
            String with the name of the model. The old name plus is appended
            with '_copy' if no name is provided.

        Returns
        -------
        ml: pastas.model.Model instance
            Copy of the original model with no references to the old model.

        Examples
        --------
        >>> ml_copy = ml.copy(name="new_name")

        """
        if name is None:
            name = self.name + "_copy"
        ml = _load_model(self.to_dict())
        ml.name = name
        return ml

    def _check_stressmodel_compatibility(self):
        """Internal method to check if the stressmodels are compatible with the
        model."""
        for sm in self.stressmodels.values():
            if hasattr(sm, '_check_stressmodel_compatibility'):
                sm._check_stressmodel_compatibility(self)
