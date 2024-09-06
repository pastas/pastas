#  This module contains the Model class in Pastas.

# Python Dependencies
from collections import OrderedDict
from itertools import combinations
from logging import getLogger
from os import getlogin

# Type Hinting
from typing import List, Optional, Tuple, Union

# External Dependencies
import numpy as np
from pandas import (
    DataFrame,
    DatetimeIndex,
    Series,
    Timedelta,
    Timestamp,
    concat,
    date_range,
)

# Internal Pastas
from pastas.decorators import (
    PastasDeprecationWarning,
    deprecate_args_or_kwargs,
    get_stressmodel,
)
from pastas.io.base import _load_model, dump
from pastas.modelstats import Statistics
from pastas.plotting.modelplots import Plotting, _table_formatter_stderr
from pastas.rfunc import HantushWellModel
from pastas.solver import LeastSquares
from pastas.stressmodels import Constant
from pastas.timeseries import TimeSeries
from pastas.timeseries_utils import (
    _frequency_is_supported,
    _get_dt,
    _get_time_offset,
    get_sample,
)
from pastas.transform import ThresholdTransform
from pastas.typing import ArrayLike, Solver, StressModel, TimestampType
from pastas.typing import Model as ModelType
from pastas.typing import NoiseModel as NoiseModelType
from pastas.utils import validate_name
from pastas.version import __version__

logger = getLogger(__name__)


class Model:
    """Class that initiates a Pastas time series model.

    Parameters
    ----------
    oseries: pandas.Series
        pandas.Series object containing the dependent time series. The observation
        can be non-equidistant.
    constant: bool, optional
        Add a constant to the model (Default=True).
    noisemodel: bool, optional
        The noisemodel argument is deprecated and will be removed in Pastas version
        2.0.0. To add a noisemodel, use ml.add_noisemodel(n), where is an instance
        of a noisemodel (e.g., n = ps.ArNoiseModel()). The use of the noisemodel
        argument will raise a ValueError.
    name: str, optional
        String with the name of the model, used in plotting and saving.
    metadata: dict, optional
        Dictionary containing metadata of the oseries, passed on to the oseries when
        creating a pastas TimeSeries object. hence, ml.oseries.metadata will give you
        the metadata.
    freq: str, optional
        String with the frequency the stressmodels are simulated. Must be one of the
        following: (D, h, m, s, ms, us, ns) or a multiple of that e.g. "7D". Default
        is "D". New in 0.18.0.

    Returns
    -------
    ml: pastas.model.Model
        Pastas Model instance, the base object in Pastas.

    Examples
    --------
    A minimal working example of the Model class is shown below:

    >>> oseries = pd.Series([1,2,1], index=pd.to_datetime(range(3), unit="D"))
    >>> ml = Model(oseries)
    """

    _accessors = set()

    def __init__(
        self,
        oseries: Series,
        constant: bool = True,
        noisemodel=None,  # will be removed in version 2.0.0
        name: Optional[str] = None,
        metadata: Optional[dict] = None,
        freq: str = "D",
    ) -> None:
        # Construct the different model components
        self.oseries = TimeSeries(oseries, settings="oseries", metadata=metadata)

        if name is None and self.oseries.name is not None:
            name = self.oseries.name
        elif name is None and self.oseries.name is None:
            name = "Observations"
        self.name = validate_name(name)

        self.parameters = DataFrame(
            columns=[
                "initial",
                "name",
                "optimal",
                "pmin",
                "pmax",
                "vary",
                "stderr",
                "dist",
            ]
        )

        # Define the model components
        self.stressmodels = OrderedDict()
        self.constant = None
        self.transform = None
        self.noisemodel = None

        # Default solve/simulation settings
        self.settings = {
            "tmin": None,
            "tmax": None,
            "freq": freq,
            "warmup": Timedelta(3650, "D"),
            "time_offset": Timedelta(0),
            "noise": False,
            "solver": None,
            "fit_constant": True,
            "freq_obs": None,
        }

        if constant:
            constant = Constant(initial=self.oseries.series.mean(), name="constant")
            self.add_constant(constant)

        if noisemodel is not None:
            if noisemodel is True:
                msg = (
                    "The new default is that no noisemodel is added "
                    "anymore and a noisemodel has to be added explicitly to a Pastas "
                    "model by the user. To fix this error, do not pass a "
                    "noisemodel keyword to Model and use `ml.add_noisemodel`, if a "
                    "noisemodel is desired. See this issue on GitHub for more "
                    "information: https://github.com/pastas/pastas/issues/735"
                )
                deprecate_args_or_kwargs(
                    "noisemodel", "2.0.0", reason=msg, force_raise=True
                )
            elif noisemodel is False:
                msg = (
                    "The new default is that no noisemodel is added "
                    "anymore, so passing noisemodel=False is not needed anymore. To "
                    "fix this error, do not pass noisemodel=False to Model."
                )
                deprecate_args_or_kwargs(
                    "noisemodel", "2.0.0", reason=msg, force_raise=True
                )

        # File Information
        self.file_info = self._get_file_info()

        # initialize some attributes for solving and simulation
        self.sim_index = None
        self.oseries_calib = None
        self.interpolate_simulation = None
        self.normalize_residuals = False
        self.solver = None
        self._solve_success = False

        # Load other modules
        self.stats = Statistics(self)
        self.plots = Plotting(self)
        self.plot = self.plots.plot  # because we are lazy

    def __repr__(self):
        """Prints a simple string representation of the model."""
        template = (
            "{cls}(oseries={os}, name={name}, constant={const}, noisemodel={noise})"
        )
        return template.format(
            cls=self.__class__.__name__,
            os=self.oseries.name,
            name=self.name,
            const=True if self.constant else False,
            noise=True if self.noisemodel else False,
        )

    def add_stressmodel(
        self, stressmodel: Union[StressModel, List[StressModel]], replace: bool = True
    ) -> None:
        """Add a stressmodel to the main model.

        Parameters
        ----------
        stressmodel: pastas.stressmodel or list of pastas.stressmodel
            instance of a pastas.stressmodel class. Multiple stress models can be
            provided (e.g., ml.add_stressmodel([sm1, sm2]) in one call.
        replace: bool, optional
            force replace the stressmodel if a stressmodel with the same name already
            exists. Not recommended but useful at times. Default is True.

        Notes
        -----
        To obtain a list of the stressmodel names, type:

        >>> ml.get_stressmodel_names()

        Examples
        --------
        >>> sm = ps.StressModel(stress, rfunc=ps.Gamma(), name="stress")
        >>> ml.add_stressmodel(sm)

        To add multiple stress models at once you can do the following:

        >>> sm1 = ps.StressModel(stress, rfunc=ps.Gamma(), name="stress1")
        >>> sm2 = ps.StressModel(stress, rfunc=ps.Gamma(), name="stress2")
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
            msg = (
                "The name for the stressmodel you are trying to add already exists "
                "for this model. Select another name."
            )
            logger.error(msg)
            raise ValueError(msg)

        else:
            if stressmodel.name in self.stressmodels.keys():
                logger.warning(
                    "The name for the stressmodel you are trying to add already "
                    "exists for this model. The stressmodel is replaced."
                )
            self.stressmodels[stressmodel.name] = stressmodel
            self.parameters = self.get_init_parameters(initial=False)
            stressmodel.update_stress(freq=self.settings["freq"])

            # Check if stress overlaps with oseries, if not give a warning
            if (stressmodel.tmin > self.oseries.series.index.max()) or (
                stressmodel.tmax < self.oseries.series.index.min()
            ):
                logger.warning(
                    "The stress of the stressmodel has no overlap with ml.oseries."
                )
        self._check_stressmodel_compatibility()

    def add_constant(self, constant: Constant) -> None:
        """Add a Constant to the time series Model.

        Parameters
        ----------
        constant: pastas.stressmodels.Constant
            Pastas constant instance.

        Examples
        --------
        >>> d = ps.Constant()
        >>> ml.add_constant(d)
        """
        self.constant = constant
        self.parameters = self.get_init_parameters(initial=False)
        self._check_stressmodel_compatibility()

    def add_transform(self, transform: ThresholdTransform):
        """Add a Transform to the time series Model.

        Parameters
        ----------
        transform: ps.ThresholdTransform
            An instance of a pastas.transform class.

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

    def add_noisemodel(self, noisemodel: NoiseModelType) -> None:
        """Adds a noisemodel to the time series Model.

        Parameters
        ----------
        noisemodel: pastas.noisemodels.NoiseModelBase
            Instance of NoiseModelBase.

        Examples
        --------
        >>> n = ps.ArNoiseModel()
        >>> ml.add_noisemodel(n)

        Notes
        -----
        As of Pastas version 1.5.0, a noisemodel should be added to the model using this
        method, and is not added by default anymore when constructing as Pastas Model.
        If a noisemodel is present, it will always be used during optimization.

        """
        self.noisemodel = noisemodel
        self.noisemodel.set_init_parameters(oseries=self.oseries.series)

        # check whether noise_alpha is not smaller than ml.settings["freq"]
        freq_in_days = _get_dt(self.settings["freq"])
        noise_alpha = self.noisemodel.parameters.initial.iat[0]
        if freq_in_days > noise_alpha:
            self.noisemodel._set_initial("noise_alpha", freq_in_days)

        self.settings["noise"] = True
        self.parameters = self.get_init_parameters(initial=False)

    @get_stressmodel
    def del_stressmodel(self, name: str):
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

    def del_constant(self) -> None:
        """Method to safely delete the Constant from the Model."""
        if self.constant is None:
            logger.warning("No constant is present in this model.")
        else:
            self.constant = None
            self.parameters = self.get_init_parameters(initial=False)

    def del_transform(self) -> None:
        """Method to safely delete the transform from the Model."""
        if self.transform is None:
            logger.warning("No transform is present in this model.")
        else:
            self.transform = None
            self.parameters = self.get_init_parameters(initial=False)

    def del_noisemodel(self) -> None:
        """Method to safely delete the noise model from the Model."""
        if self.noisemodel is None:
            logger.warning("No noisemodel is present in this model.")
        else:
            self.noisemodel = None
            self.parameters = self.get_init_parameters(initial=False)
            self.settings["noise"] = False

    def simulate(
        self,
        p: Optional[ArrayLike] = None,
        tmin: Optional[TimestampType] = None,
        tmax: Optional[TimestampType] = None,
        freq: Optional[str] = None,
        warmup: Optional[float] = None,
        return_warmup: bool = False,
    ) -> Series:
        """Method to simulate the time series model.

        Parameters
        ----------
        p: array_like, optional
            array_like object with the values as floats representing the model
            parameters. See Model.get_parameters() for more info if parameters is None.
        tmin: str, optional
            String with a start date for the simulation period (E.g. '1980'). If none
            is provided, the tmin from the oseries is used.
        tmax: str, optional
            String with an end date for the simulation period (E.g. '2010'). If none
            is provided, the tmax from the oseries is used.
        freq: str, optional
            String with the frequency the stressmodels are simulated. Must be one of
            the following: (D, h, m, s, ms, us, ns) or a multiple of that e.g. "7D".
        warmup: float, optional
            Warmup period (in Days).
        return_warmup: bool, optional
            Return the simulation including the warmup period or not, default is False.

        Returns
        -------
        sim: pandas.Series
            pandas.Series containing the simulated time series

        Notes
        -----
        This method can be used without any parameters. When the model is solved,
        the optimal parameters values are used and if not, the initial parameter
        values are used. This allows the user to get an idea of how the simulation
        looks with only the initial parameters and no calibration.
        """
        # Default options when tmin, tmax, freq and warmup are not provided.
        if tmin is None and self.settings["tmin"]:
            tmin = self.settings["tmin"]
        else:
            tmin = self.get_tmin(tmin, use_oseries=False, use_stresses=True)
        if tmax is None and self.settings["tmax"]:
            tmax = self.settings["tmax"]
        else:
            tmax = self.get_tmax(tmax, use_oseries=False, use_stresses=True)
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
        if p is None:
            p = self.get_parameters()

        sim = Series(data=np.zeros(sim_index.size, dtype=float), index=sim_index)

        istart = 0  # Track parameters index to pass to stressmodel object
        for sm in self.stressmodels.values():
            contrib = sm.simulate(
                p[istart : istart + sm.nparam], sim_index.min(), tmax, freq, dt
            )
            sim = sim.add(contrib)
            istart += sm.nparam
        if self.constant:
            sim = sim + self.constant.simulate(p[istart])
            istart += 1
        if self.transform:
            sim = self.transform.simulate(
                sim, p[istart : istart + self.transform.nparam]
            )

        # Respect provided tmin/tmax at this point, since warmup matters for
        # simulation but should not be returned, unless return_warmup=True.
        if not return_warmup:
            sim = sim.loc[tmin:tmax]

        if sim.hasnans:
            msg = (
                "Simulation contains NaN-values. Check if time series settings "
                "are provided for each stress model "
                "(e.g. `ps.StressModel(stress, settings='prec')`!"
            )
            logger.error(msg)
            raise ValueError(msg)

        sim.name = "Simulation"
        return sim

    def residuals(
        self,
        p: Optional[ArrayLike] = None,
        tmin: Optional[TimestampType] = None,
        tmax: Optional[TimestampType] = None,
        freq: Optional[str] = None,
        warmup: Optional[float] = None,
    ) -> Series:
        """Method to calculate the residual series.

        Parameters
        ----------
        p: array_like, optional
            array_like object with the values as floats representing the model
            parameters. See Model.get_parameters() for more info if parameters is None.
        tmin: str, optional
            String with a start date for the simulation period (E.g. '1980'). If none
            is provided, the tmin from the oseries is used.
        tmax: str, optional
            String with an end date for the simulation period (E.g. '2010'). If none
            is provided, the tmax from the oseries is used.
        freq: str, optional
            String with the frequency the stressmodels are simulated. Must be one of
            the following: (D, h, m, s, ms, us, ns) or a multiple of that e.g. "7D".
        warmup: float, optional
            Warmup period (in Days).

        Returns
        -------
        res: pandas.Series
            pandas.Series with the residuals.
        """
        # Default options when tmin, tmax, freq and warmup are not provided.
        if tmin is None:
            tmin = self.settings["tmin"]
        if tmax is None:
            tmax = self.settings["tmax"]
        if freq is None:
            freq = self.settings["freq"]
        if self.settings["freq_obs"] is None:
            freq_obs = freq
        else:
            freq_obs = self.settings["freq_obs"]

        # simulate model
        sim = self.simulate(p, tmin, tmax, freq, warmup, return_warmup=False)

        # Get the oseries calibration series
        oseries_calib = self.observations(tmin, tmax, freq_obs)

        # Get simulation at the correct indices
        if self.interpolate_simulation is None:
            if oseries_calib.index.difference(sim.index).size != 0:
                self.interpolate_simulation = True
                logger.info(
                    "There are observations between the simulation time steps. Linear "
                    "interpolation between simulated values is used."
                )
        if self.interpolate_simulation:
            # interpolate simulation to times of observations
            sim_interpolated = np.interp(
                oseries_calib.index.asi8, sim.index.asi8, sim.values
            )
        else:
            # All the observation indexes are in the simulation
            sim_interpolated = sim.reindex(oseries_calib.index)

        # Calculate the actual residuals here
        res = oseries_calib.subtract(sim_interpolated)

        if res.hasnans:
            res = res.dropna()
            logger.warning("Nan-values were removed from the residuals.")

        if self.normalize_residuals:
            res = res.subtract(res.values.mean())

        res.name = "Residuals"
        return res

    def noise(
        self,
        p: Optional[ArrayLike] = None,
        tmin: Optional[TimestampType] = None,
        tmax: Optional[TimestampType] = None,
        freq: Optional[str] = None,
        warmup: Optional[float] = None,
    ) -> Union[Series, None]:
        """Method to simulate the noise when a noisemodel is present.

        Parameters
        ----------
        p: array_like, optional
            array_like object with the values as floats representing the model
            parameters. See Model.get_parameters() for more info if parameters is None.
        tmin: str, optional
            String with a start date for the simulation period (E.g. '1980'). If none
            is provided, the tmin from the oseries is used.
        tmax: str, optional
            String with an end date for the simulation period (E.g. '2010'). If none
            is provided, the tmax from the oseries is used.
        freq: str, optional
            String with the frequency the stressmodels are simulated. Must be one of
            the following: (D, h, m, s, ms, us, ns) or a multiple of that e.g. "7D".
        warmup: float or int, optional
            Warmup period (in Days).

        Returns
        -------
        noise : pandas.Series or None
            Pandas series of the noise. None if no noise model is present.

        Notes
        -----
        The noise are the time series that result when applying a noise model.

        .. Note::
            The noise is sometimes also referred to as the innovations in the
            literature.

        Warnings
        --------
        This method returns None if no noise model is present in the model.
        """
        if self.noisemodel is None or self.settings["noise"] is False:
            logger.warning(
                "Noise cannot be calculated if there is no noisemodel present or is "
                "not used during parameter estimation."
            )
            return None

        # Get parameters if none are provided
        if p is None:
            p = self.get_parameters()

        # Calculate the residuals
        res = self.residuals(p, tmin, tmax, freq, warmup)
        p = p[-self.noisemodel.nparam :]

        # Calculate the noise
        noise = self.noisemodel.simulate(res, p)
        return noise

    def noise_weights(
        self,
        p: Optional[list] = None,
        tmin: Optional[TimestampType] = None,
        tmax: Optional[TimestampType] = None,
        freq: Optional[str] = None,
        warmup: Optional[float] = None,
    ) -> ArrayLike:
        """Internal method to calculate the noise weights."""
        # Get parameters if none are provided
        if p is None:
            p = self.get_parameters()

        # Calculate the residuals
        res = self.residuals(p, tmin, tmax, freq, warmup)

        # Calculate the weights
        weights = self.noisemodel.weights(res, p[-self.noisemodel.nparam :])

        return weights

    def observations(
        self,
        tmin: Optional[TimestampType] = None,
        tmax: Optional[TimestampType] = None,
        freq: Optional[str] = None,
        update_observations: bool = False,
    ) -> Series:
        """Method that returns the observations series used for calibration.

        Parameters
        ----------
        tmin: str, optional
            String with a start date for the simulation period (E.g. '1980'). If none
            is provided, the tmin from the oseries is used.
        tmax: str, optional
            String with an end date for the simulation period (E.g. '2010'). If none
            is provided, the tmax from the oseries is used.
        freq: str, optional
            String with the frequency the stressmodels are simulated. Must be one of
            the following: (D, h, m, s, ms, us, ns) or a multiple of that e.g. "7D".
        update_observations: bool, optional
            If True, force recalculation of the observations, default is False.

        Returns
        -------
        oseries_calib: pandas.Series
            pandas series of the oseries used for calibration of the model.

        Notes
        -----
        This method makes sure the simulation is compared to the nearest observation.
        It finds the index closest to sim_index, and then returns a selection of the
        oseries. In the `residuals` method, the simulation is interpolated to the
        observation-timestamps.
        """
        if tmin is None and self.settings["tmin"]:
            tmin = self.settings["tmin"]
        else:
            tmin = self.get_tmin(tmin, use_oseries=False, use_stresses=True)
        if tmax is None and self.settings["tmax"]:
            tmax = self.settings["tmax"]
        else:
            tmax = self.get_tmax(tmax, use_oseries=False, use_stresses=True)
        if freq is None:
            if self.settings["freq_obs"] is None:
                freq = self.settings["freq"]
            else:
                freq = self.settings["freq_obs"]
        for key, setting in zip([tmin, tmax, freq], ["tmin", "tmax", "freq"]):
            if key != self.settings[setting]:
                update_observations = True

        if self.oseries_calib is None or update_observations:
            oseries_calib = self.oseries.series.loc[tmin:tmax]

            # sample measurements, so that frequency is not higher than model keep
            # the original timestamps, as they will be used during interpolation of
            # the simulation
            sim_index = self._get_sim_index(tmin, tmax, freq, self.settings["warmup"])
            if not oseries_calib.empty:
                index = get_sample(oseries_calib.index, sim_index)
                oseries_calib = oseries_calib.loc[index]
        else:
            oseries_calib = self.oseries_calib
        return oseries_calib

    def initialize(
        self,
        tmin: Optional[TimestampType] = None,
        tmax: Optional[TimestampType] = None,
        freq: Optional[str] = None,
        warmup: Optional[float] = None,
        noise: Optional[bool] = None,
        weights: Optional[Series] = None,
        initial: bool = True,
        fit_constant: bool = True,
        freq_obs: Optional[str] = None,
    ) -> None:
        """Method to initialize the model.

        This method is called by the solve-method, but can also be triggered
        manually. See the solve-method for a description of the arguments.
        """

        if noise is not None:
            msg = (
                "The new behavior is that a noise model will always be "
                "used if it is present. To add a noisemodel to a model called ml, "
                "use the ml.add_noisemodel method. To solve without a noisemodel, "
                "make sure sure no noisemodel is added or remove a noisemodel with "
                "ml.del_noisemodel() before solving. See this issue on GitHub for "
                "more information: https://github.com/pastas/pastas/issues/735"
            )
            deprecate_args_or_kwargs("noise", "2.0.0", reason=msg, force_raise=True)

        # Set the settings
        self.settings["weights"] = weights
        self.settings["fit_constant"] = fit_constant
        self.settings["freq_obs"] = freq_obs

        # Set the frequency & warmup
        if freq:
            self.settings["freq"] = _frequency_is_supported(freq)

        if warmup is not None:
            self.settings["warmup"] = Timedelta(warmup, "D")

        # Set time offset from the frequency and the series in the stressmodels
        self.settings["time_offset"] = self._get_time_offset(self.settings["freq"])

        # Set tmin and tmax
        self.settings["tmin"] = self.get_tmin(tmin)
        self.settings["tmax"] = self.get_tmax(tmax)

        # make sure calibration data is renewed
        self.sim_index = self._get_sim_index(
            self.settings["tmin"],
            self.settings["tmax"],
            self.settings["freq"],
            self.settings["warmup"],
            update_sim_index=True,
        )

        # self.observations get tmin, tmax, freq, and freq_obs from self.settings
        self.oseries_calib = self.observations(update_observations=True)
        self.interpolate_simulation = None

        # Initialize parameters
        self.parameters = self.get_init_parameters(noise, initial)

        # Prepare model if not fitting the constant as a parameter
        if self.settings["fit_constant"] is False:
            if self.transform is not None:
                msg = "fit_constant needs to be True (for now) when a transform is used"
                logger.error(msg)
                raise Exception(msg)
            self.parameters.at["constant_d", "vary"] = False
            self.parameters.at["constant_d", "initial"] = 0.0
            self.normalize_residuals = True

    def add_solver(self, solver: Solver) -> None:
        """Method to add a solver to the model.

        Parameters
        ----------
        solver: pastas.solver.Solver
            Instance of a pastas Solver class used to solve the model. Options are:
            ps.LeastSquares(), ps.LmfitSolve() or ps.EmceeSolve(). An instance
            (e.g. ps.LeastSquares()) is needed as of Pastas 0.23, not a class (e.g.
            ps.LeastSquares)!

        See Also
        --------
        pastas.solver
            Different solver objects are available to estimate parameters.
        """
        self.solver = solver
        if not hasattr(self.solver, "ml") or self.solver.ml is None:
            self.solver.set_model(self)
        self.settings["solver"] = self.solver._name

    def solve(
        self,
        tmin: Optional[TimestampType] = None,
        tmax: Optional[TimestampType] = None,
        freq: Optional[str] = None,
        warmup: Optional[float] = None,
        noise=None,  # will be removed in version 2.0.0
        solver: Optional[Solver] = None,
        report: bool = True,
        initial: bool = True,
        weights: Optional[Series] = None,
        fit_constant: bool = True,
        freq_obs: Optional[str] = None,
        initialize: bool = True,
        **kwargs,
    ) -> None:
        """Method to solve the time series model.

        Parameters
        ----------
        tmin: str, optional
            String with a start date for the simulation period (E.g. '1980'). If
            none is provided, the tmin from the oseries is used.
        tmax: str, optional
            String with an end date for the simulation period (E.g. '2010'). If none
            is provided, the tmax from the oseries is used.
        freq: str, optional
            String with the frequency the stressmodels are simulated. Must be one of
            the following (D, h, m, s, ms, us, ns) or a multiple of that e.g. "7D".
        warmup: float, optional
            Warmup period (in Days) for which the simulation is calculated, but not
            used for the calibration period.
        noise: bool, optional
            This argument is deprecated and will be removed in Pastas version 2.0.0.
            To solve using a noisemodel (i.e. noise=True), add a noisemodel to the
            model using ml.add_noisemodel(n), where n is an instance of a noisemodel
            (e.g., n = ps.ArNoiseModel()). To solve without a noisemodel (noise=False),
            remove the noisemodel first (if present) using ml.del_noisemodel().
        solver: Class pastas.solver.Solver, optional
            Instance of a pastas Solver class used to solve the model. Options are:
            ps.LeastSquares() (default) or ps.LmfitSolve(). An instance is needed as
            of Pastas 0.23, not a class!
        report: bool, optional
            Print a report to the screen after optimization finished. This can also
            be manually triggered after optimization by calling print(ml.fit_report(
            )) on the Pastas model instance.
        initial: bool, optional
            Reset initial parameters from the individual stress models. Default is
            True. If False, the optimal values from an earlier optimization are used.
        weights: pandas.Series, optional
            Pandas Series with values by which the residuals are multiplied,
            index-based. Must have the same indices as the oseries.
        fit_constant: bool, optional
            Argument that determines if the constant is fitted as a parameter. If it
            is set to False, the constant is set equal to the mean of the residuals.
        freq_obs: str, optional
            String with the frequency of the observations that the model will be
            calibrated on. Must be one of the following (D, h, m, s, ms, us, ns) or a
            multiple of that e.g. "7D". Should generally be larger than the frequency
            of the original observations and the model frequency (freq). If freq_obs
            is not set, the frequency of the model (freq) will be used.
        initialize: bool, optional
            If True, the model is initialized via the Model.initialize() method
            (setting certain model settings) before solving. If False, the
            model is not initialized before solving. Note that the latter is an
            advanced option since some model settings can be missing. Default
            is True.
        **kwargs: dict, optional
            All keyword arguments will be passed onto minimization method from the
            solver. It depends on the solver used which arguments can be used.

        Notes
        -----
        - The solver instance including some results are stored as ml.solver. From here
          one can access the covariance (ml.solver.pcov) and correlation matrix (
          ml.solver.pcor).
        - Each solver returns a number of results after optimization. These solver
          specific results are stored in ml.solver.result and can be accessed from there.

        See Also
        --------
        pastas.solver
            Different solver objects are available to estimate parameters.
        """
        if noise is not None:
            if noise is True:
                msg = (
                    "To solve using a noisemodel, add a noisemodel to a "
                    "model called ml using ml.add_noisemodel(n), where n is an instance "
                    "of a noisemodel (e.g., n = ps.ArNoiseModel()). See this issue on "
                    "GitHub for more information: "
                    "https://github.com/pastas/pastas/issues/735"
                )
                deprecate_args_or_kwargs("noise", "2.0.0", reason=msg, force_raise=True)
            elif noise is False:
                msg = (
                    "To solve without a noisemodel, remove the noisemodel "
                    "(if present) from a model using ml.del_noisemodel() before "
                    "solving. See this issue on GitHub for more information: "
                    "https://github.com/pastas/pastas/issues/735"
                )
                deprecate_args_or_kwargs("noise", "2.0.0", reason=msg, force_raise=True)

        if initialize:
            self.initialize(
                tmin=tmin,
                tmax=tmax,
                freq=freq,
                warmup=warmup,
                weights=weights,
                initial=initial,
                fit_constant=fit_constant,
                freq_obs=freq_obs,
            )

        if self.oseries_calib.empty:
            msg = "Calibration series 'oseries_calib' is empty! Check 'tmin' or 'tmax'."
            logger.error(msg)
            raise ValueError(msg)

        # Create the default solver if None is provided or already present
        solver = LeastSquares() if solver is None else solver
        if self.solver is None:
            self.add_solver(solver=solver)
        elif self.solver._name != solver._name:
            logger.info(
                "Replacing original solver `%s` with new solver `%s`."
                % (self.solver._name, solver._name)
            )
            self.add_solver(solver=solver)

        # Solve model
        success, optimal, stderr = self.solver.solve(
            noise=self.settings["noise"], weights=weights, **kwargs
        )
        if not success:
            logger.warning("Model parameters could not be estimated well.")

        if self.settings["fit_constant"] is False:
            # Determine the residuals and set the constant to their mean
            self.normalize_residuals = False
            res = self.residuals(optimal).mean()
            optimal[self.parameters.name == self.constant.name] = res

        self.parameters.optimal = optimal
        self.parameters.stderr = stderr
        self._solve_success = success  # store for fit_report

        if report:
            if isinstance(report, str) and report == "full":
                print(self.fit_report(corr=True, stderr=True))
            else:
                print(self.fit_report())

    @property
    @PastasDeprecationWarning(remove_version="2.0.0", reason="Use 'ml.solver' instead.")
    def fit(self):
        """Deprecated attribute, use ml.solver instead."""
        msg = (
            "Attribute 'fit' is deprecated and will be removed in a future version. "
            "Use 'solver' instead."
        )
        logger.warning(msg)

        return self.solver

    def set_parameter(
        self,
        name: str,
        initial: Optional[float] = None,
        vary: Optional[bool] = None,
        pmin: Optional[float] = None,
        pmax: Optional[float] = None,
        optimal: Optional[float] = None,
        dist: Optional[str] = None,
        move_bounds: bool = False,
    ) -> None:
        """Method to change the parameter properties.

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
        optimal: float, optional
            optimal value for the parameter.
        dist: str, optional
            Distribution of the parameters.
        move_bounds: bool, optional
            Reset pmin/pmax based on new initial value. Of move_bounds=True, pmin and
            pmax must be None.

        Examples
        --------
        >>> ml.set_parameter(name="constant_d", initial=10, vary=True,
        >>>                  pmin=-10, pmax=20)

        Notes
        -----
        It is highly recommended to use this method to set parameter properties.
        Changing the parameter properties directly in the parameter `DataFrame` may
        not work as expected.
        """
        if name not in self.parameters.index:
            msg = "parameter %s is not present in the model"
            logger.error(msg, name)
            raise KeyError(msg % name)

        # Because either of the following is not necessarily present
        noisemodel = self.noisemodel.name if self.noisemodel else "NotPresent"
        constant = self.constant.name if self.constant else "NotPresent"
        transform = self.transform.name if self.transform else "NotPresent"

        # Get the model component for the parameter
        cat = self.parameters.at[name, "name"]

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
                msg = "Either pmin/pmax or move_bounds must be provided, but not both."
                logger.error(msg)
                raise KeyError(msg)

            factor = initial / self.parameters.at[name, "initial"]
            pmin = self.parameters.at[name, "pmin"] * factor
            pmax = self.parameters.at[name, "pmax"] * factor

        # Set the parameter properties
        if initial is not None:
            obj._set_initial(name, initial)
            self.parameters.at[name, "initial"] = initial
        if vary is not None:
            obj._set_vary(name, vary)
            self.parameters.at[name, "vary"] = bool(vary)
        if pmin is not None:
            obj._set_pmin(name, pmin)
            self.parameters.at[name, "pmin"] = pmin
        if pmax is not None:
            obj._set_pmax(name, pmax)
            self.parameters.at[name, "pmax"] = pmax
        if dist is not None:
            obj._set_dist(name, dist)
            self.parameters.at[name, "dist"] = dist
        if optimal is not None:
            self.parameters.at[name, "optimal"] = optimal

    def _get_time_offset(self, freq: str) -> Timedelta:
        """Internal method to get the time offsets from the stressmodels.

        Parameters
        ----------
        freq: str
            string with the frequency used for simulation.

        Notes
        -----

        Method to check if the StressModel timestamps match (e.g. similar hours).
        """
        time_offsets = set()
        for stressmodel in self.stressmodels.values():
            for st in stressmodel.stress:
                if st.freq_original:
                    # calculate the offset from the default frequency
                    t = st.series_original.index
                    base = t.min().ceil(freq)
                    mask = t >= base
                    if np.any(mask):
                        time_offsets.add(_get_time_offset(t[mask][0], freq))
        if len(time_offsets) > 1:
            msg = "The time-offset with the frequency is not the same for all stresses."
            logger.error(msg)
            raise Exception(msg)
        if len(time_offsets) == 1:
            return next(iter(time_offsets))
        else:
            return Timedelta(0)

    def _get_sim_index(
        self,
        tmin: Timestamp,
        tmax: Timestamp,
        freq: str,
        warmup: Timedelta,
        update_sim_index: bool = False,
    ) -> DatetimeIndex:
        """Internal method to get the simulation index, including the warmup.

        Parameters
        ----------
        tmin: pandas.Timestamp
            String with a start date for the simulation period (E.g. '1980'). If none
            is provided, the tmin from the oseries is used.
        tmax: pandas.Timestamp
            String with an end date for the simulation period (E.g. '2010'). If none
            is provided, the tmax from the oseries is used.
        freq: str
            String with the frequency the stressmodels are simulated. Must be one of
            the following: (D, h, m, s, ms, us, ns) or a multiple of that e.g. "7D".
        warmup: pandas.Timedelta
            Warmup period (in Days).
        update_sim_index : bool, optional
            if True, force recalculation of sim_index, default is False

        Returns
        -------
        sim_index: pandas.DatetimeIndex
            Pandas DatetimeIndex instance with the datetimes values for which the
            model is simulated.
        """
        # Check if any of the settings are updated
        for key, setting in zip(
            [tmin, tmax, freq, warmup], ["tmin", "tmax", "freq", "warmup"]
        ):
            if key != self.settings[setting]:
                update_sim_index = True
                break

        if self.sim_index is None or update_sim_index:
            # TODO: sort out what to do for freq > "D"
            tmin = (tmin - warmup).floor(freq) + self.settings["time_offset"]
            # tmin = (tmin - warmup) + self.settings["time_offset"]
            sim_index = date_range(tmin, tmax, freq=freq)
        else:
            sim_index = self.sim_index
        return sim_index

    def get_tmin(
        self,
        tmin: Optional[TimestampType] = None,
        use_oseries: bool = True,
        use_stresses: bool = False,
    ) -> Timestamp:
        """Method that checks and returns valid values for tmin.

        Parameters
        ----------
        tmin: str or pandas.Timestamp, optional
            string with a year or date that can be turned into a pandas Timestamp (
            e.g. pd.Timestamp(tmin)).
        use_oseries: bool, optional
            Obtain the tmin and tmax from the oseries. Default is True.
        use_stresses: bool, optional
            Obtain the tmin and tmax from the stresses. The minimum/maximum time from
            all stresses is taken.

        Returns
        -------
        tmin: pandas.Timestamp
            returns pandas timestamps for tmin.

        Notes
        -----
        The parameters tmin and tmax are leading, unless use_oseries is True, then
        these are checked against the oseries index. The tmin and tmax are checked
        and returned according to the following rules:

        A. If no value for tmin is provided:

            1. If the use_oseries argument is True, tmin is based on the oseries.
            2. If the use_stresses argument is True, tmin is based on the stressmodels.

        B. If a values for tmin is provided:

            1. A pandas timestamp is made from the string
            2. If the use_oseries argument is True, tmin is checked against oseries.
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

        return tmin

    def get_tmax(
        self,
        tmax: Optional[TimestampType] = None,
        use_oseries: bool = True,
        use_stresses: bool = False,
    ) -> Timestamp:
        """Method that checks and returns valid values for tmax.

        Parameters
        ----------
        tmax: str or pandas.Timestamp, optional
            string with a year or date that can be turned into a pandas Timestamp (
            e.g. pd.Timestamp(tmax)).
        use_oseries: bool, optional
            Obtain the tmin and tmax from the oseries. Default is True.
        use_stresses: bool, optional
            Obtain the tmin and tmax from the stresses. The minimum/maximum time from
            all stresses is taken.

        Returns
        -------
        tmax: pandas.Timestamp
            returns pandas timestamps for tmax.

        Notes
        -----
        The parameters tmin and tmax are leading, unless use_oseries is True,
        then these are checked against the oseries index. The tmin and tmax are
        checked and returned according to the following rules:

        A. If no value for tmax is provided:

            1. If the use_oseries argument is True, tmax is based on the oseries.
            2. If the use_stresses argument is True, tmax is based on the stressmodels.

        B. If a values for tmax is provided:

            1. A pandas timestamp is made from the string.
            2. if the use_oseries argument is True, tmax is checked against oseries.

        A detailed description of dealing with tmax and timesteps in general can be
        found in the developers section of the docs.
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

        return tmax

    def get_init_parameters(
        self, noise: Optional[bool] = None, initial: bool = True
    ) -> DataFrame:
        """Method to get all initial parameters from the individual objects.

        Parameters
        ----------
        noise: bool, optional
            Add the parameters for the noisemodel to the parameters Dataframe or not.
        initial: bool, optional
            True to get initial parameters, False to get optimized parameters.

        Returns
        -------
        parameters: pandas.DataFrame
            pandas.Dataframe with the parameters.
        """
        if noise is None:
            noise = self.settings["noise"]

        frames = []

        for sm in self.stressmodels.values():
            frames.append(sm.parameters)
        if self.constant:
            frames.append(self.constant.parameters)
        if self.transform:
            frames.append(self.transform.parameters)
        if self.noisemodel and noise:
            frames.append(self.noisemodel.parameters)

        if not frames:
            parameters = DataFrame(columns=self.parameters.columns)
        else:
            parameters = concat(frames)
            parameters = parameters.infer_objects()
            parameters["stderr"] = np.nan
            parameters["optimal"] = np.nan

        # Set initial parameters to optimal parameters from model
        if not initial:
            parameters.update({"initial": self.parameters.loc[:, "optimal"]})
            parameters.update({"optimal": self.parameters.loc[:, "optimal"]})
            parameters.update({"stderr": self.parameters.loc[:, "stderr"]})

        return parameters

    def get_parameters(self, name: Optional[str] = None) -> ArrayLike:
        """Method to obtain the parameters needed for calculation.

        This method is used by the simulation, residuals and the noise methods as
        well as other methods that need parameters values as arrays.

        Parameters
        ----------
        name: str, optional
            string with the name of the pastas.stressmodel object.

        Returns
        -------
        p : array_like
            NumPy array with the parameters used in the time series model.
        """
        if name:
            p = self.parameters.loc[self.parameters.name == name]
        else:
            p = self.parameters

        if p.optimal.hasnans:
            logger.warning("Model is not optimized yet, initial parameters are used.")
            parameters = p.initial
        else:
            parameters = p.optimal

        return parameters.to_numpy(dtype=float)

    def get_stressmodel_names(self) -> List[str]:
        """Returns list of stressmodel names."""
        return list(self.stressmodels.keys())

    @get_stressmodel
    def get_stressmodel_settings(self, name: str) -> Union[dict, None]:
        """Method to obtain the time series settings for a stress model.

        Parameters
        ----------
        name: str, optional
            string with the name of the pastas.stressmodel object.

        Returns
        -------
        dict or None
            Dictionary with the settings or "None" of no stress are present, e.g.,
            for a step model that uses no stress.
        """
        sm = self.stressmodels[name]
        settings = sm.get_settings()
        return settings

    @get_stressmodel
    def get_contribution(
        self,
        name: str,
        tmin: Optional[TimestampType] = None,
        tmax: Optional[TimestampType] = None,
        freq: Optional[str] = None,
        warmup: Optional[float] = None,
        istress: Optional[int] = None,
        return_warmup: bool = False,
        p: Optional[ArrayLike] = None,
    ) -> Series:
        """Method to get the contribution of a stressmodel.

        Parameters
        ----------
        name: str
            String with the name of the stressmodel.
        tmin: str, optional
            String with a start date for the simulation period (E.g. '1980'). If none
            is provided, the tmin from the oseries is used.
        tmax: str, optional
            String with an end date for the simulation period (E.g. '2010'). If none
            is provided, the tmax from the oseries is used.
        freq: str, optional
            String with the frequency the stressmodels are simulated. Must be one of
            the following: (D, h, m, s, ms, us, ns) or a multiple of that e.g. "7D".
        warmup: float or int, optional
            Warmup period (in Days).
        istress: int, optional
            When multiple stresses are present in a stressmodel, this keyword can be
            used to obtain the contribution of an individual stress.
        return_warmup: bool, optional
            Include warmup in contribution calculation or not.
        p : array_like, optional
            array_like object with the values as floats representing the model
            parameters. See Model.get_parameters() for more info if parameters is None.

        Returns
        -------
        contrib: pandas.Series
            Pandas.Series with the contribution.
        """
        if p is None:
            p = self.get_parameters(name)

        if tmin is None:
            tmin = self.settings["tmin"]
        if tmax is None:
            tmax = self.settings["tmax"]
        if freq is None:
            freq = self.settings["freq"]
        if warmup is None:
            warmup = self.settings["warmup"]
        else:
            warmup = Timedelta(warmup, "D")

        # use warmup
        if tmin:
            tmin_warm = (Timestamp(tmin) - warmup).floor(freq) + self.settings[
                "time_offset"
            ]
        else:
            tmin_warm = None

        dt = _get_dt(freq)

        kwargs = {"tmin": tmin_warm, "tmax": tmax, "freq": freq, "dt": dt}
        if istress is not None:
            kwargs["istress"] = istress
        contrib = self.stressmodels[name].simulate(p, **kwargs)

        # Respect provided tmin/tmax at this point, since warmup matters for
        # simulation but should not be returned, unless return_warmup=True.
        if not return_warmup:
            contrib = contrib.loc[tmin:tmax]

        return contrib

    def get_contributions(self, split: bool = True, **kwargs) -> List[Series]:
        """Method to get contributions of all stressmodels.

        Parameters
        ----------
        split: bool, optional
            Split the stresses in multiple stresses when possible.
        kwargs: any other arguments are passed to get_contribution

        Returns
        -------
        contribs: list of pandas.Series
            a list of Pandas.Series of the contributions.

        See Also
        --------
        pastas.model.Model.get_contribution
            This method is called to get the individual contributions, kwargs are
            passed on to this method.
        """
        contribs = []
        for name in self.stressmodels:
            nsplit = self.stressmodels[name].get_nsplit()
            if split and nsplit > 1:
                for istress in range(nsplit):
                    contrib = self.get_contribution(name, istress=istress, **kwargs)
                    contribs.append(contrib)
            else:
                contrib = self.get_contribution(name, **kwargs)
                contribs.append(contrib)
        return contribs

    def get_transform_contribution(
        self, tmin: Optional[TimestampType] = None, tmax: Optional[TimestampType] = None
    ) -> Series:
        """Method to get the contribution of a transform.

        Parameters
        ----------
        tmin: str, optional
            String with a start date for the simulation period (E.g. '1980'). If none
            is provided, the tmin from the oseries is used.
        tmax: str, optional
            String with an end date for the simulation period (E.g. '2010'). If none
            is provided, the tmax from the oseries is used.

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

    def get_output_series(
        self,
        tmin: Optional[TimestampType] = None,
        tmax: Optional[TimestampType] = None,
        add_contributions: bool = True,
        split: bool = True,
    ) -> DataFrame:
        """Method to get all the modeled output time series from the Model.

        Parameters
        ----------
        tmin: str, optional
            String with a start date for the simulation period (E.g. '1980'). If none
            is provided, the tmin from the oseries is used.
        tmax: str, optional
            String with an end date for the simulation period (E.g. '2010'). If none
            is provided, the tmax from the oseries is used.
        add_contributions: bool, optional
            Add the contributions from the different stresses or not.¬
        split: bool, optional
            Passed on to ml.get_contributions. Split the contribution from recharge
            into evaporation and precipitation. See also ml.get_contributions.

        Returns
        -------
        df: pandas.DataFrame
            Pandas DataFrame with the time series as columns and DatetimeIndex.

        Notes
        -----
        Export the observed, simulated time series, the noise and residuals series,
        and the contributions from the different stressmodels.

        Examples
        --------
        >>> df = ml.get_output_series(tmin="2000", tmax="2010")
        >>> df.to_csv("fname.csv")
        """
        obs = self.observations(tmin=tmin, tmax=tmax)
        obs.name = "Head_Calibration"

        sim = self.simulate(tmin=tmin, tmax=tmax)
        res = self.residuals(tmin=tmin, tmax=tmax)
        noise = self.noise(tmin=tmin, tmax=tmax)

        df = [obs, sim, res, noise]

        if add_contributions:
            contribs = self.get_contributions(tmin=tmin, tmax=tmax, split=split)
            for contrib in contribs:
                df.append(contrib)

        df = concat(df, axis=1, sort=True)
        return df

    def _get_response(
        self,
        block_or_step: str,
        name: str,
        p: Optional[ArrayLike] = None,
        dt: Optional[float] = None,
        add_0: bool = False,
        istress: Optional[int] = None,
        **kwargs,
    ) -> Union[Series, None]:
        """Internal method to compute the block and step response.

        Parameters
        ----------
        block_or_step: str
            String with "step" or "block"
        name: str
            string with the name of the stressmodel
        p : array_like, optional
            array_like object with the values as floats representing the model
            parameters. See Model.get_parameters() for more info if parameters is None.
        dt: float, optional
            timestep for the response function.
        add_0: bool, optional
            Add a zero at t=0.
        istress: int, optional
            When multiple stresses are present in a stressmodel, this keyword can be
            used to obtain the respone to an individual stress.
        kwargs: dict: passed to rfunc.step() or rfunc.block()

        Returns
        -------
        response: pandas.Series or None
            Pandas.Series with the response, None if not present.
        """
        if self.stressmodels[name].rfunc is None:
            logger.warning("Stressmodel %s has no rfunc.", name)
            return None
        else:
            block_or_step = getattr(self.stressmodels[name].rfunc, block_or_step)

        if p is None:
            p = self.get_parameters(name)

        if dt is None:
            dt = _get_dt(self.settings["freq"])
        if istress is not None and self.stressmodels[name].get_nsplit() > 1:
            p = self.stressmodels[name].get_parameters(model=self, istress=istress)

        response = block_or_step(p, dt, **kwargs)

        if add_0:
            if isinstance(dt, np.ndarray):
                t = dt
            else:
                t = np.linspace(0, response.size * dt, response.size + 1)
            response = np.insert(response, 0, 0.0)
        else:
            if isinstance(dt, np.ndarray):
                t = dt
            else:
                t = np.linspace(dt, response.size * dt, response.size)

        response = Series(response, index=t, name=name)
        response.index.name = "Time [days]"

        return response

    @get_stressmodel
    def get_block_response(
        self,
        name: str,
        p: Optional[ArrayLike] = None,
        add_0: bool = False,
        dt: Optional[float] = None,
        **kwargs,
    ) -> Union[Series, None]:
        """Method to obtain the block response for a stressmodel.

        The optimal parameters are used when available, initial otherwise.

        Parameters
        ----------
        name: str
            String with the name of the stressmodel.
        p: array_like, optional
            array_like object with the values as floats representing the model
            parameters. See Model.get_parameters() for more info if parameters is None.
        add_0: bool, optional
            Adds 0 at t=0 at the start of the response, defaults to False.
        dt: float, optional
            timestep for the response function.
        kwargs: dict, optional
            Kwargs are passed onto _get_response()

        Returns
        -------
        b: pandas.Series or None
            Pandas.Series with the block response. The index is based on the
            frequency that is present in the model.settings.
        """
        return self._get_response(
            block_or_step="block", name=name, dt=dt, p=p, add_0=add_0, **kwargs
        )

    @get_stressmodel
    def get_step_response(
        self,
        name,
        p: Optional[ArrayLike] = None,
        add_0: bool = False,
        dt: Optional[float] = None,
        **kwargs,
    ) -> Union[Series, None]:
        """Method to obtain the step response for a stressmodel.

        The optimal parameters are used when available, initial otherwise.

        Parameters
        ----------
        name: str
            String with the name of the stressmodel.
        p: array_like, optional
            array_like object with the values as floats representing the model
            parameters. See Model.get_parameters() for more info if parameters is None.
        add_0: bool, optional
            Adds 0 at t=0 at the start of the response, defaults to False.
        dt: float, optional
            timestep for the response function.
        kwargs: dict, optional
            Kwargs are passed onto _get_response()

        Returns
        -------
        s: pandas.Series or None
            Pandas.Series with the step response. The index is based on the frequency
            that is present in the model.settings.
        """
        return self._get_response(
            block_or_step="step", name=name, dt=dt, p=p, add_0=add_0, **kwargs
        )

    @get_stressmodel
    def get_response_tmax(
        self,
        name: str,
        p: ArrayLike = None,
        cutoff: float = 0.999,
        warn: bool = True,
    ) -> Union[float, None]:
        """Method to get the tmax used for the response function.

        Parameters
        ----------
        name: str
            A string with the name of the stressmodel.
        p: array_like, optional
            An array_like object with the values as floats representing the model
            parameters. See Model.get_parameters() for more info if parameters is None.
        cutoff: float, optional
            A float between 0 and 1. Default is 0.999 or 99.9% of the response.

        Returns
        -------
        tmax: float or None
            A float with the number of days. None is return if stressmodels has no
            response function.

        Examples
        --------
        >>> ml.get_response_tmax("recharge", cutoff=0.99)
        >>> 703

        This means that after 703 days, 99% of the response of the groundwater levels
        to a recharge pulse has taken place.
        """
        if self.stressmodels[name].rfunc is None:
            logger.warning("Stressmodel %s has no rfunc", name)
            return None
        else:
            if p is None:
                p = self.get_parameters(name)
            if isinstance(self.stressmodels[name].rfunc, HantushWellModel):
                kwargs = {"warn": warn}
            else:
                kwargs = {}
            tmax = self.stressmodels[name].rfunc.get_tmax(p=p, cutoff=cutoff, **kwargs)
            return tmax

    @get_stressmodel
    def get_stress(
        self,
        name: str,
        tmin: Optional[TimestampType] = None,
        tmax: Optional[TimestampType] = None,
        freq: Optional[str] = None,
        warmup: Optional[float] = None,
        istress: Optional[int] = None,
        return_warmup: bool = False,
        p: Optional[ArrayLike] = None,
    ) -> Union[Series, List[Series]]:
        """Method to obtain the stress(es) from the stressmodel.

        Parameters
        ----------
        name: str
            String with the name of the stressmodel.
        tmin: str, optional
            String with a start date for the simulation period (E.g. '1980'). If none
            is provided, the tmin from the oseries is used.
        tmax: str, optional
            String with an end date for the simulation period (E.g. '2010'). If none
            is provided, the tmax from the oseries is used.
        freq: str, optional
            String with the frequency the stressmodels are simulated. Must be one of
            the following: (D, h, m, s, ms, us, ns) or a multiple of that e.g. "7D".
        warmup: float, optional
            Warmup period (in Days).
        istress: int, optional
            When multiple stresses are present in a stressmodel, this keyword can be
            used to obtain the contribution of an individual stress.
        return_warmup: bool, optional
            Include warmup in contribution calculation or not.
        p: array_like, optional
            array_like object with the values as floats representing the model
            parameters. See Model.get_parameters() for more info if parameters is None.

        Returns
        -------
        stress: pandas.Series or list of pandas.Series
            If one stress is present, a pandas Series is returned. If more are
            present, a list of pandas Series is returned.
        """
        if p is None:
            p = self.get_parameters(name)

        if tmin is None:
            tmin = self.settings["tmin"]
        if tmax is None:
            tmax = self.settings["tmax"]
        if freq is None:
            freq = self.settings["freq"]
        if warmup is None:
            warmup = self.settings["warmup"]
        else:
            warmup = Timedelta(warmup, "D")

        # use warmup
        if tmin:
            tmin_warm = (Timestamp(tmin) - warmup).floor(freq) + self.settings[
                "time_offset"
            ]
        else:
            tmin_warm = None

        kwargs = {"tmin": tmin_warm, "tmax": tmax, "freq": freq}
        if istress is not None:
            kwargs["istress"] = istress

        stress = self.stressmodels[name].get_stress(p=p, **kwargs)
        if not return_warmup:
            stress = stress.loc[tmin:tmax]

        return stress

    def _get_file_info(self) -> dict:
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
        except Exception as e:
            logger.debug(e)
            file_info["owner"] = "Unknown"

        return file_info

    def fit_report(
        self,
        corr: bool = False,
        stderr: bool = False,
        warnings: bool = True,
        output: str = None,
    ) -> str:
        """Method that reports on the fit after a model is optimized.

        Parameters
        ----------
        corr : bool, optional
            If True the parameter correlations are shown.
        stderr : bool, optional
            If True the standard error of the parameter values are shown. Please be
            aware of the conditions for reliable uncertainty estimates, more information
            here:
            https://pastas.readthedocs.io/en/master/examples/diagnostic_checking.html
        warnings : bool, optional
            print warnings in case of optimization failure, parameters hitting
            bounds, or length of responses exceeding calibration period.
        output : str, optional (deprecated)
            deprecated argument, use corr and stderr arguments
            instead.

        Returns
        -------
        report: str
            String with the report.

        Examples
        --------
        This method is called by the solve method if report=True, but can also be
        called on its own::

        >>> print(ml.fit_report)

        Notes
        -----
        The reported values for the fit use the residuals time series where possible.
        If interpolation is used this means that the result may slightly differ
        compared to using ml.simulate() and ml.observations().
        """
        model = {
            "nfev": self.solver.nfev,
            "nobs": self.observations().index.size,
            "noise": str(self.settings["noise"]),
            "tmin": str(self.settings["tmin"]),
            "tmax": str(self.settings["tmax"]),
            "freq": self.settings["freq"],
            "warmup": str(self.settings["warmup"]),
            "solver": self.settings["solver"],
        }

        fit = {
            "EVP": f"{self.stats.evp():.2f}",
            "R2": f"{self.stats.rsq():.2f}",
            "RMSE": f"{self.stats.rmse():.2f}",
            "AICc": f"{self.stats.aicc():.2f}",
            "BIC": f"{self.stats.bic():.2f}",
            "Obj": f"{self.solver.obj_func:.2f}",
            "___": "",
            "Interp.": "Yes" if self.interpolate_simulation else "No",
        }

        if output is not None:
            msg = "Use 'corr=True' instead."
            deprecate_args_or_kwargs("output", "2.0.0", reason=msg)
            if isinstance(output, str) and output == "full":
                corr = True

        parameters = self.parameters.loc[:, ["optimal", "initial", "vary"]].copy()

        if stderr:
            stderr = (
                self.parameters.loc[:, "stderr"] / self.parameters.loc[:, "optimal"]
            )
            parameters.loc[:, "stderr"] = stderr.abs().apply(
                _table_formatter_stderr, na_rep="nan"
            )

        # determine width of the fit_report
        len_fit = max([len(v) for v in fit.values()]) + max(
            [len(v) for v in fit.keys()]
        )
        len_model = max([len(v) for v in model.values() if isinstance(v, str)]) + max(
            [len(v) for v in model.keys()]
        )
        len_param = len(parameters.to_string().split("\n")[1])
        width = max((len_fit + len_model + 8), len_param)
        string = "{:{fill}{align}{width}}"
        string = "{:{fill}{align}{width}}"

        # Create the first header with model information and stats
        wspace = max(width - (11 + 14 + len(self.name)), 1)
        mspace = width - wspace - (11 + 14)
        header = (
            f"Fit report {self.name:<{mspace}.{mspace}}"
            f"{string.format('', fill=' ', align='>', width=wspace)}"
            f"Fit Statistics\n"
            f"{string.format('', fill='=', align='>', width=width)}\n"
        )

        basic = ""
        len_val4 = max([len(v) for v in fit.values()])
        wspace = width - (8 + 23 + 9 + len_val4)
        for (val1, val2), (val3, val4) in zip(model.items(), fit.items()):
            basic += f"{val1:<8}{val2:<23}{val3:<9}" f"{val4:>{wspace + len_val4}}\n"

        # Create the parameters block
        params = (
            f"\nParameters ({parameters.vary.sum()} optimized)\n"
            f"{string.format('', fill='=', align='>', width=width)}\n"
            f"{parameters.to_string()}"
        )

        if corr:
            cor = DataFrame(columns=["value"])
            for idx, col in combinations(self.solver.pcor, 2):
                if np.abs(self.solver.pcor.loc[idx, col]) > 0.5:
                    cor.loc[f"{idx} {col}"] = self.solver.pcor.loc[idx, col]

            corr = (
                f"\n\nParameter correlations |rho| > 0.5\n"
                f"{string.format('', fill='=', align='>', width=width)}"
                f"\n{cor.to_string(float_format='%.2f', header=False)}"
            )
        else:
            corr = ""

        if warnings:
            msg = []
            # model optimization unsuccessful
            if not self._solve_success:
                msg.append("Model parameters could not be estimated well.")

            # parameter bound warnings
            lowerhit, upperhit = self._check_parameters_bounds()
            nhits = upperhit.sum() + lowerhit.sum()

            if nhits > 0:
                for p in upperhit.index:
                    if upperhit.at[p]:
                        msg.append(
                            f"Parameter '{p}' on upper bound: "
                            f"{self.parameters.at[p, 'pmax']:.2e}"
                        )
                    elif lowerhit.at[p]:
                        msg.append(
                            f"Parameter '{p}' on lower bound: "
                            f"{self.parameters.at[p, 'pmin']:.2e}"
                        )
            # check response t_cutoff vs length calibration period
            response_tmax_check = self._check_response_tmax()
            if (~response_tmax_check["check_ok"]).any():
                mask = ~response_tmax_check["check_ok"]
                for i in response_tmax_check.loc[mask].index:
                    msg.append(f"Response tmax for '{i}' > than calibration period.")

            # create message
            if len(msg) > 0:
                msg = [
                    f"\n\nWarnings! ({len(msg)})\n"
                    f"{string.format('', fill='=', align='>', width=width)}"
                ] + msg
                warnings = "\n".join(msg)
            else:
                warnings = ""
        else:
            warnings = ""

        report = f"{header}{basic}{params}{warnings}{corr}"

        return report

    def _check_response_tmax(self, cutoff: Optional[float] = None) -> DataFrame:
        """Internal method to check if response tmax is smaller than calibration period.

        Parameters
        ----------
        cutoff : float, optional
            cutoff for response function, by default None, which uses cutoff defined
            for each response function.

        Returns
        -------
        check : pandas.DataFrame
            dataframe containing length calibration period, response tmax for each
            stressmodel, and check result.
        """

        len_oseries_calib = (self.settings["tmax"] - self.settings["tmin"]).days

        # only check stressmodels with a response function
        sm_names = [
            key for key, item in self.stressmodels.items() if item.rfunc is not None
        ]

        check = DataFrame(
            index=sm_names,
            columns=["len_oseries_calib", "response_tmax", "check_ok"],
        )
        check["len_oseries_calib"] = len_oseries_calib

        for sm_name in self.stressmodels:
            if isinstance(self.stressmodels[sm_name].rfunc, HantushWellModel):
                kwargs = {"warn": False}
            else:
                kwargs = {}
            check.at[sm_name, "response_tmax"] = self.get_response_tmax(
                sm_name, cutoff=cutoff, **kwargs
            )

        check["check_ok"] = check["response_tmax"] < check["len_oseries_calib"]

        return check

    def _check_parameters_bounds(self) -> Tuple[Series, Series]:
        """Internal method to check if the optimal parameters are close to pmin or pmax.

        Returns
        -------
        lowerhit: pandas.Series
            pandas series with boolean values of the parameters that are close to the
            minimum (pmin) values.
        upperhit: pandas.Series
            pandas series with boolean values of the parameters that are close to the
            maximum (pmax) values.
        """
        upperhit = Series(index=self.parameters.index, dtype=bool)
        lowerhit = Series(index=self.parameters.index, dtype=bool)

        for p in self.parameters.index:
            pmax = self.parameters.at[p, "pmax"]
            pmin = self.parameters.at[p, "pmin"]

            # calculate atol based on minimum, with max 1e-8
            # otherwise set 1 order of magnitude lower than minimum value
            if pmin == 0.0 or np.isnan(pmin):
                atol = 1e-8
            else:
                atol = np.min([1e-8, 10 ** (np.floor(np.log10(np.abs(pmin))) - 1)])

            # deal with NaNs in parameter bounds
            if np.isnan(pmax):
                pmax = np.inf
            if np.isnan(pmin):
                pmax = -np.inf

            # determine hits
            upperhit.at[p] = np.allclose(
                self.parameters.at[p, "optimal"], pmax, atol=atol, rtol=1e-5
            )
            lowerhit.at[p] = np.allclose(
                self.parameters.at[p, "optimal"], pmin, atol=atol, rtol=1e-5
            )

        return lowerhit, upperhit

    def to_dict(self, series: bool = True, file_info: bool = True) -> dict:
        """Method to export a model to a dictionary.

        Parameters
        ----------
        series: bool, optional
            True to export the time series (default), False to not export them.
        file_info: bool, optional
            Export file_info or not. See method Model.get_file_info.

        Notes
        -----
        Helper function for the self.to_file method. To increase backward
        compatibility most attributes are stored in dictionaries that can be updated
        when a model is created.
        """

        # Create a dictionary to store all data
        data = {
            "name": self.name,
            "oseries": self.oseries.to_dict(series=series),
            "parameters": self.parameters,
            "settings": self.settings,
            "stressmodels": dict(),
        }

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
        if self.solver:
            data["solver"] = self.solver.to_dict()

        # Update and save file information
        if file_info:
            data["file_info"] = self._get_file_info()

        return data

    def to_file(self, fname: str, series: Union[bool, str] = True, **kwargs) -> None:
        """Method to save the Pastas model to a file.

        Parameters
        ----------
        fname: str
            String with the name and the extension of the file. File extension has to
            be supported by Pastas. E.g. "model.pas"
        series: bool or str, optional
            Export the simulated series or not. If series is "original", the original
            series are exported, if series is "modified", the series are exported
            after being changed with the time series settings. Default is True.
        **kwargs:
            any argument that is passed to :mod:`pastas.io.base.dump`.

        See Also
        --------
        :mod:`pastas.io.base.dump`
        """
        self.name = validate_name(self.name, raise_error=True)

        # Get dicts for all data sources
        data = self.to_dict(series=series)

        # Write the dicts to a file
        return dump(fname, data, **kwargs)

    def copy(self, name: Optional[str] = None) -> ModelType:
        """Method to copy a model.

        Parameters
        ----------
        name: str, optional
            String with the name of the model. The old name plus is appended with
            '_copy' if no name is provided.

        Returns
        -------
        ml: pastas.model.Model
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

    def _check_stressmodel_compatibility(self) -> None:
        """Internal method to check if the stressmodels are compatible with the
        model."""
        for sm in self.stressmodels.values():
            if hasattr(sm, "_check_stressmodel_compatibility"):
                sm._check_stressmodel_compatibility(self)
