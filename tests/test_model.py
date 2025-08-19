"""Tests for the Model class in pastas.model."""

from typing import Any

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

import pastas as ps
from pastas.model import Model


@pytest.fixture
def simple_model() -> ps.Model:
    """Create a simple model for testing without any stressmodels."""
    # Create test data
    dates = pd.date_range(start="2000-01-01", end="2005-12-31", freq="D")
    head = pd.Series(
        np.sin(np.linspace(0, 2 * np.pi * 10, len(dates))) + 5.0,
        index=dates,
        name="obs_1",
    )

    # Create a model
    ml = ps.Model(head, name="test_model")

    return ml


@pytest.fixture
def param_fixture(ml_solved: ps.Model) -> tuple[str, float, float, float, bool]:
    """Fixture to provide a consistent parameter for testing."""
    param_name = "rch_A"
    orig_value = ml_solved.parameters.at[param_name, "initial"]
    orig_pmin = ml_solved.parameters.at[param_name, "pmin"]
    orig_pmax = ml_solved.parameters.at[param_name, "pmax"]
    orig_vary = ml_solved.parameters.at[param_name, "vary"]

    yield param_name, orig_value, orig_pmin, orig_pmax, orig_vary

    # Reset parameter after test
    ml_solved.set_parameter(
        param_name, initial=orig_value, pmin=orig_pmin, pmax=orig_pmax, vary=orig_vary
    )


class TestModelInitialization:
    """Test model initialization."""

    def test_init_with_minimal_args(self) -> None:
        """Test initialization with minimal arguments."""
        dates = pd.date_range(start="2000-01-01", end="2001-12-31", freq="D")
        head = pd.Series(np.random.normal(0, 1, len(dates)), index=dates)

        model = Model(head)

        assert model.oseries is not None
        assert model.constant is not None

    def test_init_with_name(self) -> None:
        """Test initialization with a name."""
        dates = pd.date_range(start="2000-01-01", end="2001-12-31", freq="D")
        head = pd.Series(np.random.normal(0, 1, len(dates)), index=dates, name="test")

        model = Model(head)

        assert model.name == "test"

        model = Model(head, name="custom_name")

        assert model.name == "custom_name"

    def test_init_without_constant(self) -> None:
        """Test initialization without constant."""
        dates = pd.date_range(start="2000-01-01", end="2001-12-31", freq="D")
        head = pd.Series(np.random.normal(0, 1, len(dates)), index=dates)

        model = Model(head, constant=False)

        assert model.constant is None

    def test_init_with_metadata(self) -> None:
        """Test initialization with metadata."""
        dates = pd.date_range(start="2000-01-01", end="2001-12-31", freq="D")
        head = pd.Series(np.random.normal(0, 1, len(dates)), index=dates)
        metadata = {
            "location": "test well",
            "x": 100,
            "y": 200,
            "z": 300,
            "projection": "EPSG:4326",
        }

        model = Model(head, metadata=metadata)

        assert model.oseries.metadata == metadata


class TestModelComponents:
    """Test adding and removing model components."""

    def test_add_stressmodel(self, simple_model: ps.Model) -> None:
        """Test adding a stress model."""
        dates = pd.date_range(start="2000-01-01", end="2005-12-31", freq="D")
        prec = pd.Series(
            np.random.gamma(2, 1, size=len(dates)), index=dates, name="prec"
        )

        sm = ps.StressModel(stress=prec, rfunc=ps.Exponential(), name="precipitation")
        simple_model.add_stressmodel(sm)

        assert "precipitation" in simple_model.stressmodels
        assert simple_model.stressmodels["precipitation"] is sm

    def test_stressmodel_params(self, simple_model: ps.Model) -> None:
        """Test getting stress model parameters."""
        dates = pd.date_range(start="2000-01-01", end="2005-12-31", freq="D")
        prec = pd.Series(
            np.random.gamma(2, 1, size=len(dates)), index=dates, name="prec"
        )

        sm = ps.StressModel(stress=prec, rfunc=ps.Exponential(), name="precipitation")

        assert isinstance(sm.parameters, pd.DataFrame)
        assert (
            sm.parameters.columns
            == pd.Index(
                [
                    "initial",
                    "pmin",
                    "pmax",
                    "vary",
                    "name",
                    "dist",
                ]
            )
        ).all()
        assert (
            sm.parameters.dtypes.values
            == np.array(
                [
                    np.dtypes.Float64DType(),
                    np.dtypes.Float64DType(),
                    np.dtypes.Float64DType(),
                    np.dtypes.BoolDType(),
                    np.dtypes.ObjectDType(),
                    np.dtypes.ObjectDType(),
                ]
            )
        ).all()

    def test_add_multiple_stressmodels(self, simple_model: ps.Model) -> None:
        """Test adding multiple stress models at once."""
        dates = pd.date_range(start="2000-01-01", end="2005-12-31", freq="D")
        prec = pd.Series(
            np.random.gamma(2, 1, size=len(dates)), index=dates, name="prec"
        )
        evap = pd.Series(
            np.random.gamma(1, 0.5, size=len(dates)), index=dates, name="evap"
        )

        sm1 = ps.StressModel(stress=prec, rfunc=ps.Exponential(), name="precipitation")
        sm2 = ps.StressModel(stress=evap, rfunc=ps.Exponential(), name="evaporation")

        simple_model.add_stressmodel([sm1, sm2])

        assert "precipitation" in simple_model.stressmodels
        assert "evaporation" in simple_model.stressmodels

    def test_add_stressmodel_with_same_name(self, ml_solved: ps.Model) -> None:
        """Test adding a stress model with the same name."""
        # Get the first stressmodel name
        first_sm_name = list(ml_solved.stressmodels.keys())[0]

        # Create a new stress model with the same name but different response function
        dates = pd.date_range(start="2000-01-01", end="2005-12-31", freq="D")
        prec = pd.Series(
            np.random.gamma(2, 1, size=len(dates)), index=dates, name="prec"
        )
        sm = ps.StressModel(stress=prec, rfunc=ps.Gamma(), name=first_sm_name)

        # Should replace the existing stress model and log a warning
        ml_solved.add_stressmodel(sm)

        # Check that it was replaced with the new one
        assert ml_solved.stressmodels[first_sm_name].rfunc._name == "Gamma"

        # With replace=False, should raise an error
        with pytest.raises(ValueError):
            ml_solved.add_stressmodel(sm, replace=False)

    def test_del_stressmodel(self, ml_solved: ps.Model) -> None:
        """Test deleting a stress model."""
        # Get the first stressmodel name
        first_sm_name = list(ml_solved.stressmodels.keys())[0]

        ml_solved.del_stressmodel(first_sm_name)
        assert first_sm_name not in ml_solved.stressmodels

    def test_del_stressmodel_nonexistent(self, simple_model: ps.Model) -> None:
        """Test deleting a non-existent stress model."""
        with pytest.raises(KeyError):
            simple_model.del_stressmodel("nonexistent")

    def test_add_constant(self, simple_model: ps.Model) -> None:
        """Test adding a constant."""
        simple_model.del_constant()
        assert simple_model.constant is None

        constant = ps.Constant(initial=10.0, name="constant")
        simple_model.add_constant(constant)

        assert simple_model.constant is constant
        assert simple_model.constant.name == "constant"

    def test_del_constant(self, simple_model: ps.Model, caplog: Any) -> None:
        """Test deleting a constant."""
        simple_model.del_constant()
        assert simple_model.constant is None

    def test_add_transform(self, simple_model: ps.Model) -> None:
        """Test adding a transform."""
        transform = ps.ThresholdTransform()
        simple_model.add_transform(transform)

        assert simple_model.transform is transform

    def test_del_transform(self, simple_model: ps.Model, caplog: Any) -> None:
        """Test deleting a transform."""
        # First add a transform
        transform = ps.ThresholdTransform()
        simple_model.add_transform(transform)

        # Then delete it
        simple_model.del_transform()
        assert simple_model.transform is None

    def test_add_noisemodel(self, simple_model: ps.Model) -> None:
        """Test adding a noise model."""
        noise = ps.ArmaNoiseModel()
        simple_model.add_noisemodel(noise)

        assert simple_model.noisemodel is noise
        assert simple_model.settings["noise"] is True

    def test_del_noisemodel(self, simple_model: ps.Model, caplog: Any) -> None:
        """Test deleting a noise model."""
        # First add a noise model
        noise = ps.ArmaNoiseModel()
        simple_model.add_noisemodel(noise)

        # Then delete it
        simple_model.del_noisemodel()
        assert simple_model.noisemodel is None
        assert simple_model.settings["noise"] is False


@pytest.mark.integration
class TestModelSimulation:
    """Test model simulation methods."""

    def test_simulate_basic(self, ml_solved: ps.Model) -> None:
        """Test basic simulation."""
        sim = ml_solved.simulate()

        assert isinstance(sim, pd.Series)
        assert not sim.empty
        assert sim.name == "Simulation"

    def test_recharge_tmax_warning(self, ml_solved: ps.Model) -> None:
        ml_solved.set_parameter("rch_a", optimal=1e4)
        check = ml_solved._check_response_tmax()
        assert not check.loc["rch", "check_warmup"]
        assert not check.loc["rch", "check_response"]

    def test_simulate_with_tmin_tmax(self, ml_solved: ps.Model) -> None:
        """Test simulation with specified tmin and tmax."""
        # Get index range midpoints
        index_min = ml_solved.oseries.series.index.min()
        index_max = ml_solved.oseries.series.index.max()
        midpoint = index_min + (index_max - index_min) / 2

        # Use the midpoint as tmin and 3/4 point as tmax
        tmin = midpoint.strftime("%Y-%m-%d")
        tmax = (midpoint + (index_max - midpoint) / 2).strftime("%Y-%m-%d")

        sim = ml_solved.simulate(tmin=tmin, tmax=tmax)

        assert sim.index[0] >= pd.Timestamp(tmin)
        assert sim.index[-1] <= pd.Timestamp(tmax)

    def test_simulate_with_freq(self, ml_solved: ps.Model) -> None:
        """Test simulation with different frequency."""
        sim_daily = ml_solved.simulate()
        sim_weekly = ml_solved.simulate(freq="7D")

        # The weekly simulation should have fewer points than the daily one
        assert len(sim_weekly) < len(sim_daily)

    def test_simulate_with_parameters(self, ml_solved: ps.Model) -> None:
        """Test simulation with provided parameters."""
        # Solve the model first
        ml_solved.solve(report=False)

        # Get optimal parameters
        p_opt = ml_solved.get_parameters()

        # Get a copy of ml_rm with initial parameters
        ml_copy = ml_solved.copy()
        ml_copy.initialize()

        # Simulate with initial parameters
        sim_init = ml_copy.simulate()

        # Simulate with optimal parameters
        sim_opt = ml_copy.simulate(p=p_opt)

        # Should be different unless the optimization didn't change parameters
        assert not np.all(sim_init.values == sim_opt.values)

    def test_simulate_with_warmup(self, ml_solved: ps.Model) -> None:
        """Test simulation with warmup period."""
        # Standard simulation without returning warmup
        sim = ml_solved.simulate(warmup=30)

        # Simulation with warmup returned
        sim_warmup = ml_solved.simulate(warmup=30, return_warmup=True)

        # Warmup simulation should be longer
        assert len(sim_warmup) > len(sim)

        # The last part of sim_warmup should be identical to sim
        assert_series_equal(sim_warmup.loc[sim.index], sim)

    def test_residuals(self, ml_noisemodel: ps.Model) -> None:
        """Test residuals calculation."""
        res = ml_noisemodel.residuals()

        assert isinstance(res, pd.Series)
        assert res.name == "Residuals"

        # Residuals should have mean close to zero for a fitted model
        assert abs(res.mean()) < 1.0

    def test_residuals_with_normalize(self, ml_solved: ps.Model) -> None:
        """Test residuals calculation with normalization."""
        ml_solved.normalize_residuals = True
        res = ml_solved.residuals()

        # Normalized residuals should have mean very close to zero
        assert abs(res.mean()) < 1e-10

    def test_observations(self, ml_solved: ps.Model) -> None:
        """Test observations method."""
        obs = ml_solved.observations()

        assert isinstance(obs, pd.Series)
        assert not obs.empty

    def test_observations_with_time_limits(self, ml_solved: ps.Model) -> None:
        """Test observations with time limits."""
        # Get the actual time range of the observations
        oseries = ml_solved.observations()
        actual_tmin = oseries.index.min().strftime("%Y-%m-%d")
        actual_tmax = oseries.index.max().strftime("%Y-%m-%d")

        # Use actual data range for the test
        obs = ml_solved.observations(tmin=actual_tmin, tmax=actual_tmax)

        # Check that we have observations and they respect the time limits
        assert not obs.empty
        assert obs.index[0] >= pd.Timestamp(actual_tmin)
        assert obs.index[-1] <= pd.Timestamp(actual_tmax)


class TestModelParameters:
    """Test model parameter handling."""

    def test_get_init_parameters(self, ml_solved: ps.Model) -> None:
        """Test getting initial parameters."""
        params = ml_solved.get_init_parameters()

        assert isinstance(params, pd.DataFrame)
        assert not params.empty
        assert "initial" in params.columns
        assert "vary" in params.columns

    def test_get_parameters(self, ml_solved: ps.Model) -> None:
        """Test getting parameters."""
        # Unsolved model returns initial parameters
        p_init = ml_solved.get_parameters()
        assert isinstance(p_init, np.ndarray)

        # Solve the model
        ml_solved.solve(report=False)

        # Solved model returns optimal parameters
        p_opt = ml_solved.get_parameters()
        assert isinstance(p_opt, np.ndarray)

    def test_get_parameters_by_name(self, ml_sm: ps.Model) -> None:
        """Test getting parameters for a specific component."""
        # Get all stressmodel names
        sm_names = list(ml_sm.stressmodels.keys())

        p_all = ml_sm.get_parameters()
        p_sm1 = ml_sm.get_parameters(sm_names[0])

        # p_sm1 should be shorter than p_all
        assert len(p_sm1) < len(p_all)

    @pytest.mark.parametrize(
        "param_attr,value,expected",
        [
            ("initial", 10.0, 10.0),
            ("vary", False, False),
            ("vary", True, True),
        ],
    )
    def test_set_parameter_attributes(
        self,
        ml_solved: ps.Model,
        param_fixture: tuple[str, float, float, float, bool],
        param_attr: str,
        value: Any,
        expected: Any,
    ) -> None:
        """Test setting different parameter attributes."""
        param_name = param_fixture[0]

        # Set the parameter attribute
        ml_solved.set_parameter(param_name, **{param_attr: value})

        # Check that it was updated correctly
        assert ml_solved.parameters.at[param_name, param_attr] == expected

    def test_set_parameter_nonexistent(self, ml_solved: ps.Model) -> None:
        """Test setting a non-existent parameter."""
        with pytest.raises(KeyError):
            ml_solved.set_parameter("nonexistent", initial=1.0)

    def test_set_parameter_bounds(
        self, ml_solved: ps.Model, param_fixture: tuple[str, float, float, float, bool]
    ) -> None:
        """Test setting parameter bounds."""
        param_name = param_fixture[0]

        # Set bounds
        ml_solved.set_parameter(param_name, initial=5, pmin=0.1, pmax=10.0)

        assert ml_solved.parameters.at[param_name, "pmin"] == 0.1
        assert ml_solved.parameters.at[param_name, "pmax"] == 10.0

    def test_set_parameter_move_bounds(
        self, ml_solved: ps.Model, param_fixture: tuple[str, float, float, float, bool]
    ) -> None:
        """Test moving parameter bounds."""
        param_name, orig_initial, orig_pmin, orig_pmax, _ = param_fixture

        # Double the initial value and move bounds
        new_initial = orig_initial * 2
        ml_solved.set_parameter(param_name, initial=new_initial, move_bounds=True)

        # Check that bounds were moved proportionally
        assert ml_solved.parameters.at[param_name, "pmin"] == pytest.approx(
            orig_pmin * 2
        )
        assert ml_solved.parameters.at[param_name, "pmax"] == pytest.approx(
            orig_pmax * 2
        )

    def test_set_parameter_move_bounds_error(self, ml_solved: ps.Model) -> None:
        """Test error when providing both bounds and move_bounds."""
        param_name = "rch_A"

        with pytest.raises(KeyError):
            ml_solved.set_parameter(param_name, initial=2.0, pmin=0.1, move_bounds=True)


@pytest.mark.integration
class TestModelSolving:
    """Test model solving."""

    def test_initialize(self, ml_solved: ps.Model) -> None:
        """Test model initialization before solving."""
        ml_solved.initialize()

        assert ml_solved.settings["tmin"] is not None
        assert ml_solved.settings["tmax"] is not None
        assert ml_solved.oseries_calib is not None

    def test_solve(self, ml_solved: ps.Model) -> None:
        """Test solving the model."""
        ml_solved.solve(report=False)

        assert ml_solved.solver is not None
        assert ml_solved.parameters["optimal"].notna().any()
        assert ml_solved._solve_success

    def test_solve_with_weights(self, ml_solved: ps.Model) -> None:
        """Test solving with weights."""
        # Create weights series with same index as observations
        weights = ml_solved.observations().copy()
        weights[:] = 1.0

        # Lower weights for some periods
        weights.loc["2002":"2003"] = 0.5

        ml_solved.solve(weights=weights, report=False)

        assert ml_solved.settings["weights"] is weights

    def test_fit_report(self, ml_noisemodel: ps.Model) -> None:
        """Test fit report generation."""
        report = ml_noisemodel.fit_report()

        assert isinstance(report, str)
        assert "Fit report" in report
        assert "Parameters" in report

        # Test with correlation matrix
        report_corr = ml_noisemodel.fit_report(corr=True)
        assert "Parameter correlations" in report_corr

        # Test with stderr
        report_stderr = ml_noisemodel.fit_report(stderr=True)
        assert "stderr" in report_stderr


class TestModelContributions:
    """Test getting model contributions."""

    @pytest.mark.parametrize(
        "method_name,series_name",
        [
            ("get_contribution", None),  # Series name will be the stressmodel name
            ("get_block_response", None),
            ("get_step_response", None),
        ],
    )
    def test_contribution_methods(
        self, ml_noisemodel: ps.Model, method_name: str, series_name: str | None
    ) -> None:
        """Test various contribution-related methods."""
        # Get the first stressmodel name
        first_sm_name = list(ml_noisemodel.stressmodels.keys())[0]

        # Call the method
        method = getattr(ml_noisemodel, method_name)
        result = method(first_sm_name)

        # Check result
        assert isinstance(result, pd.Series)
        if series_name:
            assert result.name == series_name
        if method_name == "get_step_response":
            assert result.index.name == "Time [days]"
        if method_name == "get_block_response":
            assert result.index.name == "Time [days]"

    def test_get_contributions(self, ml_sm: ps.Model) -> None:
        """Test getting all contributions."""
        ml_sm.solve(report=False)
        contribs = ml_sm.get_contributions()

        assert isinstance(contribs, list)
        assert len(contribs) >= 2  # At least two contributions
        assert all(isinstance(c, pd.Series) for c in contribs)

    def test_get_output_series(self, ml_noisemodel: ps.Model) -> None:
        """Test getting all output series."""
        df = ml_noisemodel.get_output_series()

        assert isinstance(df, pd.DataFrame)
        assert "Head_Calibration" in df.columns
        assert "Simulation" in df.columns
        assert "Residuals" in df.columns

    def test_get_response_tmax(self, ml_noisemodel: ps.Model) -> None:
        """Test getting response tmax."""
        # Get the first stressmodel name
        first_sm_name = list(ml_noisemodel.stressmodels.keys())[0]

        tmax = ml_noisemodel.get_response_tmax(first_sm_name)

        assert isinstance(tmax, float)
        assert tmax > 0

    def test_get_stress(self, ml_noisemodel: ps.Model) -> None:
        """Test getting stress series."""
        # Get the first stressmodel name
        first_sm_name = list(ml_noisemodel.stressmodels.keys())[0]

        stress = ml_noisemodel.get_stress(first_sm_name)

        assert isinstance(stress, pd.Series)


class TestModelExportImport:
    """Test model export and import."""

    def test_to_dict(self, ml_noisemodel: ps.Model) -> None:
        """Test exporting model to dictionary."""
        data = ml_noisemodel.to_dict()

        assert isinstance(data, dict)
        assert "name" in data
        assert "oseries" in data
        assert "parameters" in data
        assert "stressmodels" in data

    @pytest.mark.slow
    def test_to_file_and_load(self, ml_noisemodel: ps.Model, tmp_path: Any) -> None:
        """Test exporting model to file and loading it."""
        # Save model to file
        fname = tmp_path / "test_model.pas"
        ml_noisemodel.to_file(fname)

        # Load model from file
        loaded_model = ps.io.load(fname)

        # Check basic properties
        assert loaded_model.name == ml_noisemodel.name
        assert loaded_model.oseries.name == ml_noisemodel.oseries.name
        assert loaded_model.stressmodels.keys() == ml_noisemodel.stressmodels.keys()

        # Check parameters
        assert_frame_equal(loaded_model.parameters, ml_noisemodel.parameters)

    def test_copy(self, ml_noisemodel: ps.Model) -> None:
        """Test copying a model."""
        copy_model = ml_noisemodel.copy(name="copy_test")

        assert copy_model.name == "copy_test"
        assert copy_model is not ml_noisemodel
        assert_frame_equal(copy_model.parameters, ml_noisemodel.parameters)
