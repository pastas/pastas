"""Tests for the stressmodels module."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

import pastas as ps
from pastas.stressmodels import (
    ChangeModel,
    Constant,
    LinearTrend,
    RechargeModel,
    StepModel,
    StressModel,
    WellModel,
)


@pytest.fixture
def stress_model(ml_sm: ps.Model) -> StressModel:
    """Create basic StressModel for testing."""
    return ml_sm.stressmodels["prec"]


class TestStressModelBase:
    """Test StressModelBase methods."""

    def test_update_stress(self, stress_model: StressModel) -> None:
        """Test updating stress settings."""
        # Get original frequency
        original_freq = stress_model.freq

        # Update to weekly frequency
        stress_model.update_stress(freq="7D")

        # Check if frequency was updated
        assert stress_model.freq == "7D"
        assert stress_model.stress.settings["freq"] == "7D"

        # Reset to original frequency
        stress_model.update_stress(freq=original_freq)

    def test_get_stress(self, stress_model: StressModel) -> None:
        """Test getting stress."""
        stress = stress_model.get_stress()
        assert isinstance(stress, pd.Series)
        assert len(stress) > 0

        # Test with time limits
        tmin = "2001-01-01"
        tmax = "2001-12-31"
        stress_limited = stress_model.get_stress(tmin=tmin, tmax=tmax)
        assert stress_limited.index[0] >= pd.Timestamp(tmin)
        assert stress_limited.index[-1] <= pd.Timestamp(tmax)


class TestStressModel:
    """Test StressModel."""

    def test_init(self, prec: pd.Series) -> None:
        """Test initialization."""
        sm = StressModel(stress=prec, rfunc=ps.Exponential(), name="test")
        assert sm.name == "test"
        assert isinstance(sm.rfunc, ps.Exponential)

        # Test with different settings
        sm = StressModel(
            stress=prec,
            rfunc=ps.Exponential(),
            name="test",
            settings={"fill_nan": "mean"},
        )
        assert sm.stress.settings["fill_nan"] == "mean"

    def test_simulate(self, stress_model: StressModel) -> None:
        """Test simulate method."""
        # Get parameters
        p = stress_model.parameters.initial.values

        # Run simulation
        sim = stress_model.simulate(p=p)

        # Check results
        assert isinstance(sim, pd.Series)
        assert len(sim) == len(stress_model.stress.series)
        assert not np.isnan(sim).any()

    def test_to_dict(self, stress_model: StressModel) -> None:
        """Test to_dict method."""
        data = stress_model.to_dict()

        # Check required keys
        required_keys = ["class", "rfunc", "name", "up", "stress"]
        for key in required_keys:
            assert key in data

        # Check values
        assert data["name"] == stress_model.name
        assert data["class"] == "StressModel"


class TestStepModel:
    """Test StepModel."""

    def test_init(self) -> None:
        """Test initialization."""
        sm = StepModel(tstart="2001-01-01", name="step1")
        assert sm.name == "step1"
        assert sm.tstart == pd.Timestamp("2001-01-01")

        # Check parameters
        assert sm.parameters.shape[0] > 0
        assert f"{sm.name}_tstart" in sm.parameters.index

    def test_simulate(self) -> None:
        """Test simulate method."""
        sm = StepModel(tstart="2001-01-01", name="step1")

        # Get parameters
        p = sm.parameters.initial.values

        # Run simulation with wide date range
        tmin = "2000-01-01"
        tmax = "2002-01-01"
        sim = sm.simulate(p=p, tmin=tmin, tmax=tmax, freq="D")

        # Check results
        assert isinstance(sim, pd.Series)
        assert sim.loc[:"2001-01-01"].mean() < sim.loc["2001-01-01":].mean()

        # Values before tstart should be 0, after should approach 1
        assert np.allclose(sim.loc[:"2001-01-01"].iloc[:-1].values, 0)
        assert sim.loc["2001-01-02":].mean() > 0.9


class TestLinearTrend:
    """Test LinearTrend."""

    def test_init(self) -> None:
        """Test initialization."""
        sm = LinearTrend(start="2001-01-01", end="2002-01-01", name="trend1")
        assert sm.name == "trend1"
        assert sm.start == "2001-01-01"
        assert sm.end == "2002-01-01"

    def test_simulate(self) -> None:
        """Test simulate method."""
        sm = LinearTrend(start="2001-01-01", end="2002-01-01", name="trend1")

        # Set positive trend
        p = sm.parameters.initial.to_numpy(copy=True)  # Make writable copy for pandas 3.0
        p[0] = 1.0  # Set slope to positive value

        # Run simulation
        tmin = "2000-01-01"
        tmax = "2003-01-01"
        sim = sm.simulate(p=p, tmin=tmin, tmax=tmax, freq="D")

        # Check results
        assert isinstance(sim, pd.Series)

        # No trend before start date
        pre_trend = sim.loc[:"2001-01-01"]
        assert np.allclose(pre_trend.diff().dropna().values, 0)

        # Positive trend during the trend period
        # Exclude the end date as it might transition to the post-trend period
        trend_period = sim.loc["2001-01-01":"2001-12-31"]
        assert (trend_period.diff().dropna() > 0).all()

        # No trend after end date
        post_trend = sim.loc["2002-01-01":]
        assert np.allclose(post_trend.diff().dropna().values, 0)


class TestConstant:
    """Test Constant."""

    def test_init(self) -> None:
        """Test initialization."""
        sm = Constant(name="constant", initial=5.0)
        assert sm.name == "constant"
        assert sm.initial == 5.0

        # Check parameter initialization
        assert sm.parameters.loc[f"{sm.name}_d", "initial"] == 5.0

    def test_simulate(self) -> None:
        """Test simulate method."""
        sm = Constant(name="constant", initial=5.0)

        # Test simulation (static value)
        result = sm.simulate(p=5.0)
        assert result == 5.0


class TestRechargeModel:
    """Test RechargeModel."""

    def setup_method(self) -> None:
        """Setup for tests."""
        # Create test data
        date_range = pd.date_range(start="2000-01-01", end="2001-12-31", freq="D")

        # Generate synthetic precipitation and evaporation (in mm/day)
        np.random.seed(42)
        self.prec = pd.Series(np.random.gamma(1, 2, len(date_range)), index=date_range)
        self.evap = pd.Series(
            np.random.gamma(0.5, 1, len(date_range)), index=date_range
        )

        # Create temperature data
        self.temp = pd.Series(
            np.sin(np.linspace(0, 4 * np.pi, len(date_range))) * 10
            + 15,  # 10-25°C range
            index=date_range,
        )

    def test_init_linear(self) -> None:
        """Test initialization with Linear recharge model."""
        rm = RechargeModel(
            prec=self.prec,
            evap=self.evap,
            name="rech1",
            recharge=ps.rch.Linear(),
        )
        assert rm.name == "rech1"
        assert rm.recharge._name == "Linear"
        assert len(rm.stresses) == 2  # prec and evap

    def test_init_flex(self) -> None:
        """Test initialization with FlexModel recharge model."""
        rm = RechargeModel(
            prec=self.prec,
            evap=self.evap,
            name="rech2",
            recharge=ps.rch.FlexModel(),
        )
        assert rm.name == "rech2"
        assert rm.recharge._name == "FlexModel"

    def test_temperature_required(self) -> None:
        """Test error when temp is required but not provided."""

        # Create recharge model that needs temperature

        with pytest.raises(TypeError) as e:
            RechargeModel(
                prec=self.prec,
                evap=self.evap,
                name="rech_error",
                recharge=ps.rch.FlexModel(snow=True),
            )
        assert "requires a temperature series" in str(e.value)

    def test_with_temperature(self) -> None:
        """Test initialization with temperature data."""

        # Create recharge model that needs temperature
        rm = RechargeModel(
            prec=self.prec,
            evap=self.evap,
            name="rech_temp",
            recharge=ps.rch.FlexModel(snow=True),
            temp=self.temp,
            settings=("prec", "evap", None),
            metadata=(None, None, None),
        )

        assert rm.name == "rech_temp"
        assert len(rm.stresses) == 3  # prec, evap, and temp
        assert rm.temp is not None

    def test_get_stress(self) -> None:
        """Test get_stress method."""
        rm = RechargeModel(
            prec=self.prec,
            evap=self.evap,
            name="rech_stress",
        )

        # Get recharge
        recharge = rm.get_stress()
        assert isinstance(recharge, pd.Series)
        assert len(recharge) == len(self.prec)

        # Get individual stresses
        prec = rm.get_stress(istress=0)
        evap = rm.get_stress(istress=1)

        assert prec.equals(rm.prec.series)
        assert evap.equals(rm.evap.series)


class TestWellModel:
    """Test WellModel."""

    def setup_method(self) -> None:
        """Setup for tests."""
        # Create test data
        date_range = pd.date_range(start="2000-01-01", end="2001-12-31", freq="D")

        # Generate synthetic pumping data
        np.random.seed(42)
        self.well1 = pd.Series(
            np.random.normal(10, 2, len(date_range)), index=date_range, name="well1"
        )
        self.well2 = pd.Series(
            np.random.normal(15, 3, len(date_range)), index=date_range, name="well2"
        )
        self.well3 = pd.Series(
            np.random.normal(8, 1, len(date_range)), index=date_range, name="well3"
        )

        self.distances = [100, 200, 300]  # in meters

    def test_init(self) -> None:
        """Test initialization."""
        wm = WellModel(
            stress=[self.well1, self.well2, self.well3],
            name="wells",
            distances=self.distances,
        )
        assert wm.name == "wells"
        assert len(wm.stresses) == 3
        assert len(wm.distances) == 3

        # Test with sorting
        wm_sorted = WellModel(
            stress=[self.well1, self.well2, self.well3],
            name="wells_sorted",
            distances=self.distances,
            sort_wells=True,
        )
        assert wm_sorted.distances.iloc[0] < wm_sorted.distances.iloc[-1]

    def test_init_error(self) -> None:
        """Test error when number of stresses and distances don't match."""
        with pytest.raises(ValueError) as e:
            WellModel(
                stress=[self.well1, self.well2],
                name="wells_error",
                distances=[100, 200, 300],
            )
        assert (
            "the number of distances must match the number of stresses."
            in str(e.value).lower()
        )

    def test_get_distances(self) -> None:
        """Test get_distances method."""
        wm = WellModel(
            stress=[self.well1, self.well2, self.well3],
            name="wells_dist",
            distances=self.distances,
            sort_wells=False,  # Keep original order for easier testing
        )

        # Get all distances
        all_distances = wm.get_distances()
        assert len(all_distances) == 3
        assert_array_equal(all_distances.values, self.distances)

        # Get specific distance
        dist1 = wm.get_distances(istress=0)
        assert dist1.iloc[0] == self.distances[0]

        # Get multiple distances
        dist_multi = wm.get_distances(istress=[0, 2])
        assert len(dist_multi) == 2
        assert dist_multi.iloc[0] == self.distances[0]
        assert dist_multi.iloc[1] == self.distances[2]

    def test_simulate(self) -> None:
        """Test simulate method."""
        wm = WellModel(
            stress=[self.well1, self.well2, self.well3],
            name="wells_sim",
            distances=self.distances,
        )

        # Get parameters
        p = wm.parameters.initial.values

        # Run simulation
        sim = wm.simulate(p=p)

        # Check results
        assert isinstance(sim, pd.Series)
        assert len(sim) == len(self.well1)
        assert sim.name == "wells_sim"
        # Test simulation of individual well
        sim1 = wm.simulate(p=p, istress=0)
        # The individual stress simulation uses the well name instead of the model name
        assert sim1.name == "well1"
        assert len(sim1) == len(self.well1)

    def test_set_stress(self) -> None:
        """Test set_stress method."""
        wm = WellModel(
            stress=[self.well1, self.well2, self.well3],
            name="wells_set",
            distances=self.distances,
        )

        # Create new stress series
        new_well1 = self.well1 * 2
        new_well2 = self.well2 * 2
        new_well3 = self.well3 * 2

        # Set new stresses
        wm.set_stress([new_well1, new_well2, new_well3])

        # Verify that stresses were updated
        assert wm.stresses[0].series.equals(new_well1)
        assert wm.stresses[1].series.equals(new_well2)
        assert wm.stresses[2].series.equals(new_well3)

        new_well4 = self.well1 * 3
        wm.stress = new_well4
        assert wm.stresses[0].series.equals(new_well4)

    def test_set_stress_error(self) -> None:
        """Test error when setting stress with incorrect number of series."""
        wm = WellModel(
            stress=[self.well1, self.well2, self.well3],
            name="wells_set_error",
            distances=self.distances,
        )

        new_well1 = self.well1 * 2
        new_well2 = self.well2 * 2

        with pytest.raises(ValueError) as e:
            wm.set_stress([new_well1, new_well2])
        assert (
            "the number of stress time series must match the number of distances."
            in str(e.value).lower()
        )

        new_well3 = (self.well3 * 2).rename("wrong_name")
        with pytest.raises(ValueError) as e:
            wm.set_stress(new_well3)
        assert "when setting a single stress series," in str(e.value).lower()

        new_well4 = (self.well1 * 2).rename("name1")
        new_well5 = (self.well2 * 2).rename("name2")
        new_well6 = (self.well3 * 2).rename("name3")
        with pytest.raises(ValueError) as e:
            wm.set_stress([new_well4, new_well5, new_well6])
        assert (
            "new names of the stress time series were provided that do not match the original names on initialization."
            in str(e.value).lower()
        )

        with pytest.raises(AttributeError) as e:
            wm.stress[0] = new_well1
        assert (
            "cannot set values in stress directly anymore. please use the `set_stress` method."
            in str(e.value).lower()
        )


class TestChangeModel:
    """Test ChangeModel."""

    def test_init(self, prec: pd.Series) -> None:
        """Test initialization."""
        cm = ChangeModel(
            stress=prec,
            rfunc1=ps.Exponential(),
            rfunc2=ps.Gamma(),
            name="change",
            tchange="2001-06-01",
        )
        assert cm.name == "change"
        assert cm.tchange == pd.Timestamp("2001-06-01")
        assert cm.rfunc1._name == "Exponential"
        assert cm.rfunc2._name == "Gamma"

    def test_simulate(self, prec: pd.Series) -> None:
        """Test simulate method."""
        cm = ChangeModel(
            stress=prec,
            rfunc1=ps.Exponential(),
            rfunc2=ps.Gamma(),
            name="change_sim",
            tchange="2010-06-01",
        )

        # Get parameters
        p = cm.parameters.initial.values

        # Run simulation
        sim = cm.simulate(p=p)

        # Check results
        assert isinstance(sim, pd.Series)
        assert len(sim) == len(prec)
        assert sim.name == "change_sim"

        # The simulation should include weighted contributions from both rfuncs
