"""Common fixtures for pastas tests."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pastas as ps

data_path = Path(__file__).parent / "data"


# Test data generation helpers
def generate_test_data(
    start_date: str = "2010-01-01", end_date: str = "2015-12-31", freq: str = "D"
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Generate test time series data."""
    dates = pd.date_range(start=start_date, end=end_date, freq=freq)

    # Observation series (head)
    head = (
        pd.read_csv(data_path / "obs.csv", index_col=0, parse_dates=True)
        .squeeze()
        .dropna()
    )
    # Precipitation series
    prec = pd.read_csv(data_path / "rain.csv", index_col=0, parse_dates=True).squeeze()

    # Evaporation series
    evap = pd.read_csv(data_path / "evap.csv", index_col=0, parse_dates=True).squeeze()

    # Temperature series
    index = (
        pd.read_csv(data_path / "evap.csv", index_col=0, parse_dates=True)
        .squeeze()
        .index
    )
    temp = pd.Series(
        index=index, data=np.sin(np.arange(index.size) / 2200), dtype=float
    )

    # Step series (e.g., pumping)
    step = pd.Series(np.zeros(len(dates)), index=dates, name="step")
    step.loc[dates[len(dates) // 3] :] = 1.0  # Step at 1/3 of the time series

    return head, prec, evap, temp, step


# Basic test data fixtures
@pytest.fixture(scope="session")
def test_data() -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Return all test data series."""
    return generate_test_data()


@pytest.fixture(scope="session")
def head(test_data: tuple[pd.Series, ...]) -> pd.Series:
    """Return head observation series."""
    return test_data[0]


@pytest.fixture(scope="session")
def prec(test_data: tuple[pd.Series, ...]) -> pd.Series:
    """Return precipitation series."""
    return test_data[1]


@pytest.fixture(scope="session")
def evap(test_data: tuple[pd.Series, ...]) -> pd.Series:
    """Return evaporation series."""
    return test_data[2]


@pytest.fixture(scope="session")
def temp(test_data: tuple[pd.Series, ...]) -> pd.Series:
    """Return temperature series."""
    return test_data[3]


@pytest.fixture(scope="session")
def step(test_data: tuple[pd.Series, ...]) -> pd.Series:
    """Return step series."""
    return test_data[4]


# Model fixtures
@pytest.fixture
def ml_basic(head: pd.Series) -> ps.Model:
    """Return a basic model with just a head series."""
    return ps.Model(head, name="basic_model")


@pytest.fixture
def ml_recharge(head: pd.Series, prec: pd.Series, evap: pd.Series) -> ps.Model:
    """Return a model with a recharge (rain) model."""
    ml = ps.Model(head, name="recharge_model")
    sm = ps.RechargeModel(prec, evap, name="rch", rfunc=ps.Exponential())
    ml.add_stressmodel(sm)
    return ml


@pytest.fixture
def ml_solved(ml_recharge: ps.Model) -> ps.Model:
    """Return a model with a recharge (rain) model."""
    ml = ml_recharge.copy()
    ml.solve(report=False)
    return ml


@pytest.fixture
def ml_sm(head: pd.Series, prec: pd.Series, evap: pd.Series) -> ps.Model:
    """Return a model with multiple stress models."""
    ml = ps.Model(head, name="multistress_model")
    sm1 = ps.StressModel(prec, name="prec", rfunc=ps.Exponential(), settings="prec")
    sm2 = ps.StressModel(evap, name="evap", rfunc=ps.Exponential(), settings="evap")
    ml.add_stressmodel([sm1, sm2])
    return ml


@pytest.fixture
def ml_step_and_exp(head: pd.Series, prec: pd.Series, step: pd.Series) -> ps.Model:
    """Return a model with step and exponential response functions."""
    ml = ps.Model(head, name="step_exp_model")
    sm1 = ps.StressModel(prec, name="prec", rfunc=ps.Exponential(), settings="prec")
    sm2 = ps.StressModel(step, name="step", rfunc=ps.StepResponse())
    ml.add_stressmodel([sm1, sm2])
    return ml


@pytest.fixture
def ml_with_transform(ml_solved: ps.Model) -> ps.Model:
    """Add a transform to the basic recharge model."""
    transform = ps.ThresholdTransform()
    ml_solved.add_transform(transform)
    return ml_solved


@pytest.fixture
def ml_noisemodel(ml_solved: ps.Model) -> ps.Model:
    """Return an already solved model."""
    ml_copy = ml_solved.copy()
    noise = ps.ArNoiseModel()
    ml_copy.add_noisemodel(noise)
    ml_copy.solve(report=False)
    return ml_copy


@pytest.fixture
def ml_bad() -> ps.Model:
    return ps.io.load(data_path / "ml_bad.pas")


# Test markers
def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest markers."""
    config.addinivalue_line("markers", "slow: mark test as slow")
    config.addinivalue_line("markers", "integration: mark as integration test")
    config.addinivalue_line("markers", "plotting: mark as test that produces plots")
