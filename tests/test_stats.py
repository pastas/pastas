import numpy as np
import pandas as pd
import pytest

import pastas as ps
from pastas.stats.tests import durbin_watson, ljung_box


def acf_func(**kwargs) -> tuple[np.ndarray, np.ndarray]:
    index = pd.to_datetime(np.arange(0, 100, 1), unit="D", origin="2000")
    index = kwargs.pop("index", index)
    data = np.sin(np.linspace(0, 10 * np.pi, 100))
    r = pd.Series(data=data, index=index)
    lags = np.arange(1.0, 11.0)
    lags = kwargs.pop("lags", lags)
    acf_true_len = lags + 1 if isinstance(lags, int) else len(lags) + 1
    acf_true = np.cos(np.linspace(0.0, np.pi, acf_true_len))[1:]
    acf = ps.stats.acf(r, lags=lags, min_obs=1, **kwargs).values
    return acf, acf_true


def test_acf_rectangle() -> None:
    acf, acf_true = acf_func(bin_method="rectangle")
    assert abs((acf - acf_true)).max() < 0.05


def test_acf_gaussian() -> None:
    acf, acf_true = acf_func(bin_method="gaussian")
    assert abs((acf - acf_true)).max() < 0.05


def test_acf_hourly() -> None:
    index = pd.to_datetime(np.arange(0, 100, 1), unit="h", origin="2000")
    lags = 10
    acf, acf_true = acf_func(index=index, lags=lags)
    assert abs((acf - acf_true)).max() < 0.05


def test_runs_test() -> None:
    """
    http://www.itl.nist.gov/div898/handbook/eda/section3/eda35d.htm
    True Z-statistic = 2.69
    Read NIST test data
    """
    data = pd.read_csv("tests/data/nist.csv")
    test, _ = ps.stats.runs_test(data)
    assert test[0] - 2.69 < 0.02


def test_stoffer_toloi() -> None:
    res = pd.Series(
        index=pd.date_range(start=0, periods=1000, freq="D"), data=np.random.rand(1000)
    )
    _, pval = ps.stats.stoffer_toloi(res)
    assert pval > 1e-10


@pytest.fixture
def random_series() -> pd.Series:
    """Create a random time series for testing."""
    np.random.seed(42)  # For reproducibility
    index = pd.date_range(start="2000-01-01", periods=1000, freq="D")
    data = np.random.normal(0, 1, 1000)
    return pd.Series(data=data, index=index)


@pytest.fixture
def autocorrelated_series() -> pd.Series:
    """Create an autocorrelated time series for testing."""
    np.random.seed(42)  # For reproducibility
    index = pd.date_range(start="2000-01-01", periods=1000, freq="D")

    # AR(1) process with phi=0.8 to create autocorrelation
    data = np.zeros(1000)
    data[0] = np.random.normal(0, 1)
    for i in range(1, 1000):
        data[i] = 0.8 * data[i - 1] + np.random.normal(0, 0.5)

    return pd.Series(data=data, index=index)


def test_durbin_watson_random(random_series: pd.Series) -> None:
    """Test durbin_watson on random data with no autocorrelation."""
    dw_stat, _ = durbin_watson(random_series)
    # For random data, DW should be close to 2
    assert 1.8 < dw_stat < 2.2


def test_durbin_watson_autocorrelated(autocorrelated_series: pd.Series) -> None:
    """Test durbin_watson on autocorrelated data."""
    dw_stat, _ = durbin_watson(autocorrelated_series)
    # For positively autocorrelated data, DW should be < 2
    assert dw_stat < 1.5


def test_ljung_box_random(random_series: pd.Series) -> None:
    """Test ljung_box on random data with no autocorrelation."""
    q_stat, p_value = ljung_box(random_series, lags=15)

    # For random data, p-value should be high (fail to reject H0)
    assert p_value > 0.05
    # Q-statistic should be relatively low
    assert q_stat < 25


def test_ljung_box_autocorrelated(autocorrelated_series: pd.Series) -> None:
    """Test ljung_box on autocorrelated data."""
    q_stat, p_value = ljung_box(autocorrelated_series, lags=15)

    # For autocorrelated data, p-value should be low (reject H0)
    assert p_value < 0.05
    # Q-statistic should be relatively high
    assert q_stat > 100


def test_ljung_box_with_parameters(random_series: pd.Series) -> None:
    """Test ljung_box with model parameters."""
    # Test with nparam=2 (simulating 2 parameters in model)
    q_stat, p_value = ljung_box(random_series, lags=15, nparam=2)

    # Degrees of freedom should be reduced, affecting p-value
    q_stat_no_param, p_value_no_param = ljung_box(random_series, lags=15, nparam=0)

    # Same Q-statistic but different p-values due to different degrees of freedom
    assert np.isclose(q_stat, q_stat_no_param)
    assert p_value != p_value_no_param


def test_ljung_box_full_output(random_series: pd.Series) -> None:
    """Test ljung_box with full_output=True."""
    result = ljung_box(random_series, lags=15, full_output=True)

    # Should return a DataFrame
    assert isinstance(result, pd.DataFrame)
    # Should contain Q Stat and P-value columns
    assert "Q Stat" in result.columns
    assert "P-value" in result.columns
    # Should have lags rows (excluding zero lag)
    assert len(result) <= 15  # May be less if some lags couldn't be computed
