import numpy as np
import pandas as pd
import pytest
from scipy import fft as sp_fft

import pastas as ps
from pastas._options import options
from pastas.timeseries_utils import convolve


def test_frequency_is_supported() -> None:
    ps.ts._frequency_is_supported("D")
    ps.ts._frequency_is_supported("7D")
    with pytest.raises(Exception):  # noqa: B017
        ps.ts._frequency_is_supported("SMS")


def test_get_stress_dt() -> None:
    assert ps.ts._get_stress_dt("D") == 1.0
    assert ps.ts._get_stress_dt("7D") == 7.0
    assert ps.ts._get_stress_dt("W") == 7.0
    assert ps.ts._get_stress_dt("SMS") == 15.0


def test_time_series_sampling_methods() -> None:
    # some helper functions to compute differences in performance
    def values_kept(s: pd.Series, original: pd.Series) -> int:
        diff = set(original.dropna().values) & set(s.dropna().values)
        return len(diff)

    def n_duplicates(s: pd.Series) -> int:
        return (s.value_counts() >= 2).sum()

    # Create timeseries
    freq = "2h"
    freq2 = "1h"
    idx0 = pd.date_range("2000-01-01 18:00:00", freq=freq, periods=3).tolist()
    idx1 = pd.date_range("2000-01-02 01:30:00", freq=freq2, periods=10).tolist()
    idx0 = idx0 + idx1
    idx0[3] = pd.Timestamp("2000-01-02 01:31:00")
    series = pd.Series(index=idx0, data=np.arange(len(idx0), dtype=float))
    series.iloc[8:10] = np.nan

    # Create equidistant timeseries
    s_pd1 = ps.ts.pandas_equidistant_sample(series, freq)
    s_pd2 = ps.ts.pandas_equidistant_nearest(series, freq)
    s_pd3 = ps.ts.pandas_equidistant_asfreq(series, freq)
    s_pastas1 = ps.ts.get_equidistant_series_nearest(
        series, freq, minimize_data_loss=False
    )
    s_pastas2 = ps.ts.get_equidistant_series_nearest(
        series, freq, minimize_data_loss=True
    )

    dfall = pd.concat(
        [series, s_pd1, s_pd2, s_pd3, s_pastas1, s_pastas2], axis=1, sort=True
    )
    dfall.columns = [
        "original",
        "pandas_equidistant_sample",
        "pandas_equidistant_nearest",
        "pandas_equidistant_asfreq",
        "get_equidistant_series_nearest (default)",
        "get_equidistant_series_nearest (minimize data loss)",
    ]
    valueskept = dfall.apply(values_kept, args=(dfall["original"],))
    duplicates = dfall.apply(n_duplicates)
    print(valueskept)
    assert np.all(valueskept.values == np.array([11, 4, 7, 8, 8, 9]))
    assert np.all(duplicates.values == np.array([0, 0, 2, 0, 0, 0]))


class TestConvolve:
    """Tests for convolve vs scipy.signal.fftconvolve."""

    RTOL = 1e-12

    @staticmethod
    def _ref(stress, b):
        """Reference: fftconvolve(..., 'full')[:n_points] per row."""
        from scipy.signal import fftconvolve

        stress = np.asarray(stress)
        b = np.asarray(b)
        n_points = stress.shape[-1]
        if stress.ndim == 1 and b.ndim == 1:
            return fftconvolve(stress, b, mode="full")[:n_points]
        stress2 = stress[np.newaxis, :] if stress.ndim == 1 else stress
        b2 = b[np.newaxis, :] if b.ndim == 1 else b
        n_real = max(stress2.shape[0], b2.shape[0])
        out = np.empty((n_real, n_points))
        for i in range(n_real):
            s = stress2[min(i, stress2.shape[0] - 1)]
            bi = b2[min(i, b2.shape[0] - 1)]
            out[i] = fftconvolve(s, bi, mode="full")[:n_points]
        return out

    def test_1d_fast_path(self) -> None:
        rng = np.random.default_rng(1)
        stress = rng.normal(size=500)
        b = rng.normal(size=37)
        out = convolve(stress, b)
        assert out.shape == (500,)
        np.testing.assert_allclose(
            out, self._ref(stress, b), rtol=self.RTOL, atol=1e-12
        )

    def test_1d_stress_2d_B(self) -> None:
        rng = np.random.default_rng(2)
        stress = rng.normal(size=500)
        B = rng.normal(size=(7, 123))
        out = convolve(stress, B)
        assert out.shape == (7, 500)
        np.testing.assert_allclose(
            out, self._ref(stress, B), rtol=self.RTOL, atol=1e-12
        )

    def test_2d_stress_2d_B(self) -> None:
        rng = np.random.default_rng(4)
        stress = rng.normal(size=(5, 400))
        B = rng.normal(size=(5, 80))
        out = convolve(stress, B)
        assert out.shape == (5, 400)
        np.testing.assert_allclose(
            out, self._ref(stress, B), rtol=self.RTOL, atol=1e-12
        )

    def test_2d_stress_1d_B(self) -> None:
        rng = np.random.default_rng(5)
        stress = rng.normal(size=(5, 400))
        b = rng.normal(size=80)
        ref = self._ref(stress, b)
        for B in (b, b[np.newaxis, :]):
            out = convolve(stress, B)
            assert out.shape == (5, 400)
            np.testing.assert_allclose(out, ref, rtol=self.RTOL, atol=1e-12)

    def test_prime_factor_lengths(self) -> None:
        # n_points + n_b - 1 = 2003 (prime): must still match fftconvolve
        rng = np.random.default_rng(6)
        stress = rng.normal(size=1000)
        B = rng.normal(size=(4, 1004))
        out = convolve(stress, B)
        np.testing.assert_allclose(
            out, self._ref(stress, B), rtol=self.RTOL, atol=1e-12
        )

    def test_complex_input(self) -> None:
        rng = np.random.default_rng(7)
        stress = rng.normal(size=300)
        B = rng.normal(size=(3, 50)) + 1j * 1e-20
        out = convolve(stress, B)
        assert np.iscomplexobj(out)
        np.testing.assert_allclose(
            out, self._ref(stress, B), rtol=self.RTOL, atol=1e-12
        )

    def test_workers_resolution_passed_to_fft(self, monkeypatch) -> None:
        # spy on the scipy transforms to capture the workers argument
        calls = []
        real_rfft, real_irfft = sp_fft.rfft, sp_fft.irfft

        def spy_rfft(x, n=None, axis=-1, **kwargs):
            calls.append(kwargs.get("workers"))
            return real_rfft(x, n=n, axis=axis, **kwargs)

        def spy_irfft(x, n=None, axis=-1, **kwargs):
            calls.append(kwargs.get("workers"))
            return real_irfft(x, n=n, axis=axis, **kwargs)

        monkeypatch.setattr(sp_fft, "rfft", spy_rfft)
        monkeypatch.setattr(sp_fft, "irfft", spy_irfft)

        rng = np.random.default_rng(8)
        stress = rng.normal(size=100)
        B = rng.normal(size=(3, 20))

        # parallel on + scipy workers untouched -> all cores (-1)
        # parallel off, or user-configured scipy workers -> None (scipy default)
        old = options.parallel
        try:
            with sp_fft.set_workers(1):
                options.parallel = True
                calls.clear()
                out = convolve(stress, B)
                np.testing.assert_allclose(
                    out, self._ref(stress, B), rtol=self.RTOL, atol=1e-12
                )
                assert calls and all(w == -1 for w in calls)
                options.parallel = False
                calls.clear()
                convolve(stress, B)
                assert calls and all(w is None for w in calls)
            with sp_fft.set_workers(2):
                options.parallel = True
                calls.clear()
                convolve(stress, B)
                assert calls and all(w is None for w in calls)  # user setting kept
        finally:
            options.parallel = old
