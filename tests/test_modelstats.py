"""Tests for the modelstats module."""

import numpy as np
import pandas as pd
import pytest

import pastas as ps


class TestStatistics:
    """Test Statistics class methods."""

    def test_rmse(self, ml_solved: ps.Model) -> None:
        """Test RMSE calculation."""
        # Calculate RMSE
        rmse = ml_solved.stats.rmse()

        # Should return a float value
        assert isinstance(rmse, float)

        # RMSE should be positive
        assert rmse > 0

        # Test weighted version
        rmse_weighted = ml_solved.stats.rmse(weighted=True)
        assert isinstance(rmse_weighted, float)

    def test_rmsn(self, ml_solved: ps.Model, ml_noisemodel: ps.Model) -> None:
        """Test RMSN calculation."""
        # Without noise model, should return nan
        rmsn = ml_solved.stats.rmsn()
        assert np.isnan(rmsn)

        # With noise model, should return a value
        rmsn = ml_noisemodel.stats.rmsn()
        assert isinstance(rmsn, float)
        assert rmsn > 0

    def test_sse(self, ml_solved: ps.Model) -> None:
        """Test SSE calculation."""
        sse = ml_solved.stats.sse()

        # Should return a float value
        assert isinstance(sse, float)

        # SSE should be positive
        assert sse > 0

        # SSE should be related to RMSE by n * RMSE^2
        n = len(ml_solved.observations())
        rmse = ml_solved.stats.rmse()
        assert sse == pytest.approx(n * rmse**2, rel=1e-10)

    def test_mae(self, ml_solved: ps.Model) -> None:
        """Test MAE calculation."""
        mae = ml_solved.stats.mae()

        # Should return a float value
        assert isinstance(mae, float)

        # MAE should be positive
        assert mae > 0

        # Test weighted version
        mae_weighted = ml_solved.stats.mae(weighted=True)
        assert isinstance(mae_weighted, float)

    def test_nse(self, ml_solved: ps.Model) -> None:
        """Test NSE calculation."""
        nse = ml_solved.stats.nse()

        # Should return a float value
        assert isinstance(nse, float)

        # NSE should be <= 1.0 (theoretical maximum)
        assert nse <= 1.0

    def test_nnse(self, ml_solved: ps.Model) -> None:
        """Test NNSE calculation."""
        nnse = ml_solved.stats.nnse()

        # Should return a float value
        assert isinstance(nnse, float)

        # NNSE should be between 0 and 1
        assert 0 <= nnse <= 1.0

    def test_pearsonr(self, ml_solved: ps.Model) -> None:
        """Test Pearson r calculation."""
        r = ml_solved.stats.pearsonr()

        # Should return a float value
        assert isinstance(r, float)

        # Pearson r should be between -1 and 1
        assert -1.0 <= r <= 1.0

    def test_evp(self, ml_solved: ps.Model) -> None:
        """Test EVP calculation."""
        evp = ml_solved.stats.evp()

        # Should return a float value
        assert isinstance(evp, float)

        # EVP should be between 0 and 100
        assert 0 <= evp <= 100

    def test_rsq(self, ml_solved: ps.Model) -> None:
        """Test R-squared calculation."""
        rsq = ml_solved.stats.rsq()

        # Should return a float value
        assert isinstance(rsq, float)

        # R-squared should be between 0 and 1
        assert rsq <= 1.0

    def test_kge(self, ml_solved: ps.Model) -> None:
        """Test KGE calculation."""
        kge = ml_solved.stats.kge()

        # Should return a float value
        assert isinstance(kge, float)

        # KGE should be <= 1.0 (theoretical maximum)
        assert kge <= 1.0

        # Test modified KGE
        kge_mod = ml_solved.stats.kge(modified=True)
        assert isinstance(kge_mod, float)

    def test_kge_2012(self, ml_solved: ps.Model) -> None:
        """Test KGE 2012 calculation."""
        kge_2012 = ml_solved.stats.kge_2012()

        # Should return a float value
        assert isinstance(kge_2012, float)

        # KGE 2012 should be <= 1.0 (theoretical maximum)
        assert kge_2012 <= 1.0

    def test_information_criteria(self, ml_solved: ps.Model) -> None:
        """Test information criteria calculations."""
        # Test AIC
        aic = ml_solved.stats.aic()
        assert isinstance(aic, float)

        # Test BIC
        bic = ml_solved.stats.bic()
        assert isinstance(bic, float)

        # Test AICc
        aicc = ml_solved.stats.aicc()
        assert isinstance(aicc, float)

        # BIC should be larger than AIC for models with multiple parameters
        assert bic > aic

        # AICc should be larger than AIC
        assert aicc > aic

    def test_summary(self, ml_solved: ps.Model) -> None:
        """Test summary method."""
        # Get summary with default stats
        summary = ml_solved.stats.summary()

        # Should return a DataFrame
        assert isinstance(summary, pd.DataFrame)

        # Should contain default stats
        for stat in ["rmse", "sse", "mae", "rsq", "evp"]:
            assert stat in summary.index

        # Get summary with specific stats
        selected_stats = ["rmse", "evp"]
        summary_selected = ml_solved.stats.summary(stats=selected_stats)

        # Should only contain specified stats
        assert len(summary_selected) == len(selected_stats)
        for stat in selected_stats:
            assert stat in summary_selected.index

    def test_diagnostics(self, ml_solved: ps.Model, ml_noisemodel: ps.Model) -> None:
        """Test diagnostics method."""
        # Get diagnostics for model without noise model
        diag = ml_solved.stats.diagnostics()

        # Should return a DataFrame
        assert isinstance(diag, pd.DataFrame)

        # Get diagnostics for model with noise model
        diag_noise = ml_noisemodel.stats.diagnostics()
        assert isinstance(diag_noise, pd.DataFrame)

    def test_tmin_tmax_filtering(self, ml_solved: ps.Model) -> None:
        """Test statistics with tmin/tmax filtering."""
        # Get full period statistic
        full_rmse = ml_solved.stats.rmse()

        # Get statistic for partial period
        dates = ml_solved.observations().index
        mid_point = dates[len(dates) // 2]
        partial_rmse = ml_solved.stats.rmse(tmin=mid_point)

        # Statistics should be different when calculated over different periods
        assert full_rmse != partial_rmse
