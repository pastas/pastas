# -*- coding: utf-8 -*-
"""

Author: T. van Steijn, R.A. Collenteur, 2017

"""

import numpy as np
import pandas as pd

import pastas as ps


class TestQGXG(object):
    def test_q_ghg(self):
        n = 101
        idx = pd.date_range("20160101", freq="d", periods=n)
        s = pd.Series(np.arange(n), index=idx)
        v = ps.stats.q_ghg(s, q=0.94)
        assert v == 94.0

    def test_q_glg(self):
        n = 101
        idx = pd.date_range("20160101", freq="d", periods=n)
        s = pd.Series(np.arange(n), index=idx)
        v = ps.stats.q_glg(s, q=0.06)
        assert v == 6.0

    def test_q_ghg_nan(self):
        idx = pd.date_range("20160101", freq="d", periods=4)
        s = pd.Series([1, np.nan, 3, np.nan], index=idx)
        v = ps.stats.q_ghg(s, q=0.5)
        assert v == 2.0

    def test_q_gvg(self):
        idx = pd.to_datetime(["20160320", "20160401", "20160420"])
        s = pd.Series([0, 5, 10], index=idx)
        v = ps.stats.q_gvg(s)
        assert v == 2.5

    def test_q_gvg_nan(self):
        idx = pd.to_datetime(["20160820", "20160901", "20161120"])
        s = pd.Series([0, 5, 10], index=idx)
        v = ps.stats.q_gvg(s)
        assert np.isnan(v)

    def test_q_glg_tmin(self):
        tmin = "20160301"
        idx = pd.date_range("20160101", "20160331", freq="d")
        s = pd.Series(np.arange(len(idx)), index=idx)
        v = ps.stats.q_glg(s, q=0.06, tmin=tmin)
        assert v == 61.8

    def test_q_ghg_tmax(self):
        n = 101
        tmax = "20160301"
        idx = pd.date_range("20160101", freq="d", periods=n)
        s = pd.Series(np.arange(n), index=idx)
        v = ps.stats.q_ghg(s, q=0.94, tmax=tmax)
        assert v == 56.4

    def test_q_gvg_tmin_tmax(self):
        tmin = "20170301"
        tmax = "20170401"
        idx = pd.to_datetime(["20160401", "20170401", "20180401"])
        s = pd.Series([0, 5, 10], index=idx)
        v = ps.stats.q_gvg(s, tmin=tmin, tmax=tmax)
        assert v == 5

    def test_q_gxg_series(self):
        s = pd.read_csv(
            "tests/data/hseries_gxg.csv",
            index_col=0,
            header=0,
            parse_dates=True,
            dayfirst=True,
        ).squeeze("columns")
        ghg = ps.stats.q_ghg(s)
        glg = ps.stats.q_glg(s)
        gvg = ps.stats.q_gvg(s)
        print("\n")
        print("calculated GXG's percentile method: \n")
        print(
            (
                "GHG: {ghg:.2f} m+NAP\n"
                "GLG: {glg:.2f} m+NAP\n"
                "GVG: {gvg:.2f} m+NAP\n"
            ).format(ghg=ghg, glg=glg, gvg=gvg)
        )
        print("Menyanthes GXG's: \n")
        print(
            (
                "GHG: {ghg:.2f} m+NAP\n"
                "GLG: {glg:.2f} m+NAP\n"
                "GVG: {gvg:.2f} m+NAP\n"
            ).format(ghg=-3.23, glg=-3.82, gvg=-3.43)
        )
