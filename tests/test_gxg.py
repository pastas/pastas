# -*- coding: utf-8 -*-
"""

Author: T. van Steijn, R.A. Collenteur, 2017

"""

import numpy as np
import pandas as pd

import pastas as ps


class TestGXG(object):
    def test_ghg(self):
        idx = pd.to_datetime(["20160114", "20160115", "20160128", "20160214"])
        s = pd.Series([10.0, 3.0, 30.0, 20.0], index=idx)
        v = ps.stats.ghg(s, min_n_meas=1, min_n_years=1)
        assert v == 30.0

    def test_ghg_ffill(self):
        idx = pd.to_datetime(["20160101", "20160115", "20160130"])
        s = pd.Series([0.0, 0.0, 10.0], index=idx)
        v = ps.stats.ghg(s, fill_method="ffill", limit=15, min_n_meas=1, min_n_years=1)
        assert v == 0.0

    def test_ghg_bfill(self):
        idx = pd.to_datetime(["20160101", "20160115", "20160130"])
        s = pd.Series([0.0, 0.0, 10.0], index=idx)
        v = ps.stats.ghg(s, fill_method="bfill", limit=15, min_n_meas=1, min_n_years=1)
        # TODO is this correct?
        assert v == 10.0

    def test_ghg_linear(self):
        idx = pd.to_datetime(["20160101", "20160110", "20160120", "20160130"])
        s = pd.Series([0.0, 0.0, 10.0, 10.0], index=idx)
        v = ps.stats.ghg(s, fill_method="linear", min_n_meas=1, min_n_years=1, limit=8)
        # TODO is this correct?
        assert v == 10.0

    def test_ghg_len_yearly(self):
        idx = pd.date_range("20000101", "20550101", freq="d")
        s = pd.Series(np.ones(len(idx)), index=idx)
        v = ps.stats.ghg(s, output="yearly")
        assert v.notna().sum() == 55

    def test_glg(self):
        idx = pd.date_range("20000101", "20550101", freq="d")
        s = pd.Series(
            [x.month + x.day for x in idx],
            index=idx,
        )
        v = ps.stats.glg(s, year_offset="a")
        assert v == 16.0

    def test_glg_fill_limit(self):
        idx = pd.to_datetime(["20170115", "20170130", "20200101"])
        s = pd.Series(np.ones(len(idx)), index=idx)
        v = ps.stats.glg(
            s,
            fill_method="linear",
            limit=15,
            output="yearly",
            year_offset="a",
            min_n_meas=1,
        )
        assert v.notna().sum() == 2

    def test_glg_fill_limit_null(self):
        idx = pd.to_datetime(["20170101", "20170131", "20200101"])
        s = pd.Series(np.ones(len(idx)), index=idx)
        v = ps.stats.glg(
            s,
            fill_method="linear",
            limit=None,
            output="yearly",
            year_offset="a",
            min_n_meas=1,
        )
        assert v.notna().sum() == 3

    def test_gvg(self):
        idx = pd.to_datetime(["20170314", "20170328", "20170414", "20170428"])
        s = pd.Series([1.0, 2.0, 3.0, 4], index=idx)
        v = ps.stats.gvg(
            s, fill_method="linear", output="mean", min_n_meas=1, min_n_years=1
        )
        assert v == 2.0

    def test_gvg_nan(self):
        idx = pd.to_datetime(["20170228", "20170428", "20170429"])
        s = pd.Series([1.0, 2.0, 3.0], index=idx)
        v = ps.stats.gvg(
            s, fill_method=None, output="mean", min_n_meas=1, min_n_years=1
        )
        assert np.isnan(v)

        # def test_gxg_series(self):
        #     s = pd.read_csv('data\\hseries_gxg.csv', index_col=0, header=0,
        #                     parse_dates=True, dayfirst=True).squeeze()
        #     ps = Model(s)
        #     ps.freq = 'D'
        #     ghg = ps.stats.ghg(s)
        #     glg = ps.stats.glg(s)
        #     gvg = ps.stats.gvg(s)
        #     print('\n')
        #     print('calculated GXG\'s classic method: \n')
        #     print(('GHG: {ghg:.2f} m+NAP\n'
        #            'GLG: {glg:.2f} m+NAP\n'
        #            'GVG: {gvg:.2f} m+NAP\n').format(
        #         ghg=ghg, glg=glg, gvg=gvg))
        #     print('Menyanthes GXG\'s: \n')
        #     print(('GHG: {ghg:.2f} m+NAP\n'
        #            'GLG: {glg:.2f} m+NAP\n'
        #            'GVG: {gvg:.2f} m+NAP\n').format(
        #         ghg=-3.23, glg=-3.82, gvg=-3.43))

        # def test_gxg_series(self, capsys):
        #     s = pd.read_csv(r'data/hseries_gxg.csv', index_col=0, header=0,
        #         parse_dates=True, dayfirst=True).squeeze()
        #     ps = Model(s)
        #     ghg = ps.stats.ghg(s)
        #     glg = ps.stats.glg(s)
        #     gvg = ps.stats.gvg(s)
        #     with capsys.disabled():
        #         print('\n')
        #         print('calculated GXG\'s: \n')
        #         print(('GHG: {ghg:.2f} m+NAP\n'
        #                'GLG: {glg:.2f} m+NAP\n'
        #                'GVG: {gvg:.2f} m+NAP\n').format(
        #                ghg=ghg, glg=glg, gvg=gvg))
        #         print('Menyanthes GXG\'s: \n')
        #         print(('GHG: {ghg:.2f} m+NAP\n'
        #                'GLG: {glg:.2f} m+NAP\n'
        #                'GVG: {gvg:.2f} m+NAP\n').format(
        #                ghg=-3.23, glg=-3.82, gvg=-3.43))
