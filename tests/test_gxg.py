#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Uses pytest

run:
> python -m pytest test_GXG.py

"""

import numpy as np
import pandas as pd

from pastas import Model


class TestGXG(object):
    def test_ghg(self):
        idx = pd.to_datetime(['20160114', '20160115', '20160128', '20160214'])
        s = pd.Series([10., 3., 30., 20.], index=idx)
        ml = Model(s)
        v = ml.stats.ghg(key='observations')
        assert v == 20.0

    def test_ghg_ffill(self):
        idx = pd.to_datetime(['20160101', '20160115', '20160130'])
        s = pd.Series([0., 0., 10.], index=idx)
        ml = Model(s)
        v = ml.stats.ghg(key='observations', fill_method='ffill')
        assert v == 0.

    def test_ghg_bfill(self):
        idx = pd.to_datetime(['20160101', '20160115', '20160130'])
        s = pd.Series([0., 0., 10.], index=idx)
        ml = Model(s)
        v = ml.stats.ghg(key='observations', fill_method='bfill')
        # TODO is this correct?
        assert v == 5.

    def test_ghg_linear(self):
        idx = pd.to_datetime(['20160101', '20160110', '20160120', '20160130'])
        s = pd.Series([0., 0., 10., 10.], index=idx)
        ml = Model(s)
        v = ml.stats.ghg(key='observations', fill_method='linear')
        # TODO is this correct?
        assert v == 7.

    def test_ghg_len_yearly(self):
        idx = pd.date_range('20000101', '20550101', freq='d')
        s = pd.Series(np.ones(len(idx)), index=idx)
        ml = Model(s)
        v = ml.stats.ghg(key='observations', output='yearly')
        assert len(v) == 55

    def test_glg(self):
        idx = pd.date_range('20000101', '20550101', freq='d')
        s = pd.Series([x.month + x.day for x in idx], index=idx)
        ml = Model(s)
        v = ml.stats.glg(key='observations')
        assert v == 16.

    def test_glg_fill_limit(self):
        idx = pd.to_datetime(['20170115', '20170130', '20200101'])
        s = pd.Series(np.ones(len(idx)), index=idx)
        ml = Model(s)
        v = ml.stats.glg(key='observations', fill_method='linear', limit=15,
                         output='yearly')
        assert v.count() == 1

    def test_glg_fill_limit_null(self):
        idx = pd.to_datetime(['20170101', '20170131', '20200101'])
        s = pd.Series(np.ones(len(idx)), index=idx)
        ml = Model(s)
        ml.freq = 'D'
        v = ml.stats.glg(key='observations', fill_method='linear', limit=10,
                         output='yearly')
        assert v.count() == 0

    def test_gvg(self):
        idx = pd.to_datetime(['20170314', '20170328', '20170414', '20170428'])
        s = pd.Series([1., 2., 3., 4], index=idx)
        ml = Model(s)
        ml.freq = 'D'
        v = ml.stats.gvg(key='observations', fill_method='linear',
                         output='mean')
        assert v == 2.

    def test_gvg_nan(self):
        idx = pd.to_datetime(['20170228', '20170428', '20170429'])
        s = pd.Series([1., 2., 3.], index=idx)
        ml = Model(s)
        ml.freq = 'D'
        v = ml.stats.gvg(key='observations', fill_method=None, output='mean', )
        assert np.isnan(v)

        # def test_gxg_series(self):
        #     s = pd.read_csv('data\\hseries_gxg.csv', index_col=0, header=0,
        #                     parse_dates=True, dayfirst=True, squeeze=True)
        #     ml = Model(s)
        #     ml.freq = 'D'
        #     ghg = ml.stats.ghg(key='observations')
        #     glg = ml.stats.glg(key='observations')
        #     gvg = ml.stats.gvg(key='observations')
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
        #         parse_dates=True, dayfirst=True,
        #         squeeze=True,)
        #     ml = Model(s)
        #     ghg = ml.stats.ghg(key='observations')
        #     glg = ml.stats.glg(key='observations')
        #     gvg = ml.stats.gvg(key='observations')
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
