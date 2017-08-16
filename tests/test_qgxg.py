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


class TestQGXG(object):
    def test_q_ghg(self):
        n = 101
        idx = pd.date_range('20160101', freq='d', periods=n)
        s = pd.Series(np.arange(n), index=idx)
        ml = Model(s)
        ml.freq = 'D'
        v = ml.stats.q_ghg(key='observations', q=.94)
        assert v == 94.

    def test_q_glg(self):
        n = 101
        idx = pd.date_range('20160101', freq='d', periods=n)
        s = pd.Series(np.arange(n), index=idx)
        ml = Model(s)
        ml.freq = 'D'
        v = ml.stats.q_glg(key='observations', q=.06)
        assert v == 6.

    def test_q_gxg_nan(self):
        idx = pd.date_range('20160101', freq='d', periods=4)
        s = pd.Series([1, np.nan, 3, np.nan], index=idx)
        ml = Model(s)
        ml.freq = 'D'
        v = ml.stats.q_ghg(key='observations', q=.5)
        assert v == 2.

    def test_q_gvg(self):
        idx = pd.to_datetime(['20160320', '20160401', '20160420'])
        s = pd.Series([0, 5, 10], index=idx)
        ml = Model(s)
        ml.freq = 'D'
        v = ml.stats.q_gvg(key='observations')
        assert v == 2.5

    def test_q_gvg_nan(self):
        idx = pd.to_datetime(['20160820', '20160901', '20161120'])
        s = pd.Series([0, 5, 10], index=idx)
        ml = Model(s)
        ml.freq = 'D'
        v = ml.stats.q_gvg(key='observations')
        assert np.isnan(v)

    def test_q_gxg_series(self):
        s = pd.read_csv('tests/data/hseries_gxg.csv', index_col=0, header=0,
                        parse_dates=True, dayfirst=True, squeeze=True)
        ml = Model(s)
        ml.freq = 'D'
        ghg = ml.stats.q_ghg(key='observations')
        glg = ml.stats.q_glg(key='observations')
        gvg = ml.stats.q_gvg(key='observations')
        print('\n')
        print('calculated GXG\'s percentile method: \n')
        print(('GHG: {ghg:.2f} m+NAP\n'
               'GLG: {glg:.2f} m+NAP\n'
               'GVG: {gvg:.2f} m+NAP\n').format(
            ghg=ghg, glg=glg, gvg=gvg))
        print('Menyanthes GXG\'s: \n')
        print(('GHG: {ghg:.2f} m+NAP\n'
               'GLG: {glg:.2f} m+NAP\n'
               'GVG: {gvg:.2f} m+NAP\n').format(
            ghg=-3.23, glg=-3.82, gvg=-3.43))

        # def test_q_gxg_series(self, capsys):
        #     s = pd.read_csv(r'data/hseries_gxg.csv', index_col=0, header=0,
        #         parse_dates=True, dayfirst=True,
        #         squeeze=True,)
        #     ml = Model(s)
        #     ghg = ml.stats.q_ghg(key='observations')
        #     glg = ml.stats.q_glg(key='observations')
        #     gvg = ml.stats.q_gvg(key='observations')
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
