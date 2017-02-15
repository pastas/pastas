#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Uses pytest

run:
> python -m pytest test_GXG.py

"""

from pastas import Model, Recharge, Gamma, Linear, NoiseModel
import pandas as pd
import numpy as np

# debugging
import pdb


class TestGXG(object):
    def test_qGHG(self):
        n = 101
        idx = pd.date_range('20160101', freq='d', periods=n)
        s = pd.Series(np.arange(n), index=idx)
        ml = Model(s)
        v = ml.stats.qGHG(key='observations', q=.94)
        assert v == 94.

    def test_qGLG(self):
        n = 101
        idx = pd.date_range('20160101', freq='d', periods=n)
        s = pd.Series(np.arange(n), index=idx)
        ml = Model(s)
        v = ml.stats.qGLG(key='observations', q=.06)
        assert v == 6.

    def test_qGXG_nan(self):
        idx = pd.date_range('20160101', freq='d', periods=3)
        s = pd.Series([1, 3, np.nan], index=idx)
        ml = Model(s)
        v = ml.stats.qGHG(key='observations', q=.5)
        assert v == 2.

    def test_qGVG(self):
        idx = pd.to_datetime(['20160320', '20160401', '20160420'])        
        s = pd.Series([0, 5, 10], index=idx)
        ml = Model(s)
        v = ml.stats.qGVG(key='observations')
        assert v == 2.5

    def test_qGVG_nan(self):
        idx = pd.to_datetime(['20160820', '20160901', '20161120'])        
        s = pd.Series([0, 5, 10], index=idx)
        ml = Model(s)
        v = ml.stats.qGVG(key='observations')
        assert np.isnan(v)

    def test_qGXG_series(self, capsys):
        s = pd.read_csv(r'data/hseries_gxg.csv', index_col=0, header=0,
            parse_dates=True, dayfirst=True,
            squeeze=True,)
        ml = Model(s)
        ghg = ml.stats.qGHG(key='observations')
        glg = ml.stats.qGLG(key='observations')
        gvg = ml.stats.qGVG(key='observations')
        with capsys.disabled():
            print('\n')
            print('calculated GXG\'s: \n')
            print(('GHG: {ghg:.2f} m+NAP\n'
                   'GLG: {glg:.2f} m+NAP\n'
                   'GVG: {gvg:.2f} m+NAP\n').format(
                   ghg=ghg, glg=glg, gvg=gvg))
            print('Menyanthes GXG\'s: \n')
            print(('GHG: {ghg:.2f} m+NAP\n'
                   'GLG: {glg:.2f} m+NAP\n'
                   'GVG: {gvg:.2f} m+NAP\n').format(
                   ghg=-3.23, glg=-3.82, gvg=-3.43))
