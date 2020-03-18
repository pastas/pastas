Examples
========
This page provides a short example and a list of example applications in
Jupyter Notebooks. Examples in the form of Python scripts can also be found
on the `examples directory on GitHub <https://github
.com/pastas/pastas/tree/master/examples>`_.

Below is a list of Jupyter Notebooks with code, comments and figures:

.. toctree::
    :maxdepth: 1
    :numbered:
    :glob:

    ./*

.. tip::
    The latest versions of the Jupyter Notebooks can be found in the
    examples folder on GitHub!

Short Example
-------------
Below is an example of a short script to simulate groundwater levels (the
csv-files with the data can be found in the examples-directory on GitHub)::

    import pandas as pd
    import pastas as ps

    oseries = pd.read_csv('data/head_nb1.csv', parse_dates=['date'],
                          index_col='date', squeeze=True)
    rain = pd.read_csv('data/rain_nb1.csv', parse_dates=['date'],
                       index_col='date', squeeze=True)
    evap = pd.read_csv('data/evap_nb1.csv', parse_dates=['date'],
                       index_col='date', squeeze=True)

    ml = ps.Model(oseries)
    sm = ps.RechargeModel(rain, evap, ps.Gamma, name='recharge')
    ml.add_stressmodel(sm)

    ml.solve()
    ml.plots.decomposition()

