Examples
=========
This page provides a list of example applications. For each example a Jupyter
Notebook is available at the `github examples repository <https://github.com/pastas/
pasta/tree/master/examples>`_.

.. toctree::
  :maxdepth: 1
  :glob:

  examples/**


Short Example
-------------
Examples of a short script to simulate groundwater levels::

   ml = Model(oseries)
   ts1 = Tseries2([rain, evap], Gamma(), name='recharge')
   ml.addtseries(ts1)
   d = Constant()
   ml.addtseries(d)
   n = NoiseModel()
   ml.addnoisemodel(n)
   ml.solve()
   ml.plot_results()
