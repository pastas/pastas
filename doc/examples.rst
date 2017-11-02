========
Examples
========
This page provides a list of example applications in Jupyter Notebooks.
Examples in the form of Python scripts can also be found on the `examples directory on GitHub <https://github.com/pastas/pastas/tree/master/examples>`_.

**Example Notebooks**

.. toctree::
  :maxdepth: 1
  :glob:

  ../examples/**

**Short Example**

Examples of a short script to simulate groundwater levels::

   ml = Model(oseries)
   ts1 = StressModel2([rain, evap], Gamma(), name='recharge')
   ml.addtseries(ts1)
   d = Constant()
   ml.addtseries(d)
   ml.solve()
   ml.plot_results()

