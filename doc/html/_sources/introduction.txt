============doc

Introduction
============
GWTSA is an open source python package for simulating time series in the field of
hydrology. The object oriented stucture allows for the quick implementation of new
model components. Time series models can be created, calibrated, and analysed with
just a few lines of python code with the built-in optimization, visualisation, and
statistical analysis tools.

If you think you have found a bug in GWTSA, or if you would like to suggest an
improvement or enhancement, please submit a new Issue through the Github Issue
tracker toward the upper-right corner of the Github repository. Pull requests will
only be accepted on the development branch (dev) of the repository.

========================
Quick installation guide
========================
To get the latest development version, use::

   git clone http://github.com/gwtwa/gwtsa.git

and install using::

   python setup.py install


========
Examples
========
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

========
Tutorial
========
Tutorial will be available soon, please refer to the examples folder for the time
 being.
