Getting started
===============
After installation




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
