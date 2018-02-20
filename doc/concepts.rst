==================
Concepts of PASTAS
==================
A time-series model consists of one or more StressModels which together with a Constant and a NoiseModel form the simulation.
Most StressModels use a response-function, from the rfunc-module, that transform the stress in its contribution in the simulation.
Examples of response-functions are Gamma, Exponential or One (which is used for the Constant).
Each StressModel has a number of parameters, which are optimized by the Solver.
During optimization the innovations (or the residuals, the difference from the observations, when no NoiseModel is used) are minimized.


.. toctree::
  :maxdepth: 1
  :glob:

  concepts/sources.rst
  concepts/timesteps.rst
  concepts/stressmodels.rst
  concepts/solver.rst
  concepts/statistics.rst
  concepts/visualization.rst