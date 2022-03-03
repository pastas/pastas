Version 0.18 (2nd of September 2021)
------------------------------------
Minor update introducing a few new features and fixes some smaller bugs.

New Features / Enhancements
***************************

- Improved documentation of the uncertainty quantification methods
- Add section to fit report if parameters hit or are close to parameter bounds after optimization.
- Improve method :meth:`pastas.Model._check_parameters_bounds()` (used by
  above).
- Improve parameter bounds for HantushWellModel, no longer limited by maximum distance.
- `pastas.Model.__init__()` now accepts the `freq` argument, which can be
  used to set the simulation frequency of the Model at creating.
- :meth:`ps.plots.series` is added to quickly visualise the input time series.

Deprecations
************
- No methods have been deprecated in this version.

Backwards incompatible API changes
**********************************
- Official support for Python 3.6 is dropped in this version. Python 3.9
  support is added.

New Example Notebooks
*********************

- An example notebook on uncertainty quantification has been added.
