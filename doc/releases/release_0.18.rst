Version 0.18 (31st of August 2021)
--------------------------------


New Features / Enhancements
***************************

- Improved documentation of the uncertainty quantification methods
- Add section to fit report if parameters hit or are close to parameter bounds after optimization.
- Improve method `pastas.Model._check_parameters_bounds()` (used by above).
- Improve parameter bounds for HantushWellModel, no longer limited by maximum distance.
- `pastas.Model.__init__()` now accepts the `freq` argument, which can be
  used to set the simulation frequency of the Model at creating.

Deprecations
************


Backwards incompatible API changes
**********************************


New Example Notebooks
*********************

- An example notebook on uncertainty quantification has been added.
-
