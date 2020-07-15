Release Notes
=============

Starting with the release of Pastas 0.15 changes to the API will be
reported here. The release notes for previous releases up to 0.14 can be found
at the `GitHub Release page <https://github.com/pastas/pastas/releases>`_.
For full details of all changes check the commit log.

Version 0.15 (Expected End of July 2020)
----------------------------------------
This release will be partly be backward incompatible

New Features / Enhancements
***************************

- `ml.set_parameter` method on the Model class is introduced to set the
  initial, minimum, maximum and vary settings for a parameters in one line.

Deprecations
************

- `ml.set_vary`, `ml.set_initial`, `ml.set_pmin`, and `ml.set_pmax` are
  deprecated and will be removed in a future release. The use of `ml
  .set_parameter` method is now recommended.

Backwards incompatible API changes
**********************************

- The parameters of the Hantush response function have new names. This will
  cause problems when loading models using this function to be loaded from
  .pas-file. No fix is available for this.
- The first value of the noise series has changes (see Issue #152 for
  details), causing changes in the optimal parameter values.

