
Version 0.17.1 (12th of May 2021)
---------------------------------

.. note::
    Intermediate release to solve an Issue with the LmFitSolver that is
    experienced by many users. See
    `Issue 295 <https://github.com/pastas/pastas/issues/295>`_.


Version 0.17 (2nd of April 2021)
--------------------------------

This version primarily fixes a number of Bugs and introduced some
performance improvements. This version should be backward-compatible with
the previous release.

.. note::
    The :class:`Pastas.Project` class has been removed from Pastas in this
    release. `Pastastore <https://github.com/pastas/pastastore>`_ is now
    recommended to deal with multiple time series models.

New Features / Enhancements
***************************

- :meth:`ml.set_parameter` now also takes an `optimal` argument.
- :meth:`ml.get_stressmodel_settings` has been added. This method can be
  used to obtains the TimeSeries settings in a stressmodel.
- Predefined TimeSeries settings can now be accessed through `ps.rcParams`
- The Parameters of the response function :class:`ps.Hantush` have been
  rescaled such that parameter 'A' is the gain.
- The metrics :meth:`AIC` and :meth:`BIC` now use the noise when available,
  and residuals otherwise.
- The use of f-strings is now the preferred method for strings. For the
  logger messages s-strings are now consistently applied to further speed-up
  the code.
- The method :meth:`ml.plots.cum_frequency` is added to the plotting library.


Deprecations
************

No methods were deprecated in this version.

Backwards incompatible API changes
**********************************

- :class:`pastas.Project` class is removed from pastas. Use `Pastastore
  <https://github.com/pastas/pastastore>`_ instead.
- :class:`ps.FactorModel` stressmodel is removed. Use :class:`ps.StressModel`
  with :class:`ps.One` response function instead.


New Example Notebooks
*********************

- A new example notebook on model calibration has been added.
- A new example notebook on recharge estimation has been added.
- A new example notebook on changing the time step has been added.
