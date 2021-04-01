Version 0.17 (2nd of April 2021)
--------------------------------

.. note::
    The :class:`Pastas.Project` class has been removed from Pastas in this
release. `Pastastore <https://github.com/pastas/pastastore>`_ is now
recommended to deal with multiple time series models.

New Features / Enhancements
***************************

- :meth:`ml.set_parameter` now also takes an `optimal` argument.
- ...

Deprecations
************


Backwards incompatible API changes
**********************************

- :class:`pastas.Project` class is removed from pastas. Use `Pastastore
  <https://github.com/pastas/pastastore>`_ instead.

New Example Notebooks
*********************

- A new example notebook on model calibration has been added.
- A new example notebook on recharge estimation has been added.
- A new example notebook on changing the time step has been added.
