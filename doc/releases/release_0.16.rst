Version 0.16 (16th of November 2020)
------------------------------------

.. note::
    This release will introduce backward incompatible changes to Pastas, most
    notably due to the renaming of the parameter input argument. This change
    is mostly internally and will only affect users that explicitly pass
    parameters into a method.

New Features / Enhancements
***************************

- A new stress model (:class:`ps.LinearTrend`) to simulate linear trends is
  added to the list of stable stress models.
- New method to compute the Standardized Groundwater Index. See
  (:meth:`ps.stats.sgi`) for more details.
- Most of the goodness-of-fit metrics now allow providing a "weighted"
  argument. This may result in more realistic values for time series with
  irregular time steps.
- The documentation website is further improved. We now separate 'Examples'
  and 'Concepts'. The First are worked-out example notebooks using Pastas to
  analyse a problem, the second are notebooks showing a underlying methods.

Deprecations
************

- The following methods to set parameter properties are now deprecated
  and replaced by the single method `ml.set_parameter`: `ml.set_vary`,
  `ml.set_pmin`, `ml.set_pmax`, `ml.set_initial`.
- The name of the input argument for the parameters was made consistent
  throughout Pastas. If the input argument is named `p` an array-like object
  is expected, whereas if the input is `parameters` a Pandas DataFrame object
  is expected.
- :class:`ps.FactorModel` is deprecated and will be removed in a future
  version. Use :class:`ps.StressModel` with `rfunc=ps.One` instead.

Backwards incompatible API changes
**********************************


New Example (Notebooks)
***********************

- New notebook on computing Standardized Groundwater Index using Pastas.
- New Notebook on simulated step and linear trend.