Release Notes
=============

Starting with the release of Pastas 0.15 changes to the API are
reported here. The release notes for previous releases up to 0.14 can be found
at the `GitHub Release page <https://github.com/pastas/pastas/releases>`_.
For full details of all changes check the commit log.

Version 0.16 (16th of November 2020)
--------------------------------------------

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

Version 0.15 (31st of July 2020)
--------------------------------

.. note::
    This release will introduce backward incompatible changes to Pastas, most
    notably due to the weighting of the first value of the noise. This will
    cause the calibrated values to be slightly different but better for most
    models. It is highly recommended to upgrade to this new version of Pastas.

New Features / Enhancements
***************************

- :meth:`Model.noise()` now returns the noise and not the weighted noise.
  Weights may now be obtained through :meth:`Model.noise_weights()`.
- Private methods are now identified by a leading underscore issue 74.
- :meth:`Model.set_parameter` method on the Model class is introduced to set
  the initial, minimum, maximum and vary settings for a parameters in one line.
- the ps.stats subpackage has been completely restructured. All methods may now
  also be used as separate methods.

    - :func:`ps.stats.diagnostics`: perform multiple diagnostic tests at once.
    - :func:`ps.stats.stoffer_toloi`: Ljung-box test adapted for missing data.
    - :func:`ps.stats.plot_diagnostics`: stand-alone version of the plot for
      diagnostic checking
    - :func:`ps.stats.plot_acf`: convenience method to plot the
      autocorrelation function.
    - all goodness-of-fit metrics are now available as separate functions e.g.,
      :func:`ps.stats.nse()`. See the API docs for all available methods.

- A new experimental noise model is added: :class:`ArmaModel`. This model
  computes the noise from the residuals according to a
  autoregressive-moving-average model (ARMA(1,1)). Currently this method is
  experimental and only applicable to time series with equidistant time steps.
- The response functions have been standardized to all fit the same formula
  for the impulse response function, when some parameters are fixed to certain
  values.
- new function :func:`ps.show_versions()` is introduced. This function may
  be used to show the version of package dependencies that are installed.
- New method :meth:`ml.get_response_tmax` is introduced. This method may be
  used to obtain the tmax of the response function.

Deprecations
************

- :meth:`ml.set_vary`, :meth:`ml.set_initial`, :meth:`ml.set_pmin`, and
  :meth:`ml.set_pmax` are deprecated and will be removed in a future release
  . The use of :meth:`ml.set_parameter` method is now recommended.

Backwards incompatible API changes
**********************************

- The parameters of the Hantush response function have new names. This will
  cause problems when loading models using this function to be loaded from
  .pas-file. No fix is available for this.
- The first value of the noise series has changes (see issue 152 for
  details), causing changes in the optimal parameter values.

New Example (Notebooks)
***********************

- Notebook on diagnostic checking of Pastas models.
- Notebook on the new ArmaModel noise model.
- Notebook on reading Dutch datasets.
- Notebook on the autocorrelation function with irregular time steps.

