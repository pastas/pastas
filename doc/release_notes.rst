Release Notes
=============

Starting with the release of Pastas 0.15 changes to the API will be
reported here. The release notes for previous releases up to 0.14 can be found
at the `GitHub Release page <https://github.com/pastas/pastas/releases>`_.
For full details of all changes check the commit log.

Version 0.15 (Expected End of July 2020)
----------------------------------------
This release will introduce backward incompatible changes to Pastas, most
notably due to the weighting of the first value of the noise. This will
cause the calibrated values to be slightly different but better for most
models.

New behaviour
*************
- ml.noise() now returns the noise and not the weighted noise.
-

New Features / Enhancements
***************************

- `ml.set_parameter` method on the Model class is introduced to set the
  initial, minimum, maximum and vary settings for a parameters in one line.
- the stats subpackage has been completely restructured. All methods may now
  also be used as separate methods.

    - diagnostics: perform multiple diagnostic tests at once
    - stoffer_toloi: Ljung-box test adapted for missing data.
    - plot_diagnostics: stand-alone version of the plot for diagnostic checking
    - plot_acf: convenience method to plot the autocorrelation function.
    - all goodness-of-fit metrics are now available as separate functions e.g.,
      ps.stats.nse(). See the API docs for all available methods.

- A new experimental noise model is added: ArmaModel. This model computes
  the noise from the residuals according to a autoregressive-moving-average
  model (ARMA(1,1)). Currently this method is experimental and only applicable
  to time series with equidistant time steps.

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

New Notebooks
*************

