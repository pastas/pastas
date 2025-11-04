Examples
========

Below you can find examples of how Pastas models are used for the analysis
of groundwater levels. Examples in the form of Python scripts can also be found
on the `examples directory on GitHub <https://github.com/pastas/pastas/tree/master/doc/examples>`_.

.. toctree::
    :maxdepth: 4
    :hidden:
    :glob:

    prepare_timeseries
    basic_model
    fix_parameters
    calibration_options
    adding_rivers
    adding_wells
    multiple_wells
    adding_trends
    changing_responses
    threshold_non_linear
    non_linear_recharge
    recharge_estimation
    snowmodel
    comparing_models
    diagnostic_checking
    timestep_analysis
    uncertainty
    uncertainty_emcee
    uncertainty_ls_mcmc
    standardized_groundwater_index
    signatures
    ensemble_predictions


Basics
------

`Preprocessing user-provided time series`_

`A basic model`_

`Fixing parameters while fitting`_

`Calibration`_

`Modeling with different timesteps`_

.. _Preprocessing user-provided time series: prepare_timeseries.html
.. _A basic model: basic_model.html
.. _Fixing parameters while fitting: fix_parameters.html
.. _Calibration: calibration_options.html
.. _Modeling with different timesteps: modeling_timestep.html

Stressmodels
------------

`Adding surface water levels`_

`Adding pumping wells`_

`Adding multiple wells`_

`Adding trends`_

`Changing response functions`_

.. _Adding surface water levels: adding_rivers.html
.. _Adding pumping wells: adding_wells.html
.. _Adding multiple wells: multiple_wells.html
.. _Adding trends: adding_trends.html
.. _Changing response functions: changing_responses.html

Non-linear (Recharge) Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`Threshold non-linearities`_

`Non-linear recharge models`_

`Estimating recharge`_

`Modeling snow`_

.. _Threshold non-linearities: threshold_non_linear.html
.. _Non-linear recharge models: non_linear_recharge.html
.. _Estimating recharge: recharge_estimation.html
.. _Modeling snow: snowmodel.html


Model Evaluation
----------------

`Comparing models visually`_

`Diagnostic checking`_

`Reducing autocorrelation`_

`Uncertainty quantification`_

`MCMC uncertainty`_

`MCMC vs. LS`_

.. _Comparing models visually: comparing_models.html
.. _Diagnostic checking: diagnostic_checking.html
.. _`Reducing autocorrelation`: timestep_analysis.html
.. _Uncertainty quantification: uncertainty.html
.. _MCMC uncertainty: uncertainty_emcee.html
.. _MCMC vs. LS: uncertainty_ls_mcmc.html


Applications
------------

`Standardized Groundwater Index`_

`Groundwater signatures`_

`Ensemble predictions`_

.. _Standardized Groundwater Index: standardized_groundwater_index.html
.. _Groundwater signatures: signatures.html
.. _Ensemble predictions: ensemble_predictions.html


Time Series Analysis Manual
---------------------------

The `notebooks <https://github.com/ArtesiaWater/stowa_handleiding_tijdreeksanalyse>`_ from
the Dutch Manual on Time Series Analysis, which use Pastas, have been translated into English
and are available below. The full manual can be found here: Von Asmuth, J., Baggelaar, P.,
Bakker, M., Brakenhoff, D., Collenteur, R., Ebbens, O., Mondeel, H., Klop, S., & Schaars, F. (2021).
Handleiding Tijdreeksanalyse (`STOWA rapport nr. 32 <https://www.stowa.nl/publicaties/handleiding-voor-het-uitvoeren-van-tijdreeksanalyses>`_).
Stichting Toegepast Onderzoek Waterbeheer, Amersfoort.

`Preprocessing`_

`Model structure`_

`Model calibration`_

`Model assessment`_

`Case Study 1 Assessing contributions`_

`Case Study 2 Determining characteristics`_

`Case Study 3 System analysis`_

`Case Study 4 Forecasting`_

.. _Preprocessing: stowa_preprocessing.html
.. _Model structure: stowa_model_structure.html
.. _Model calibration: stowa_calibration.html
.. _Model assessment: stowa_assessment.html
.. _Case Study 1 Assessing contributions: stowa_cases_contribution_assessment.html
.. _Case Study 2 Determining characteristics: stowa_cases_characteristics.html
.. _Case Study 3 System analysis: stowa_cases_system_analysis.html
.. _Case Study 4 Forecasting: stowa_cases_forecasting.html


Pastas Performance
------------------
`Caching for performance`_

.. _Caching for performance: caching_for_performance.html
