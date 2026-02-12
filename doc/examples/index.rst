Examples
========

Below you can find examples of how Pastas models are used for the analysis
of groundwater levels. Examples in the form of Python scripts can also be found
on the `examples directory on GitHub <https://github.com/pastas/pastas/tree/master/doc/examples>`_.

.. toctree::
    :maxdepth: 4
    :hidden:
    :caption: Basics

    prepare_timeseries
    basic_model
    fix_parameters
    calibration_options
    modeling_timestep

.. toctree::
    :maxdepth: 4
    :hidden:
    :caption: Stressmodels

    adding_rivers
    adding_wells
    multiple_wells
    hantush_response
    adding_trends
    changing_responses
    threshold_non_linear
    non_linear_recharge
    recharge_estimation
    snowmodel

.. toctree::
    :maxdepth: 4
    :hidden:
    :caption: Model Evaluation

    comparing_models
    diagnostic_checking
    model_check_module
    timestep_analysis
    uncertainty
    uncertainty_emcee
    uncertainty_ls_mcmc

.. toctree::
    :maxdepth: 4
    :hidden:
    :caption: Applications

    standardized_groundwater_index
    signatures
    ensemble_predictions

.. toctree::
    :maxdepth: 4
    :hidden:
    :caption: Time Series Analysis Manual

    stowa_preprocessing
    stowa_model_structure
    stowa_calibration
    stowa_assessment
    stowa_cases_contribution_assessment
    stowa_cases_characteristics
    stowa_cases_system_analysis
    stowa_cases_forecasting

.. toctree::
    :maxdepth: 4
    :hidden:
    :caption: Pastas Performance

    caching_for_performance

.. toctree::
    :maxdepth: 4
    :hidden:
    :caption: Groundwater Article

    groundwater_paper/Ex1_simple_model/Example1
    groundwater_paper/Ex2_monitoring_network/Example2


Basics
------

:doc:`Preprocessing user-provided time series <prepare_timeseries>`

:doc:`A basic model <basic_model>`

:doc:`Fixing parameters while fitting <fix_parameters>`

:doc:`Calibration <calibration_options>`

:doc:`Modeling with different timesteps <modeling_timestep>`

Stressmodels
------------

:doc:`Adding surface water levels <adding_rivers>`

:doc:`Adding pumping wells <adding_wells>`

:doc:`Adding multiple wells <multiple_wells>`

:doc:`Adding trends <adding_trends>`

:doc:`Changing response functions <changing_responses>`

Non-linear (Recharge) Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:doc:`Threshold non-linearities <threshold_non_linear>`

:doc:`Non-linear recharge models <non_linear_recharge>`

:doc:`Estimating recharge <recharge_estimation>`

:doc:`Modeling snow <snowmodel>`


Model Evaluation
----------------

:doc:`Comparing models visually <comparing_models>`

:doc:`Diagnostic checking <diagnostic_checking>`

:doc:`Model check module <model_check_module>`

:doc:`Reducing autocorrelation <timestep_analysis>`

:doc:`Uncertainty quantification <uncertainty>`

:doc:`MCMC uncertainty <uncertainty_emcee>`

:doc:`MCMC vs. LS <uncertainty_ls_mcmc>`


Applications
------------

:doc:`Standardized Groundwater Index <standardized_groundwater_index>`

:doc:`Groundwater signatures <signatures>`

:doc:`Ensemble predictions <ensemble_predictions>`


Time Series Analysis Manual
---------------------------

The `notebooks <https://github.com/ArtesiaWater/stowa_handleiding_tijdreeksanalyse>`_ from
the Dutch Manual on Time Series Analysis, which use Pastas, have been translated into English
and are available below. The full manual can be found here: Von Asmuth, J., Baggelaar, P.,
Bakker, M., Brakenhoff, D., Collenteur, R., Ebbens, O., Mondeel, H., Klop, S., & Schaars, F. (2021).
Handleiding Tijdreeksanalyse (`STOWA rapport nr. 32 <https://www.stowa.nl/publicaties/handleiding-voor-het-uitvoeren-van-tijdreeksanalyses>`_).
Stichting Toegepast Onderzoek Waterbeheer, Amersfoort.

:doc:`Preprocessing <stowa_preprocessing>`

:doc:`Model structure <stowa_model_structure>`

:doc:`Model calibration <stowa_calibration>`

:doc:`Model assessment <stowa_assessment>`

:doc:`Case Study 1 Assessing contributions <stowa_cases_contribution_assessment>`

:doc:`Case Study 2 Determining characteristics <stowa_cases_characteristics>`

:doc:`Case Study 3 System analysis <stowa_cases_system_analysis>`

:doc:`Case Study 4 Forecasting <stowa_cases_forecasting>`


Pastas Performance
------------------

:doc:`Caching for performance <caching_for_performance>`


Groundwater Article
-------------------
These notebooks are supplementary material to the following article in Groundwater:
Collenteur, R.A., Bakker, M., Caljé, R., Klop, S.A. and Schaars, F. (2019).
Pastas: Open Source Software for the Analysis of Groundwater Time Series.
Groundwater, 57: 877-885. `doi:10.1111/gwat.12925 <https://doi.org/10.1111/gwat.12925>`_

:doc:`Example 1 Modeling Groundwater Levels with Pastas <./groundwater_paper/Ex1_simple_model/Example1>`

:doc:`Example 2 Analysis of groundwater monitoring networks using Pastas <./groundwater_paper/Ex2_monitoring_network/Example2>`