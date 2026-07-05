"""
Category configuration for sphinx-gallery examples.

This file defines which notebooks belong to which categories.
When you add a new notebook, add it to the appropriate category list below.
"""

gallery_categories = {
    "Basics": [
        "prepare_timeseries.ipynb",
        "basic_model.ipynb",
        "fix_parameters.ipynb",
        "calibration_options.ipynb",
        "modeling_timestep.ipynb"
    ],
    "Stressmodels": [
        "adding_rivers.ipynb",
        "adding_wells.ipynb",
        "multiple_wells.ipynb",
        "hantush_response.ipynb",
        "adding_trends.ipynb",
        "changing_responses.ipynb",
        "threshold_non_linear.ipynb",
        "non_linear_recharge.ipynb",
        "recharge_estimation.ipynb",
        "snowmodel.ipynb"
    ],
    "Model Evaluation": [
        "comparing_models.ipynb",
        "diagnostic_checking.ipynb",
        "model_check_module.ipynb",
        "timestep_analysis.ipynb",
        "uncertainty.ipynb",
        "uncertainty_emcee.ipynb",
        "uncertainty_ls_mcmc.ipynb"
    ],
    "Applications": [
        "standardized_groundwater_index.ipynb",
        "signatures.ipynb",
        "ensemble_predictions.ipynb"
    ],
    "Time Series Analysis Manual": [
        "stowa_preprocessing.ipynb",
        "stowa_model_structure.ipynb",
        "stowa_calibration.ipynb",
        "stowa_assessment.ipynb",
        "stowa_cases_contribution_assessment.ipynb",
        "stowa_cases_characteristics.ipynb",
        "stowa_cases_system_analysis.ipynb",
        "stowa_cases_forecasting.ipynb"
    ],
    "Pastas Performance": [
        "caching_for_performance.ipynb"
    ],
    "Groundwater Article": [
        "groundwater_paper/Ex1_simple_model/Example1.ipynb",
        "groundwater_paper/Ex2_monitoring_network/Example2.ipynb"
    ]
}
