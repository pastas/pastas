from .typing import OseriesSettingsDict, StressSettingsDict

rcParams = {
    "timeseries": {
        "oseries": OseriesSettingsDict(
            fill_nan="drop",
            sample_down="drop",
        ),
        "prec": StressSettingsDict(
            sample_up="bfill",
            sample_down="mean",
            fill_nan=0.0,
            fill_before="mean",
            fill_after="mean",
        ),
        "evap": StressSettingsDict(
            sample_up="bfill",
            sample_down="mean",
            fill_before="mean",
            fill_after="mean",
            fill_nan="interpolate",
        ),
        "well": StressSettingsDict(
            sample_up="bfill",
            sample_down="mean",
            fill_nan=0.0,
            fill_before=0.0,
            fill_after=0.0,
        ),
        "waterlevel": StressSettingsDict(
            sample_up="interpolate",
            sample_down="mean",
            fill_before="mean",
            fill_after="mean",
            fill_nan="interpolate",
        ),
        "level": StressSettingsDict(
            sample_up="interpolate",
            sample_down="mean",
            fill_before="mean",
            fill_after="mean",
            fill_nan="interpolate",
        ),
        "flux": StressSettingsDict(
            sample_up="bfill",
            sample_down="mean",
            fill_before="mean",
            fill_after="mean",
            fill_nan=0.0,
        ),
        "quantity": StressSettingsDict(
            sample_up="divide",
            sample_down="sum",
            fill_before="mean",
            fill_after="mean",
            fill_nan=0.0,
        ),
    }
}
