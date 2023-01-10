rcParams = {
    "timeseries": {
        "oseries": {"fill_nan": "drop", "sample_down": "drop"},
        "prec": {
            "sample_up": "bfill",
            "sample_down": "mean",
            "fill_nan": 0.0,
            "fill_before": "mean",
            "fill_after": "mean",
        },
        "evap": {
            "sample_up": "bfill",
            "sample_down": "mean",
            "fill_before": "mean",
            "fill_after": "mean",
            "fill_nan": "interpolate",
        },
        "well": {
            "sample_up": "bfill",
            "sample_down": "mean",
            "fill_nan": 0.0,
            "fill_before": 0.0,
            "fill_after": 0.0,
        },
        "waterlevel": {
            "sample_up": "interpolate",
            "sample_down": "mean",
            "fill_before": "mean",
            "fill_after": "mean",
            "fill_nan": "interpolate",
        },
        "level": {
            "sample_up": "interpolate",
            "sample_down": "mean",
            "fill_before": "mean",
            "fill_after": "mean",
            "fill_nan": "interpolate",
        },
        "flux": {
            "sample_up": "bfill",
            "sample_down": "mean",
            "fill_before": "mean",
            "fill_after": "mean",
            "fill_nan": 0.0,
        },
        "quantity": {
            "sample_up": "divide",
            "sample_down": "sum",
            "fill_before": "mean",
            "fill_after": "mean",
            "fill_nan": 0.0,
        },
    }
}
