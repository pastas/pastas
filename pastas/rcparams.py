rcParams = {
    "timeseries": {
        "oseries": {"sample_down": "drop"},
        "prec": {
            "sample_up": "bfill",
            "sample_down": "mean",
            "fill_before": "mean",
            "fill_after": "mean",
        },
        "evap": {
            "sample_up": "bfill",
            "sample_down": "mean",
            "fill_before": "mean",
            "fill_after": "mean",
        },
        "well": {
            "sample_up": "bfill",
            "sample_down": "mean",
            "fill_before": 0.0,
            "fill_after": 0.0,
        },
        "waterlevel": {
            "sample_up": "interpolate",
            "sample_down": "mean",
            "fill_before": "mean",
            "fill_after": "mean",
        },
        "level": {
            "sample_up": "interpolate",
            "sample_down": "mean",
            "fill_before": "mean",
            "fill_after": "mean",
        },
        "flux": {
            "sample_up": "bfill",
            "sample_down": "mean",
            "fill_before": "mean",
            "fill_after": "mean",
        },
        "quantity": {
            "sample_up": "divide",
            "sample_down": "sum",
            "fill_before": "mean",
            "fill_after": "mean",
        },
    }
}
