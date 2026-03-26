from pandas import Timedelta


class Settings(dict):
    """Class to the store the Pastas Model settings."""

    def __init__(self,
                 tmin=None,
                 tmax=None,
                 freq=None,
                 warmup=Timedelta(3650, "D"), time_offset=Timedelta(0),
                 solver=None,
                 fit_constant=True,
                 freq_obs=None):
        super().__init__()

        # Settings for the model
        self._settings = {
            "tmin": tmin,
            "tmax": tmax,
            "freq": freq,
            "warmup": warmup,
            "time_offset": time_offset,
            "solver": solver,
            "fit_constant": fit_constant,
            "freq_obs": freq_obs,
        }

    def __setitem__(self, key, value):
        raise PermissionError("Settings cannot be changed directly. Use the ml.set_settings() method instead.")

    def __getitem__(self, key):
        return self._settings[key]

    def __repr__(self):
        return f"Settings({self._settings})"

    def __str__(self):
        return f"Settings({self._settings})"

    def update(self, *args, **kwargs):
        raise PermissionError("Settings cannot be changed directly. Use the ml.set_settings() method instead.")


