import pandas as pd

import pastas as ps
from pastas.extensions import register_model_accessor


def test_model_extension():
    ml = ps.Model(
        pd.Series(index=pd.date_range("2020", periods=10, freq="D"), data=1.0)
    )

    @register_model_accessor("extension")
    class FakeExtension:
        def __init__(self, model):
            self._model = model
            self.useless = True

    assert hasattr(ml, "extension")
    assert ml.extension.useless
