pastas.io
=========

This section lists all the read methods that are available in Pastas to
import time series and the input/output methods to store and load Pastas
models.

IO Methods
----------
.. currentmodule:: pastas.io

.. autosummary::
   :nosignatures:
   :toctree: ./generated

   load

Using Pandas
------------
While Pastas provides a few basic methods for reading time series data, we
highly recommend using the `Pandas read methods <https://pandas.pydata
.org/docs/reference/io.html>`_ to load your time series. Pastas expects
:class:`pandas.Series` with a :class:`pandas.DatetimeIndex`, so
using Pandas is generally a good idea, for example::

    import pandas as pd

    ts = pd.read_csv("your_file.csv", parse_dates=True,
                     infer_datetime_format=True).squeeze()
