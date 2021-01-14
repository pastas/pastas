"""
This file contains the import routine for import of groundwater observations
from RWS Waterbase / WaterInfo database. (http://waterinfo.rws.nl/)

Author: R.A. Collenteur, Artesia Water 2017

"""

from pandas import read_csv

from ..timeseries import TimeSeries


def read_waterbase(fname, locations=None, variable="NUMERIEKEWAARDE",
                   kind="waterlevel", freq="10min"):
    """Method to import waterlevel ts from waterbase.

    Parameters
    ----------
    fname: str
        string with the path and filename of the waterbase file.
    variable: str
        name of the variable to collect the time series from. Only one
        variable name is allowed.
    kind: str
    freq: str

    Returns
    -------
    ts: pastas.TimeSeries
        returns a Pastas TimeSeries object or a list of objects.

    Notes
    -----
    More information on the ts provided by the Waterbase database see:
    http://waterinfo.rws.nl/

    the xy-coordinates are calculates as the mean xy-coordinate in case these
    values are not unique.

    """
    ts = []
    df = read_csv(fname, delimiter=";", index_col="Date", decimal=",",
                  usecols=["MEETPUNT_IDENTIFICATIE", "WAARNEMINGDATUM",
                           "WAARNEMINGTIJD", variable, "EPSG", "X", "Y"],
                  parse_dates={"Date": ["WAARNEMINGDATUM", "WAARNEMINGTIJD"]},
                  infer_datetime_format=True, dayfirst=True,
                  na_values=[-999999999, 999999999],
                  encoding="ISO-8859-1")

    if locations is None:
        locations = df.MEETPUNT_IDENTIFICATIE.unique()
    elif isinstance(locations, str):
        locations = [locations]

    for name in locations:
        series = df.loc[df["MEETPUNT_IDENTIFICATIE"].isin([name])]
        metadata = {
            "x": series.X.mean(),
            "y": series.Y.mean(),
            "z": 0,
            "projection": "epsg:" + str(series.loc[:, "EPSG"].unique()[0]),
            "units": "cm"
        }
        series = series.loc[:, variable].sort_index()
        ts.append(TimeSeries(series, name=name, metadata=metadata,
                             settings=kind, freq_original=freq))

    if len(ts) == 1:
        ts = ts[0]

    return ts
