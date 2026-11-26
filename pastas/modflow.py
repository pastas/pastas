"""Extract time series from MODFLOW models and create Pastas models.

This module provides utilities to extract time series (heads, well stresses
and recharge) from a MODFLOW model through `flopy
<https://github.com/modflowpy/flopy>`_ and to automatically create Pastas
time series models from them. Both MODFLOW-2005 style models
(``flopy.modflow.Modflow``) and MODFLOW 6 models (``flopy.mf6`` model
objects) are supported.

flopy is an optional dependency of Pastas. Install it with
``pip install flopy`` or ``pip install pastas[flopy]``.

Examples
--------
>>> import flopy
>>> import pastas as ps
>>> sim = flopy.mf6.MFSimulation.load(sim_ws="path/to/sim")
>>> gwf = sim.get_model()
>>> ml = ps.modflow.model_from_flopy(gwf, cellids=[(0, 10, 10)], solve=True)

"""

import logging
import os
from collections.abc import Iterable
from typing import Any

import numpy as np
from pandas import (
    DatetimeIndex,
    Series,
    Timestamp,
    date_range,
    to_datetime,
    to_timedelta,
)

from pastas.model import Model
from pastas.rfunc import Exponential, HantushWellModel
from pastas.stressmodels import StressModel, WellModel
from pastas.typing import RFunc, Solver

logger = logging.getLogger(__name__)

# MODFLOW time units mapped to a (pandas Timedelta unit, multiplier) tuple
_TIME_UNITS = {
    "seconds": ("s", 1.0),
    "minutes": ("min", 1.0),
    "hours": ("h", 1.0),
    "days": ("D", 1.0),
    "years": ("D", 365.25),
    "undefined": ("D", 1.0),
}

# Column names (in order of preference) that hold the stress value in
# list-based stress packages, for MODFLOW 6 ("q", "head", ...) and
# MODFLOW-2005 ("flux", "shead", ...) packages
_VALUE_COLUMNS = (
    "q",
    "flux",
    "shead",
    "ehead",
    "bhead",
    "stage",
    "elev",
    "evtr",
    "recharge",
    "rate",
    "surf",
    "cond",
)

# Default settings for a recharge stress extracted from a MODFLOW model
_RECHARGE_SETTINGS = {
    "sample_up": "bfill",
    "sample_down": "mean",
    "fill_nan": 0.0,
    "fill_before": "mean",
    "fill_after": "mean",
}


def _import_flopy() -> Any:
    """Import flopy and raise an informative error if it is not installed."""
    try:
        import flopy
    except ImportError:
        raise ImportError(
            "The flopy package is required to use pastas.modflow. Install flopy "
            "using 'pip install flopy' or 'pip install pastas[flopy]'."
        ) from None
    return flopy


def _is_mf6(model: Any) -> bool:
    """Return True if the model is a MODFLOW 6 model, False for MODFLOW-2005."""
    return hasattr(model, "simulation")


def _get_package(model: Any, name: str) -> Any:
    """Return the package with the given name, or None if not present."""
    if not isinstance(name, str):
        return name
    pkg = model.get_package(name)
    if pkg is None:
        pkg = model.get_package(name.upper())
    return pkg


def _get_time_info(model: Any) -> tuple[Timestamp, str, float, np.ndarray]:
    """Return start datetime, Timedelta unit, unit factor and perlen array."""
    if _is_mf6(model):
        tdis = model.simulation.tdis
        perlen = np.asarray(tdis.perioddata.array["perlen"], dtype=float)
        unit_name = str(np.asarray(tdis.time_units.array).flat[0]).lower()
        start = to_datetime(str(np.asarray(tdis.start_date_time.array).flat[0]))
    else:
        dis = model.get_package("DIS")
        perlen = np.asarray(dis.perlen.array, dtype=float)
        itmuni = int(np.asarray(dis.itmuni).flat[0])
        unit_name = dis.itmuni_dict.get(itmuni, "days").lower()
        start = to_datetime(str(getattr(dis, "start_datetime", None) or "1970-01-01"))
    unit, factor = _TIME_UNITS.get(unit_name, _TIME_UNITS["days"])
    return start, unit, factor, perlen


def _validate_cellid(model: Any, cellid: tuple | None = None) -> tuple:
    """Validate a cellid tuple against the model grid, default to first cell."""
    grid_type = model.modelgrid.grid_type
    ndim = {"structured": 3, "vertex": 2, "unstructured": 1}.get(grid_type)
    if ndim is None:
        raise ValueError(f"Unsupported grid type: {grid_type}.")
    if cellid is None:
        return (0,) * ndim
    cellid = tuple(int(c) for c in cellid)
    if len(cellid) != ndim:
        raise ValueError(
            f"cellid must have {ndim} elements for a {grid_type} grid, got {cellid}."
        )
    return cellid


def _cellid_to_name(cellid: tuple) -> str:
    """Return a readable name for a cellid, e.g. (0, 2, 3) -> 'L0R2C3'."""
    if len(cellid) == 3:
        return f"L{cellid[0]}R{cellid[1]}C{cellid[2]}"
    if len(cellid) == 2:
        return f"L{cellid[0]}C{cellid[1]}"
    return f"C{cellid[0]}"


def _cell_center(model: Any, cellid: tuple) -> np.ndarray:
    """Return the (x, y) center of a cell as the centroid of its vertices."""
    grid = model.modelgrid
    if grid.grid_type not in ("structured", "vertex"):
        raise ValueError(
            "Cell centers can only be computed for structured and vertex grids, "
            f"not for a {grid.grid_type} grid."
        )
    vertices = np.asarray(grid.get_cell_vertices(cellid), dtype=float)
    return vertices.mean(axis=0)


def _cell_metadata(model: Any, cellid: tuple) -> dict[str, Any]:
    """Return x, y, z metadata for a cell of the model grid."""
    x, y = (float(v) for v in _cell_center(model, cellid))
    z = np.nan
    try:
        grid = model.modelgrid
        top = np.asarray(grid.top.array, dtype=float)
        botm = np.asarray(grid.botm.array, dtype=float)
        if grid.grid_type == "structured":
            z = 0.5 * (top[cellid[1], cellid[2]] + botm[cellid])
        elif grid.grid_type == "vertex":
            z = 0.5 * (top[cellid[1]] + botm[cellid])
    except (AttributeError, IndexError, TypeError):
        pass
    return {"x": x, "y": y, "z": float(z), "projection": None}


def _dry_sentinels(model: Any) -> list[float]:
    """Return head values used by the model for dry/inactive cells."""
    values = []
    for pkg in getattr(model, "packagelist", []):
        for attr in ("hdry", "hnoflo"):
            value = getattr(pkg, attr, None)
            if value is None:
                continue
            try:
                values.append(float(np.asarray(value).flat[0]))
            except (TypeError, ValueError):
                continue
    return values


def _get_headfile(model: Any, headfile: Any = None) -> Any:
    """Return a flopy HeadFile for the model, optionally from a given file."""
    flopy = _import_flopy()
    if headfile is not None:
        if isinstance(headfile, (str, os.PathLike)):
            headfile = flopy.utils.HeadFile(str(headfile), modelgrid=model.modelgrid)
        hf = headfile
    elif hasattr(model, "output"):
        try:
            hf = model.output.head()
        except Exception as exc:
            raise FileNotFoundError(
                "Could not find a head output file for this MODFLOW 6 model. Make "
                "sure the model was run with head output saved, or pass the path to "
                "a head file with the `headfile` argument."
            ) from exc
    else:
        fnames = [
            f
            for f in getattr(model, "output_fnames", [])
            if str(f).lower().endswith(".hds")
        ]
        if fnames:
            path = os.path.join(str(model.model_ws), str(fnames[0]))
        else:
            path = os.path.join(str(model.model_ws), f"{model.name}.hds")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No head output file found at {path}. Make sure the model was run "
                "with head output saved, or pass the path to a head file with the "
                "`headfile` argument."
            )
        hf = flopy.utils.HeadFile(path, modelgrid=model.modelgrid)
    # make sure cellid-based get_ts works for the model grid
    hf.modelgrid = model.modelgrid
    return hf


def get_stress_period_times(model: Any, when: str = "start") -> DatetimeIndex:
    """Return the times of the stress periods of a MODFLOW model.

    Parameters
    ----------
    model: flopy model
        A flopy model object, e.g. a ``flopy.modflow.Modflow`` or a
        ``flopy.mf6`` groundwater flow model.
    when: str, optional
        Which time of the stress period to return: "start" (default), "end"
        or "middle" of each stress period.

    Returns
    -------
    pandas.DatetimeIndex
        DatetimeIndex with one entry per stress period.

    Examples
    --------
    >>> import pastas as ps
    >>> times = ps.modflow.get_stress_period_times(gwf)  # doctest: +SKIP

    """
    _import_flopy()
    start, unit, factor, perlen = _get_time_info(model)
    end = np.cumsum(perlen)
    begin = end - perlen
    if when == "start":
        totim = begin
    elif when == "end":
        totim = end
    elif when == "middle":
        totim = 0.5 * (begin + end)
    else:
        raise ValueError(
            f"Invalid value for 'when': {when!r}. Use 'start', 'end' or 'middle'."
        )
    return DatetimeIndex(start + to_timedelta(totim * factor, unit=unit))


def _stress_period_series(
    model: Any, values: np.ndarray, freq: str | None = "D", name: str | None = None
) -> Series:
    """Return per-stress-period values as a Series with an equidistant index.

    MODFLOW applies a stress constantly from the start of a stress period
    until the end of that stress period. The values are therefore assigned
    to the start of each stress period and forward-filled onto an equidistant
    index with the given frequency, as required by Pastas time series. If
    freq is None, the (possibly non-equidistant) stress period start times
    are used as the index.

    """
    start, unit, factor, perlen = _get_time_info(model)
    sp_end = np.cumsum(perlen)
    sp_begin = sp_end - perlen
    period_starts = DatetimeIndex(start + to_timedelta(sp_begin * factor, unit=unit))
    series = Series(np.asarray(values, dtype=float), index=period_starts, name=name)
    if freq is None:
        return series
    index = DatetimeIndex(
        date_range(
            period_starts[0],
            start + to_timedelta(sp_end[-1] * factor, unit=unit),
            freq=freq,
        )
    )
    return series.reindex(index, method="ffill")


def get_head_series(
    model: Any,
    cellid: tuple | None = None,
    headfile: Any = None,
    name: str | None = None,
    mask_dry: bool = True,
) -> Series:
    """Extract the simulated head time series for a single cell.

    Parameters
    ----------
    model: flopy model
        A flopy model object, e.g. a ``flopy.modflow.Modflow`` or a
        ``flopy.mf6`` groundwater flow model. The model must have been run
        with head output saved.
    cellid: tuple of int, optional
        Cell identifier of the cell to extract the heads from: (layer, row,
        column) for a structured grid, (layer, cell2d) for a vertex grid and
        (node,) for an unstructured grid. Default is the first cell.
    headfile: str, path or flopy HeadFile, optional
        Path to a head output file or an open flopy HeadFile object. If None,
        the head file of the model output is used.
    name: str, optional
        Name of the returned pandas Series. Default is derived from the
        cellid, e.g. "L0R2C3".
    mask_dry: bool, optional
        If True (default), replace head values of dry or inactive cells
        (hdry/hnoflo) with NaN.

    Returns
    -------
    pandas.Series
        Simulated heads with a DatetimeIndex.

    Examples
    --------
    >>> import pastas as ps
    >>> heads = ps.modflow.get_head_series(gwf, cellid=(0, 10, 10))  # doctest: +SKIP

    """
    _import_flopy()
    cellid = _validate_cellid(model, cellid)
    hf = _get_headfile(model, headfile)
    data = np.asarray(hf.get_ts(cellid), dtype=float)
    totim = data[:, 0]
    values = data[:, 1]
    start, unit, factor, _ = _get_time_info(model)
    index = DatetimeIndex(start + to_timedelta(totim * factor, unit=unit))
    series = Series(values, index=index, name=name or _cellid_to_name(cellid))
    if mask_dry:
        for sentinel in _dry_sentinels(model):
            series = series.mask(np.isclose(series, sentinel, rtol=1e-5, atol=1e-8))
    return series


def _record_cellid(record: Any, names: tuple) -> tuple:
    """Return the cellid tuple of a single stress period data record."""
    if "cellid" in names:
        return tuple(int(c) for c in record["cellid"])
    return (int(record["k"]), int(record["i"]), int(record["j"]))


def _iter_stress_period_data(
    model: Any, spd: Any, nper: int
) -> Iterable[tuple[int, Any]]:
    """Yield (kper, record) for each stress period, repeating the last record.

    MODFLOW repeats the stresses of the previous stress period when no data
    is specified for a stress period.

    """
    mf6 = _is_mf6(model)
    last = None
    for kper in range(nper):
        if mf6:
            record = spd.get_data(kper)
            if record is not None and len(record) > 0:
                last = record
        else:
            try:
                record = spd[kper]
            except (KeyError, IndexError):
                record = None
            if record is not None and len(record) > 0:
                last = record
            else:
                record = last
        yield kper, (last if mf6 else record)


def _get_list_stress_series(
    model: Any,
    package: Any,
    column: str | None = None,
) -> tuple[dict[tuple, np.ndarray], np.ndarray, np.ndarray, int]:
    """Extract per-cell and aggregated stress values for a list-based package."""
    pkg = _get_package(model, package)
    if pkg is None:
        raise ValueError(f"The model has no {package} package.")
    spd = getattr(pkg, "stress_period_data", None)
    if spd is None:
        raise ValueError(
            f"Package {package} has no stress_period_data. Array-based packages "
            "such as RCH are handled by get_recharge_series."
        )
    _, _, _, perlen = _get_time_info(model)
    nper = len(perlen)

    per_cell: dict[tuple, np.ndarray] = {}
    totals = np.zeros(nper)
    counts = np.zeros(nper)
    for kper, record in _iter_stress_period_data(model, spd, nper):
        if record is None or len(record) == 0:
            continue
        names = record.dtype.names
        if column is None:
            colname = next((c for c in _VALUE_COLUMNS if c in names), None)
            if colname is None:
                raise ValueError(
                    f"Could not determine the stress column of package {package} "
                    f"from columns {names}. Pass the `column` argument."
                )
        else:
            if column not in names:
                raise ValueError(
                    f"Column {column!r} not found in package {package} columns {names}."
                )
            colname = column
        for row in record:
            cellid = _record_cellid(row, names)
            value = float(row[colname])
            if cellid not in per_cell:
                per_cell[cellid] = np.zeros(nper)
            per_cell[cellid][kper] += value
            totals[kper] += value
            counts[kper] += 1
    return per_cell, totals, counts, nper


def get_stress_series(
    model: Any,
    package: str | Any = "WEL",
    column: str | None = None,
    as_dict: bool = False,
    agg: str = "sum",
    freq: str | None = "D",
) -> Series | dict[str, Series]:
    """Extract stress time series from a list-based stress package.

    Works with packages that store their stresses per cell per stress period,
    such as WEL, CHD, DRN, GHB and RIV, for both MODFLOW-2005 and MODFLOW 6
    models. The stresses are constant within each stress period (MODFLOW
    semantics), so the returned series have an equidistant index.

    Parameters
    ----------
    model: flopy model
        A flopy model object, e.g. a ``flopy.modflow.Modflow`` or a
        ``flopy.mf6`` groundwater flow model.
    package: str or flopy package, optional
        Name of the package (e.g. "WEL", default) or a flopy package object.
    column: str, optional
        Name of the column in the stress period data holding the stress
        value. Default is the first of q, flux, shead, ehead, bhead, stage,
        elev, evtr, recharge, rate, surf, cond present in the package, which
        resolves to "flux" for a MODFLOW-2005 WEL package and "q" for a
        MODFLOW 6 WEL package.
    as_dict: bool, optional
        If True, return a dictionary with one pandas Series per cell (keyed
        by cell name, e.g. "L0R2C3"). If False (default), return a single
        pandas Series with the aggregated stress per stress period.
    agg: str, optional
        Aggregation for the single series when as_dict is False: "sum"
        (default) or "mean".
    freq: str or None, optional
        Frequency of the equidistant index of the returned series, "D" by
        default. The stress of each stress period is forward-filled from the
        start of the stress period. If None, the (possibly non-equidistant)
        stress period start times are used as the index.

    Returns
    -------
    pandas.Series or dict of pandas.Series
        The stress time series, or per cell when as_dict is True.

    Examples
    --------
    >>> import pastas as ps
    >>> wells = ps.modflow.get_stress_series(gwf, package="WEL", as_dict=True)

    """
    _import_flopy()
    if agg not in ("sum", "mean"):
        raise ValueError(f"Invalid value for 'agg': {agg!r}. Use 'sum' or 'mean'.")
    pkg = _get_package(model, package)
    pname = package if isinstance(package, str) else type(pkg).__name__
    per_cell, totals, counts, _ = _get_list_stress_series(model, package, column=column)
    if as_dict:
        return {
            _cellid_to_name(cellid): _stress_period_series(
                model,
                values,
                freq=freq,
                name=f"{pname.lower()}_{_cellid_to_name(cellid)}",
            )
            for cellid, values in per_cell.items()
        }
    data = (
        totals
        if agg == "sum"
        else np.divide(totals, counts, out=np.zeros_like(totals), where=counts > 0)
    )
    return _stress_period_series(model, data, freq=freq, name=pname.lower())


def get_recharge_series(
    model: Any,
    package: str | Any = "RCH",
    agg: str = "mean",
    freq: str | None = "D",
) -> Series:
    """Extract the recharge time series from a RCH(A) package.

    Parameters
    ----------
    model: flopy model
        A flopy model object, e.g. a ``flopy.modflow.Modflow`` or a
        ``flopy.mf6`` groundwater flow model.
    package: str or flopy package, optional
        Name of the recharge package, "RCH" (default) or "RCHA", or a flopy
        package object.
    agg: str, optional
        Aggregation over the model cells: "mean" (default, giving the areal
        recharge rate) or "sum".
    freq: str or None, optional
        Frequency of the equidistant index of the returned series, "D" by
        default. The recharge of each stress period is forward-filled from
        the start of the stress period. If None, the (possibly
        non-equidistant) stress period start times are used as the index.

    Returns
    -------
    pandas.Series
        Recharge time series with an equidistant DatetimeIndex.

    Examples
    --------
    >>> import pastas as ps
    >>> rch = ps.modflow.get_recharge_series(gwf)  # doctest: +SKIP

    """
    _import_flopy()
    pkg = _get_package(model, package)
    if pkg is None:
        raise ValueError(f"The model has no {package} package.")
    _, _, _, perlen = _get_time_info(model)
    nper = len(perlen)

    # array-based recharge (MODFLOW-2005 RCH, MODFLOW 6 RCHA)
    for attr in ("recharge", "rech"):
        arr = getattr(pkg, attr, None)
        if arr is not None:
            data = np.asarray(arr.array if hasattr(arr, "array") else arr, dtype=float)
            if data.shape[0] != nper:
                data = np.broadcast_to(data, (nper,) + data.shape).copy()
            data = data.reshape(nper, -1)
            if agg == "mean":
                values = np.nanmean(data, axis=1)
            elif agg == "sum":
                values = np.nansum(data, axis=1)
            else:
                raise ValueError(
                    f"Invalid value for 'agg': {agg!r}. Use 'mean' or 'sum'."
                )
            return _stress_period_series(model, values, freq=freq, name="recharge")

    # list-based recharge (MODFLOW 6 RCH)
    if getattr(pkg, "stress_period_data", None) is not None:
        return get_stress_series(
            model, package=pkg, as_dict=False, agg=agg, freq=freq
        ).rename("recharge")

    raise ValueError(f"Package {package} has no recharge data.")


def get_cell_distance(model: Any, cellid1: tuple, cellid2: tuple) -> float:
    """Return the horizontal distance between the centers of two cells.

    Parameters
    ----------
    model: flopy model
        A flopy model object with a structured or vertex grid.
    cellid1: tuple of int
        First cell identifier, e.g. (layer, row, column).
    cellid2: tuple of int
        Second cell identifier, e.g. (layer, row, column).

    Returns
    -------
    float
        Distance between the cell centers in model length units.

    """
    _import_flopy()
    center1 = _cell_center(model, _validate_cellid(model, cellid1))
    center2 = _cell_center(model, _validate_cellid(model, cellid2))
    return float(np.hypot(*(center1 - center2)))


def model_from_flopy(
    model: Any,
    cellids: tuple | list[tuple] | None = None,
    name: str | None = None,
    headfile: Any = None,
    add_wells: bool = True,
    well_package: str | Any = "WEL",
    well_max_distance: float | None = None,
    add_recharge: bool = True,
    recharge_package: str | Any = "RCH",
    rfunc_well: RFunc | None = None,
    rfunc_recharge: RFunc | None = None,
    well_settings: Any = "well",
    recharge_settings: Any = None,
    freq: str = "D",
    solve: bool = False,
    solver: Solver | None = None,
    tmin: Timestamp | str | None = None,
    tmax: Timestamp | str | None = None,
    report: bool = False,
    **kwargs: Any,
) -> Model | dict[str, Model]:
    """Create Pastas model(s) from a flopy MODFLOW model.

    The simulated heads at the given cell(s) are used as the oseries. A
    WellModel is created from the stresses of the well package (e.g. WEL),
    with distances computed from the model grid, and a StressModel is
    created from the recharge package (e.g. RCH).

    Parameters
    ----------
    model: flopy model
        A flopy model object, e.g. a ``flopy.modflow.Modflow`` or a
        ``flopy.mf6`` groundwater flow model. The model must have been run
        with head output saved.
    cellids: tuple or list of tuple, optional
        Cell identifier(s) of the observation location(s): (layer, row,
        column) for a structured grid or (layer, cell2d) for a vertex grid.
        Default is the first cell. If multiple cellids are given, a
        dictionary with one Pastas model per cell is returned.
    name: str, optional
        Name of the Pastas model. Default is the model name combined with
        the cell name, e.g. "gwf_L0R10C10".
    headfile: str, path or flopy HeadFile, optional
        Path to a head output file or an open flopy HeadFile object. If None,
        the head file of the model output is used.
    add_wells: bool, optional
        Add a WellModel for the well package (default True).
    well_package: str or flopy package, optional
        Name of the well package, "WEL" by default.
    well_max_distance: float, optional
        If given, only wells within this distance (in model length units)
        from the observation cell are included in the WellModel.
    add_recharge: bool, optional
        Add a StressModel for the recharge package (default True).
    recharge_package: str or flopy package, optional
        Name of the recharge package, "RCH" by default.
    rfunc_well: pastas.rfunc instance, optional
        Response function for the WellModel, default HantushWellModel.
    rfunc_recharge: pastas.rfunc instance, optional
        Response function for the recharge StressModel, default Exponential.
    well_settings: str or dict, optional
        Settings for the well stresses, default "well".
    recharge_settings: str or dict, optional
        Settings for the recharge stress. Default uses mean fill before and
        after and bfill up-sampling, similar to precipitation.
    freq: str, optional
        Frequency of the Pastas model simulation, default "D".
    solve: bool, optional
        Solve each model directly (default False).
    solver: pastas.solver.Solver, optional
        Solver used when solve is True. Default is the Pastas default solver.
    tmin: pandas.Timestamp or str, optional
        Start of the calibration period, only used when solve is True.
    tmax: pandas.Timestamp or str, optional
        End of the calibration period, only used when solve is True.
    report: bool, optional
        Print a report when solving, default False.
    **kwargs: Any
        Additional keyword arguments passed to Model.solve when solve is
        True, e.g. warmup.

    Returns
    -------
    pastas.Model or dict of pastas.Model
        A single Pastas model if one cellid is given, otherwise a dictionary
        mapping cell names (e.g. "L0R10C10") to Pastas models.

    Examples
    --------
    >>> import flopy
    >>> import pastas as ps
    >>> sim = flopy.mf6.MFSimulation.load(sim_ws="path/to/sim")
    >>> gwf = sim.get_model()
    >>> ml = ps.modflow.model_from_flopy(gwf, cellids=[(0, 10, 10)], solve=True)

    """
    _import_flopy()
    single = not isinstance(cellids, list)
    if cellids is None:
        cellids = [_validate_cellid(model, None)]
    elif single:
        cellids = [cellids]
    cellids = [_validate_cellid(model, cellid) for cellid in cellids]

    well_series: dict[tuple, Series] | None = None
    if add_wells:
        if _get_package(model, well_package) is None:
            logger.warning("No %s package found, no WellModel added.", well_package)
        else:
            per_cell, _, _, _ = _get_list_stress_series(model, well_package)
            pname = (
                well_package.lower()
                if isinstance(well_package, str)
                else type(_get_package(model, well_package)).__name__.lower()
            )
            well_series = {
                cellid: _stress_period_series(
                    model,
                    values,
                    freq=freq,
                    name=f"{pname}_{_cellid_to_name(cellid)}",
                )
                for cellid, values in per_cell.items()
            }

    recharge_series: Series | None = None
    if add_recharge:
        if _get_package(model, recharge_package) is None:
            logger.warning(
                "No %s package found, no recharge stress added.", recharge_package
            )
        else:
            recharge_series = get_recharge_series(
                model, package=recharge_package, freq=freq
            )

    models: dict[str, Model] = {}
    for cellid in cellids:
        cell_name = _cellid_to_name(cellid)
        model_name = (
            name if name is not None and single else f"{name or model.name}_{cell_name}"
        )
        oseries = get_head_series(
            model, cellid=cellid, headfile=headfile, name=model_name
        )
        ml = Model(
            oseries,
            name=model_name,
            metadata=_cell_metadata(model, cellid),
            freq=freq,
        )

        if recharge_series is not None:
            StressModel(
                ml,
                stress=recharge_series,
                rfunc=Exponential() if rfunc_recharge is None else rfunc_recharge,
                name="recharge",
                up=True,
                settings=(
                    _RECHARGE_SETTINGS
                    if recharge_settings is None
                    else recharge_settings
                ),
            )

        if well_series:
            stresses = []
            distances = []
            for well_cellid, series in well_series.items():
                distance = get_cell_distance(model, well_cellid, cellid)
                if distance <= 0.0:
                    logger.warning(
                        "Well %s is located in the observation cell %s, using a "
                        "distance of 1.0.",
                        _cellid_to_name(well_cellid),
                        cell_name,
                    )
                    distance = 1.0
                if well_max_distance is not None and distance > well_max_distance:
                    continue
                stresses.append(series)
                distances.append(distance)
            if stresses:
                WellModel(
                    ml,
                    stress=stresses,
                    rfunc=HantushWellModel() if rfunc_well is None else rfunc_well,
                    name="wells",
                    up=False,
                    distances=distances,
                    settings=well_settings,
                )

        if solve:
            ml.solve(tmin=tmin, tmax=tmax, solver=solver, report=report, **kwargs)

        models[cell_name] = ml

    if single:
        return models[_cellid_to_name(cellids[0])]
    return models
