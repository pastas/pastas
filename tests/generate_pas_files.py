# /// script
# dependencies = [
#   "lmfit"
# ]
# ///
"""Generate .pas files covering all model components for a specific Pastas version.

Run with uv to pin the Pastas version used for generation, e.g.::

    uv run --with "pastas==1.13.1" generate_pas_files.py

The generated files are written to ``data/pas_files_<version>/``.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import pastas as ps

# ---------------------------------------------------------------------------
# Output directory and reproducible synthetic data
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path(f"data/pas_files_{ps.__version__}")

_rng = np.random.default_rng(42)
_idx = pd.date_range("2000-01-01", periods=365, freq="D")
DATA = pd.Series(_rng.random(365), index=_idx, name="head")


# ---------------------------------------------------------------------------
# Model-builder helpers
# ---------------------------------------------------------------------------


def create_model(oseries: pd.Series, constant: bool = True) -> ps.Model:
    """Create a basic Pastas model from an observation series.

    Parameters
    ----------
    oseries : pd.Series
        Observation time series.
    constant : bool, optional
        Whether to include a constant in the model. Default is True.

    Returns
    -------
    ps.Model
        Pastas model instance.
    """
    return ps.Model(oseries, constant=constant)


def add_stressmodel(
    ml: ps.Model,
    stress: pd.Series,
    rfunc_name: str,
    rfunc_kwargs: dict[str, Any],
) -> None:
    """Add a StressModel with the specified response function.

    Parameters
    ----------
    ml : ps.Model
        Pastas model to which the stressmodel is added.
    stress : pd.Series
        Stress time series.
    rfunc_name : str
        Name of the response function class (e.g. ``"Exponential"``).
    rfunc_kwargs : dict
        Keyword arguments forwarded to the response function constructor.
    """
    rfunc_cls = getattr(ps, rfunc_name)
    sm = ps.StressModel(
        stress, name="stress", rfunc=rfunc_cls(**rfunc_kwargs), settings="prec"
    )
    ml.add_stressmodel(sm)


def add_rechargemodel(
    ml: ps.Model,
    stresses: dict[str, pd.Series],
    recharge_name: str,
    rech_kwargs: dict[str, Any],
) -> None:
    """Add a RechargeModel with the specified recharge class.

    Parameters
    ----------
    ml : ps.Model
        Pastas model to which the recharge model is added.
    stresses : dict
        Mapping of ``{"prec": ..., "evap": ..., "temp": ...}`` series.
    recharge_name : str
        Name of the recharge class (e.g. ``"Linear"``).
    rech_kwargs : dict
        Keyword arguments forwarded to the recharge constructor.
    """
    recharge_cls = getattr(ps.recharge, recharge_name)
    rm = ps.RechargeModel(
        **stresses,
        rfunc=ps.Exponential(),
        name="recharge",
        recharge=recharge_cls(**rech_kwargs),
    )
    ml.add_stressmodel(rm)


def add_tarsomodel(
    ml: ps.Model,
    prec: pd.Series,
    evap: pd.Series,
) -> None:
    """Add a TarsoModel to the model using explicit drainage level bounds.

    Parameters
    ----------
    ml : ps.Model
        Pastas model to which the TarsoModel is added.
    prec : pd.Series
        Precipitation time series.
    evap : pd.Series
        Evaporation time series.

    Notes
    -----
    TarsoModel is incompatible with other stressmodels, a constant, or a
    transform. Create the model with ``constant=False`` and add no other
    stressmodels beforehand.
    """
    tm = ps.TarsoModel(prec, evap, oseries=ml.oseries.series_original, name="tarso")
    ml.add_stressmodel(tm)


def add_thresholdtransform(ml: ps.Model) -> None:
    """Add a ThresholdTransform to the model.

    Parameters
    ----------
    ml : ps.Model
        Pastas model to which the transform is added.
    """
    tt = ps.ThresholdTransform(name="ThresholdTransform")
    ml.add_transform(tt)


def add_wellmodel(
    ml: ps.Model,
    stresses: list[pd.Series],
    distances: list[float],
    rfunc_kwargs: dict[str, Any],
) -> None:
    """Add a WellModel with the HantushWellModel response function.

    Parameters
    ----------
    ml : ps.Model
        Pastas model to which the WellModel is added.
    stresses : list of pd.Series
        List of pumping well time series; each must have a unique ``name``.
    distances : list of float
        Distances (m) between each well and the observation well.
    rfunc_kwargs : dict
        Keyword arguments forwarded to ``ps.HantushWellModel``.
    """
    wm = ps.WellModel(
        stresses,
        "wellmodel",
        distances=distances,
        rfunc=ps.HantushWellModel(**rfunc_kwargs),
    )
    ml.add_stressmodel(wm)


def add_changemodel(ml: ps.Model, stress: pd.Series) -> None:
    """Add a ChangeModel with Exponential → Gamma response functions.

    Parameters
    ----------
    ml : ps.Model
        Pastas model to which the ChangeModel is added.
    stress : pd.Series
        Stress time series driving the change model.
    """
    cm = ps.ChangeModel(
        stress,
        rfunc1=ps.Exponential(),
        rfunc2=ps.Gamma(),
        name="changemodel",
        tchange="2000-07-01",
    )
    ml.add_stressmodel(cm)


def add_noisemodel(ml: ps.Model, noise_name: str) -> None:
    """Add a noise model by class name.

    Parameters
    ----------
    ml : ps.Model
        Pastas model to which the noise model is added.
    noise_name : str
        Name of the noise model class (e.g. ``"ArNoiseModel"``).
    """
    noise_cls = getattr(ps, noise_name)
    ml.add_noisemodel(noise_cls())


def add_lineartrend(ml: ps.Model) -> None:
    """Add a LinearTrend stressmodel spanning the synthetic data period.

    Parameters
    ----------
    ml : ps.Model
        Pastas model to which the LinearTrend is added.
    """
    lt = ps.LinearTrend(start="2000-01-01", end="2000-12-31", name="lineartrend")
    ml.add_stressmodel(lt)


def add_stepmodel(ml: ps.Model) -> None:
    """Add a StepModel with an instantaneous response function.

    Parameters
    ----------
    ml : ps.Model
        Pastas model to which the StepModel is added.
    """
    sm = ps.StepModel(tstart="2000-07-01", rfunc=ps.One(), name="steptrend")
    ml.add_stressmodel(sm)


def add_solver(ml: ps.Model, stress: pd.Series, solver_name: str) -> None:
    """Add a StressModel and solve the model with the specified solver.

    Parameters
    ----------
    ml : ps.Model
        Pastas model to solve.
    stress : pd.Series
        Stress time series used in the StressModel.
    solver_name : str
        Name of the solver class (e.g. ``"LeastSquares"``).
    """
    add_stressmodel(ml, stress, rfunc_name="Exponential", rfunc_kwargs={})
    solver_cls = getattr(ps, solver_name)
    ml.solve(solver=solver_cls(), report=False)


# ---------------------------------------------------------------------------
# Save / validate helpers
# ---------------------------------------------------------------------------


def save_model(ml: ps.Model, output_dir: Path, filename: str) -> Path:
    """Save a Pastas model to a .pas file.

    Parameters
    ----------
    ml : ps.Model
        Pastas model to save.
    output_dir : Path
        Directory in which to write the file.
    filename : str
        File name (including ``.pas`` extension).

    Returns
    -------
    Path
        Absolute path of the written file.
    """
    fname = output_dir / filename
    fname.parent.mkdir(parents=True, exist_ok=True)
    ml.to_file(fname)
    return fname


def round_trip(fname: Path) -> None:
    """Validate that *fname* can be loaded back without errors.

    Parameters
    ----------
    fname : Path
        Path to the .pas file to load.

    Raises
    ------
    Exception
        Re-raises any exception raised by ``ps.io.load``.
    """
    ps.io.load(fname)


def _variant_fname(base: str, kwargs: dict[str, Any]) -> str:
    """Build a canonical filename from a base name and variant kwargs.

    Parameters
    ----------
    base : str
        Base class name (e.g. ``"Hantush"``).
    kwargs : dict
        Variant keyword arguments (e.g. ``{"quad": True}``).

    Returns
    -------
    str
        Filename string such as ``"Hantush_quad-True.pas"``.
    """
    suffix = "_".join(f"{k}-{v}" for k, v in kwargs.items())
    return f"{base}{'_' if suffix else ''}{suffix}.pas"


def composer(
    func: Any,
    fname: str,
    oseries: pd.Series,
    output_dir: Path,
    constant: bool = True,
    **kwargs: Any,
) -> None:
    """Build a model, apply *func*, save it, and round-trip validate it.

    Parameters
    ----------
    func : callable
        One of the ``add_*`` builder functions.
    fname : str
        Output filename (relative to *output_dir*).
    oseries : pd.Series
        Observation series passed to ``create_model``.
    output_dir : Path
        Directory in which to write the .pas file.
    constant : bool, optional
        Whether to include a constant in the model. Default is True.
    **kwargs
        Additional keyword arguments forwarded to *func*.
    """
    ml = create_model(oseries, constant=constant)
    func(ml, **kwargs)
    _saved = save_model(ml, output_dir=output_dir, filename=fname)
    # round_trip(saved)


# ---------------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------------


def generate(series: pd.Series, output_dir: Path) -> None:
    """Generate all .pas files for the current Pastas version.

    Parameters
    ----------
    oseries : pd.Series
        Synthetic observation series used for all models.
    output_dir : Path
        Root directory where .pas files are written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # -- StressModel with various response functions -------------------------
    rfunc_variants: dict[str, dict[str, dict]] = {
        "FourParam": {"default": {}, "quad": {"quad": True}},
        "Hantush": {"default": {}, "quad": {"quad": True}},
    }
    rfunc_names = [
        "Exponential",
        "Gamma",
        "DoubleExponential",
        "FourParam",
        "Hantush",
        "Kraijenhoff",
        "One",
        "Polder",
        "Spline",
    ]
    for rfunc in rfunc_names:
        for kwargs in rfunc_variants.get(rfunc, {"default": {}}).values():
            composer(
                add_stressmodel,
                _variant_fname(rfunc, kwargs),
                series,
                output_dir,
                stress=series,
                rfunc_name=rfunc,
                rfunc_kwargs=kwargs,
            )

    # -- RechargeModel with various recharge classes -------------------------
    recharge_variants: dict[str, dict[str, dict]] = {
        "FlexModel": {
            "default": {},
            "interception": {"interception": True},
            "snow": {"snow": True},
            "gw_uptake": {"gw_uptake": True},
        },
    }
    recharge_names = ["Linear", "FlexModel", "Berendrecht", "Peterson"]
    stresses = {"prec": series, "evap": series, "temp": series}
    for recharge in recharge_names:
        for kwargs in recharge_variants.get(recharge, {"default": {}}).values():
            composer(
                add_rechargemodel,
                _variant_fname(recharge, kwargs),
                series,
                output_dir,
                stresses=stresses,
                recharge_name=recharge,
                rech_kwargs=kwargs,
            )

    # -- TarsoModel (incompatible with constant and other stressmodels) ------
    composer(
        add_tarsomodel,
        "TarsoModel.pas",
        series,
        output_dir,
        constant=False,
        prec=series,
        evap=series,
    )

    # -- WellModel -----------------------------------------------------------
    well_stresses = [series.rename("well1"), series.rename("well2")]
    composer(
        add_wellmodel,
        "WellModel.pas",
        series,
        output_dir,
        stresses=well_stresses,
        distances=[100.0, 200.0],
        rfunc_kwargs={},
    )

    # -- ChangeModel ---------------------------------------------------------
    composer(
        add_changemodel,
        "ChangeModel.pas",
        series,
        output_dir,
        stress=series,
    )

    # -- LinearTrend ---------------------------------------------------------
    composer(add_lineartrend, "LinearTrend.pas", series, output_dir)

    # -- StepModel -----------------------------------------------------------
    composer(add_stepmodel, "StepModel.pas", series, output_dir)

    # -- ThresholdTransform (requires at least one stressmodel) --------------
    ml = create_model(series)
    add_rechargemodel(ml, stresses, "Linear", {})
    add_thresholdtransform(ml)
    save_model(ml, output_dir, "ThresholdTransform.pas")

    # -- Noise models --------------------------------------------------------
    for noise in ["ArNoiseModel", "ArmaNoiseModel"]:
        composer(
            add_noisemodel,
            f"{noise}.pas",
            series,
            output_dir,
            noise_name=noise,
        )

    # -- Solvers (solved models) ---------------------------------------------
    for solver in ["LeastSquares", "LmfitSolve"]:
        composer(
            add_solver,
            f"solver_{solver}.pas",
            series,
            output_dir,
            stress=series,
            solver_name=solver,
        )

    # -- Model with / without Constant ---------------------------------------
    for constant in [True, False]:
        ml = create_model(series, constant=constant)
        save_model(ml, output_dir, f"model_constant-{constant}.pas")


if __name__ == "__main__":
    generate(DATA, OUTPUT_DIR)
