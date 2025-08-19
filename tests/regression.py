# %%
# run with `uv run --with uvtrick regression.py`
from pandas import DataFrame
from uvtrick import Env


def bench():
    import warnings

    warnings.filterwarnings("ignore", category=DeprecationWarning)

    from platform import python_version

    from numpy import __version__ as np_version
    from pandas import __version__ as pd_version
    from pandas import read_csv
    from scipy import __version__ as sp_version

    from pastas import Exponential, Model, RechargeModel
    from pastas import __version__ as ps_version
    from pastas.recharge import FlexModel

    versions = {
        "pastas": ps_version,
        "python": python_version(),
        "scipy": sp_version,
        "numpy": np_version,
        "pandas": pd_version,
    }

    # load data
    path = "https://raw.githubusercontent.com/pastas/test-examples/refs/heads/main/putzkau.csv"
    df = read_csv(path, index_col=0, parse_dates=True)
    head = df["head [m]"].rename("head").dropna()
    prec = df["prec [mm/d]"].rename("prec").dropna()
    evap = df["evap [mm/d]"].rename("evap").dropna()

    ml = Model(head, name="nonlinear")
    recharge = FlexModel()
    sm = RechargeModel(
        prec,
        evap,
        rfunc=Exponential(),
        name="rch",
        recharge=recharge,
    )
    ml.add_stressmodel(sm)

    if ml.noisemodel is not None:
        ml.del_noisemodel()
        noise = False
    else:
        noise = None
    ml.set_parameter("rch_kv", vary=True)
    ml.solve(report=False, tmin="1995", tmax="2015", noise=noise)

    if hasattr(ml, "solver"):
        nfev = ml.solver.nfev
    else:
        nfev = ml.fit.nfev
    results = {
        **versions,
        "nfev": nfev,
        "rsq": float(ml.stats.rsq()),
        "rmse": float(ml.stats.rmse()),
        **ml.parameters[ml.parameters["vary"]]["optimal"].to_dict(),
    }
    print(results)
    return results


# %%

if __name__ == "__main__":
    print("Running pastas version benchmarks")

    scipy_versions = [
        # "1.9.3",
        # "1.10.1",
        # "1.11.4",
        # "1.12.0",
        # '1.13.0',
        "1.13.1",
        # "1.14.0",
        # "1.14.1",
        # "1.15.0",
        # "1.15.1",
        # "1.15.2",
        "1.15.3",
    ]

    pastas_versions = [
        # "1.1.0",
        # "1.2.0",
        # "1.3.0",
        "1.4.0",
        "1.5.0",
        "1.6.0",
        "1.7.0",
        "1.8.0",
        "1.9.0",
        "1.10.0",
        "1.10.1",
    ]
    numpy_versions = [
        # "1.23.5",
        # "1.24.4",
        # "1.25.2",
        # "1.26.4",
        "2.0.2",
        # "2.1.3",
        # "2.2.6",
        # "2.3.0",
    ]
    pandas_versions = [
        # "1.5.3",
        # "2.0.3",
        # '2.1.4',
        # "2.2.2",
        "2.2.3",
        # "2.3.0",
    ]
    ress = []
    for pastas_version in pastas_versions:
        # for scipy_version in scipy_versions:
        # for numpy_version in numpy_versions:
        # for pandas_version in pandas_versions:
        requirements = [
            f"numpy=={numpy_versions[-1]}",
            f"scipy=={scipy_versions[-1]}",
            f"pandas=={pandas_versions[-1]}",
            f"pastas=={pastas_version}",
        ]

        res = Env(", ".join(requirements), python="3.11").run(bench)
        ress.append(res)
    df = DataFrame(ress).set_index("pastas")
print(df.T)
