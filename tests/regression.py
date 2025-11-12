"""
Regression Testing for Pastas
==============================

This script performs regression testing to verify that model results (ml.solve())
remain consistent across different versions of Pastas and its dependencies.

How it Works
------------
Uses uvtrick (https://github.com/koaning/uvtrick) to run code in isolated
environments with different package versions using the uv package manager.

What is Tested
--------------
1. Creates a standardized model using test data from the Pastas test-examples repository
2. Solves the model using ml.solve() with fixed parameters
3. Compares results across multiple versions of Pastas, NumPy, SciPy, and Pandas
4. Tracks key metrics: R-squared (rsq), RMSE, number of function evaluations (nfev),
   and model parameters (optimal values)

Running Locally with Tox
-------------------------
The recommended way to run regression tests is using tox:
    tox -e regression

This uses the tox configuration in pyproject.toml and ensures consistent
execution with the CI environment.

Alternative: Direct Execution
-----------------------------
Prerequisites:
    pip install uv uvtrick

Usage:
    cd tests
    uv run --with uvtrick regression.py

This will:
- Test multiple Pastas versions with the same dependencies
- Print results to the console
- Save detailed results to regression_results.json

Modifying Test Versions
------------------------
Edit the version lists below (pastas_versions, scipy_versions, etc.) to customize
which versions are tested. Uncomment versions you want to include in the tests.

Continuous Integration
----------------------
The regression tests run automatically via GitHub Actions:
- Trigger: Pushes to the master branch
- Workflow file: .github/workflows/test_regression.yml
- Uses tox for consistent execution with local testing
- Manual trigger: Can be run manually from the Actions tab
- Results: Stored as GitHub Actions artifacts with 90-day retention

Understanding Results
---------------------
The output shows metrics for each tested version. Look for:
- Consistent metrics: rsq, rmse, and parameters should be identical or very similar
- Unexpected changes: Large differences may indicate breaking changes, dependency
  behavior changes, or numerical stability issues

Future Enhancements
-------------------
- Automated comparison with previous runs
- Tolerance thresholds for acceptable deviations
- Alerts for significant regressions
- Historical tracking and visualization
- Multiple test cases with different configurations
"""

# %%
# run with `uv run --with uvtrick regression.py`
import json
from pathlib import Path

from pandas import DataFrame, Timestamp
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

        res = Env(", ".join(requirements)).run(bench)
        ress.append(res)
    df = DataFrame(ress).set_index("pastas")
    print(df.T)

    # Save results to JSON for CI/CD tracking
    output_data = {
        "timestamp": Timestamp.now().isoformat(),
        "results": ress,
        "summary": df.to_dict(orient="index"),
    }

    output_file = Path(__file__).parent / "regression_results.json"
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_file}")
