# Breaking Changes in Pastas 2.0

This document tracks all breaking changes introduced in the Pastas 2.0 release.

## Python Version Support

### Drop Python 3.10 Support
- **Commit:** ec608ce
- **PR:** Part of dev2 branch
- **Impact:** Python 3.10 is no longer supported. Minimum required version is now Python 3.11.
- **Migration:** Users must upgrade to Python 3.11 or later.
- **Files Changed:**
  - `pyproject.toml`: Updated `requires-python` from `>= 3.10` to `>= 3.11`
  - `.github/workflows/test_unit_pytest.yml`: Removed Python 3.10 from test matrix
  - Removed Python 3.10 from classifiers and tox environment list

## API Changes

### StressModel.stress Changed to Property
- **Commit:** 492d18d
- **PR:** #1065
- **Impact:** The `stress` attribute on StressModel classes is now a property instead of a direct attribute.
- **Migration:** 
  - For single stress models (e.g., `StressModel`): Access stress via `sm.stress` (now returns TimeSeries directly)
  - For multiple stress models (e.g., `RechargeModel`): Access individual stresses via named properties like `sm.prec`, `sm.evap`, etc.
  - New `add_stress()` method added for adding stress data
  - `set_stress()` method now available for setting stress data
  - `update_stress()` signature changed
- **Old Code:**
  ```python
  sm = ml.stressmodels["Recharge"]
  prec = sm.stress[0].series_original
  evap = sm.stress[1].series_original
  ```
- **New Code:**
  ```python
  sm = ml.stressmodels["Recharge"]
  prec = sm.prec.series_original
  evap = sm.evap.series_original
  ```
- **Files Changed:**
  - `pastas/stressmodels.py`: Major refactoring of stress handling
  - `pastas/forecast.py`: Updated to use new stress property
  - `pastas/model.py`, `pastas/plotting/modelplots.py`, `pastas/plotting/plotutil.py`: Updates to work with new API
  - Multiple example notebooks and test files updated

### Forecast Data Structure Change
- **Commit:** 492d18d (same as stress property change)
- **PR:** #1065
- **Impact:** The structure of forecast data dictionaries has changed for multi-stress models.
- **Old Code:**
  ```python
  fc = {
      "rch": [
          pd.read_csv("ensemble_prec.csv", index_col=0, parse_dates=True),
          pd.read_csv("ensemble_evap.csv", index_col=0, parse_dates=True),
          pd.read_csv("ensemble_temp.csv", index_col=0, parse_dates=True),
      ]
  }
  ```
- **New Code:**
  ```python
  fc = {
      "rch": {
          "prec": pd.read_csv("ensemble_prec.csv", index_col=0, parse_dates=True),
          "evap": pd.read_csv("ensemble_evap.csv", index_col=0, parse_dates=True),
          "temp": pd.read_csv("ensemble_temp.csv", index_col=0, parse_dates=True),
      }
  }
  ```

### RechargeModel Constructor Change
- **Commit:** 492d18d
- **PR:** #1065
- **Impact:** RechargeModel now requires keyword arguments for stress inputs.
- **Old Code:**
  ```python
  ml.add_stressmodel(
      ps.RechargeModel(
          prec,
          evap,
          rfunc=ps.Gamma(),
          recharge=ps.rch.FlexModel(snow=True),
          temp=temp,
          name="rch",
      )
  )
  ```
- **New Code:**
  ```python
  ml.add_stressmodel(
      ps.RechargeModel(
          prec=prec,
          evap=evap,
          rfunc=ps.Gamma(),
          recharge=ps.rch.FlexModel(snow=True),
          temp=temp,
          name="rch",
      )
  )
  ```

## Internal Changes

### Removed pd_version Configuration
- **Commit:** 60522c5
- **Impact:** Removed `pd_options.future.infer_string = True` from `pastas/version.py`
- **Migration:** No user action required. This was an internal pandas configuration.

### KGE Metric Name Change
- **Commit:** 790bc06
- **Impact:** The metric name `kge_2012` has been changed to `kge` in examples.
- **Migration:** Use `kge` instead of `kge_2012` when referencing this metric.
- **Files Changed:**
  - `doc/examples/recharge_estimation.ipynb`

## Documentation Updates

### Model.fit References Updated to Model.solver
- **Commit:** 42df832, e4bb953
- **Impact:** Documentation and examples updated to reflect the correct API (`.solver` instead of `.fit`)
- **Note:** This appears to be a documentation fix rather than a breaking change, as `.fit` was likely already deprecated.
- **Files Changed:**
  - `doc/examples/hantush_response.ipynb`
  - `doc/examples/stowa_assessment.ipynb`
  - `doc/examples/stowa_cases_contribution_assessment.ipynb`
  - `doc/examples/stowa_cases_forecasting.ipynb`
  - `tests/regression.py`

## Summary

The major breaking changes in Pastas 2.0 are:
1. **Python 3.10 is no longer supported** - Minimum version is Python 3.11
2. **StressModel.stress is now a property** - Access via `sm.stress` or named properties like `sm.prec`
3. **Forecast data structure changed** - Use nested dictionaries with named stress inputs
4. **RechargeModel requires keyword arguments** - Use `prec=prec, evap=evap` instead of positional arguments

These changes improve the API consistency and make the code more maintainable, but will require updates to existing code.
