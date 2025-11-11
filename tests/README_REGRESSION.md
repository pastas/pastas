# Regression Testing for Pastas

## Overview

This directory contains regression testing infrastructure for Pastas. The regression tests verify that model results (`ml.solve()`) remain consistent across different versions of Pastas and its dependencies.

## How it Works

The regression testing uses [`uvtrick`](https://github.com/baggiponte/uvtrick), which allows running code in isolated environments with different package versions using the `uv` package manager.

### What is Tested

The regression test (`regression.py`) performs the following:

1. **Creates a standardized model** using test data from the Pastas test-examples repository
2. **Solves the model** using `ml.solve()` with fixed parameters
3. **Compares results** across multiple versions of:
   - Pastas (different releases)
   - NumPy
   - SciPy
   - Pandas
4. **Tracks key metrics**:
   - R-squared (rsq)
   - Root Mean Square Error (rmse)
   - Number of function evaluations (nfev)
   - Model parameters (optimal values)

## Running Regression Tests

### Prerequisites

Install the required tools:

```bash
pip install uv uvtrick
```

### Running Locally

To run regression tests manually:

```bash
cd tests
uv run --with uvtrick regression.py
```

This will:
- Test multiple Pastas versions with the same dependencies
- Print results to the console
- Save detailed results to `regression_results.json`

### Modifying Test Versions

Edit `regression.py` to customize which versions are tested:

```python
pastas_versions = [
    "1.4.0",
    "1.5.0",
    # ... add or remove versions
]

scipy_versions = ["1.15.3"]  # Test different scipy versions
numpy_versions = ["2.0.2"]   # Test different numpy versions
pandas_versions = ["2.2.3"]  # Test different pandas versions
```

## Continuous Integration

### GitHub Actions Workflow

The regression tests run automatically via GitHub Actions:

- **Trigger**: Pushes to the `master` branch
- **Workflow file**: `.github/workflows/test_regression.yml`
- **Manual trigger**: Can be run manually from the Actions tab

### Viewing Results

1. Go to the [Actions tab](https://github.com/pastas/pastas/actions)
2. Select the "CI REGRESSION" workflow
3. Click on a workflow run
4. Download the `regression-results-*` artifact to view detailed results

### Artifact Storage

- Results are stored as GitHub Actions artifacts
- Retention period: 90 days
- Each run is tagged with the commit SHA

## Understanding Results

The regression test output includes:

```
pastas           1.4.0       1.5.0      1.6.0      ...
python         3.11.14     3.11.14    3.11.14      ...
scipy           1.15.3      1.15.3     1.15.3      ...
numpy            2.0.2       2.0.2      2.0.2      ...
pandas           2.2.3       2.2.3      2.2.3      ...
nfev               134         134        134      ...
rsq           0.915723    0.915723   0.915723      ...
rmse          0.089082    0.089082   0.089082      ...
rch_A           1.7471      1.7471     1.7471      ...
...
```

### What to Look For

- **Consistent metrics**: rsq, rmse, and parameters should be identical or very similar across versions
- **Unexpected changes**: Large differences in results may indicate:
  - Breaking changes in Pastas
  - Changes in dependency behavior
  - Numerical stability issues
  
## Future Enhancements

Potential improvements to the regression testing framework:

1. **Automated comparison**: Automatically compare results with previous runs
2. **Tolerance thresholds**: Define acceptable deviation ranges
3. **Alerts**: Notify maintainers when significant regressions are detected
4. **Historical tracking**: Store and visualize results over time
5. **Multiple test cases**: Test different model configurations
6. **Performance tracking**: Monitor solve time and memory usage

## Contributing

When adding new features to Pastas:

1. Run regression tests locally before submitting PR
2. Document any expected changes in model behavior
3. Update test configurations if new dependencies are added

## Questions?

For questions or issues with regression testing, please open an issue on the [Pastas GitHub repository](https://github.com/pastas/pastas/issues).
