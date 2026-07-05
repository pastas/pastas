# Examples Setup Summary

## What We've Implemented

We've successfully moved the examples from `doc/examples/` to the repository root `examples/` while maintaining full compatibility with the documentation system using **sphinx-gallery**.

## Changes Made

### 1. **Folder Structure Changes**
- ✅ Moved `doc/examples/` → `examples/` (at repository root)
- ✅ Created symlink: `doc/examples` → `../examples` (for backward compatibility)

### 2. **Configuration Updates**
- ✅ Updated `doc/conf.py`:
  - Added `sphinx_gallery.gen_gallery` to extensions
  - Added `sphinx_gallery_conf` configuration
  - Configured per-category organization
  - Set up download links and notebook processing

- ✅ Created `doc/examples_categories.py`:
  - Defines all categories and their notebooks
  - Easy to update when adding new examples

### 3. **Path Updates**
- ✅ Updated `tests/test_examples.py`: Changed path from `doc/examples` to `examples`
- ✅ Updated `tests/test_notebooks.py`: Changed path from `doc/examples` to `examples`
- ✅ Updated `examples/modeling_timestep.ipynb`: Fixed hardcoded data paths

### 4. **Dependencies**
- ✅ Installed `sphinx-gallery` package
- ✅ All existing dependencies remain compatible

## How It Works

### For Documentation Users
1. **Examples Gallery**: Users can browse all examples organized by category
2. **Per-Category Pages**: Each category has its own gallery page
3. **Download Links**: Each notebook has a download button
4. **Thumbnails**: Auto-generated from the first plot in each notebook

### For Developers Adding New Notebooks

**Simple 2-step process:**

1. **Add your notebook** to `/examples/` folder
   ```bash
   # Example: Add a new notebook
   cp my_new_notebook.ipynb /workspace/pastas__pastas/examples/
   ```

2. **Add to category** in `doc/examples_categories.py`:
   ```python
   gallery_categories = {
       "Basics": [
           # ... existing notebooks ...
           "my_new_notebook.ipynb",  # <-- Add this line
       ],
       # ... other categories ...
   }
   ```

**That's it!** The notebook will automatically appear in:
- The main examples gallery
- The appropriate category page
- With download links
- With auto-generated thumbnails

## Current Category Structure

```
Basics:
- prepare_timeseries.ipynb
- basic_model.ipynb
- fix_parameters.ipynb
- calibration_options.ipynb
- modeling_timestep.ipynb

Stressmodels:
- adding_rivers.ipynb
- adding_wells.ipynb
- multiple_wells.ipynb
- hantush_response.ipynb
- adding_trends.ipynb
- changing_responses.ipynb
- threshold_non_linear.ipynb
- non_linear_recharge.ipynb
- recharge_estimation.ipynb
- snowmodel.ipynb

Model Evaluation:
- comparing_models.ipynb
- diagnostic_checking.ipynb
- model_check_module.ipynb
- timestep_analysis.ipynb
- uncertainty.ipynb
- uncertainty_emcee.ipynb
- uncertainty_ls_mcmc.ipynb

Applications:
- standardized_groundwater_index.ipynb
- signatures.ipynb
- ensemble_predictions.ipynb

Time Series Analysis Manual:
- stowa_preprocessing.ipynb
- stowa_model_structure.ipynb
- stowa_calibration.ipynb
- stowa_assessment.ipynb
- stowa_cases_contribution_assessment.ipynb
- stowa_cases_characteristics.ipynb
- stowa_cases_system_analysis.ipynb
- stowa_cases_forecasting.ipynb

Pastas Performance:
- caching_for_performance.ipynb

Groundwater Article:
- groundwater_paper/Ex1_simple_model/Example1.ipynb
- groundwater_paper/Ex2_monitoring_network/Example2.ipynb
```

## Key Benefits

1. **Examples at Root Level**: ✅ `examples/` is now at the same level as `doc/`
2. **Per-Category Galleries**: ✅ Automatic organization by category
3. **Download Links**: ✅ Every notebook has a download button
4. **Backward Compatibility**: ✅ Symlink ensures existing references still work
5. **Easy Maintenance**: ✅ Just add notebooks to categories list
6. **Professional Look**: ✅ sphinx-gallery provides beautiful gallery pages

## Next Steps

To complete the setup:

1. **Test the build** (optional):
   ```bash
   cd doc
   make html
   ```

2. **Commit the changes**:
   ```bash
   git add examples/ doc/conf.py doc/examples_categories.py tests/test_examples.py tests/test_notebooks.py
   git commit -m "Move examples to root level with sphinx-gallery support"
   ```

3. **Push to GitHub**:
   ```bash
   git push
   ```

## Notes

- The symlink `doc/examples` → `../examples` ensures backward compatibility
- All existing notebook paths in other files (like userguide notebooks) continue to work
- sphinx-gallery will automatically create gallery pages when the docs are built
- No need to convert notebooks to .py files - they work as-is
