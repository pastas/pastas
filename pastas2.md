# Introducing Pastas 2.0

Pastas 2.0 has several new features, but also some new syntax that replaces old syntax.
The new syntax will make Pastas scripts shorter and, more importantly, will make it
possible to implement some new features that are in our future plans. This post
introduces the most important changes and provides guidance for migrating your scripts
to Pastas 2.0.

## Contents

- [New syntax for adding components to Models](#new-syntax-for-adding-components-to-models)
- [Changes in defaults](#changes-in-defaults)
- [Renamed objects and access locations](#renamed-objects-and-access-locations)
- [New features](#new-features)
- [Notable deprecations](#notable-deprecations)
- [Pastas Sustaining Membership Program](#pastas-sustaining-membership-program)

---

## New syntax for adding components to Models

The biggest change for users is that we are introducing a new method to add components
to your model. Rather than first creating a `StressModel` and then adding it to the
model with `add_stressmodel`, this is now done in one statement, where the model is
provided as the first argument when creating a `StressModel`. The new syntax is:

```python
# Pastas 2.0
ml = ps.Model(oseries)
sm = ps.StressModel(ml, stress, rfunc=ps.Exponential(), name="rain")
#                   ^^ Note the model is now the first argument!
```

Whereas in previous versions of Pastas, the `ml.add_stressmodel()` was necessary:

```python
# Pastas <= 1.4
ml = ps.Model(oseries)
sm = ps.StressModel(stress, rfunc=ps.Exponential(), name="rain")
ml.add_stressmodel(sm)
```

The affected Pastas objects are listed below. Each of these classes now expects the
model as the first argument:

- Stress models (`StressModel`, `RechargeModel`, `WellModel`, `TarsoModel`, `StepTrend`, etc.)
- Noise models (`ArNoiseModel`, `ArmaNoiseModel`)
- Solvers (`Lmfit`, `LeastSquares`, etc.)
- Transforms (`ThresholdTransform`)
- Constant (`Constant`)

Deprecated functions that will log a warning in Pastas 2.0:

- `ml.add_stressmodel()`
- `ml.add_noisemodel()`
- `ml.add_solver()`
- `ml.add_transform()`
- `ml.add_constant()`

Note that there will be a grace period in which we support both syntaxes. When using
the old method, you will receive a warning, but it will continue to work until Pastas
2.4. We hope this gives our users sufficient time to adapt their scripts to the new
syntax.

### So why are we making this big change to the syntax?

The main reason for this syntax change is that all objects (stress models, noise
models, solvers, etc.) will know to which Model they are added. This will make it
possible for each object to use information from the Model. For example, the head
observations (`oseries`) may be used to set initial values of parameters. Another
example is that for the development of new solvers for Pastas (inspired by PEST++), the
solver classes need to be able to access the model in order to set up the solve.

Another reason for the syntax change has to do with the original Pastas design. One of
the original design ideas was that a user could make a `StressModel`, for example to
simulate the effect of rainfall, and then add this `StressModel` to multiple Pastas
models, as several observation wells probably use the same weather data. But Pastas has
evolved over the years, and nowadays this is not possible anymore. So if you cannot add
the same `StressModel` to multiple Pastas models, the whole advantage of a separate
`add_stressmodel` function has disappeared (and may even be confusing).

Some other smaller advantages of the new syntax are:

- shorter scripts (no more `ml.add_stressmodels()`, etc.)
- matches behavior in other packages frequently used by hydrologists, e.g. flopy:
`flopy.mf6.Modflowgwfwel(gwf_model, ...)` and timflow `timflow.steady.Well(model, ...)`

We are aware that syntax changes are annoying, and we did not make this change lightly.
There was quite a bit of debate, but in the end we decided the benefits outweighed the
(temporary) annoyance of getting used to a syntax change.

## Changes in defaults

The following changes to default behavior of Pastas are introduced in Pastas 2.0. These
changes mean that when relying on default behavior, models solved with Pastas 2.0 might
give different results than models solved with older versions of Pastas. We think these
changes will generally result in better models.

- The default arguments for `ps.solver.LeastSquares` have changed slightly. We now
default to a more accurate 3-point method to compute the Jacobian `jac="3-point"`. The
previous default was `jac="2-point"`. This change introduces a small extra
computational effort but the advantage of a more accurate estimate of the Jacobian can
lead to better performing models. Users can always go back to 2-point.

- The parameter update in `ps.solver.LeastSquares` is now controlled by `x_scale="jac"`
which uses the Jacobian to determine the step change for each parameter during
optimization. Using the Jacobian gives better parameter updates, especially when
parameter scales are very different.

- The order of arguments in `WellModel`, `TarsoModel`, `StepModel`, `LinearTrend`, and
`ChangeModel` was adjusted to more closely match the other StressModels. All
StressModels now adhere to the following order: stress/time input, rfunc, name, etc. Be
aware that old code that used positional arguments for these StressModels may be
affected and result in errors when run with Pastas 2.0.

- Adjustments to the parameter boundaries for `ps.rch.FlexModel`. **fill in here**

- The `HantushWellModel` response function used natural log scaling to avoid very small
values for parameter `b` in the optimization. This has now been updated to use log10
scaling (which makes more sense), which can be controlled with the `log_b=True|False`
keyword argument. The default is True, so this means any old Pastas models with a
WellModel will yield somewhat different estimates of parameter `b` when solved with
Pastas 2.0. Take care when loading older models, be sure to solve them again with
Pastas 2.0 prior to doing any simulations.

- Parameter `_d` was renamed to `_A` in the `ps.One()` response function.

## Renamed objects and access locations

We put a lot of effort into Pastas 2.0 to make the software more consistent and
intuitive while maintaining ease of use. As a part of that effort, several classes were
renamed and some were moved. Of course all these changes come with warnings, so the old
methods will still work but will generate warnings about future deprecations.

### Solvers

As part of a major internal refactor of the solver classes they were renamed and
moved to `ps.solvers`:

- `ps.LeastSquares() --> ps.solvers.LeastSquares()`
- `ps.LmfitSolve() --> ps.solvers.Lmfit()`
- `ps.EmceeSolve() --> ps.solvers.Emcee()`

Instead of making each solver have `"Solve"` in its name, we figured it was much
clearer to just move all these classes to the `ps.solvers` submodule.

### Options

Pastas had certain global option variables that could be set through
`ps.set_use_numba()` or `ps.set_use_cache()`. These are now collected in `ps.options`
and can be set from there:

```python
ps.options.cache = True  # to turn on caching, which can speed up model solves by caching simulate results
ps.options.numba = False  # to turn off numba (not recommended, except for debugging)
ps.options.parallel = True  # to turn on parallel processing for the Emcee solver.
```

The `ps.options` replaces the `ps.rcParams` module.

### Time series settings

Pastas uses certain logic to fill gaps, extend time series into the future or past, or
up- or downsample time series to other timesteps. These default settings were defined
in `ps.rcParams["timeseries"]` which contained dictionaries containing default settings
for precipitation (`"prec"`), evaporation (`"evap"`), waterlevels (`"waterlevel"`) and
wells (`"well"`), etc. The placement of these settings within Pastas made them hard to
find, so they were moved to `ps.timeseries.settings`:

```python
ps.timeseries.settings  # <-- see all settings
prec_settings = ps.timeseries.settings["prec"]  # <-- get settings for "prec"
```

Hopefully that feels a lot more intuitive.

### StressModel stresses attribute refactoring

We have updated how stress time series are stored and accessed in `StressModel` and its subclasses (such as `RechargeModel`, `WellModel`, and `TarsoModel`). So what's changed?

* **`sm.stresses` (NamedTuple of TimeSeries):** The primary attribute for accessing all stresses in a model is now `sm.stresses`. It returns an immutable **named tuple** of `pastas.TimeSeries` objects (e.g., `(stress,)` or `(prec, evap)`). Because it is a named tuple, you can access stresses either by index or directly by name (e.g., `rm.stresses.prec` is equivalent to `rm.stresses[0]`).
* **Direct Property Access:**
    - Single-stress models (`StressModel`): Expose their primary `pastas.TimeSeries` object via `sm.stress`.
    - Multi-stress models (`RechargeModel`, `TarsoModel`, etc.): Expose dedicated attributes for each stress, such as `rm.prec`, `rm.evap`, or `rm.temp`.
* **Simpler Updates:** Updating a stress series is now much more intuitive. You can directly assign a `pandas.Series` (or `pastas.TimeSeries`) to the attribute, or use the `.set_stress()` method.

This is what it would look like in practice:

```python
# --- Single StressModel ---

# New way: Update stress using pandas.Series or pastas.TimeSeries
sm.set_stress(new_series)  # Recommended method
sm.stress = new_series  # Direct property assignment

# Old way (deprecated)
sm.stress[0] = new_ts_object  # Required a pastas.TimeSeries object

# --- Multi-stress Model (e.g., RechargeModel) ---

# Accessing individual stress objects
prec_ts = rm.prec
evap_ts = rm.evap

# Accessing via named tuple
prec_ts = rm.stresses.prec

# Updating stress series
rm.set_stress(prec=new_prec_series, evap=new_evap_series)
# or alternatively
rm.prec = new_prec_series
rm.evap = new_evap_series

# --- Iterating Over Stresses ---

# Old way (deprecated)
for ts in sm.stress:
    ts.series.plot()

# New way (Pastas 2.0)
for ts in sm.stresses:
    ts.series.plot()
```

## New features

Among the many new features introduced by Pastas 2.0, these are some of the highlights:

- Use the impulse response instead of the block response to simulate heads. This can be
set by adding `use_block=False` in the response function definitions. This is faster,
since it avoids having to compute two step responses, and produces comparable results
in our tests. Set with e.g. `ps.FourParam(use_block=False)`.

- Use `jac="cs"` in `ps.solver.LeastSquares()` to compute the Jacobian using a complex
step. This method is more accurate and more efficient than other jacobian estimation
routines despite the added computational burden of doing complex arithmetic. Pastas
internals were adjusted to support complex math in order to make this possible. If more
accurate Jacobians are your thing, try this out.

- The `tmax` estimates of the `Hantush` and `FourParam` response functions for a given
`cutoff` can now be computed more accurately using fast numerical integration. This is
slightly slower than the default conservative analytical approximations but can be used
to avoid very long response function tails that sometimes crop up in models.

## Notable deprecations

- The `noise=True|False` argument was removed from `ml.solve()` in Pastas 2.0. Whether
or not a noise model is applied is completely dependent on the presence of a
NoiseModel. We understand it is slightly more verbose to solve a model with and then
without a noise model which is why we are working on so-called "solve strategies"
(#1175) that can automate certain solve steps for users. This is still work in progress
so stay tuned for that in a future Pastas release.

- `ps.plots.contributions_pie()` was removed. This chart is difficult to generalize to
all Pastas models and can sometimes be misleading. We leave it to the users to decide
whether it is appropriate to present their data this way.

- `ps.stats.kge_2012` was removed. Use `ps.stats.kge(..., modified=True)` instead.

- The specification of a Solver in the `solve` method of a Pastas model is deprecated.
The solver must be defined prior to solving the model. So while it used to be possible
to solve a model called `ml` with the Lmfit solver by typing
`ml.solve(solver=ps.LmfitSolve())`, it now requires the specification of a solver first
`ps.solvers.Lmfit(model=ml)`, after which a simple `ml.solve()` will solve the model
with the specified solver.

## Pastas Sustaining Membership Program

We are introducing the [Pastas Sustaining Membership
Program](https://pastas.dev/about/support.html) to allow users to financially
contribute to the development and upkeep of Pastas. To be clear, **Pastas will always
remain open-source and free**! However, it is important to realize that a lot of the
maintenance has historically been carried out on a voluntary basis by the Pastas
development team. This includes investigating and solving bug reports, responding to
questions on the Discussions page, requests for new features, etc. Mind you, the Pastas
development team really enjoys (many of) these tasks, but time and budgets are limited.
As the Pastas user base grows and the software becomes more capable, this maintenance
requires additional resources. Yes, Pastas is free and open-source software but that
doesn't mean it doesn't cost any money to maintain. So if you or your company are
frequent users of Pastas software, we kindly ask you to consider supporting Pastas
financially. Check out the Sustainable Membership Program and join!
