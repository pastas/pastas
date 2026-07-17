# Introducing Pastas 2.0

This post introduces the lastest version of Pastas, that includes loads of new
features! The biggest changes are explained and guidance is provided for migrating your
scripts to Pastas 2.0.

## Contents

- [New syntax for adding components to Models](#new-syntax-for-adding-components-to-models)
- [Changes in defaults](#changes-in-defaults)
- [Renamed objects and access locations](#renamed-objects-and-access-locations)
- [New features](#new-features)
- [Notable deprecations](#notable-deprecations) 
- [Pastas Sustaining Membership Program](#pastas-sustaining-membership-program)

---

## New syntax for adding components to Models

The biggest change for users in Pastas 2.0 is that we are introducing a new method to
add components to your model, so for example to add a StressModel the new syntax is now:

```python
# Pastas 2.0
ml = ps.Model(oseries)
sm = ps.StressModel(ml, stress, rfunc=ps.Exponential(), name="rain")  # <-- Note the model is now the first argument!
```

Whereas in previous versions of Pastas, the the `ml.add_stressmodel()` was necessary:

```python
# Pastas <= 1.4
ml = ps.Model(oseries)
sm = ps.StressModel(stress, rfunc=ps.Exponential(), name="rain")
ml.add_stressmodel(sm)
```

The affected pastas objects are listed below, each of these classes now expects the
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

The reason is that in many cases, we want to use information from the Model class, e.g.
the head observations to set up certain things in an underlying class (e.g a
StressModel). A good example of this is `TarsoModel` which currently accepts the
`oseries` in order to give a good intitial estimate of parameters `d1` and `d2`. With
this new change `TarsoModel` directly has access to the model object, so it can add
itself, and retrieve information about the oseries to setup good guesses for the
initial parameters. Another example is that for the development of new solvers for
Pastas (inspired by PEST++) the solver classes need to access to the model in order to
set up the solve. We think implementing this change will make it easier for objects to
communicate with one another and simplify those developments in the future.

Some other smaller advantages of the new syntax are:

- shorter scripts (no more `ml.add_stressmodels()`, etc.)
- matches behavior in other packages frequently used by hydrologists, e.g. flopy:
`flopy.mf6.Modflowgwfwel(gwf_model, ...)` and timflow `timflow.stead.Well(model, ...)`

We are aware that syntax changes are annoying, and we did not make this change lightly.
There was quite a bit of debate, but in the end we decided the benefits outweighed the
downsides.

## Changes in defaults

The following changes to default behavior of Pastas are introduced in Pastas 2.0. These
changes mean that when relying on default behavior, models solved with Pastas 2.0 might
give different results than models solved with older versions of Pastas. We think these
changes will generally result in better models.

- The default arguments for `ps.solver.LeastSquares` have changed slightly. We now
default to a more accurate 3-point method to compute the Jacobian `jac="3-point"`. The
previous default was `jac="2-point"`. This change introduces a tiny extra computational
effort but the advantage of a more accurate estimate of the Jacobian can lead to better
performing models so is worth it in our opinion. You can always go back to 2-point if
you want.

- The parameter update in `ps.solver.LeastSquares` is now controlled by `x_scale="jac"`
which uses the Jacobian to determine the step change for each parameter during
optimization. Using the Jacobian gives better parameter updates, especially when
parameter scales are very different.

- The order of arguments in `WellModel`, `TarsoModel`, `StepModel`, `LinearTrend`,
`ChangeModel` was adjusted to more closely match the other StressModels. All
StressModels now adhere to the following order: stress/time input, rfunc, name, etc. Be
aware that old code that used positional arguments for these StressModels may be
affected and result in errors when run with Pastas 2.0.

- Adjustments to the parameter boundaries for `ps.rch.FlexModel`. **fill in here**

- The `HantushWellModel` response function used natural log scaling to avoid very small
values for parameter `b` in the optimization. This has now been updated to use log10
scaling (which makes more sense), which can be controlled with the `log_b=True|False`
keyword argument. The default is True, so this means any old pastas models with a
WellModel will yield different estimates of parameter `b` when solved with Pastas 2.0.
Take care when loading older models, be sure to solve them again with Pastas 2.0 prior
to doing any simulations.

- Parameter `_d` was renamed to `_A` in the `ps.One()` response function.

## Renamed objects and access locations

We put a lot of effort into Pastas 2.0 to make the software more consistent and
intuitive while maintaining ease of use. As a part of that effort, several classes were
renamed and some were moved. Of course all these changes come with warnings, so the old
methods will still work, but will generate warnings about future deprecations.

### Solvers

As part of a major internal refactor of the solver classes they were also renamed and
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
up- or downsample time series to other timesteps. These default setings were defined in
`ps.rcParams["timeseries"]` which contained dictionaries containing default settings
for precipitation (`"prec"`), evaporation (`"evap"`), waterlevels (`"waterlevel"`) and
wells (`"well"`), etc. The placement of these settings within Pastas made them very
hard to find, so it was moved to:

```python
ps.timeseries.settings  # <-- see all settings
prec_settings = ps.timeseries.settings["prec"]  # <-- get settings for "prec"
```

Hopefully that feels a lot more intuitive.

## New features

Among the many new features introduced by Pastas 2.0, these are some of the highlights:

- Use the impulse response instead of the block response to simulate heads. This can be
set by adding `use_block=False` in the response function definitions. This is faster,
since it avoids having to compute two step responses, and produces the comparable
results in our tests. Set with e.g. `ps.FourParam(use_block=False)`.

- Use `jac="cs"` in `ps.solver.LeastSquares()` to compute the Jacobian using a complex
step. This method is more accurate and more efficient than other jacobian estimation
routines despite the added computational burden of doing complex arithmetic. Pastas
internals were adjusted to support complex math in order to make this possible. If more
accurate Jacobians are your thing, try this out.

- The `tmax` estimates of the `Hantush` and `FourParam` response functions for a given
`cutoff` can now be computed more accurately using fast numerical integration. This is
slightly slower than the default conservative analytical approximations but can
be used to avoid very long response function tails that sometimes crop up in models.

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


## Pastas Sustaining Membership Program

We are introducing the [Pastas Sustaining Membership
Program](https://pastas.dev/about/support.html) to allow users to financially
contribute to the development and upkeep of Pastas. To be clear, __Pastas will always
remain open-source and free__! However, we would also like to note that a lot of the
maintenance has historically been carried out on a voluntary basis by the Pastas
development team. As our user base grows and the software becomes more capable this
maintenance requires additional resources. So if you or your company are frequent users
of Pastas software, we kindly ask you to consider supporting us financially. Community
backing will secure the future of Pastas and ensure that everyone can continue enjoying
this software.