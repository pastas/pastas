import pastas as ps

# load a model with precipitation, evaporation and a well
ml = ps.io.load('data/B28H1808_2.pas')

# first solve and plot without a transform to see the bad model-performance
ml.solve(noise=False)
ax = ml.plots.decomposition(figsize=(10,6))
ax[0].legend()
ax[0].figure.tight_layout(pad=0.0)

# then solve and plot with a ThresholdTransform
ml.add_transform(ps.ThresholdTransform())
ml.solve(noise=False)
ax = ml.plots.decomposition(figsize=(10,6))
ax[0].legend()
ax[0].figure.tight_layout(pad=0.0)