# %%
import pastas as ps

# %%
data = ps.load_dataset("collenteur_2023")
# %%
ml = ps.Model(data["heads"]["Davos"].dropna())
rm = ps.RechargeModel(
    data["precipitation"]["Davos"].copy().rename("davos_rain"),
    data["evaporation"]["Davos"].copy().rename("davos_evap"),
    ps.Gamma(),
    temp=data["temperature"]["Davos"].copy().rename("davos_temp"),
    recharge=ps.rch.FlexModel(snow=True),
    name="rech",
)
sm = ps.StepModel("2010", "step", rfunc=ps.Exponential())
ml.add_stressmodel([rm, sm])
ml.add_noisemodel(ps.ArNoiseModel())
ml.solve()
# %%
ml.plots.results()
# %%
ml.plots.results_mosaic()
