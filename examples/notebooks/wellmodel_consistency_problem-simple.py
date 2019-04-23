#!/usr/bin/env python
# coding: utf-8

# # Testing Pastas WellModel: The consistency problem (simple)
# 
# Comparison between adding well normally (as StressModel with Hantush response) and adding the same well as a WellModel.
# 
# ## What is WellModel?
# The idea behind the WellModel is that the same response function determines the influence of multiple wells ( e.g. because they are located in the same aquifer) and the influence at the observation point is scaled only by the distance between the well and the observation point. The WellModel uses the Hantush response function which contains a parameter $\rho = \tfrac{r}{\lambda}$. Each well in the WellModel then has its own paramater $\rho_i = \tfrac{r_i}{\lambda}$ which scales the response with the distance to the observation point.
# 
# ## How is the scaling implemented?
# The scaling is implemented by passing an extra parameter onto the Hantush response function. The original three parameters are:
# 1. gain
# 2. $\rho$
# 3. $cS$
# 
# A fourth parameter is added which contains the distance to the observation point.
# 4. $r$
# 
# The assumption then is that the parameter passed to the LeastSquares solver is: $\rho' = \tfrac{1}{\lambda}$ (the distance is set to 1 m). In `Hantush.step` this parameter $\rho'$ is multiplied by $r$ to obtain the original $\rho$. The initial, minimum and maximum parameters values of $\rho'$ are scaled accordingly by dividing the original guess by the distance to the well. (For several wells the maximum distance is used, but we're only looking at one well in this notebook). The scaling is currently performed manually.
# 
# ## What is the problem?
# An issue occurs where subsequent runs of seemingly identical models yield different fit results when using a WellModel. The calculated EVP is sometimes similar to the Normal Model (model with normal StressModel) and sometimes much lower. The possible EVP outcomes do seem limited to a limited set of values that are obtained seemingly at random.

# ### Import packages
# Using wellmodel branch of pastas!
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(1, "C:/GitHub/pastas")  # edit for your PC/Mac!
import pastas as ps
import scipy
print("pastas is installed in {}".format(os.path.dirname(ps.__file__)))

print("Python version:", sys.version_info)
print("Pandas version:", pd.__version__)
print("Numpy version:", np.__version__)
print("Scipy version:", scipy.__version__)
print("Pastas version:", ps.__version__)

N = 50  # no. of models to build and run one after another

# ## Load data
oseries = pd.read_csv(r"./data_notebook_wellmodel/B44E0126_10.csv", parse_dates=True, index_col=[0])
drongelen = pd.read_csv(r"./data_notebook_wellmodel/drongelen.csv", parse_dates=True, index_col=[0])
prec = pd.read_csv(r"./data_notebook_wellmodel/RD Capelle (Nb).csv", parse_dates=True, index_col=[0])
evap = pd.read_csv(r"./data_notebook_wellmodel/EV24 Gilze-Rijen.csv", parse_dates=True, index_col=[0])


# If I (David Brakenhoff) comment out this next cell, I get the old behavior, where I randomly get either an EVP of about 75% or about 30%. If slice the timeseries prior to doing all the Pastas stuff, so far I get a consistent fit...
# tmin = "1987-09-03"  # this is the tmin pastas determines if nothing is passed
# tmax = "2018-10-04"  # this is the tmax pastas determines if nothing is passed

# tmin = "2008"
# tmax = "2018"

# oseries = oseries.loc[tmin:tmax]
# drongelen = drongelen.loc[tmin:tmax]
# prec = prec.loc[tmin:tmax]
# evap = evap.loc[tmin:tmax]

# Convert to pastas.TimeSeries
# ps.TimeSeries._predefined_settings.keys()
oseries = ps.TimeSeries(oseries, name="B44E0126_10", settings="oseries")

drongelen = ps.TimeSeries(drongelen, name="Drongelen", settings="well")

prec = ps.TimeSeries(prec, name="RD Capelle", settings="prec")
evap = ps.TimeSeries(evap, name="EV24 Gilze-Rijen", settings="evap")

welldistances = np.array([48.08856265])  # in order listed above

# ## Using WellModel with one well with recharge
results3 = pd.DataFrame()
distances = welldistances[0:1]

ml2 = ps.Model(oseries, name="B44E0126_10", log_level="ERROR")

rm = ps.StressModel2([prec, evap], ps.Gamma, name="recharge")
ml2.add_stressmodel(rm)

wm = ps.stressmodels.WellModel([drongelen], ps.Hantush, name="DrongelenWM", 
                               distances=distances, up=False, settings=[drongelen.settings])
ml2.add_stressmodel(wm)

# Reset initial paramter value to be scaled by the maximum distances in the well Model
ml2.set_parameter(wm.name + "_rho", 1./np.max(distances), "initial")  # this works
ml2.set_parameter(wm.name + "_rho", 1e-4/np.max(distances), "pmin")  # this works
ml2.set_parameter(wm.name + "_rho", 10./np.max(distances), "pmax")  # this works

for i in range(N):    
    ml2.solve(freq="14D", report=False, noise=True, method="trf")
    dt = pd.datetime.now()
    results3.loc[dt, "evp"] = ml2.stats.evp()
    results3.loc[dt, "sum_sim"] = ml2.simulate().sum()
    print(".", end="", flush=True)


fig, ax = plt.subplots(1, 1, figsize=(12, 6))
ax.plot(range(1, N+1), results3.loc[:, "evp"].dropna().values, 
        label="EVP WellModel (with recharge)", marker="*")
ax.grid(b=True)
ax.set_ylabel("EVP (%)")
ax.set_xticks(range(1, N+1))
ax.set_xlabel("Run no.")
ax.legend(loc="best")
ax.set_ylim(bottom=0.0, top=100.)
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(12, 6))
ax.plot(range(1, N+1), results3.loc[:, "sum_sim"].dropna().values, 
        label="Sum of simulate", marker="*")
ax.grid(b=True)
ax.set_ylabel("Sum of simulate (m)")
ax.set_xticks(range(1, N+1))
ax.set_xlabel("Run no.")
ax.legend(loc="best")
plt.show()
