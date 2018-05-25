"""This file contains an example of the use of the StepmModel stressmodel.

Author: R.A. Collenteur, Artesia B.V. 2018-05-25

"""


import pastas as ps
import pandas as pd

ml = ps.io.load("data\\step_model.pas")
s = ps.stressmodels.StepModel("2018-02-12", "step", rfunc=ps.rfunc.One)
ml.add_stressmodel(s)

ml.set_initial("noise_alpha", 1)

ml.solve()
