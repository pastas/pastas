#%%
import pastas as ps
import pandas as pd

from IPython.core.display import display, HTML

%load_ext autoreload
%autoreload 2
#%%
ho = pd.read_csv("../tests/data/obs.csv", index_col=0, parse_dates=True)
ml = ps.Model(ho)
# %%
display(HTML(ml._repr_html_()))
# %%
