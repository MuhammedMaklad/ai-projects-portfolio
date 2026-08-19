# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# ___
#
# <a href='http://www.pieriandata.com'> <img src='../Pierian_Data_Logo.png' /></a>
# ___

# %% [markdown]
# # Missing Data
#
# Let's show a few convenient methods to deal with Missing Data in pandas:

# %%
import numpy as np
import pandas as pd

# %%
df = pd.DataFrame({'A':[1,2,np.nan],
                  'B':[5,np.nan,np.nan],
                  'C':[1,2,3]})

# %%
df

# %%
df.dropna()

# %%
df.dropna(axis=1)

# %%
df.dropna(thresh=2) # notice that default axis = 0 mean row thresh mean if the row contain 2 or greater NaN remove its

# %%
df.fillna(value='FILL VALUE')

# %%
df['A'].fillna(value=df['A'].mean())

# %% [markdown]
# # Great Job!
