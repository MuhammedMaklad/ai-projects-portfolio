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
# # Groupby
#
# The groupby method allows you to group rows of data together and call aggregate functions

# %%
import pandas as pd
# Create dataframe
data = {'Company':['GOOG','GOOG','MSFT','MSFT','FB','FB'],
       'Person':['Sam','Charlie','Amy','Vanessa','Carl','Sarah'],
       'Sales':[200,120,340,124,243,350]}

# %%
df = pd.DataFrame(data)

# %%
df

# %% [markdown]
# ** Now you can use the .groupby() method to group rows together based off of a column name. For instance let's group based off of Company. This will create a DataFrameGroupBy object:**

# %%
df.groupby('Company')

# %% [markdown]
# You can save this object as a new variable:

# %%
by_comp = df.groupby("Company")

# %% [markdown]
# And then call aggregate methods off the object:

# %%
by_comp.mean()

# %%
df.groupby('Company').mean()

# %% [markdown]
# More examples of aggregate methods:

# %%
by_comp.std()

# %%
by_comp.min()

# %%
by_comp.max()

# %%
by_comp.count()

# %%
by_comp.describe()

# %%
by_comp.describe().transpose()

# %%
by_comp.describe().transpose()['GOOG']

# %% [markdown]
# # Great Job!
