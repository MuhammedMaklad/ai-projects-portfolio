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
# # Plotly and Cufflinks

# %% [markdown]
# Plotly is a library that allows you to create interactive plots that you can use in dashboards or websites (you can save them as html files or static images).
#
# ## Installation
#
# In order for this all to work, you'll need to install plotly and cufflinks to call plots directly off of a pandas dataframe. These libraries are not currently available through **conda** but are available through **pip**. Install the libraries at your command line/terminal using:
#
#     pip install plotly
#     pip install cufflinks
#
# ** NOTE: Make sure you only have one installation of Python on your computer when you do this, otherwise the installation may not work. **
#
# ## Imports and Set-up

# %%
import pandas as pd
import numpy as np
# %matplotlib inline

# %%
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

print(__version__) # requires version >= 1.9.0

# %%
import cufflinks as cf

# %%
# For Notebooks
init_notebook_mode(connected=True)

# %%
# For offline use
cf.go_offline()

# %% [markdown]
# ### Fake Data

# %%
df = pd.DataFrame(np.random.randn(100,4),columns='A B C D'.split())

# %%
df.head()

# %%
df2 = pd.DataFrame({'Category':['A','B','C'],'Values':[32,43,50]})

# %%
df2.head()

# %% [markdown]
# ## Using Cufflinks and iplot()
#
# * scatter
# * bar
# * box
# * spread
# * ratio
# * heatmap
# * surface
# * histogram
# * bubble

# %% [markdown]
# ## Scatter

# %%
df.iplot(kind='scatter',x='A',y='B',mode='markers',size=10)

# %% [markdown]
# ## Bar Plots

# %%
df2.iplot(kind='bar',x='Category',y='Values')

# %%
df.count().iplot(kind='bar')

# %% [markdown]
# ## Boxplots

# %%
df.iplot(kind='box')

# %% [markdown]
# ## 3d Surface

# %%
df3 = pd.DataFrame({'x':[1,2,3,4,5],'y':[10,20,30,20,10],'z':[5,4,3,2,1]})
df3.iplot(kind='surface',colorscale='rdylbu')

# %% [markdown]
# ## Spread

# %%
df[['A','B']].iplot(kind='spread')

# %% [markdown]
# ## histogram

# %%
df['A'].iplot(kind='hist',bins=25)

# %%
df.iplot(kind='bubble',x='A',y='B',size='C')

# %% [markdown]
# ## scatter_matrix()
#
# Similar to sns.pairplot()

# %%
df.scatter_matrix()

# %% [markdown]
# # Great Job!
