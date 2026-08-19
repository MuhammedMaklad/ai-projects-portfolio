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
# # Matrix Plots
#
# Matrix plots allow you to plot data as color-encoded matrices and can also be used to indicate clusters within the data (later in the machine learning section we will learn how to formally cluster data).
#
# Let's begin by exploring seaborn's heatmap and clutermap:

# %%
import seaborn as sns
# %matplotlib inline

# %%
flights = sns.load_dataset('flights')

# %%
tips = sns.load_dataset('tips')

# %%
tips.head()

# %%
flights.head()

# %% [markdown]
# ## Heatmap
#
# In order for a heatmap to work properly, your data should already be in a matrix form, the sns.heatmap function basically just colors it in for you. For example:

# %%
tips.head()

# %%
# Matrix form for correlation data
tips.corr()

# %%
sns.heatmap(tips.corr())

# %%
sns.heatmap(tips.corr(),cmap='coolwarm',annot=True)

# %% [markdown]
# Or for the flights data:

# %%
flights.pivot_table(values='passengers',index='month',columns='year')

# %%
pvflights = flights.pivot_table(values='passengers',index='month',columns='year')
sns.heatmap(pvflights)

# %%
sns.heatmap(pvflights,cmap='magma',linecolor='white',linewidths=1)

# %% [markdown]
# ## clustermap
#
# The clustermap uses hierarchal clustering to produce a clustered version of the heatmap. For example:

# %%
sns.clustermap(pvflights)

# %% [markdown]
# Notice now how the years and months are no longer in order, instead they are grouped by similarity in value (passenger count). That means we can begin to infer things from this plot, such as August and July being similar (makes sense, since they are both summer travel months)

# %%
# More options to get the information a little clearer like normalization
sns.clustermap(pvflights,cmap='coolwarm',standard_scale=1)

# %% [markdown]
# # Great Job!
