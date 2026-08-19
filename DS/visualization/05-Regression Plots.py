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
# # Regression Plots
#
# Seaborn has many built-in capabilities for regression plots, however we won't really discuss regression until the machine learning section of the course, so we will only cover the **lmplot()** function for now.
#
# **lmplot** allows you to display linear models, but it also conveniently allows you to split up those plots based off of features, as well as coloring the hue based off of features.
#
# Let's explore how this works:

# %%
import seaborn as sns
# %matplotlib inline

# %%
tips = sns.load_dataset('tips')

# %%
tips.head()

# %% [markdown]
# ## lmplot()

# %%
sns.lmplot(x='total_bill',y='tip',data=tips)

# %%
sns.lmplot(x='total_bill',y='tip',data=tips,hue='sex')

# %%
sns.lmplot(x='total_bill',y='tip',data=tips,hue='sex',palette='coolwarm')

# %% [markdown]
# ### Working with Markers
#
# lmplot kwargs get passed through to **regplot** which is a more general form of lmplot(). regplot has a scatter_kws parameter that gets passed to plt.scatter. So you want to set the s parameter in that dictionary, which corresponds (a bit confusingly) to the squared markersize. In other words you end up passing a dictionary with the base matplotlib arguments, in this case, s for size of a scatter plot. In general, you probably won't remember this off the top of your head, but instead reference the documentation.

# %%
# http://matplotlib.org/api/markers_api.html
sns.lmplot(x='total_bill',y='tip',data=tips,hue='sex',palette='coolwarm',
           markers=['o','v'],scatter_kws={'s':100})

# %% [markdown]
# ## Using a Grid
#
# We can add more variable separation through columns and rows with the use of a grid. Just indicate this with the col or row arguments:

# %%
sns.lmplot(x='total_bill',y='tip',data=tips,col='sex')

# %%
sns.lmplot(x="total_bill", y="tip", row="sex", col="time",data=tips)

# %%
sns.lmplot(x='total_bill',y='tip',data=tips,col='day',hue='sex',palette='coolwarm')

# %% [markdown]
# ## Aspect and Size
#
# Seaborn figures can have their size and aspect ratio adjusted with the **size** and **aspect** parameters:

# %%
sns.lmplot(x='total_bill',y='tip',data=tips,col='day',hue='sex',palette='coolwarm',
          aspect=0.6,size=8)

# %% [markdown]
# You're probably wondering how to change the font size or control the aesthetics even more, check out the Style and Color Lecture and Notebook for more info on that!
#
# # Great Job!
