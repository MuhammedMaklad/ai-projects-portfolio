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
# # Series

# %% [markdown]
# The first main data type we will learn about for pandas is the Series data type. Let's import Pandas and explore the Series object.
#
# A Series is very similar to a NumPy array (in fact it is built on top of the NumPy array object). What differentiates the NumPy array from a Series, is that a Series can have axis labels, meaning it can be indexed by a label, instead of just a number location. It also doesn't need to hold numeric data, it can hold any arbitrary Python Object.
#
# Let's explore this concept through some examples:

# %%
import numpy as np
import pandas as pd

# %% [markdown]
# ### Creating a Series
#
# You can convert a list,numpy array, or dictionary to a Series:

# %%
labels = ['a','b','c']
my_list = [10,20,30]
arr = np.array([10,20,30])
d = {'a':10,'b':20,'c':30}

# %% [markdown]
# ** Using Lists**

# %%
pd.Series(data=my_list)

# %%
pd.Series(data=my_list,index=labels)

# %%
pd.Series(my_list,labels)

# %% [markdown]
# ** NumPy Arrays **

# %%
pd.Series(arr)

# %%
pd.Series(arr,labels)

# %% [markdown]
# ** Dictionary**

# %%
pd.Series(d)

# %% [markdown]
# ### Data in a Series
#
# A pandas Series can hold a variety of object types:

# %%
pd.Series(data=labels)

# %%
# Even functions (although unlikely that you will use this)
pd.Series([sum,print,len])

# %% [markdown]
# ## Using an Index
#
# The key to using a Series is understanding its index. Pandas makes use of these index names or numbers by allowing for fast look ups of information (works like a hash table or dictionary).
#
# Let's see some examples of how to grab information from a Series. Let us create two sereis, ser1 and ser2:

# %%
ser1 = pd.Series([1,2,3,4],index = ['USA', 'Germany','USSR', 'Japan'])                                   
ser2 = pd.Series([1,2,5,4],index = ['USA', 'Germany','Italy', 'Japan'])                                   

# %%
ser1  

# %%
ser2 = pd.Series([1,2,5,4],index = ['USA', 'Germany','Italy', 'Japan'])                                   

# %%
ser2

# %%
ser1['USA']

# %% [markdown]
# Operations are then also done based off of index:

# %%
ser1 + ser2

# %% [markdown]
# Let's stop here for now and move on to DataFrames, which will expand on the concept of Series!
# # Great Job!

# %%
