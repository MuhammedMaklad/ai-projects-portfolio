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
#     display_name: Python [conda env:base] *
#     language: python
#     name: conda-base-py
# ---

# %% [markdown]
# ___
#
# <a href='http://www.pieriandata.com'> <img src='../Pierian_Data_Logo.png' /></a>
# ___

# %% [markdown]
# # NumPy Exercises 
#
# Now that we've learned about NumPy let's test your knowledge. We'll start off with a few simple tasks, and then you'll be asked some more complicated questions.

# %% [markdown]
# #### Import NumPy as np

# %%
import numpy as np

# %% [markdown]
# #### Create an array of 10 zeros 

# %%
print(np.zeros(10))

# %% [markdown]
# #### Create an array of 10 ones

# %%
np.ones(10)

# %% [markdown]
# #### Create an array of 10 fives

# %%
np.array([ 5. for i in range(5)])

# %% [markdown]
# #### Create an array of the integers from 10 to 50

# %%
np.arange(10,51,dtype=int)

# %% [markdown]
# #### Create an array of all the even integers from 10 to 50

# %%
np.arange(10,51,2)

# %% [markdown]
# #### Create a 3x3 matrix with values ranging from 0 to 8

# %%
np.arange(0,9).reshape(3,3)

# %% [markdown]
# #### Create a 3x3 identity matrix

# %%
np.eye(3,3)

# %% [markdown]
# #### Use NumPy to generate a random number between 0 and 1

# %%
np.random.rand()

# %% [markdown]
# #### Use NumPy to generate an array of 25 random numbers sampled from a standard normal distribution

# %%
np.random.randn(25)

# %% [markdown]
# #### Create the following matrix:

# %%
np.linspace(.01,1.0,100).reshape(10,10)

# %% [markdown]
# #### Create an array of 20 linearly spaced points between 0 and 1:

 # %%
 np.linspace(0, 1, 20)

# %% [markdown]
# ## Numpy Indexing and Selection
#
# Now you will be given a few matrices, and be asked to replicate the resulting matrix outputs:

# %%
mat = np.arange(1,26).reshape(5,5)
print(mat)

# %%
# WRITE CODE HERE THAT REPRODUCES THE OUTPUT OF THE CELL BELOW
# BE CAREFUL NOT TO RUN THE CELL BELOW, OTHERWISE YOU WON'T
# BE ABLE TO SEE THE OUTPUT ANY MORE
mat[2:,1:]

# %%

# %%
# WRITE CODE HERE THAT REPRODUCES THE OUTPUT OF THE CELL BELOW
# BE CAREFUL NOT TO RUN THE CELL BELOW, OTHERWISE YOU WON'T
# BE ABLE TO SEE THE OUTPUT ANY MORE
print(mat[3,4])

# %%

# %%
# WRITE CODE HERE THAT REPRODUCES THE OUTPUT OF THE CELL BELOW
# BE CAREFUL NOT TO RUN THE CELL BELOW, OTHERWISE YOU WON'T
# BE ABLE TO SEE THE OUTPUT ANY MORE
print(mat[:3,1].reshape(-1,1))

# %%

# %%
# WRITE CODE HERE THAT REPRODUCES THE OUTPUT OF THE CELL BELOW
# BE CAREFUL NOT TO RUN THE CELL BELOW, OTHERWISE YOU WON'T
# BE ABLE TO SEE THE OUTPUT ANY MORE
mat[-1]

# %%

# %%
# WRITE CODE HERE THAT REPRODUCES THE OUTPUT OF THE CELL BELOW
# BE CAREFUL NOT TO RUN THE CELL BELOW, OTHERWISE YOU WON'T
# BE ABLE TO SEE THE OUTPUT ANY MORE
print(mat[3:,:])

# %%

# %% [markdown]
# ### Now do the following

# %% [markdown]
# #### Get the sum of all the values in mat

# %%
# 325
mat.sum()

# %% [markdown]
# #### Get the standard deviation of the values in mat

# %%
np.std(mat)

# %% [markdown]
# #### Get the sum of all the columns in mat

# %%
mat.sum(axis=0)
# array([55, 60, 65, 70, 75])

# %% [markdown]
# # Great Job!
