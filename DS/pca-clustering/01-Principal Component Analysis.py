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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ___
#
# <a href='http://www.pieriandata.com'> <img src='../Pierian_Data_Logo.png' /></a>
# ___

# %% [markdown]
# # Principal Component Analysis
#
# Let's discuss PCA! Since this isn't exactly a full machine learning algorithm, but instead an unsupervised learning algorithm, we will just have a lecture on this topic, but no full machine learning project (although we will walk through the cancer set with PCA).
#
# ## PCA Review
#
# Make sure to watch the video lecture and theory presentation for a full overview of PCA! 
# Remember that PCA is just a transformation of your data and attempts to find out what features explain the most variance in your data. For example:

# %% [markdown]
# <img src='PCA.png' />

# %% [markdown]
# ## Libraries

# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
# %matplotlib inline

# %% [markdown]
# ## The Data
#
# Let's work with the cancer data set again since it had so many features.

# %%
from sklearn.datasets import load_breast_cancer

# %%
cancer = load_breast_cancer()

# %% jupyter={"outputs_hidden": false}
cancer.keys()

# %% jupyter={"outputs_hidden": false}
print(cancer['DESCR'])

# %%
df = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
#(['DESCR', 'data', 'feature_names', 'target_names', 'target'])

# %% jupyter={"outputs_hidden": false}
df.head()

# %% [markdown]
# ## PCA Visualization
#
# As we've noticed before it is difficult to visualize high dimensional data, we can use PCA to find the first two principal components, and visualize the data in this new, two-dimensional space, with a single scatter-plot. Before we do this though, we'll need to scale our data so that each feature has a single unit variance.

# %%
from sklearn.preprocessing import StandardScaler

# %% jupyter={"outputs_hidden": false}
scaler = StandardScaler()
scaler.fit(df)

# %%
scaled_data = scaler.transform(df)

# %% [markdown]
# PCA with Scikit Learn uses a very similar process to other preprocessing functions that come with SciKit Learn. We instantiate a PCA object, find the principal components using the fit method, then apply the rotation and dimensionality reduction by calling transform().
#
# We can also specify how many components we want to keep when creating the PCA object.

# %%
from sklearn.decomposition import PCA

# %%
pca = PCA(n_components=2)

# %% jupyter={"outputs_hidden": false}
pca.fit(scaled_data)

# %% [markdown]
# Now we can transform this data to its first 2 principal components.

# %%
x_pca = pca.transform(scaled_data)

# %% jupyter={"outputs_hidden": false}
scaled_data.shape

# %% jupyter={"outputs_hidden": false}
x_pca.shape

# %% [markdown]
# Great! We've reduced 30 dimensions to just 2! Let's plot these two dimensions out!

# %% jupyter={"outputs_hidden": false}
plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0],x_pca[:,1],c=cancer['target'],cmap='plasma')
plt.xlabel('First principal component')
plt.ylabel('Second Principal Component')

# %% [markdown]
# Clearly by using these two components we can easily separate these two classes.
#
# ## Interpreting the components 
#
# Unfortunately, with this great power of dimensionality reduction, comes the cost of being able to easily understand what these components represent.
#
# The components correspond to combinations of the original features, the components themselves are stored as an attribute of the fitted PCA object:

# %% jupyter={"outputs_hidden": false}
pca.components_

# %% [markdown]
# In this numpy matrix array, each row represents a principal component, and each column relates back to the original features. we can visualize this relationship with a heatmap:

# %%
df_comp = pd.DataFrame(pca.components_,columns=cancer['feature_names'])

# %% jupyter={"outputs_hidden": false}
plt.figure(figsize=(12,6))
sns.heatmap(df_comp,cmap='plasma',)

# %% [markdown]
# This heatmap and the color bar basically represent the correlation between the various feature and the principal component itself.
#
# ## Conclusion
#
# Hopefully this information is useful to you when dealing with high dimensional data!

# %% [markdown]
# # Great Job!
