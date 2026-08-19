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
# # Decision Trees and Random Forests in Python

# %% [markdown]
# This is the code for the lecture video which goes over tree methods in Python. Reference the video lecture for the full explanation of the code!
#
# I also wrote a [blog post](https://medium.com/@josemarcialportilla/enchanted-random-forest-b08d418cb411#.hh7n1co54) explaining the general logic of decision trees and random forests which you can check out. 
#
# ## Import Libraries

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

# %% [markdown]
# ## Get the Data

# %%
df = pd.read_csv('kyphosis.csv')

# %% jupyter={"outputs_hidden": false}
df.head()

# %% [markdown]
# ## EDA
#
# We'll just check out a simple pairplot for this small dataset.

# %% jupyter={"outputs_hidden": false}
sns.pairplot(df,hue='Kyphosis',palette='Set1')

# %% [markdown]
# ## Train Test Split
#
# Let's split up the data into a training set and a test set!

# %%
from sklearn.model_selection import train_test_split

# %%
X = df.drop('Kyphosis',axis=1)
y = df['Kyphosis']

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

# %% [markdown]
# ## Decision Trees
#
# We'll start just by training a single decision tree.

# %%
from sklearn.tree import DecisionTreeClassifier, plot_tree

# %%
dtree = DecisionTreeClassifier()

# %% jupyter={"outputs_hidden": false}
dtree.fit(X_train,y_train)

# %% [markdown]
# ## Prediction and Evaluation 
#
# Let's evaluate our decision tree.

# %%
predictions = dtree.predict(X_test)

# %%
from sklearn.metrics import classification_report,confusion_matrix

# %% jupyter={"outputs_hidden": false}
print(classification_report(y_test,predictions))

# %% jupyter={"outputs_hidden": false}
print(confusion_matrix(y_test,predictions))

# %% [markdown]
# ## Tree Visualization
#
# Scikit learn actually has some built-in visualization capabilities for decision trees, you won't use this often and it requires you to install the pydot library, but here is an example of what it looks like and the code to execute this:

# %% jupyter={"outputs_hidden": false}
features = list(df.columns[1:])
features

# %%
import matplotlib.pyplot as plt
# Plot the decision tree
plt.figure(figsize=(12, 8))
plot_tree(dtree, filled=True, feature_names=features)
plt.show()

# %% [markdown]
# ## Random Forests
#
# Now let's compare the decision tree model to a random forest.

# %% jupyter={"outputs_hidden": false}
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)

# %% jupyter={"outputs_hidden": false}
rfc_pred = rfc.predict(X_test)

# %% jupyter={"outputs_hidden": false}
print(confusion_matrix(y_test,rfc_pred))

# %% jupyter={"outputs_hidden": false}
print(classification_report(y_test,rfc_pred))

# %% [markdown]
# # Great Job!
