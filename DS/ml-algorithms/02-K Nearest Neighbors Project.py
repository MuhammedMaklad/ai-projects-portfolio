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
# # K Nearest Neighbors Project 
#
# Welcome to the KNN Project! This will be a simple project very similar to the lecture, except you'll be given another data set. Go ahead and just follow the directions below.
# ## Import Libraries
# **Import pandas,seaborn, and the usual libraries.**

# %% jupyter={"outputs_hidden": true}

# %% [markdown]
# ## Get the Data
# ** Read the 'KNN_Project_Data csv file into a dataframe **

# %% jupyter={"outputs_hidden": true}

# %% [markdown]
# **Check the head of the dataframe.**

# %% jupyter={"outputs_hidden": false}

# %% [markdown]
# # EDA
#
# Since this data is artificial, we'll just do a large pairplot with seaborn.
#
# **Use seaborn on the dataframe to create a pairplot with the hue indicated by the TARGET CLASS column.**

# %% jupyter={"outputs_hidden": false}

# %% [markdown]
# # Standardize the Variables
#
# Time to standardize the variables.
#
# ** Import StandardScaler from Scikit learn.**

# %% jupyter={"outputs_hidden": true}

# %% [markdown]
# ** Create a StandardScaler() object called scaler.**

# %% jupyter={"outputs_hidden": true}

# %% [markdown]
# ** Fit scaler to the features.**

# %% jupyter={"outputs_hidden": false}

# %% [markdown]
# **Use the .transform() method to transform the features to a scaled version.**

# %% jupyter={"outputs_hidden": true}

# %% [markdown]
# **Convert the scaled features to a dataframe and check the head of this dataframe to make sure the scaling worked.**

# %% jupyter={"outputs_hidden": false}

# %% [markdown]
# # Train Test Split
#
# **Use train_test_split to split your data into a training set and a testing set.**

# %% jupyter={"outputs_hidden": true}

# %% jupyter={"outputs_hidden": true}

# %% [markdown]
# # Using KNN
#
# **Import KNeighborsClassifier from scikit learn.**

# %% jupyter={"outputs_hidden": true}

# %% [markdown]
# **Create a KNN model instance with n_neighbors=1**

# %% jupyter={"outputs_hidden": true}

# %% [markdown]
# **Fit this KNN model to the training data.**

# %% jupyter={"outputs_hidden": false}

# %% [markdown]
# # Predictions and Evaluations
# Let's evaluate our KNN model!

# %% [markdown]
# **Use the predict method to predict values using your KNN model and X_test.**

# %% jupyter={"outputs_hidden": false}

# %% [markdown]
# ** Create a confusion matrix and classification report.**

# %% jupyter={"outputs_hidden": true}

# %% jupyter={"outputs_hidden": false}

# %% jupyter={"outputs_hidden": false}

# %% [markdown]
# # Choosing a K Value
# Let's go ahead and use the elbow method to pick a good K Value!
#
# ** Create a for loop that trains various KNN models with different k values, then keep track of the error_rate for each of these models with a list. Refer to the lecture if you are confused on this step.**

# %% jupyter={"outputs_hidden": true}

# %% [markdown]
# **Now create the following plot using the information from your for loop.**

# %% jupyter={"outputs_hidden": false}

# %% [markdown]
# ## Retrain with new K Value
#
# **Retrain your model with the best K value (up to you to decide what you want) and re-do the classification report and the confusion matrix.**

# %% jupyter={"outputs_hidden": false}

# %% [markdown]
# # Great Job!
