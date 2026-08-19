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
# # K Means Clustering Project 
#
# For this project we will attempt to use KMeans Clustering to cluster Universities into to two groups, Private and Public.
#
# ___
# It is **very important to note, we actually have the labels for this data set, but we will NOT use them for the KMeans clustering algorithm, since that is an unsupervised learning algorithm.** 
#
# When using the Kmeans algorithm under normal circumstances, it is because you don't have labels. In this case we will use the labels to try to get an idea of how well the algorithm performed, but you won't usually do this for Kmeans, so the classification report and confusion matrix at the end of this project, don't truly make sense in a real world setting!.
# ___
#
# ## The Data
#
# We will use a data frame with 777 observations on the following 18 variables.
# * Private A factor with levels No and Yes indicating private or public university
# * Apps Number of applications received
# * Accept Number of applications accepted
# * Enroll Number of new students enrolled
# * Top10perc Pct. new students from top 10% of H.S. class
# * Top25perc Pct. new students from top 25% of H.S. class
# * F.Undergrad Number of fulltime undergraduates
# * P.Undergrad Number of parttime undergraduates
# * Outstate Out-of-state tuition
# * Room.Board Room and board costs
# * Books Estimated book costs
# * Personal Estimated personal spending
# * PhD Pct. of faculty with Ph.D.’s
# * Terminal Pct. of faculty with terminal degree
# * S.F.Ratio Student/faculty ratio
# * perc.alumni Pct. alumni who donate
# * Expend Instructional expenditure per student
# * Grad.Rate Graduation rate

# %% [markdown]
# ## Import Libraries
#
# ** Import the libraries you usually use for data analysis.**

# %%

# %% [markdown]
# ## Get the Data

# %% [markdown]
# ** Read in the College_Data file using read_csv. Figure out how to set the first column as the index.**

# %%

# %% [markdown]
# **Check the head of the data**

# %%

# %% [markdown]
# ** Check the info() and describe() methods on the data.**

# %%

# %%

# %% [markdown]
# ## EDA
#
# It's time to create some data visualizations!
#
# ** Create a scatterplot of Grad.Rate versus Room.Board where the points are colored by the Private column. **

# %%

# %% [markdown]
# **Create a scatterplot of F.Undergrad versus Outstate where the points are colored by the Private column.**

# %%

# %% [markdown]
# ** Create a stacked histogram showing Out of State Tuition based on the Private column. Try doing this using [sns.FacetGrid](https://stanford.edu/~mwaskom/software/seaborn/generated/seaborn.FacetGrid.html). If that is too tricky, see if you can do it just by using two instances of pandas.plot(kind='hist'). **

# %%

# %% [markdown]
# **Create a similar histogram for the Grad.Rate column.**

# %%

# %% [markdown]
# ** Notice how there seems to be a private school with a graduation rate of higher than 100%.What is the name of that school?**

# %%

# %% [markdown]
# ** Set that school's graduation rate to 100 so it makes sense. You may get a warning not an error) when doing this operation, so use dataframe operations or just re-do the histogram visualization to make sure it actually went through.**

# %%

# %%

# %%

# %% [markdown]
# ## K Means Cluster Creation
#
# Now it is time to create the Cluster labels!
#
# ** Import KMeans from SciKit Learn.**

# %%

# %% [markdown]
# ** Create an instance of a K Means model with 2 clusters.**

# %%

# %% [markdown]
# **Fit the model to all the data except for the Private label.**

# %%

# %% [markdown]
# ** What are the cluster center vectors?**

# %%

# %% [markdown]
# ## Evaluation
#
# There is no perfect way to evaluate clustering if you don't have the labels, however since this is just an exercise, we do have the labels, so we take advantage of this to evaluate our clusters, keep in mind, you usually won't have this luxury in the real world.
#
# ** Create a new column for df called 'Cluster', which is a 1 for a Private school, and a 0 for a public school.**

# %%

# %%

# %%

# %% [markdown]
# ** Create a confusion matrix and classification report to see how well the Kmeans clustering worked without being given any labels.**

# %%

# %% [markdown]
# Not so bad considering the algorithm is purely using the features to cluster the universities into 2 distinct groups! Hopefully you can begin to see how K Means is useful for clustering un-labeled data!
#
# ## Great Job!
