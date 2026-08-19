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

# %% [markdown] id="J8yT2y73EM4F"
# <a class="anchor" id="0"></a>
# # **K-Means Clustering with Python**
#
# Hello friends,
#
# **K-Means clustering** is the most popular unsupervised machine learning algorithm. K-Means clustering is used to find intrinsic groups within the unlabelled dataset and draw inferences from them. In this kernel, I implement K-Means clustering to find intrinsic groups within the dataset that display the same `status_type` behaviour. The `status_type` behaviour variable consists of posts of a different nature (video, photos, statuses and links).
#
#
# So, let's get started.

# %% [markdown] id="BgelEloIEM4I"
# <a class="anchor" id="0.1"></a>
# # **Table of Contents**
#
#
# 1.	[Introduction to K-Means Clustering](#1)
# 1.  [Applications of clustering](#2)
# 1.	[K-Means Clustering intuition](#3)
# 1.	[Choosing the value of K](#4)
# 1.	[The elbow method](#5)
# 1.  [Import libraries](#6)
# 1.	[Import dataset](#7)
# 1.	[Exploratory data analysis](#8)
# 1.	[Declare feature vector and target variable](#9)
# 1.	[Convert categorical variable into integers](#10)
# 1.	[Feature scaling](#11)
# 1.	[K-Means model with two clusters](#12)
# 1.	[K-Means model parameters study](#13)
# 1.	[Check quality of weak classification by the model](#14)
# 1.	[Use elbow method to find optimal number of clusters](#15)
# 1.	[K-Means model with different clusters](#16)
# 1.	[Results and conclusion](#17)
# 1.  [References](#18)
#

# %% [markdown] id="lQzo2F64EM4I"
# # **1. Introduction to K-Means Clustering** <a class="anchor" id="1"></a>
#
# [Table of Contents](#0.1)
#
#
# Machine learning algorithms can be broadly classified into two categories - supervised and unsupervised learning. There are other categories also like semi-supervised learning and reinforcement learning. But, most of the algorithms are classified as supervised or unsupervised learning. The difference between them happens because of presence of target variable. In unsupervised learning, there is no target variable. The dataset only has input variables which describe the data. This is called unsupervised learning.
#
# **K-Means clustering** is the most popular unsupervised learning algorithm. It is used when we have unlabelled data which is data without defined categories or groups. The algorithm follows an easy or simple way to classify a given data set through a certain number of clusters, fixed apriori. K-Means algorithm works iteratively to assign each data point to one of K groups based on the features that are provided. Data points are clustered based on feature similarity.
#
# K-Means clustering can be represented diagrammatically as follows:-
#

# %% [markdown] id="DZlgxQGOEM4J"
# ## K-Means
#
# ![K-Means](https://miro.medium.com/max/2160/1*tWaaZX75oumVwBMcKN-eHA.png)

# %% [markdown] id="J_ghCfzGEM4J"
# # **2. Applications of clustering** <a class="anchor" id="2"></a>
#
# [Table of Contents](#0.1)
#
#
# - K-Means clustering is the most common unsupervised machine learning algorithm. It is widely used for many applications which include-
#
#   1. Image segmentation
#
#   2. Customer segmentation
#
#   3. Species clustering
#
#   4. Anomaly detection
#
#   5. Clustering languages

# %% [markdown] id="b1BkEJlYEM4J"
# # **3. K-Means Clustering intuition** <a class="anchor" id="3"></a>
#
# [Table of Contents](#0.1)
#
#
# K-Means clustering is used to find intrinsic groups within the unlabelled dataset and draw inferences from them. It is based on centroid-based clustering.
#
#
# **Centroid** - A centroid is a data point at the centre of a cluster. In centroid-based clustering, clusters are represented by a centroid. It is an iterative algorithm in which the notion of similarity is derived by how close a data point is to the centroid of the cluster.
# K-Means clustering works as follows:-
# The K-Means clustering algorithm uses an iterative procedure to deliver a final result. The algorithm requires number of clusters K and the data set as input. The data set is a collection of features for each data point. The algorithm starts with initial estimates for the K centroids. The algorithm then iterates between two steps:-
#
#
# ## **3.1 Data assignment step**
#
#
# Each centroid defines one of the clusters. In this step, each data point is assigned to its nearest centroid, which is based on the squared Euclidean distance. So, if ci is the collection of centroids in set C, then each data point is assigned to a cluster based on minimum Euclidean distance.
#
#
#
# ## **3.2 Centroid update step**
#
#
# In this step, the centroids are recomputed and updated. This is done by taking the mean of all data points assigned to that centroid’s cluster.
#
#
# The algorithm then iterates between step 1 and step 2 until a stopping criteria is met. Stopping criteria means no data points change the clusters, the sum of the distances is minimized or some maximum number of iterations is reached.
# This algorithm is guaranteed to converge to a result. The result may be a local optimum meaning that assessing more than one run of the algorithm with randomized starting centroids may give a better outcome.
#
# The K-Means intuition can be represented with the help of following diagram:-
#

# %% [markdown] id="6PNRGLBnEM4J"
# ## K-Means intuition
# ![K-Means intuition](https://i.ytimg.com/vi/_aWzGGNrcic/hqdefault.jpg)

# %% [markdown] id="d2Z-htzQEM4K"
# # **4. Choosing the value of K** <a class="anchor" id="4"></a>
#
# [Table of Contents](#0.1)
#
#
# The K-Means algorithm depends upon finding the number of clusters and data labels for a pre-defined value of K. To find the number of clusters in the data, we need to run the K-Means clustering algorithm for different values of K and compare the results. So, the performance of K-Means algorithm depends upon the value of K. We should choose the optimal value of K that gives us best performance. There are different techniques available to find the optimal value of K. The most common technique is the **elbow method** which is described below.
#

# %% [markdown] id="Sz4t2JcLEM4K"
# # **5. The elbow method** <a class="anchor" id="5"></a>
#
# [Table of Contents](#0.1)
#
#
# The elbow method is used to determine the optimal number of clusters in K-means clustering. The elbow method plots the value of the cost function produced by different values of K. The below diagram shows how the elbow method works:-

# %% [markdown] id="ADtIYtu4EM4K"
# ## The elbow method
#
# ![Elbow method in K-Means](https://www.oreilly.com/library/view/statistics-for-machine/9781788295758/assets/995b8b58-06f1-4884-a2a1-f3648428e947.png)

# %% [markdown] id="MVytlClWEM4K"
# We can see that if K increases, average distortion will decrease.  Then each cluster will have fewer constituent instances, and the instances will be closer to their respective centroids. However, the improvements in average distortion will decline as K increases. The value of K at which improvement in distortion declines the most is called the elbow, at which we should stop dividing the data into further clusters.
#

# %% [markdown] id="ocmgGhltEM4K"
# # **6. Import libraries** <a class="anchor" id="6"></a>
#
# [Table of Contents](#0.1)
#

# %% executionInfo={"elapsed": 2150, "status": "ok", "timestamp": 1725951697752, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="qGANPr0YEM4K"
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for data visualization
import seaborn as sns # for statistical data visualization
# %matplotlib inline


# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 20418, "status": "ok", "timestamp": 1725951718166, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="hvNqBwr_ETL9" outputId="af588e6e-63c6-47b3-b271-f63c4b7ed208"
from google.colab import drive
drive.mount('/content/drive')

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 853, "status": "ok", "timestamp": 1725951719013, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="g89G05zsEVHC" outputId="5f117a11-df17-40fe-8aeb-cb9e8beb6d60"
# cd /content/drive/MyDrive/Machine learning 2024/Notebooks - Machine Learning/kmeans

# %% [markdown] id="kHXj6x5NEM4L"
# ### Ignore warnings
#

# %% executionInfo={"elapsed": 6, "status": "ok", "timestamp": 1725951719013, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="_QfiNIVaEM4L"
import warnings

warnings.filterwarnings('ignore')

# %% [markdown] id="lIvMUirnEM4M"
# # **7. Import dataset** <a class="anchor" id="7"></a>
#
# [Table of Contents](#0.1)
#
#

# %% executionInfo={"elapsed": 1471, "status": "ok", "timestamp": 1725951720478, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="PyuogSY7EM4M"
df = pd.read_csv("Live.csv")

# %% [markdown] id="LDJSZV-wEM4M"
# # **8. Exploratory data analysis** <a class="anchor" id="8"></a>
#
# [Table of Contents](#0.1)
#

# %% [markdown] id="f-0H_KO8EM4M"
# ### Check shape of the dataset

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 6, "status": "ok", "timestamp": 1725951720479, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="5ubo6oeWEM4M" outputId="4f50c1ee-9de6-41d7-a8bd-b88d983d37b7"
df.shape

# %% [markdown] id="rWCn9f9iEM4M"
# We can see that there are 7050 instances and 16 attributes in the dataset. In the dataset description, it is given that there are 7051 instances and 12 attributes in the dataset.
#
# So, we can infer that the first instance is the row header and there are 4 extra attributes in the dataset. Next, we should take a look at the dataset to gain more insight about it.

# %% [markdown] id="lxvl894QEM4M"
# ### Preview the dataset

# %% colab={"base_uri": "https://localhost:8080/", "height": 696} executionInfo={"elapsed": 338, "status": "ok", "timestamp": 1725951731883, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="RC42KHo8EM4M" outputId="0f4afc96-fcf0-4916-c740-fc1876acc3b6"
df.head(20)

# %% [markdown] id="Zt0BGbeAEM4M"
# ### View summary of dataset

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 328, "status": "ok", "timestamp": 1725951768645, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="9e8SET-rEM4M" outputId="e117ba81-c5ea-42bc-e4f3-93c2e7374bee"
df.info()

# %% [markdown] id="rM0d8FxeEM4M"
# ### Check for missing values in dataset

# %% colab={"base_uri": "https://localhost:8080/", "height": 585} executionInfo={"elapsed": 332, "status": "ok", "timestamp": 1725951772559, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="Kxt71GK9EM4M" outputId="9eea2a33-5c9e-4ae6-a7cb-011bce416520"
df.isnull().sum()

# %% [markdown] id="soLzkpLcEM4N"
# We can see that there are 4 redundant columns in the dataset. We should drop them before proceeding further.

# %% [markdown] id="Xl6_HnGQEM4N"
# ### Drop redundant columns

# %% executionInfo={"elapsed": 295, "status": "ok", "timestamp": 1725951789697, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="AdfcUzO0EM4N"
df.drop(['Column1', 'Column2', 'Column3', 'Column4'], axis=1, inplace=True)

# %% [markdown] id="GyMZD2YjEM4N"
# ### Again view summary of dataset

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 279, "status": "ok", "timestamp": 1725951792415, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="dDKfE1XFEM4N" outputId="cb820c1b-1287-48b4-eae5-c0fbb098b0cb"
df.info()

# %% [markdown] id="vpfJHpF-EM4N"
# Now, we can see that redundant columns have been removed from the dataset.
#
# We can see that, there are 3 character variables (data type = object) and remaining 9 numerical variables (data type = int64).
#

# %% [markdown] id="YHAoOjelEM4N"
# ### View the statistical summary of numerical variables

# %% colab={"base_uri": "https://localhost:8080/", "height": 320} executionInfo={"elapsed": 338, "status": "ok", "timestamp": 1725951805535, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="IB2VMmdeEM4N" outputId="6a3ae422-12c3-4553-8e01-66220fb0dd83"
df.describe()

# %% [markdown] id="r4Cyo-yQEM4N"
# There are 3 categorical variables in the dataset. I will explore them one by one.

# %% [markdown] id="Luf111mTEM4N"
# ### Explore `status_id` variable

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 301, "status": "ok", "timestamp": 1725951828240, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="tDGQs4pxEM4N" outputId="40d9cf53-96fc-4644-b16a-a3dc217fea5c"
# view the labels in the variable

df['status_id'].unique()

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 350, "status": "ok", "timestamp": 1725951829700, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="xF541EqJEM4R" outputId="68a0a62d-6942-4787-9d19-0f85e72b9ae0"
# view how many different types of variables are there

len(df['status_id'].unique())

# %% [markdown] id="sTTKWt8oEM4R"
# We can see that there are 6997 unique labels in the `status_id` variable. The total number of instances in the dataset is 7050. So, it is approximately a unique identifier for each of the instances. Thus this is not a variable that we can use. Hence, I will drop it.

# %% [markdown] id="SwJzYTuIEM4R"
# ### Explore `status_published` variable

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 322, "status": "ok", "timestamp": 1725951839408, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="9ewtKUo4EM4R" outputId="4b795b3c-4501-4442-b167-772e57098e6c"
# view the labels in the variable

df['status_published'].unique()

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 344, "status": "ok", "timestamp": 1725951841168, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="s-pftocFEM4R" outputId="3ff46318-8424-4077-ea4e-0f9a7b29c037"
# view how many different types of variables are there

len(df['status_published'].unique())

# %% [markdown] id="j6FWc7cKEM4S"
# Again, we can see that there are 6913 unique labels in the `status_published` variable. The total number of instances in the dataset is 7050. So, it is also a approximately a unique identifier for each of the instances. Thus this is not a variable that we can use. Hence, I will drop it also.

# %% [markdown] id="QxcVqLPNEM4S"
# ### Explore `status_type` variable

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 331, "status": "ok", "timestamp": 1725951853499, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="_LM3moMpEM4S" outputId="faa3b1b9-7777-48ae-e3f2-ea49aefe0594"
# view the labels in the variable

df['status_type'].unique()

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 287, "status": "ok", "timestamp": 1725951860998, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="VxtTL6NhEM4S" outputId="9c2bba8b-c0d5-4807-98be-ef5ac65e529e"
# view how many different types of variables are there

len(df['status_type'].unique())

# %% [markdown] id="7BmXuK7mEM4S"
# We can see that there are 4 categories of labels in the `status_type` variable.

# %% [markdown] id="MqXREzFwEM4S"
# ### Drop `status_id` and `status_published` variable from the dataset

# %% executionInfo={"elapsed": 295, "status": "ok", "timestamp": 1725951877775, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="yE8gXXRcEM4S"
df.drop(['status_id', 'status_published'], axis=1, inplace=True)

# %% [markdown] id="UqpVI-FJEM4S"
# ### View the summary of dataset again

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 334, "status": "ok", "timestamp": 1725951880890, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="75-I3-vCEM4S" outputId="7e4875e9-8235-47e0-9f7b-784d524ed251"
df.info()

# %% [markdown] id="1MedG9zrEM4S"
# ### Preview the dataset again

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} executionInfo={"elapsed": 305, "status": "ok", "timestamp": 1725951891621, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="JfbOZZIiEM4T" outputId="0efd105d-ba40-4abf-e3ea-a05ad06d02fa"
df.head()

# %% [markdown] id="dtt3G0bhEM4T"
# We can see that there is 1 non-numeric column `status_type` in the dataset. I will convert it into integer equivalents.

# %% [markdown] id="uYhmLlmvEM4T"
# # **9. Declare feature vector and target variable** <a class="anchor" id="9"></a>
#
# [Table of Contents](#0.1)
#

# %% executionInfo={"elapsed": 314, "status": "ok", "timestamp": 1725951962971, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="KPANbVRrEM4T"
X = df

y = df['status_type']

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} executionInfo={"elapsed": 335, "status": "ok", "timestamp": 1725951964543, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="4UJLp28hCdm-" outputId="0ec5b535-2532-4b8a-c421-225df9b6e06c"
X[:5]

# %% [markdown] id="mylZRhQKEM4T"
# # **10. Convert categorical variable into integers** <a class="anchor" id="10"></a>
#
# [Table of Contents](#0.1)
#

# %% executionInfo={"elapsed": 323, "status": "ok", "timestamp": 1725952014256, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="FrekShRfEM4T"
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

X['status_type'] = le.fit_transform(X['status_type'])

y = le.transform(y)

# %% [markdown] id="HDPxwzmiEM4T"
# ### View the summary of X

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 633, "status": "ok", "timestamp": 1725952018038, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="E7aXs4SWEM4T" outputId="ea36a3c4-994a-4af6-d4c8-28e237663f83"
X.info()

# %% [markdown] id="QzFy3ryrEM4T"
# ### Preview the dataset X

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} executionInfo={"elapsed": 336, "status": "ok", "timestamp": 1725952023536, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="649hS6TiEM4T" outputId="6cb0f54d-fe5a-4d8a-cbd3-5d6f374dabc2"
X.head()

# %% [markdown] id="VHNzqU1dEM4U"
# # **11. Feature Scaling** <a class="anchor" id="11"></a>
#
# [Table of Contents](#0.1)

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 316, "status": "ok", "timestamp": 1725952030049, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="hGKQGKYOEM4U" outputId="d0f435c4-3d6f-4d90-d48d-22e3989d3878"
cols = X.columns
print(cols)

# %% executionInfo={"elapsed": 305, "status": "ok", "timestamp": 1725952075294, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="Tt69TTuMEM4U"
from sklearn.preprocessing import MinMaxScaler

ms = MinMaxScaler()

X = ms.fit_transform(X)   # convert Dataframe to array

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 329, "status": "ok", "timestamp": 1725952085246, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="jn6j0WhivXB2" outputId="ef7d4b91-6891-4b6b-f3b3-65b39da4ebec"
X[:2]

# %% executionInfo={"elapsed": 317, "status": "ok", "timestamp": 1725952108689, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="QvEOtGtvEM4U"
X = pd.DataFrame(X, columns=[cols])

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} executionInfo={"elapsed": 11, "status": "ok", "timestamp": 1725952109965, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="vpvY4JTdEM4U" outputId="9f89763c-46b7-45f3-8f62-53465994be44"
X.head()

# %% [markdown] id="FIrr39P1EM4U"
# # **12. K-Means model with two clusters** <a class="anchor" id="12"></a>
#
# [Table of Contents](#0.1)

# %% colab={"base_uri": "https://localhost:8080/", "height": 74} executionInfo={"elapsed": 924, "status": "ok", "timestamp": 1725952266014, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="SCQLgOQWEM4U" outputId="a784a3e4-699b-4331-feb9-bb8d71bc7467"
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=0)   #n_clusters = K

kmeans.fit(X)

# %% [markdown] id="Ksv73bz6EM4U"
# # **13. K-Means model parameters study** <a class="anchor" id="13"></a>
#
# [Table of Contents](#0.1)

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 368, "status": "ok", "timestamp": 1725952276723, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="Slfi-XJWEM4U" outputId="d2aa13ee-42cc-4634-80b3-2e7229d75466"
kmeans.cluster_centers_

# %% [markdown] id="8vqFOCCNEM4U"
# - The KMeans algorithm clusters data by trying to separate samples in n groups of equal variances, minimizing a criterion known as **inertia**, or within-cluster sum-of-squares Inertia, or the within-cluster sum of squares criterion, can be recognized as a measure of how internally coherent clusters are.
#
#
# - The k-means algorithm divides a set of N samples X into K disjoint clusters C, each described by the mean j of the samples in the cluster. The means are commonly called the cluster **centroids**.
#
#
# - The K-means algorithm aims to choose centroids that minimize the inertia, or within-cluster sum of squared criterion.

# %% [markdown] id="EREkZeWrEM4V"
# ### Inertia
#
#
# - **Inertia** is not a normalized metric.
#
# - The lower values of inertia are better and zero is optimal.
#
# - But in very high-dimensional spaces, euclidean distances tend to become inflated (this is an instance of `curse of dimensionality`).
#
# - Running a dimensionality reduction algorithm such as PCA prior to k-means clustering can alleviate this problem and speed up the computations.
#
# - We can calculate model inertia as follows:-

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 305, "status": "ok", "timestamp": 1725952291506, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="rbCmLGTBEM4V" outputId="b124f588-47e6-4419-c1a9-eb59d7baa2f2"
kmeans.inertia_

# %% [markdown] id="a2G-9Cg3EM4V"
# - The lesser the model inertia, the better the model fit.
#
# - We can see that the model has very high inertia. So, this is not a good model fit to the data.

# %% [markdown] id="I_mwg1rDEM4V"
#  # **14. Check quality of weak classification by the model** <a class="anchor" id="14"></a>
#
# [Table of Contents](#0.1)

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 333, "status": "ok", "timestamp": 1725952441456, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="9X1edS_lEM4V" outputId="ea0ac91c-3c5d-4baa-983e-8b2491864c89"
labels = kmeans.labels_
print(labels[:10])

# %% colab={"base_uri": "https://localhost:8080/", "height": 363} executionInfo={"elapsed": 305, "status": "ok", "timestamp": 1725952448879, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="hlHvJLt8yEBl" outputId="b1d0d07d-26a5-4d7a-94c7-19a5edec5b44"
X[:10]

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1542, "status": "ok", "timestamp": 1725952460100, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="1N689l9bSU8Z" outputId="77d0913b-d8e2-470c-cbe6-aee5d9abe120"
from sklearn.metrics import silhouette_score
silhouette_avg = silhouette_score(X, kmeans.labels_)
print("The average silhouette score is :", silhouette_avg)

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 338, "status": "ok", "timestamp": 1723569681604, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="qd9CCYELF5Ro" outputId="14c0cd5c-a10f-4f09-cd20-c5ab63aab8a3"
print(labels[100:150])
print(y[100:150])

# %% [markdown] id="Rnp_vunXEM4V"
# We have achieved a weak classification accuracy of 1% by our unsupervised model.

# %% [markdown] id="ltIB8BA4EM4V"
# # **15. Use elbow method to find optimal number of clusters** <a class="anchor" id="15"></a>
#
# [Table of Contents](#0.1)

# %% colab={"base_uri": "https://localhost:8080/", "height": 472} executionInfo={"elapsed": 5055, "status": "ok", "timestamp": 1725952712577, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="XKYhpLDrEM4V" outputId="6b19bf37-fa62-4ca2-f9fb-56d6ed8a6aee"
from sklearn.cluster import KMeans
cs = []
for i in range(1, 21):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    cs.append(kmeans.inertia_)

plt.plot(range(1, 21), cs)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('CS')
plt.show()


# %% [markdown] id="1jWYoFfNLm5h"
# ![image.png](data:image/png;base64,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)

# %% [markdown] id="252swnFpL5UR"
# **n_clusters = i:** This sets the number of clusters you want to form.
#
# **init = 'k-means++'**: This specifies that the k-means++ method should be used for initializing the cluster centers. This means points that are farther from the existing centroids are more likely to be selected.
#
# **max_iter = 300:** This sets the maximum number of iterations the algorithm will run for a single initialization.
#
# **n_init = 10:** This specifies the number of times the algorithm will be run with different centroid seeds. The final result will be the best output of n_init consecutive runs in terms of inertia (within-cluster sum-of-squares).
#
# **random_state = 0:** This ensures reproducibility of results by setting a seed for the random number generator used in centroid initialization

# %% [markdown] id="cJHUU8jcEM4W"
# I will check the model accuracy with different number of clusters.

# %% [markdown] id="3Wi1gOAIEM4W"
# # **16. K-Means model with different clusters** <a class="anchor" id="16"></a>
#
# [Table of Contents](#0.1)

# %% [markdown] id="NIDjWvyuEM4W"
# ### K-Means model with 3 clusters

# %% executionInfo={"elapsed": 332, "status": "ok", "timestamp": 1725953026383, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="rRxDQWmVEM4W"
kmeans = KMeans(n_clusters=3, random_state=0)

kmeans.fit(X)

# check how many of the samples were correctly labeled
labels = kmeans.labels_

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1261, "status": "ok", "timestamp": 1725953030616, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="utAF0sQsSOro" outputId="ecfa2ddb-29bc-4570-ec57-a448eee37bd9"
from sklearn.metrics import silhouette_score
silhouette_avg = silhouette_score(X, kmeans.labels_)
print("The average silhouette score is :", silhouette_avg)

# %% [markdown] id="Iq7EaRSpEM4W"
# ### K-Means model with 4 clusters

# %% executionInfo={"elapsed": 307, "status": "ok", "timestamp": 1725953042779, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="0sOGKV4pEM4W"
kmeans = KMeans(n_clusters=4, random_state=0)

kmeans.fit(X)

# check how many of the samples were correctly labeled
labels = kmeans.labels_

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 2782, "status": "ok", "timestamp": 1725953046958, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="yK_jAVVJ5bDs" outputId="8b1853a7-b932-4c5a-814c-5414e3a2eb42"
from sklearn.metrics import silhouette_score
silhouette_avg = silhouette_score(X, kmeans.labels_)
print("The average silhouette score is :", silhouette_avg)


# %% executionInfo={"elapsed": 301, "status": "ok", "timestamp": 1725953075933, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="EEz_q5Rc0bat"
kmeans = KMeans(n_clusters=2, random_state=0)

kmeans.fit(X)

# check how many of the samples were correctly labeled
labels = kmeans.labels_

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1197, "status": "ok", "timestamp": 1725953079905, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="Lk7_re_R0beY" outputId="718d5b00-e76c-4776-e175-b8d2bbb3547d"
from sklearn.metrics import silhouette_score
silhouette_avg = silhouette_score(X, kmeans.labels_)
print("The average silhouette score is :", silhouette_avg)

# %% executionInfo={"elapsed": 298, "status": "ok", "timestamp": 1725953117836, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="dS2Z5gq20lqH"
kmeans = KMeans(n_clusters=5, random_state=0)

kmeans.fit(X)

# check how many of the samples were correctly labeled
labels = kmeans.labels_

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1174, "status": "ok", "timestamp": 1725953121205, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="Pm6x7iRT0nHa" outputId="49b9c3c0-7555-4309-c9cb-e867e0ff1785"
from sklearn.metrics import silhouette_score
silhouette_avg = silhouette_score(X, kmeans.labels_)
print("The average silhouette score is :", silhouette_avg)

# %% colab={"base_uri": "https://localhost:8080/", "height": 472} executionInfo={"elapsed": 23887, "status": "ok", "timestamp": 1725954359561, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="4exOqp3k48fk" outputId="e6eb47e2-a256-4168-b804-1f98c561acf0"
# Need to compute the kmeans with k from 1 to 20 and compute silhoutte score for them and then plot graph between them

import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

silhouette_scores = []
for i in range(2, 21):
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(X)
    labels = kmeans.labels_
    silhouette_avg = silhouette_score(X, labels)
    silhouette_scores.append(silhouette_avg)

# Plotting the silhouette scores
plt.plot(range(2, 21), silhouette_scores)
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score vs. Number of Clusters")
plt.show()


# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 466, "status": "ok", "timestamp": 1725954389187, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="pKtTsa-15c_Q" outputId="62e0a526-b2df-47ef-ec8b-101bde9d88fb"
max(silhouette_scores)
