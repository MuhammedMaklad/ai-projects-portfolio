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

# %% _uuid="8f2839f25d086af736a60e9eeb907d3b93b6e0e5" _cell_guid="b1076dfc-b9ad-4769-8c92-a6c4dae69d19" id="aQcUOnGKzBQt" executionInfo={"status": "ok", "timestamp": 1726209809044, "user_tz": -180, "elapsed": 410, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# %% colab={"base_uri": "https://localhost:8080/"} id="D5mEHBFWzJb_" executionInfo={"status": "ok", "timestamp": 1726209811762, "user_tz": -180, "elapsed": 2273, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="36b17387-1487-4dae-ac6e-25c27cc6f271"
from google.colab import drive
drive.mount('/content/drive')

# %% id="k5yj9-yQS2We" executionInfo={"status": "ok", "timestamp": 1726209811762, "user_tz": -180, "elapsed": 30, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
import warnings
warnings.filterwarnings('ignore')

# %% colab={"base_uri": "https://localhost:8080/"} id="IBtJXu8IzLZ0" executionInfo={"status": "ok", "timestamp": 1726209811762, "user_tz": -180, "elapsed": 30, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="69bc6d16-aaf7-4c80-a6b2-b51bc76c1556"
# cd /content/drive/MyDrive/Machine learning 2024/Notebooks - Machine Learning/DBSCAN & Hierarchical clustering & kmeans

# %% colab={"base_uri": "https://localhost:8080/"} id="qfSsQD__KKWZ" executionInfo={"status": "ok", "timestamp": 1726209811762, "user_tz": -180, "elapsed": 29, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="8701af21-5563-4d67-bab2-a270c677b08a"
# ls

# %% _uuid="d629ff2d2480ee46fbb7e2d37f6b5fab8052498a" _cell_guid="79c7e3d0-c299-4dcb-8224-4455121ee9b0" id="mQ4ek87szBQu" executionInfo={"status": "ok", "timestamp": 1726209811762, "user_tz": -180, "elapsed": 24, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
df=pd.read_csv("Mall_Customers.csv")

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} id="90-S3iCfzBQv" executionInfo={"status": "ok", "timestamp": 1726209811762, "user_tz": -180, "elapsed": 24, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="b4c5019b-482f-4f3c-e6a2-727c83b15c87"
df.head()

# %% colab={"base_uri": "https://localhost:8080/"} id="hb9Pf0aWzBQv" executionInfo={"status": "ok", "timestamp": 1726209811762, "user_tz": -180, "elapsed": 23, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="9071d8e8-081c-416a-f33a-dbbd01e98fc5"
df.info()

# %% colab={"base_uri": "https://localhost:8080/", "height": 300} id="xBO97N3LzBQv" executionInfo={"status": "ok", "timestamp": 1726209811762, "user_tz": -180, "elapsed": 21, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="a122af04-9ef5-4f43-f021-93b61864e730"
df.describe()

# %% id="ZwN3trknzBQv" executionInfo={"status": "ok", "timestamp": 1726209811762, "user_tz": -180, "elapsed": 21, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
df.rename(columns={'Annual Income (k$)':'Income','Spending Score (1-100)':'SpendScore'},inplace=True)

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} id="thoDsxNBzBQv" executionInfo={"status": "ok", "timestamp": 1726209811762, "user_tz": -180, "elapsed": 20, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="b1311ff3-f67f-4a65-e1e6-b8fe82da0381"
df.head()

# %% id="uAdfPNtmzBQx" executionInfo={"status": "ok", "timestamp": 1726209811762, "user_tz": -180, "elapsed": 20, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
df=df.drop(['CustomerID'],axis=1)

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} id="oUSaYJDTzBQx" executionInfo={"status": "ok", "timestamp": 1726209811762, "user_tz": -180, "elapsed": 19, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="72a1cb63-fb3d-49fa-c15f-545f87a9c794"
df.head()

# %% colab={"base_uri": "https://localhost:8080/", "height": 178} id="ClPpiqLnBD-R" executionInfo={"status": "ok", "timestamp": 1726209811763, "user_tz": -180, "elapsed": 20, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="e12f0f47-e710-4059-9fae-2103bb7ae75d"
df['Gender'].value_counts()

# %% colab={"base_uri": "https://localhost:8080/", "height": 576} id="92FV4gChzBQy" executionInfo={"status": "ok", "timestamp": 1726209812308, "user_tz": -180, "elapsed": 564, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="b4a1c709-a7e4-4fcd-f17a-f45bdda0918f"
plt.figure(figsize=(7,7))
size=df['Gender'].value_counts()
label=['Female','Male']
color=['Pink','Blue']
explode=[0,0.1]
plt.pie(size,explode=explode,labels=label,colors=color,shadow=True)
plt.legend()
plt.show()

# %% colab={"base_uri": "https://localhost:8080/", "height": 178} id="M2UkWYE1bUo-" executionInfo={"status": "ok", "timestamp": 1726209812309, "user_tz": -180, "elapsed": 31, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="76d751fd-de6c-4d15-8370-015fdd93c1cd"
size

# %% [markdown] id="D8VL0XICzBQy"
# **From the diagram we can say that females are more visiting to mall than males**

# %% colab={"base_uri": "https://localhost:8080/", "height": 497} id="EBlWqUaEzBQz" executionInfo={"status": "ok", "timestamp": 1726209813187, "user_tz": -180, "elapsed": 907, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="de18ceac-bf59-4036-cf08-834e073565de"
plt.bar(df['Income'],df['SpendScore'])
plt.title('Spendscore over income',fontsize=20)
plt.xlabel('Income')
plt.ylabel('Spendscore')

# %% [markdown] id="LYJ0M_ASzBQz"
# **Peoples of income in the range of 20k-40k and 70k-100k have the highest spend score**

# %% [markdown] id="iE1-iebGzBQz"
# # **Density Based Spacial Clustering of Applications with noise (DBSCAN)**

# %% [markdown] id="TEnPMhPmzBQz"
# **We are going to use the DBSCAN  for algorithm for the purpose of clustering. It is an unsupervised machine learning algorithm. It is used for clusters of high density. It automatically predicts the outliers and removes it. It is better than hierarchical and k-means clustering algorithm. It makes the clusters based on the parameters like epsilon,min points and noise.It separately predicts the core points, border points and outliers efficiently.**

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} id="92rysORhzBQz" executionInfo={"status": "ok", "timestamp": 1726209813187, "user_tz": -180, "elapsed": 19, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="8ad3ad24-8b60-466f-d8be-d4446f8f97a1"
df.head()

# %% id="Bt4lCVe-zBQ0" executionInfo={"status": "ok", "timestamp": 1726209813187, "user_tz": -180, "elapsed": 18, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
x=df.iloc[:,[2,3]]

# %% colab={"base_uri": "https://localhost:8080/"} id="-OHxpB4mce5e" executionInfo={"status": "ok", "timestamp": 1726209813187, "user_tz": -180, "elapsed": 18, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="b98d60fe-57c9-4872-c2b3-7b524cc00a3c"
print(type(x))
print(x[:5])

# %% id="vIKjjKEHzBQ0" executionInfo={"status": "ok", "timestamp": 1726209813188, "user_tz": -180, "elapsed": 18, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
x=x.values

# %% colab={"base_uri": "https://localhost:8080/"} id="7XiwQFgjcyeJ" executionInfo={"status": "ok", "timestamp": 1726209813188, "user_tz": -180, "elapsed": 17, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="dcb0619e-dd1e-4c49-9b3f-bcbebe268fa5"
print(type(x))
print(x[:5])

# %% id="KrSuwB_azBQ0" executionInfo={"status": "ok", "timestamp": 1726209813188, "user_tz": -180, "elapsed": 16, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
from sklearn.cluster import DBSCAN
db=DBSCAN(eps=5,min_samples=6,metric='euclidean')

# %% id="oTpcFzjIzBQ0" executionInfo={"status": "ok", "timestamp": 1726209813188, "user_tz": -180, "elapsed": 16, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
model=db.fit(x)

# %% id="tf-yY261zBQ0" executionInfo={"status": "ok", "timestamp": 1726209813188, "user_tz": -180, "elapsed": 16, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
label=model.labels_

# %% colab={"base_uri": "https://localhost:8080/"} id="UBs8Bn3lzBQ0" executionInfo={"status": "ok", "timestamp": 1726209813188, "user_tz": -180, "elapsed": 16, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="a334c483-4dae-4626-b64c-e9bab5a03e5f"
label

# %% colab={"base_uri": "https://localhost:8080/"} id="3wNvbXFQzBQ0" executionInfo={"status": "ok", "timestamp": 1726209813188, "user_tz": -180, "elapsed": 15, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="7f28f44c-a37c-47ac-aa96-585d6b7cae4e"
from sklearn import metrics

#Calculating the number of clusters

n_clusters=len(set(label))- (1 if -1 in label else 0)
print('No of clusters:',n_clusters)

# %% id="Cmv_9WnTEDqp" executionInfo={"status": "ok", "timestamp": 1726209819996, "user_tz": -180, "elapsed": 489, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
y_means = db.fit_predict(x)

# %% colab={"base_uri": "https://localhost:8080/"} id="xnlyJy-HENGg" executionInfo={"status": "ok", "timestamp": 1726209822441, "user_tz": -180, "elapsed": 4, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="f79b913b-b3f9-4fdc-c673-bfe3eca2b5ad"
y_means == 0, 0

# %% [markdown] id="Txpcdy7rRlcS"
# **Hint**
#
# x[y_means == i, 0] and x[y_means == i, 1] select the x and y coordinates of points in the i-th cluster.

# %% colab={"base_uri": "https://localhost:8080/", "height": 487} id="GbIpSQjXzBQ1" executionInfo={"status": "ok", "timestamp": 1726209823889, "user_tz": -180, "elapsed": 9, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="d20dacfe-2997-4e83-9a78-8c6f438f78eb"
plt.figure(figsize=(7,5))
plt.scatter(x[y_means == 0, 0], x[y_means == 0, 1], s = 50, c = 'pink')
plt.scatter(x[y_means == 1, 0], x[y_means == 1, 1], s = 50, c = 'yellow')
plt.scatter(x[y_means == 2, 0], x[y_means == 2, 1], s = 50, c = 'cyan')
plt.scatter(x[y_means == 3, 0], x[y_means == 3, 1], s = 50, c = 'magenta')
plt.scatter(x[y_means == 4, 0], x[y_means == 4, 1], s = 50, c = 'orange')
plt.scatter(x[y_means == 5, 0], x[y_means == 5, 1], s = 50, c = 'blue')
plt.scatter(x[y_means == 6, 0], x[y_means == 6, 1], s = 50, c = 'red')
plt.scatter(x[y_means == 7, 0], x[y_means == 7, 1], s = 50, c = 'black')
plt.scatter(x[y_means == 8, 0], x[y_means == 8, 1], s = 50, c = 'violet')
plt.xlabel('Annual Income in (1k)')
plt.ylabel('Spending Score from 1-100')
plt.title('Clusters of data')
plt.show()



# %% colab={"base_uri": "https://localhost:8080/"} id="1UtU-7zE4Tok" executionInfo={"status": "ok", "timestamp": 1726209838116, "user_tz": -180, "elapsed": 436, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="eb054f8d-43eb-4fbe-f2bf-872a5622a2d6"
# Calculate Silhouette Score for DBSCAN
silhouette_score_dbscan = metrics.silhouette_score(x, y_means)
print("Silhouette Score for DBSCAN: ", silhouette_score_dbscan)


# %% colab={"base_uri": "https://localhost:8080/"} id="hRNe0_56NuHa" executionInfo={"status": "ok", "timestamp": 1726210004634, "user_tz": -180, "elapsed": 2315, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="2c4d65e0-ee94-451b-9314-6962b1924db6"
# Automatic code for selecting the best value of min samples and epslon based on siheoute score

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics

# Range of epsilon and min_samples to explore
eps_range = np.arange(2, 15, 1)
min_samples_range = np.arange(2, 15, 1)

best_score = -1
best_eps = None
best_min_samples = None

for eps in eps_range:
  for min_samples in min_samples_range:
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    model = db.fit(x)
    labels = model.labels_

    # Ignore cases where all points are noise
    if len(set(labels)) > 1:
      silhouette_score = metrics.silhouette_score(x, labels)
      if silhouette_score > best_score:
        best_score = silhouette_score
        best_eps = eps
        best_min_samples = min_samples

print("Best Silhouette Score:", best_score)
print("Best Epsilon:", best_eps)
print("Best Min Samples:", best_min_samples)


# %% [markdown] id="1OSilt3KzBQ1"
# # HIERARCHICAL CLUSTERING

# %% colab={"base_uri": "https://localhost:8080/", "height": 479} id="5xSHcmeSzBQ1" executionInfo={"status": "ok", "timestamp": 1723896316640, "user_tz": -180, "elapsed": 2683, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="9fa3c646-873f-47dc-eaa3-773bafd0178d"
import scipy.cluster.hierarchy as sch

dendrogram = sch.dendrogram(sch.linkage(x, method = 'ward'))
plt.title('Dendrogam', fontsize = 20)
plt.xlabel('Customers')
plt.ylabel('Ecuclidean Distance')
plt.show()

# %% [markdown] id="8oPUJW1PSJWR"
# **sch.linkage(x, method='ward')** computes the linkage matrix using the Ward variance minimization algorithm.
# x is the dataset you are clustering.
#
# **method='ward'** specifies that the Ward method should be used, which minimizes the variance of the clusters being merged. This is a popular method for hierarchical clustering.

# %% colab={"base_uri": "https://localhost:8080/", "height": 517} id="uk1u2PDKzBQ1" executionInfo={"status": "ok", "timestamp": 1726210217266, "user_tz": -180, "elapsed": 990, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="d86eb4b5-db7e-4e76-e8b8-c69e2ff19ae0"
from sklearn.cluster import AgglomerativeClustering

hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(x)

plt.scatter(x[y_hc == 0, 0], x[y_hc == 0, 1], s = 50, c = 'pink')
plt.scatter(x[y_hc == 1, 0], x[y_hc == 1, 1], s = 50, c = 'yellow')
plt.scatter(x[y_hc == 2, 0], x[y_hc == 2, 1], s = 50, c = 'cyan')
plt.scatter(x[y_hc == 3, 0], x[y_hc == 3, 1], s = 50, c = 'magenta')
plt.scatter(x[y_hc == 4, 0], x[y_hc == 4, 1], s = 50, c = 'orange')
plt.scatter(x[y_hc == 5, 0], x[y_hc == 5, 1], s = 50, c = 'blue')
plt.scatter(x[y_hc == 6, 0], x[y_hc == 6, 1], s = 50, c = 'red')
plt.scatter(x[y_hc == 7, 0], x[y_hc == 7, 1], s = 50, c = 'black')
plt.scatter(x[y_hc == 8, 0], x[y_hc == 8, 1], s = 50, c = 'violet')


plt.title('Hierarchial Clustering', fontsize = 20)
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.grid()
plt.show()

# %% colab={"base_uri": "https://localhost:8080/"} id="J3TofFIszBQ1" executionInfo={"status": "ok", "timestamp": 1726210229948, "user_tz": -180, "elapsed": 446, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="c633f4b5-21e2-4740-c971-3a318111749b"
# Calculate Silhouette Score for Hierarchical Clustering
silhouette_score_hc = metrics.silhouette_score(x, y_hc)
print("Silhouette Score for Hierarchical Clustering: ", silhouette_score_hc)


# %% id="uuTMuPE2O06e" executionInfo={"status": "ok", "timestamp": 1726210413216, "user_tz": -180, "elapsed": 423, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
# Need to make automatic code for selecting the best value n_clusters based on silheoute score

import numpy as np
# Try different numbers of clusters and calculate Silhouette Score
silhouette_scores = []
for n_clusters in range(2, 15):
  hc = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
  y_hc = hc.fit_predict(x)
  silhouette_scores.append(metrics.silhouette_score(x, y_hc))

# %% colab={"base_uri": "https://localhost:8080/"} id="UGYVeNsyKHbY" executionInfo={"status": "ok", "timestamp": 1726210416552, "user_tz": -180, "elapsed": 409, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="ad6819f5-1a62-4ae2-88a8-28dbf31efef0"
silhouette_scores

# %% colab={"base_uri": "https://localhost:8080/"} id="bLZCGCkgKGoJ" executionInfo={"status": "ok", "timestamp": 1726210649055, "user_tz": -180, "elapsed": 426, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="aa96e90c-bfe3-48f0-8dc3-04bce19b2914"
# Find the optimal number of clusters
optimal_n_clusters = np.argmax(silhouette_scores) + 2  # Add 2 because range starts from 2
print("Optimal number of clusters:", optimal_n_clusters)

# Perform Hierarchical Clustering with the optimal number of clusters
hc = AgglomerativeClustering(n_clusters=optimal_n_clusters, affinity='euclidean', linkage='ward')
y_hc = hc.fit_predict(x)

# Calculate Silhouette Score for Hierarchical Clustering
silhouette_score_hc = metrics.silhouette_score(x, y_hc)
print("Silhouette Score for Hierarchical Clustering: ", silhouette_score_hc)


# %% colab={"base_uri": "https://localhost:8080/", "height": 472} id="M2-icoWKSgqv" executionInfo={"status": "ok", "timestamp": 1726210752455, "user_tz": -180, "elapsed": 1848, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="8a31f6f5-c869-4e89-a30b-4012db0b198f"
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Determine optimal number of clusters using the Elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()


# %% colab={"base_uri": "https://localhost:8080/", "height": 497} id="RHyhrQ_tLb6O" executionInfo={"status": "ok", "timestamp": 1726210771123, "user_tz": -180, "elapsed": 578, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="d26b278e-4c33-4b16-d9de-4801b6e78b2d"
# Based on the Elbow method, choose the optimal number of clusters (let's say 5)
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(x)

# Visualize the clusters
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s=50, c='pink')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s=50, c='yellow')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s=50, c='cyan')
plt.scatter(x[y_kmeans == 3, 0], x[y_kmeans == 3, 1], s=50, c='magenta')
plt.scatter(x[y_kmeans == 4, 0], x[y_kmeans == 4, 1], s=50, c='orange')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='black', label='Centroids')

plt.title('K-Means Clustering', fontsize=20)
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.grid()
plt.show()

# Calculate Silhouette Score for K-Means
silhouette_score_kmeans = metrics.silhouette_score(x, y_kmeans)
print("Silhouette Score for K-Means: ", silhouette_score_kmeans)

# %% colab={"base_uri": "https://localhost:8080/", "height": 497} id="o2A07m0CLkFq" executionInfo={"status": "ok", "timestamp": 1726210793760, "user_tz": -180, "elapsed": 1131, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="f7e992d0-fed3-44e7-85a1-e75ec8fdbf47"
# Based on the Elbow method, choose the optimal number of clusters (let's say 5)
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(x)

# Visualize the clusters
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s=50, c='pink')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s=50, c='yellow')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s=50, c='cyan')
plt.scatter(x[y_kmeans == 3, 0], x[y_kmeans == 3, 1], s=50, c='magenta')
plt.scatter(x[y_kmeans == 4, 0], x[y_kmeans == 4, 1], s=50, c='orange')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='black', label='Centroids')

plt.title('K-Means Clustering', fontsize=20)
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.grid()
plt.show()

# Calculate Silhouette Score for K-Means
silhouette_score_kmeans = metrics.silhouette_score(x, y_kmeans)
print("Silhouette Score for K-Means: ", silhouette_score_kmeans)
