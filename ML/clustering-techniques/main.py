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

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
import warnings
warnings.filterwarnings('ignore')

# %%
df = pd.read_csv("Mall_Customers.csv")

# %%
df.head()

# %%
df.tail()

# %%
df.info()

# %% [markdown]
#  - Remove `CustomerID` column   

# %%
df.describe()

# %%
df.drop(columns=['CustomerID'], axis=1,inplace=True)
df.head()

# %%
df.rename(columns={
    "Annual Income (k$)":"Income",
    "Spending Score (1-100)":"SpendingScore"
},inplace=True)

# %%
df.head()

# %%
df['Gender'].value_counts()

# %%
plt.figure(figsize=(7,7))
size  = df['Gender'].value_counts()
label = ['Female','Male']
color = ['Pink','Blue']
explode=[0,0.1]
plt.pie(size, explode=explode, labels=label, colors=color, shadow=True)
plt.legend()
plt.show()

# %% [markdown]
# - from the diagram we can say that females are more visiting to mall than males

# %%
plt.bar(df.Income,df.SpendingScore)
plt.title("Spend score over income",fontsize = 20)
plt.xlabel("Income")
plt.ylabel("SpendingScore")

# %% [markdown]
# - from the diagram we can say that peoples in the range `20 - 40 K and 70 - 100 K` have the highest spend score

# %% [markdown]
# # Density Based Spacial Clustering `DBSCAN`

# %%
df.head()

# %%
X = df.iloc[:,[2,3]]

# %%
print(type(X))

# %%
X.head()

# %%
x = X.values

# %%
print(f"Max value {np.max(x)} , Min value {np.min(x)}")

# %%
# preprocessing 
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# x = scaler.fit_transform(x)

# %% [markdown]
# # Applying DBSCAN with Hyperparameter 

# %%
from itertools import product

# %%
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

# Define Params ranges
esp_range = np.arange(1, 15,1)
min_samples_range = range(2, 15)

# Results Storage 
results = []

best_score = -1
best_params = {}
best_n_clusters = 2
y_means = None
# Iterate through all parameter combinations
for eps, min_samples in product(esp_range, min_samples_range):
    
    # build and fit
    dbScan = DBSCAN(eps=eps, min_samples=min_samples,metric='euclidean')
    labels = dbScan.fit_predict(x)
    
    # Calculate metrices if there are at least two clusters
    n_clusters = len(set(labels)) - 1 if -1 in labels else 0
    
    if n_clusters >= 2:
        silhouette = silhouette_score(x, labels)
    else:
        silhouette = -1
    
    # store results
    results.append({
        'eps': eps,
       'min_samples': min_samples,
        'n_clusters': n_clusters,
        'n_noise':np.count_nonzero(labels == -1),
       'silhouette_score': silhouette
    })
    
    # update best score and params
    if silhouette > best_score:
        best_score = silhouette
        best_params = {
            'eps': eps,
           'min_samples': min_samples,
        }
        best_n_clusters = n_clusters
        y_means = labels
    
results_df = pd.DataFrame(results)

# %%
# Print results
print("Best silhouette score:",best_score)
print("Best parameters:",best_params)
print("Best n_clusters:",best_n_clusters)


# %%
results_df

# %%
plt.figure(figsize=(7,5))
plt.scatter(x[y_means == 0, 0], x[y_means == 0, 1], s = 50, c = 'pink')
plt.scatter(x[y_means == 1, 0], x[y_means == 1, 1], s = 50, c = 'yellow')
plt.scatter(x[y_means == 2, 0], x[y_means == 2, 1], s = 50, c = 'cyan')
plt.scatter(x[y_means == 3, 0], x[y_means == 3, 1], s = 50, c = 'magenta')
plt.scatter(x[y_means == 4, 0], x[y_means == 4, 1], s = 50, c = 'orange')
plt.scatter(x[y_means == 5, 0], x[y_means == 5, 1], s = 50, c = 'blue')
plt.scatter(x[y_means == 6, 0], x[y_means == 6, 1], s = 50, c = 'red')
plt.scatter(x[y_means == 7, 0], x[y_means == 7, 1], s = 50, c = 'black')
plt.scatter(x[y_means == -1, 0], x[y_means == -1, 1], s = 50, c = 'green') # outliers
plt.xlabel('Annual Income in (1k)')
plt.ylabel('Spending Score from 1-100')
plt.title('Clusters of data')
plt.show()



# %% [markdown]
# # HIERARCHICAL CLUSTERING

# %%
# applying dendrogam
import scipy.cluster.hierarchy as sch

dendrogram = sch.dendrogram(sch.linkage(x, method = 'ward'))
plt.title('Dendrogam', fontsize = 20)
plt.xlabel('Customers')
plt.ylabel('Ecuclidean Distance')
plt.show()

# %% [markdown]
# **sch.linkage(x, method='ward')** computes the linkage matrix using the Ward variance minimization algorithm.
# x is the dataset you are clustering.
#
# **method='ward'** specifies that the Ward method should be used, which minimizes the variance of the clusters being merged. This is a popular method for hierarchical clustering.

# %%
from sklearn.cluster import AgglomerativeClustering

hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(x)



# %%
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

# %%
# Calculate Silhouette Score for Hierarchical Clustering
silhouette_score_hc = silhouette_score(x, y_hc)
print("Silhoutte score for Hierarchical Clustering ",silhouette_score_hc)

# %%
# Need to make automatic code for selecting the best value n_clusters based on silheoute score

import numpy as np
# Try different numbers of clusters and calculate Silhouette Score
silhouette_scores = []
for n_clusters in range(2, 15):
  hc = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
  y_hc = hc.fit_predict(x)
  silhouette_scores.append(silhouette_score(x, y_hc))

# %%
silhouette_scores

# %%
# Find the optimal number of clusters
optimal_n_clusters = np.argmax(silhouette_scores) + 2  # Add 2 because range starts from 2
print("Optimal number of clusters:", optimal_n_clusters)

# Perform Hierarchical Clustering with the optimal number of clusters
hc = AgglomerativeClustering(n_clusters=optimal_n_clusters, affinity='euclidean', linkage='ward')
y_hc = hc.fit_predict(x)

# Calculate Silhouette Score for Hierarchical Clustering
silhouette_score_hc = silhouette_score(x, y_hc)
print("Silhouette Score for Hierarchical Clustering: ", silhouette_score_hc)


# %%
# Determine Optimal number of clusters using Elbow method 
from sklearn.cluster import KMeans

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=150, random_state=42)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# %%
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
silhouette_score_kmeans = silhouette_score(x, y_kmeans)
print("Silhouette Score for K-Means: ", silhouette_score_kmeans)
