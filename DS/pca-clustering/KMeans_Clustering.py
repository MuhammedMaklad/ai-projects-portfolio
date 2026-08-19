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
#     name: python3
# ---

# %% id="fKs-rjKfiW8q"
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.datasets import make_blobs

# %% [markdown] id="vbvAckmti6lX"
# ### Create Data

# %% id="-OK8wh86i5Wb"
X, y = make_blobs(n_samples=500, n_features=4,
                           centers=4, cluster_std=1.5,random_state=101)

# %% colab={"base_uri": "https://localhost:8080/"} id="uoOX-dTpjFrh" outputId="3b9401b8-6f82-4881-a4dc-4532fe572f5f"
print(X.shape)
print(y.shape)

# %% colab={"base_uri": "https://localhost:8080/", "height": 430} id="MFVtr1RhjAm5" outputId="34c083b6-1353-4ae6-a8f0-2a742cb3231e"
plt.scatter(X[:,0],X[:,1],c=y,cmap='rainbow')
plt.show("Distribution of Data")
plt.show()

# %% colab={"base_uri": "https://localhost:8080/", "height": 80} id="yj1FFWJflTik" outputId="c530b16c-2ab1-4cca-a854-4b2d706f277d"
model = KMeans(
    n_clusters=4,          # Critical parameter
    init='k-means++',       # Default (smart initialization)
    n_init=10,             # Number of initializations
    max_iter=300,           # Max iterations per run
    tol=1e-4,              # Tolerance for convergence
    algorithm='elkan',
    random_state=42        # Reproducibility
)
model.fit(X)

# %% colab={"base_uri": "https://localhost:8080/"} id="hzML5at-l1dH" outputId="95f97df6-1f0b-4f9c-8074-9cff5bfb9c08"
print(f"centers: {model.cluster_centers_}")

# %% colab={"base_uri": "https://localhost:8080/"} id="fhkCizDbmDnL" outputId="e98f9d5d-031c-4cd2-b2a6-e24b21bb7c48"
print(f"model : {model.inertia_}")

# %% colab={"base_uri": "https://localhost:8080/"} id="2SnLZ1qZl6Od" outputId="eacb0a3d-2012-422f-e5c0-7ff44d190e76"
print(f"labels: {model.labels_}")

# %% colab={"base_uri": "https://localhost:8080/", "height": 430} id="2BxBcR31mKyW" outputId="abe44297-b3a4-4967-e8ed-baebc57e3ab1"
plt.scatter(X[:,0],X[:,1],c=model.labels_,cmap='rainbow')
plt.show("Distribution of Data")
plt.show()

# %% colab={"base_uri": "https://localhost:8080/"} id="kbhshA-MnLir" outputId="4a57be1e-2751-4d66-bc7b-355f8f2f86f3"
print(model.labels_)
print(y)

# %% colab={"base_uri": "https://localhost:8080/", "height": 430} id="33WVohGtne6L" outputId="59cb6b8e-21a8-4f4d-dbc9-411d010538c7"
plt.scatter(X[:,0],X[:,1],c=model.labels_,cmap='rainbow')
plt.show("Distribution of Data")
plt.show()

# %% colab={"base_uri": "https://localhost:8080/", "height": 582} id="kl-z96POmRan" outputId="1183b0a0-5fb8-4fbf-d5cb-d181dc2e4afb"
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(10,6))
f.suptitle('K Means', fontsize=16)
ax1.set_title('K Means')
ax1.scatter(X[:,0],X[:,1],c=model.labels_,cmap='rainbow')
ax2.set_title("Original")
ax2.scatter(X[:,0],X[:,1],c=y,cmap='rainbow')
plt.show()

# %% colab={"base_uri": "https://localhost:8080/"} id="_65I96JCn-N6" outputId="57db8de7-71cd-463d-bc97-f4d1f9e497eb"
# check how many of the samples were correctly labeled
from sklearn.metrics import silhouette_score
silhouette_avg = silhouette_score(X, model.labels_)
print("The average silhouette score is :", silhouette_avg)

# %% id="HGT9cxJooExV"
# Need to compute the kmeans with k from 1 to 20 and compute silhoutte score for them and then plot graph between them

silhouette_scores = []
for i in range(2, 21):
    kmeans = KMeans( n_clusters=i,          # Critical parameter
    init='k-means++',       # Default (smart initialization)
    n_init=10,             # Number of initializations
    max_iter=300,           # Max iterations per run
    tol=1e-4,              # Tolerance for convergence
    algorithm='elkan',
    random_state=42
    )
    kmeans.fit(X)
    labels = kmeans.labels_
    silhouette_avg = silhouette_score(X, labels)
    silhouette_scores.append(silhouette_avg)


# %% colab={"base_uri": "https://localhost:8080/", "height": 564} id="VDS49lqmpFDp" outputId="c5734ec4-27aa-4801-fcf5-5194641c0a5f"
# Plotting the silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(range(2, 21), silhouette_scores)
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score vs. Number of Clusters")
plt.show()

# %% colab={"base_uri": "https://localhost:8080/", "height": 564} id="NIjUYnzXpZt5" outputId="0c1fb7e1-9998-421e-d8d3-3586360b1264"
# Plotting the silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(range(2, 21), silhouette_scores, marker='o')  # Add markers for clarity
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score vs. Number of Clusters")

# Force integer ticks on x-axis from 1 to 21
plt.xticks(range(1, 22))  # 1 to 21 (inclusive)

plt.grid(True, linestyle='--', alpha=0.6)  # Optional: Add gridlines
plt.show()

# %% [markdown] id="X9AL9nV5pgqN"
# ## best `K`= 4

# %% colab={"base_uri": "https://localhost:8080/"} id="lkia-8udoo6B" outputId="1d894ab7-e91d-4477-f74b-d912975ca8f3"
max(silhouette_scores)
