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

# %% id="TNtx17HakFYS" executionInfo={"status": "ok", "timestamp": 1727420575489, "user_tz": -180, "elapsed": 4646, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load and prepare the dataset
data = load_iris()
X, y = data.data, data.target

# %% colab={"base_uri": "https://localhost:8080/"} id="EhC8auViSbCv" executionInfo={"status": "ok", "timestamp": 1727420598896, "user_tz": -180, "elapsed": 321, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="0badb190-7fce-4184-aee7-2795316b28ab"
feature_names = data.feature_names
target_names = data.target_names
print("Feature Names:", feature_names)
print("Target Names:", target_names)

# %% id="k6QB9S1X1j5U" executionInfo={"status": "ok", "timestamp": 1727420639001, "user_tz": -180, "elapsed": 300, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="50496068-8ae9-43c9-a86d-0846e2d3ca71" colab={"base_uri": "https://localhost:8080/"}
X.shape

# %% colab={"base_uri": "https://localhost:8080/"} id="6arTq37iNnhu" executionInfo={"status": "ok", "timestamp": 1727420648339, "user_tz": -180, "elapsed": 318, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="2eb3919e-8bdb-4462-a463-9091a7a8915b"
X[:5]

# %% colab={"base_uri": "https://localhost:8080/"} id="VETwgQ0rNuAp" executionInfo={"status": "ok", "timestamp": 1727420670255, "user_tz": -180, "elapsed": 330, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="ef161e7a-3e3d-44f3-8d99-499a41f4b75a"
y[:20]

# %% id="8cYwPnG51H63" executionInfo={"status": "ok", "timestamp": 1727420721891, "user_tz": -180, "elapsed": 334, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
# Standardize the dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# %% colab={"base_uri": "https://localhost:8080/"} id="jsUw4CuD1J9g" executionInfo={"status": "ok", "timestamp": 1727420725030, "user_tz": -180, "elapsed": 465, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="08e5f73a-c6df-4091-bbe0-2b3652d5364d"
# f_classif Feature Selection
selector = SelectKBest(score_func=f_classif, k=2)
X_selected = selector.fit_transform(X_scaled, y)
selected_features = [feature_names[i] for i in selector.get_support(indices=True)]

print(f"Selected features based on Z-score: {selected_features}")

# %% id="XenFcGPE1NYN" executionInfo={"status": "ok", "timestamp": 1727420728852, "user_tz": -180, "elapsed": 315, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)


# %% id="Zu0ZbBBF1AW8" executionInfo={"status": "ok", "timestamp": 1727420751499, "user_tz": -180, "elapsed": 276, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}}
# Function to train and evaluate the model
def train_evaluate(X_train, X_test, y_train, y_test, DR_name):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)
    print(f"\n{DR_name} - Accuracy: {accuracy}\n")
    print(f"{DR_name} - Classification Report:\n{report}\n")



# %% colab={"base_uri": "https://localhost:8080/"} id="eSA9xhYr1U0b" executionInfo={"status": "ok", "timestamp": 1727420813966, "user_tz": -180, "elapsed": 290, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="b3930d46-b6b3-48ea-ddba-8a7501988584"
# Train and evaluate the model on the selected features
train_evaluate(X_train, X_test, y_train, y_test, "f_classif Feature Selection")

# %% colab={"base_uri": "https://localhost:8080/"} id="bqV0d4HQ1CRH" executionInfo={"status": "ok", "timestamp": 1727421019259, "user_tz": -180, "elapsed": 3262, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="96ccf073-523c-40f5-ee7e-271c7a26320b"
# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.3, random_state=42)

# Train and evaluate the model on PCA features
train_evaluate(X_train_pca, X_test_pca, y_train_pca, y_test_pca, "PCA")

# Apply LDA
lda = LDA(n_components=2)
X_lda = lda.fit_transform(X_scaled, y)
X_train_lda, X_test_lda, y_train_lda, y_test_lda = train_test_split(X_lda, y, test_size=0.3, random_state=42)

# Train and evaluate the model on LDA features
train_evaluate(X_train_lda, X_test_lda, y_train_lda, y_test_lda, "LDA")

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)
X_train_tsne, X_test_tsne, y_train_tsne, y_test_tsne = train_test_split(X_tsne, y, test_size=0.3, random_state=42)
# Train and evaluate the model on t-SNE features
train_evaluate(X_train_tsne, X_test_tsne, y_train_tsne, y_test_tsne, "t-SNE")
