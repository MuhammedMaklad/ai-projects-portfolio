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

# %% executionInfo={"elapsed": 5820, "status": "ok", "timestamp": 1727421269626, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="YoIqJ7_neyNZ"
import pandas as pd
import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 6, "status": "ok", "timestamp": 1727421269627, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="DS9JA4tsevlF" outputId="0eeaccda-c651-4827-a8e7-03495469ad89"
# Load the digits dataset
digits = load_digits()
X = digits.data
y = digits.target
feature_names = digits.feature_names
print(feature_names)

# %% colab={"base_uri": "https://localhost:8080/"} id="FLbJUuIVVKjC" executionInfo={"status": "ok", "timestamp": 1727421286030, "user_tz": -180, "elapsed": 4, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="564f6a79-645c-4e5a-c970-a6281d1c686d"
len(X[0])

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 6, "status": "ok", "timestamp": 1725105015472, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="6xkSqSySgrSE" outputId="cc77a230-87c1-4c4d-d71a-01b897416950"
len(X)

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 4, "status": "ok", "timestamp": 1725105015472, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="0Qwl52nifhiB" outputId="2cb35f93-db7d-4fc4-b07d-84a856e7a663"
X[:1]

# %% colab={"base_uri": "https://localhost:8080/", "height": 174} executionInfo={"elapsed": 2274, "status": "ok", "timestamp": 1725105017743, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="K64Rf5tLgKma" outputId="0087eb8b-020e-446d-e92b-70c6d0746c12"
import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(20,4))
for index, (image, label) in enumerate(zip(X[0:5], y[0:5])):
  plt.subplot(1, 5, index + 1)
  plt.imshow(np.reshape(image, (8,8)), cmap=plt.cm.gray)
  plt.title('Training: %i\n' % label, fontsize = 20)


# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 9, "status": "ok", "timestamp": 1725105017743, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="ZNYujUcCf4eX" outputId="1a85f83d-a41d-4df5-de2b-d9b39640d174"
y[:5]

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 9, "status": "ok", "timestamp": 1725105017744, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="k6QB9S1X1j5U" outputId="c6d02ab9-5cd7-47cd-da2a-eae26a203d4b"
X.shape

# %% id="bFVVPGD5hDYM"
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 8, "status": "ok", "timestamp": 1725105017744, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="jsUw4CuD1J9g" outputId="0ae597a9-218b-4119-b705-4887ba46667e"
# f_classif Feature Selection
selector = SelectKBest(score_func=f_classif, k=30)
X_selected = selector.fit_transform(X_scaled, y)
selected_features = [feature_names[i] for i in selector.get_support(indices=True)]

print(f"Selected features based on Z-score: {selected_features}")

# %% id="XenFcGPE1NYN"
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)


# %% id="Zu0ZbBBF1AW8"
# Function to train and evaluate the model
def train_evaluate(X_train, X_test, y_train, y_test, DR_name):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)
    print(f"\n{DR_name} - Accuracy: {accuracy}\n")
    print(f"{DR_name} - Classification Report:\n{report}\n")



# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 626, "status": "ok", "timestamp": 1725105018364, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="eSA9xhYr1U0b" outputId="7c18f3c7-73c9-41aa-d4e6-f5e380a19302"
# Train and evaluate the model on the selected features
train_evaluate(X_train, X_test, y_train, y_test, "f_classif Feature Selection")

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 5, "status": "ok", "timestamp": 1724081545746, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="ACXvicwyiTdF" outputId="03087715-74f0-4266-b6fe-d9e448c5b2cc"
len(X_scaled)
len(y)

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 4, "status": "ok", "timestamp": 1724081545746, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="AvsayZ4RjAnY" outputId="4c94ac3f-0542-463c-e32c-1d77c4115b9b"
print(min(X_scaled.shape[1], len(np.unique(y)) - 1))

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1431, "status": "ok", "timestamp": 1725105113507, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}, "user_tz": -180} id="2S7zO_xXj5TQ" outputId="e8aa2e94-7df9-447c-dba0-2cd201d255fb"
# Apply PCA
pca = PCA(n_components=20)
X_pca = pca.fit_transform(X_selected)
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Train and evaluate the model on PCA features
train_evaluate(X_train_pca, X_test_pca, y_train_pca, y_test_pca, "PCA")

# %% colab={"base_uri": "https://localhost:8080/"} id="bqV0d4HQ1CRH" executionInfo={"status": "ok", "timestamp": 1725105429111, "user_tz": -180, "elapsed": 41061, "user": {"displayName": "Ahmed Yousry", "userId": "15736517196575255940"}} outputId="f25015e0-5bb5-4743-cb9f-d384840f543c"
# Apply PCA
pca = PCA(n_components=20)
X_pca = pca.fit_transform(X_selected)
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Train and evaluate the model on PCA features
train_evaluate(X_train_pca, X_test_pca, y_train_pca, y_test_pca, "PCA")

# Apply LDA
lda = LDA(n_components=min(X_selected.shape[1], len(np.unique(y)) - 1)) # Set n_components based on features and classes
X_lda = lda.fit_transform(X_selected, y)
X_train_lda, X_test_lda, y_train_lda, y_test_lda = train_test_split(X_lda, y, test_size=0.2, random_state=42)

# Train and evaluate the model on LDA features
train_evaluate(X_train_lda, X_test_lda, y_train_lda, y_test_lda, "LDA")

# Apply t-SNE
tsne = TSNE(n_components=3, random_state=42)
X_tsne = tsne.fit_transform(X_selected)
X_train_tsne, X_test_tsne, y_train_tsne, y_test_tsne = train_test_split(X_tsne, y, test_size=0.2, random_state=42)
# Train and evaluate the model on LDA features
train_evaluate(X_train_lda, X_test_lda, y_train_lda, y_test_lda, "TSNE")

