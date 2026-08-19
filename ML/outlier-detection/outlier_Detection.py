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

# %% id="_9rrqKaLsquN"
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

# %% [markdown] id="TCh4-XUq7okW"
# # ***Introduction***
#
# ---
#
# **`Outlier detection`** is a critical step in **`data preprocessing`**, helping to identify `anomalies` that may distort statistical analyses or machine learning models. This document explains five widely used outlier detection methods: **Z-Score, IQR, Isolation Forest, LOF, and Elliptic Envelope**

# %% [markdown] id="8z3XrFxP8flf"
# ## **Z - Score**
# - Measures how many standard deviations a data point is from the mean.
# - A common threshold is |Z-Score| > `threshold`
# ### **Pros & Cons**
#
# ✔ Simple, works well for Gaussian data.
#
# ✖ Sensitive to extreme values (mean & std deviation can be skewed). Sensitive to extreme values (mean & std deviation can be skewed).
#
# ### **Common Threshold**
# * |Z| > 2 → potential outlier (less strict)
#
# * |Z| > 3 → strong outlier (standard rule)
#
# * |Z| > 4 or 5 → very extreme cases (very strict)

# %% id="27--CouO7Oam"
import numpy as np

def remove_outliers_zscore(X, threshold=3.0, return_outliers=False):
    """
    Remove outliers using Z-score method.

    Parameters:
    - X: np.array or pd.DataFrame
    - threshold: float, Z-score threshold (default 3.0)
    - return_outliers: bool, whether to also return detected outliers

    Returns:
    - X_clean: array or DataFrame without outliers
    - mask: boolean array where True = inlier
    - (optional) outliers: the detected outliers
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0, ddof=0)  # Population std dev
    z_scores = np.abs((X - mean) / std)

    mask = (z_scores < threshold).all(axis=1)
    X_clean = X[mask]

    if return_outliers:
        outliers = X[~mask]
        return X_clean, mask, outliers
    else:
        return X_clean, mask


# %% [markdown] id="tSUrFCWk_hX0"
# ## **Interquartile Range (IQR)**
# ### **Concept**
#
# - Uses quartiles (Q1 = 25th percentile, Q3 = 75th percentile).
# - Outliers lie outside:
# - [ Q1 − 1.5 × IQR, Q3 + 1.5 × IQR ]
#
# ### **Pros & Cons**
# ✔ Robust to skewed distributions.
#
# ✖ Less effective for high-dimensional data.

# %% id="Yh2zNRqm_B3K"
def remove_outliers_iqr(X, factor=1.5, return_outliers=False):
    """
    Remove outliers using the IQR method.

    Parameters:
    - X: np.array or pd.DataFrame
    - factor: float, the multiplier for the IQR (default 1.5)
    - return_outliers: bool, whether to also return the outliers

    Returns:
    - X_clean: array or DataFrame without outliers
    - mask: boolean array where True = inlier
    - (optional) outliers: the detected outliers
    """
    Q1 = np.percentile(X, 25, axis=0)
    Q3 = np.percentile(X, 75, axis=0)
    IQR = Q3 - Q1

    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR

    mask = ((X >= lower_bound) & (X <= upper_bound)).all(axis=1)
    X_clean = X[mask]

    if return_outliers:
        outliers = X[~mask]
        return X_clean, mask, outliers
    else:
        return X_clean, mask


# %% [markdown] id="T_reprWWBbjG"
# ## **3. Isolation Forest**
# ### **Concept**
# - An `unsupervised algorithm` that isolates anomalies instead of profiling normal data.
#
# - Anomalies require fewer splits to be isolated.
#
# ### **How It Works**
# 1. Randomly selects a feature and split value.
#
# 2. Repeats until data points are isolated.
#
# 3. Points with shorter path lengths are outliers.
#
#
# ### **Pros & Cons**
# ✔ Efficient for high-dimensional data.
#
# ✖ May struggle with local outliers near dense clusters.

# %% id="c0rd4qrsAS7f"
from sklearn.ensemble import IsolationForest

def remove_outliers_isolation_forest(X, contamination=0.1, random_state=42, return_outliers=False):
    """
    Detect and remove outliers using Isolation Forest.

    Parameters:
    - X: np.array or pd.DataFrame
    - contamination: float, the proportion of outliers (e.g., 0.1 means 10%)
    - random_state: int, random seed for reproducibility
    - return_outliers: bool, if True also return detected outliers

    Returns:
    - X_clean: cleaned data without outliers
    - mask: boolean array where True = inlier, False = outlier
    - (optional) outliers: the outlier points
    """
    iso = IsolationForest(contamination=contamination, random_state=random_state)
    yhat = iso.fit_predict(X)
    mask = yhat != -1  # Inliers are labeled as 1, outliers as -1
    X_clean = X[mask]

    if return_outliers:
        outliers = X[~mask]
        return X_clean, mask, outliers
    else:
        return X_clean, mask


# %% [markdown] id="kum9fzpEB7NB"
# ## **Local Outlier Factor (LOF)**
#
# ###**Concept**
# - A density-based method comparing local density of a point with its neighbors.
#
# - LOF >> 1 → Outlier.
#
# ### **Formula**
#
# ### **Pros & Cons**
#
# ✔ Detects local outliers in varying densities.
#
# ✖ Computationally expensive for large datasets.

# %% id="fQrA0R0TBzdV"
from sklearn.neighbors import LocalOutlierFactor
import numpy as np

def remove_outliers_lof(X, contamination=0.1, n_neighbors=20, return_outliers=False):
    """
    Remove outliers using Local Outlier Factor (LOF).

    Parameters:
    - X: np.array or pd.DataFrame
    - contamination: float, expected proportion of outliers (default 0.1)
    - n_neighbors: int, number of neighbors to use for LOF (default 20)
    - return_outliers: bool, whether to also return detected outliers

    Returns:
    - X_clean: array or DataFrame without outliers
    - mask: boolean array where True = inlier
    - (optional) outliers: the detected outliers
    """
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination, novelty=False)
    yhat = lof.fit_predict(X)

    mask = yhat != -1
    X_clean = X[mask]

    if return_outliers:
        outliers = X[~mask]
        return X_clean, mask, outliers
    else:
        return X_clean, mask


# %% [markdown] id="oFoIUYhGClQz"
# ## **Elliptic Envelope**
# ### **Concept**
# - Assumes data is Gaussian-distributed and fits an ellipse.
#
# - Points outside the ellipse are outliers.
#
# ### **Pros & Cons**
# ✔ Works well for normally distributed data.
#
# ✖ Fails if data is non-Gaussian.

# %% id="2LxKksE-Cy9u"
from sklearn.covariance import EllipticEnvelope

def remove_outliers_elliptic_envelope(X, contamination=0.1, return_outliers=False):
    """
    Remove outliers using Elliptic Envelope method (assumes Gaussian distributed data).

    Parameters:
    - X: np.array or pd.DataFrame
    - contamination: float, proportion of outliers (default 0.1)
    - return_outliers: bool, whether to also return detected outliers

    Returns:
    - X_clean: array or DataFrame without outliers
    - mask: boolean array where True = inlier
    - (optional) outliers: the detected outliers
    """
    envelope = EllipticEnvelope(contamination=contamination)
    yhat = envelope.fit_predict(X)

    mask = yhat != -1
    X_clean = X[mask]

    if return_outliers:
        outliers = X[~mask]
        return X_clean, mask, outliers
    else:
        return X_clean, mask


# %% [markdown] id="W8FUq0hlFYBv"
# # Load Data

# %% id="zCqxchRtE6ob"
data = load_iris()
X, y = data.data, data.target
features_name = data.feature_names

# %% colab={"base_uri": "https://localhost:8080/"} id="Nnc9s7ESFGLA" outputId="ec9a8740-0bb5-4f70-e9ac-99a5016de6c0"
print(features_name)

# %% id="L32pwxSTFfBP"
# dict contain all outlier detection method
outliers_method = {
    'Z-Score': remove_outliers_zscore,
    'IQR': remove_outliers_iqr,
    'Isolation Forest': remove_outliers_isolation_forest,
    'LOF': remove_outliers_lof,
    'Elliptic Envelope': remove_outliers_elliptic_envelope
}

# %% id="-QRJtQMaF9Iq"
from sklearn.model_selection import train_test_split
def split_data(X, y):
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  return X_train, X_test, y_train, y_test


# %% id="wSHa9XZ7GAJ7"

# %% id="iGFOaPzXGCfJ"
# scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)


# %% id="kREZoMRHF1uT"
def evaluate_model(X_train, y_train, X_test, y_test, model):
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred)

  print(f'Accuracy: {accuracy}')
  print(f'F1 Score: {f1_score(y_test, y_pred, average="weighted")}')

  return y_pred


# %% id="_W9bCaG8F1rp"
def display_confusion_matrix(y_pred, y_test, features_name):
  cm = confusion_matrix(y_test, y_pred)
  sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
  plt.xlabel('Predicted')
  plt.ylabel('Actual')
  plt.title('Confusion Matrix')
  plt.show()


# %% id="NqkqTNnNGksJ"
#code to visualize and auc
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score

def visualize_and_auc(y_pred, y_test, model):
    # Assuming y_pred are probabilities for positive class
    # If y_pred are class labels, you need to get probabilities first
    try:
        # Binarize the output
        y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
        n_classes = y_test_bin.shape[1]

        y_prob = model.predict_proba(X_test)

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_prob.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # Plot ROC curve for a specific class (e.g., class 0)
        plt.figure()
        plt.plot(fpr[0], tpr[0], color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc[0]:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()

    except AttributeError:
        print("Model does not have predict_proba method. AUC cannot be calculated.")
        return


# %% [markdown] id="4oxJtPNZG8VA"
# # Build & Evaluate Model

# %% id="ZXEpr71JG5vS"
model = RandomForestClassifier(random_state=42,n_estimators=50)

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="FigadB0OHPSs" outputId="eb2ea4b3-2568-4326-9b30-3bab48c7b3ef"
# Evaluate model in each outliers mehtod
for method_name, method in outliers_method.items():
  print(f'Outliers Detection Method: {method_name}')

  X, mask = method(X)
  y = y[mask]

  X_train, X_test, y_train, y_test = split_data(X, y)
  y_pred = evaluate_model(X_train, y_train, X_test, y_test, model)

  display_confusion_matrix(y_pred, y_test, features_name)
  visualize_and_auc(y_pred, y_test, model)

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="YbzEjTt6Jeao" outputId="a042c982-c57a-40e7-940a-8c2eeda3d36a"
import matplotlib.pyplot as plt

# ... (rest of the code remains the same) ...

# Evaluate model in each outliers method
for method_name, method in outliers_method.items():
    print(f'Outliers Detection Method: {method_name}')

    X, mask = method(X)
    y = y[mask]

    X_train, X_test, y_train, y_test = split_data(X, y)
    y_pred = evaluate_model(X_train, y_train, X_test, y_test, model)

    # Create subplots for confusion matrix and AUC
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))  # 1 row, 2 columns

    # Display confusion matrix on the first subplot (ax1)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    ax1.set_title('Confusion Matrix')

    # Visualize AUC on the second subplot (ax2)
    try:
        # Binarize the output
        y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
        n_classes = y_test_bin.shape[1]

        y_prob = model.predict_proba(X_test)

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_prob.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # Now you can plot using fpr, tpr, and roc_auc:
        ax2.plot(fpr[0], tpr[0], color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc[0]:.2f})')
        ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('Receiver Operating Characteristic')
        ax2.legend(loc="lower right")

    except AttributeError:
        print("Model does not have predict_proba method. AUC cannot be calculated.")

    plt.tight_layout()  # Adjust subplot parameters for a tight layout
    plt.show()
