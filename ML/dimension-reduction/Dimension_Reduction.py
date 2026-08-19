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

# %% [markdown] id="gtIdxrZVNxhC"
# # **Dimension Reduction Techniques: PCA, LDA, t-SNE**
# ### Dimension reduction is a crucial step in machine learning and data analysis, helping to reduce the number of features (variables) while retaining meaningful information. It improves computational efficiency, reduces overfitting, and aids in visualization. Below are three key techniques:

# %% id="UK4cHZFpKIhL"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

# %% [markdown] id="gzsm4ozvOFWN"
# ## **Principal Component Analysis (PCA)**
# ### **What is PCA?**
#   PCA is `an unsupervised linear dimensionality reduction technique` that transforms data into a `new coordinate system` where the `greatest variance` lies along the first axis (principal component), the second greatest variance along the second axis, and so on.
#
# ### **How PCA Works:**
#     - Standardize the Data (mean = 0, variance = 1).
#
#     - Compute Covariance Matrix (showing how variables relate).
#
#     - Calculate Eigenvectors & Eigenvalues (identify principal components).
#
#     - Select Top-k Eigenvectors (retain components explaining most variance).
#
#     - Project Data onto New Axes (transform original data).
#
# ### **Key Features:**
#     ✅ Linear transformation
#
#     ✅ Preserves global structure
#
#     ✅ Sensitive to scaling (requires normalization)
#
#     ✅ Used for noise reduction & feature extraction
#
# ### **Applications:**
#     - Image compression (eigenfaces)
#
#     - Genomics (gene expression analysis)
#
#     - Financial risk modeling

# %% id="5L7_5CgdQfNT"
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def apply_pca(X, variance_threshold=0.95, standardize=False):
  """
  Applies Principal Component Analysis (PCA) to reduce dimensionality while retaining a specified variance threshold.

  Args:
      X (array-like of shape (n_samples, n_features)): Input data to transform.
      variance_threshold (float, optional): Minimum fraction of variance to retain.
          PCA will select the smallest number of components such that the explained variance ≥ this threshold.
          Defaults to 0.95 (95% variance retained).
      standardize (bool, optional): If True, standardizes the data (mean=0, variance=1) before PCA.
          Recommended if features are on different scales. Defaults to False.

  Returns:
      array-like of shape (n_samples, n_components): Transformed data with reduced dimensions.
          `n_components` is automatically determined by `variance_threshold`.

  Prints:
      - Explained variance ratio per component.
      - Total variance explained by the selected components.

  Example:
      >>> from sklearn.datasets import load_iris
      >>> X = load_iris().data
      >>> X_pca = apply_pca(X, variance_threshold=0.95, standardize=True)
      Explained variance ratio: [0.73, 0.23]  # Example output
      Selected 2 components explaining 96.00% variance
  """
  if standardize:
      X = StandardScaler().fit_transform(X)

  pca = PCA(n_components=variance_threshold)
  X_pca = pca.fit_transform(X)

  print("Explained variance ratio:", pca.explained_variance_ratio_)
  print(f"Selected {pca.n_components_} components explaining "
        f"{100*pca.explained_variance_ratio_.sum():.2f}% variance")

  return X_pca


# %% [markdown] id="lItkZ3pWOx7l"
# ## **Linear Discriminant Analysis (LDA)**
# ### **What is LDA?**
#   **LDA** is a `supervised` dimensionality reduction method that `maximizes class separability` by *projecting data onto a lower-dimensional space.*
#
# ### **How LDA Works:**
#     - Compute Mean Vectors for each class.
#
#     - Calculate Scatter Matrices:
#
#     - Within-class scatter (SW) (variability within each class).
#
#     - Between-class scatter (SB) (variability between class means).
#
#     - Solve for Eigenvectors of
#
#     - Select Top Discriminant Components.
#
#     - Project Data onto New Subspace.
#
# ### **Key Features:**
#     ✅ Supervised (uses class labels)
#
#     ✅Maximizes class separation
#
#     ✅ Assumes Gaussian-distributed data
#
#     ✅ Limited by the number of classes (max components = C-1)
#
# ### **Applications:**
#     - Face recognition
#
#     - Biomedical data classification
#
#     - Customer segmentation

# %% id="LPM-DVTJTJXA"
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
import numpy as np

def apply_lda(X, y, variance_threshold=None, standardize=False):
  """
  Applies Linear Discriminant Analysis (LDA) with dynamic component selection.

  Automatically determines optimal n_components either:
  - By keeping components that capture specified variance (if variance_threshold is set)
  - Or using max possible components (min(n_features, n_classes-1)) if variance_threshold=None

  Args:
    X (array-like of shape (n_samples, n_features)): Input data
    y (array-like of shape (n_samples,)): Target labels
    variance_threshold (float, optional): Minimum fraction of variance to retain.
      If None, uses max possible components. Defaults to None.
    standardize (bool, optional): Standardize features before LDA. Defaults to False.

  Returns:
    tuple: (transformed_data, optimal_n_components)

  Example:
    >>> X, y = load_iris(return_X_y=True)
    >>> X_lda, n = apply_lda(X, y, variance_threshold=0.95)
    >>> print(f"Used {n} components")
  """
  if standardize:
    X = StandardScaler().fit_transform(X)

  # Calculate max possible components
  n_classes = len(np.unique(y))
  max_components = min(X.shape[1], n_classes - 1)

  if variance_threshold is None:
    # Use max possible components
    lda = LDA(n_components=max_components)
    X_lda = lda.fit_transform(X, y)
    return X_lda, max_components
  else:
    # Fit with max components first to get variance ratios
    lda = LDA(n_components=max_components)
    lda.fit(X, y)

    if not hasattr(lda, 'explained_variance_ratio_'):
        raise RuntimeError("Your scikit-learn version doesn't support explained_variance_ratio_ for LDA")

    # Calculate cumulative variance
    cum_var = np.cumsum(lda.explained_variance_ratio_)
    optimal_n = np.argmax(cum_var >= variance_threshold) + 1

    # Refit with optimal components
    lda = LDA(n_components=optimal_n)
    X_lda = lda.fit_transform(X, y)

    print(f"Selected {optimal_n}/{max_components} components explaining {cum_var[optimal_n-1]:.2%} variance")

    return X_lda, optimal_n


# %% [markdown] id="btgxG5u9P8rG"
# ## **t-Distributed Stochastic Neighbor Embedding (t-SNE)**
# ### **What is t-SNE?**
# t-SNE is a `nonlinear`, `unsupervised` technique primarily used for `visualizing high-dimensional` data in 2D/3D by preserving local similarities.
#
# ### **How t-SNE Works:**
#     1. Compute Pairwise Similarities (using Gaussian probabilities in high-dim space).
#
#     2. Construct a Probability Distribution where similar points have high probability.
#
#     3. Embed in Low-Dimensional Space (using Student’s t-distribution to avoid crowding).
#
#     4. Minimize KL Divergence between high-dim & low-dim distributions.
#
# ### **Key Features:**
#     ✅ Nonlinear transformation
#     ✅ Preserves local structure (clusters)
#     ✅ Great for visualization
#     ❌ Computationally expensive
#     ❌ Results vary with perplexity parameter
#
# ### **Applications:**
#     - Visualizing word embeddings (Word2Vec, GloVe)
#
#     - Single-cell RNA sequencing data
#
#     - Anomaly detection in high-dim data

# %% id="KDsMyKzOPauV"
from sklearn.manifold import TSNE
import numpy as np

def apply_tsne(X, n_components='auto', perplexity='auto', random_state=42, **kwargs):
  """
  Applies t-Distributed Stochastic Neighbor Embedding (t-SNE) for nonlinear dimensionality reduction,
  with optional dynamic component selection based on silhouette analysis.

  Note: t-SNE is primarily for visualization (n_components=2/3). Higher dimensions lose the method's benefits.

  Args:
      X (array-like of shape (n_samples, n_features)): Input data. Standardization recommended.
      n_components (int or 'auto'): Number of dimensions for embedding.
          If 'auto', uses 2 for visualization unless data has >50 features (then uses 3).
      perplexity (int or 'auto'): t-SNE perplexity parameter.
          If 'auto', sets to min(30, n_samples-1).
      random_state (int): Random seed for reproducibility.
      **kwargs: Additional t-SNE parameters (e.g., learning_rate, n_iter).

  Returns:
      array-like of shape (n_samples, n_components)): Embedded data.

  Example:
      >>> from sklearn.datasets import load_digits
      >>> X, _ = load_digits(return_X_y=True)
      >>> X_tsne = apply_tsne(X, n_components='auto')
      >>> X_tsne.shape  # (1797, 2)
  """
  # Auto-set perplexity (rule of thumb: 5-50, max=n_samples-1)
  if perplexity == 'auto':
    perplexity = min(30, len(X)-1)

  # Dynamic n_components logic
  if n_components == 'auto':
    n_components = 3 if X.shape[1] > 50 else 2  # Use 3D for high-dim data

  # Validate n_components (t-SNE is practically limited to 2/3D)
  if n_components not in [2, 3] and n_components != 'auto':
    print("Warning: t-SNE is typically used for 2D/3D visualization. Higher dimensions may not preserve structure well.")

  # Run t-SNE
  tsne = TSNE(
    n_components=n_components,
    perplexity=perplexity,
    random_state=random_state,
    **kwargs
  )
  X_tsne = tsne.fit_transform(X)

  return X_tsne


# %% [markdown] id="Irzdpf__VUBn"
# ## **Load** **dataset**

# %% id="q5zsHrM6VYwC"
from sklearn.datasets import load_digits

data = load_digits()

X, y = data.data, data.target

features_names = data.feature_names

# %% colab={"base_uri": "https://localhost:8080/"} id="DYx-emugWFjY" outputId="ea263fda-fe6c-4aac-bf39-5fe3fa0b6dde"
X.shape

# %% colab={"base_uri": "https://localhost:8080/"} id="DOeLraggWPWx" outputId="6ffde482-76f7-4763-b106-0f6069d955c7"
y.shape

# %% colab={"base_uri": "https://localhost:8080/", "height": 538} id="NPlIo0-oVtGy" outputId="18aa036b-59e2-4bd7-ef23-6c12051976dc"
plt.figure(figsize=(20, 10))

for i in range(10):
  plt.subplot(2, 5, i+1)
  plt.imshow(X[i].reshape(8, 8), cmap='gray')
  plt.title(f"Label: {y[i]}")

# %% id="L3dZ1-4lWtdr"
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# %% id="8pJm_iGSXLGU"
def split_data(X, y, test_size=0.2, random_state=42):
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
  return X_train, X_test, y_train, y_test


# %% id="tiGSrUZZWo6j"
def evaluate_model(model, X_train, y_train, X_test, y_test):
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)

  print("Accuracy:", accuracy_score(y_test, y_pred))
  print("Classification Report", classification_report(y_test, y_pred))

  cm = confusion_matrix(y_test, y_pred)
  plt.figure(figsize=(10, 8))
  sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')


# %% [markdown] id="twOcazrzWTk-"
# ## Will Apply Feature Selection Technique First and Then Compare With Dimension Reduction Techiques

# %% id="z4W2eAWqWRUj"
from sklearn.feature_selection import SelectKBest, f_classif

# %% colab={"base_uri": "https://localhost:8080/"} id="OuxW05PoXQmu" outputId="9744eae3-db69-452d-c1f3-ac6bc3184ffe"
kbets = SelectKBest(f_classif, k=min(30, X.shape[1]))

X_scaled = X / 255
X_new = kbets.fit_transform(X, y)

selected_features = np.array(features_names)[kbets.get_support()]
print(f"Selected features based on Z-score: {selected_features}")

# %% id="mlfJguiwYH5p"
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=42, n_estimators=50)

# %% colab={"base_uri": "https://localhost:8080/", "height": 988} id="whLbZOVUZFdn" outputId="348f3729-2b18-4a90-9d02-ad1a204a9ddf"
# evaluate model
X_train, X_test, y_train, y_test = split_data(X_new, y)
evaluate_model(model, X_train, y_train, X_test, y_test)

# %% [markdown] id="SEdaNXC5ZbQw"
# ### Apply dataset with Dimension Reduction Techniques

# %% [markdown] id="4kq6-iSWaYSU"
# ### apply pca

# %% colab={"base_uri": "https://localhost:8080/"} id="68JQ5ApXaPSE" outputId="fdc73dbd-81d9-402d-e043-28b7c7e33c59"
X_new = apply_pca(X, variance_threshold=0.95, standardize=True)

# %% colab={"base_uri": "https://localhost:8080/", "height": 988} id="eTNk7bBBZOuV" outputId="013db535-6693-423f-af22-8f69dd20e0d6"
X_train, X_test, y_train, y_test = split_data(X_new, y)
evaluate_model(model, X_train, y_train, X_test, y_test)

# %% [markdown] id="UCzlYz0eac6R"
# ### Apply LDA

# %% colab={"base_uri": "https://localhost:8080/"} id="LrDD40qUacTF" outputId="38f39c16-9651-4031-b260-aff706bf5922"
X_new, optimal_n = apply_lda(X, y, variance_threshold=0.95, standardize=True)
print(f"Selected {optimal_n} components")

# %% colab={"base_uri": "https://localhost:8080/", "height": 988} id="wEug7ZHSaqId" outputId="6b9100b9-cf6c-4497-d405-479d098efde3"
X_train, X_test, y_train, y_test = split_data(X_new, y)
evaluate_model(model, X_train, y_train, X_test, y_test)

# %% [markdown] id="L5fDeha5axSb"
# ### Apply X_tsne

# %% id="3pYx0evIa2_3"
X_new = apply_tsne(X, n_components='auto')

# %% colab={"base_uri": "https://localhost:8080/", "height": 988} id="829cAprZa6Qb" outputId="2aa2351a-a950-4458-9458-64f6126ded63"
X_train, X_test, y_train, y_test = split_data(X_new, y)
evaluate_model(model, X_train, y_train, X_test, y_test)

# %% [markdown] id="NxYM12wHbITZ"
# **# Best Dimension Reduction Techinque Applyed on this data set is `X_tsne`**
