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
import matplotlib.pyplot as plt

# %%
import warnings
warnings.filterwarnings('ignore')

# %%
# Read dataset
df = pd.read_csv("Live.csv")

# %%
# explore dataset
df.head(10)

# %%
# get info about dataset
df.info()

# %%
# get features that have more than 50 unique values
ignored_features = []
for col in df.columns:
    if df[col].nunique() > 50 and df[col].dtype != 'int64':
        ignored_features.append(col)
ignored_features

# %%
# Remove ignored features
df.drop(columns=ignored_features, axis=1, inplace=True)

# %%
df.info()

# %%
# get features that have missing values
features_with_null = []
for col in df.columns:
    if df[col].isnull().sum() > 0:
        features_with_null.append(col)
features_with_null

# %%
# operations on features with null values
for feature in features_with_null:
    print(f"value count for feature {df[feature].value_counts()}")
    print("---" * 10)

# %%
# drop the feature with null values 
df.drop(columns=features_with_null, axis=1, inplace=True)

# %%
df.head()

# %%
df.info()

# %%
# some statistics analyzer
df.describe()

# %%
columns_names = []
numerical_features = []
categorical_features = []
for col in df.columns:
    columns_names.append(col)
    if df[col].dtype == 'O':
        categorical_features.append(col)
    else:
        numerical_features.append(col)
print(columns_names)
print(numerical_features)
print(categorical_features)

# %%
import seaborn as sns

# %%
# checking for outliers in numeric_features and treating it

# for feature in numerical_features:
#     Q1 = df[feature].quantile(.25)
#     Q3 = df[feature].quantile(.75)
#     IQR = Q3 - Q1
#     
#     # outlier boundaries
#     lower_bound = Q1 - 1.5 * IQR
#     upper_bound = Q1 + 1.5 * IQR
#     
#     outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]
#     
#     # print number of outliers
#     print(f"Number of outlier in feature : {feature} : {outliers.shape[0]}")
#     
#     # Visualize the outliers using a boxplot
#     plt.figure(figsize=(10, 5))
#     sns.boxplot(x=df[feature])
#     plt.title("Boxplot of {feature}")
#     plt.show()
#     
#     # treating outliers
#     df[feature].clip(lower_bound, upper_bound, inplace=True)
#     
#     # Visualize the outliers using a boxplot
#     plt.figure(figsize=(10, 5))
#     sns.boxplot(x=df[feature])
#     plt.title("Boxplot of {feature}")
#     plt.show()
    

# %%
# encode categorical features using one-Hot ending
# df = pd.get_dummies(df, columns=categorical_features)

# %%
# encode categorical features using Label encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['status_type'] = le.fit_transform(df['status_type'])

# %%
df.head()

# %%
columns_name = df.columns

# %%
print(columns_names)

# %%
#feature scaling
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

df = scaler.fit_transform(df)    

# %%
df = pd.DataFrame(df, columns=columns_name)

# %%
df.head()

# %%
# build model using K-means and GridSearch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import ParameterGrid

# Define hyperparameter grid
param_grid = {
    'n_clusters': [2, 3, 4, 5, 6],  # Number of clusters
    'init': ['k-means++', 'random'],  # Initialization methods
    'max_iter': [300, 500]           # Maximum number of iterations
}

# Initialize variables to store the best results
best_params = None
best_score = -1
results = []

# Perform manual grid search
for params in ParameterGrid(param_grid):
    # Create a KMeans model with the current parameters
    kmeans = KMeans(
        n_clusters=params['n_clusters'],
        init=params['init'],
        max_iter=params['max_iter'],
        random_state=42
    )
    
    # Fit the model
    kmeans.fit(df)
    
    # Calculate silhouette score (higher is better)
    if params['n_clusters'] > 1:  # Silhouette is undefined for 1 cluster
        score = silhouette_score(df, kmeans.labels_)
        results.append((params, score))
        
        # Update best score and parameters
        if score > best_score:
            best_score = score
            best_params = params

# Display the best parameters and score
print("Best Parameters:", best_params)
print("Best Silhouette Score:", best_score)

# Optionally: Convert results to a DataFrame for better readability
results_df = pd.DataFrame(results, columns=['Parameters', 'Silhouette Score'])
print(results_df.sort_values(by='Silhouette Score', ascending=False))

# %%
from sklearn.cluster import KMeans
cs = []
for i in range(1, 21):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(df)
    cs.append(kmeans.inertia_)

plt.plot(range(1, 21), cs)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('CS')
plt.show()

# %% [markdown]
# ## best number of clusters is `3`
