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

# %% id="BCx7NhwUTuFj" colab={"base_uri": "https://localhost:8080/", "height": 73} outputId="9aea35f6-c380-47b5-8bfa-93870d29ed88"
# prompt: code to upload dataset from local machine

from google.colab import files
uploaded = files.upload()

# %% id="R6kTH-N9VS2t"
import io
import pandas as pd
df = pd.read_csv(io.BytesIO(uploaded['USA_Housing.csv']))

# %% colab={"base_uri": "https://localhost:8080/", "height": 276} id="sewHbM2KtVML" outputId="2fe5bad6-0024-4d25-bc3f-2d94302d42f9"
df.head()

# %% colab={"base_uri": "https://localhost:8080/"} id="h5YXqULRtXcj" outputId="3fce9488-6cb8-4468-d121-3b71d9ea1d1c"
df.info()

# %% colab={"base_uri": "https://localhost:8080/", "height": 300} id="Xn_pXuaBtZfS" outputId="a295d161-640a-42fe-cc36-0e756c41e611"
df.describe()

# %% id="M5Gej6JUxIeD"
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="v0OHXWVAtdpt" outputId="17de8d3a-d057-4a41-8a67-7e7c7a3f29fc"
#  showing relationships between multiple variables in a dataset

sns.pairplot(df)
plt.show()

# %% [markdown] id="oQU3QKEWysuo"
# - relation between all features and `number of bedrooms` feature is const, can ignore this featuer

# %% colab={"base_uri": "https://localhost:8080/", "height": 449} id="wlTgl85Evfsw" outputId="d1dcb18b-5494-4b2a-f571-5e36ed462b92"
# distribution for house pricing
sns.histplot(df['Price'],kde=True)
plt.show()

# %% colab={"base_uri": "https://localhost:8080/", "height": 641} id="2xC4iC-YwhYf" outputId="f244b68d-9eb7-432d-a461-64d3acb35f15"
# visulaize corrolection betweeen featuers
numerical_columns = df.select_dtypes(include=[np.number]).columns
sns.heatmap(df[numerical_columns].corr(), annot=True,cmap='coolwarm')
plt.show()

# %% [markdown] id="K9889GkoxkDD"
# - corrlation between `price` and `Avg. Area Number of Bedrooms` is very low --> we can remove this feature and not influnce our target  

# %% [markdown] id="PB941tmP0gNg"
# ### Detect Outlier
# - Linear regression is highly sensitive to outliers

# %% colab={"base_uri": "https://localhost:8080/"} id="WZNsCaET00jS" outputId="ee22fda4-859f-4e3d-90fb-98dea33c929d"
features_contain_outliers = []
# Calculate quartiles and IQR for each numerical feature
for col in numerical_columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    print(f"Outliers for {col}:")
    print(f"Lower bound: {lower_bound}, Upper bound: {upper_bound}")
    print(f"Number of outlier in col {col} = {outliers[[col]].sum()}")  # Display only the outlier values for the current feature
    if not outliers.empty:
        features_contain_outliers.append(col)
    print("-" * 20)


# %% colab={"base_uri": "https://localhost:8080/"} id="pTMgNbBX5J90" outputId="528d2fc7-ad6d-4033-d371-7029aa9df554"
features_contain_outliers

# %% colab={"base_uri": "https://localhost:8080/", "height": 299} id="iEbGs7Jx5W1n" outputId="2efaef94-4d71-4ffa-c2b2-fa0c0020d6e0"
plt.figure(figsize=(20, 10))

# Iterate through numerical columns and create boxplots
for i, col in enumerate(features_contain_outliers):
    plt.subplot(2, 4, i + 1)  # Create subplots in 2 rows and 4 columns
    sns.boxplot(y=df[col])
    plt.title(col)

plt.tight_layout()  # Adjust spacing between subplots
plt.show()


# %% colab={"base_uri": "https://localhost:8080/", "height": 299} id="qJtirSua59ZU" outputId="8a7871d3-0448-4b98-ad49-48b59a726778"

# Function to replace outliers with the upper or lower bound
def impute_outliers_iqr(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
    df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

    return df

# Impute outliers for each feature identified
for col in features_contain_outliers:
    df = impute_outliers_iqr(df, col)

# Now you can verify that the outliers have been imputed.
# Display boxplots again to show the effect of outlier imputation.
plt.figure(figsize=(20, 10))
for i, col in enumerate(features_contain_outliers):
    plt.subplot(2, 4, i + 1)
    sns.boxplot(y=df[col])
    plt.title(col)
plt.tight_layout()
plt.show()


# %% colab={"base_uri": "https://localhost:8080/"} id="b-gCTqOY6Nok" outputId="c2180a0d-3831-4451-9bf1-543bb579b42d"
df.columns

# %% id="z98PklEv6I9m"
# remove `Address` feature no need it
df.drop('Address',axis=1,inplace=True)

# %% id="D9H4hI1K6TKf"
X = df.drop('Price',axis=1)
y = df['Price']

# %% id="RQnZDrkX6mbJ"
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# %% id="FrA0s2XK6rnV"
# prompt: code to normalize dataset

from sklearn.preprocessing import MinMaxScaler

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Fit the scaler on the training data and transform both training and testing data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert the scaled data back to Pandas DataFrames (optional, but recommended)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)


# %% colab={"base_uri": "https://localhost:8080/", "height": 80} id="yzvVj_BR7a2I" outputId="0b964c3c-92d8-4458-dd39-d35d90bb9e24"
from sklearn.linear_model import LinearRegression

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# %% colab={"base_uri": "https://localhost:8080/"} id="hV0ejW-77oYL" outputId="73c4f208-ec4e-449b-a2f1-9cccda981d50"
print(model.intercept_)

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} id="SxGH-fDs7r9p" outputId="9b77c466-a797-4033-91a4-e0ba52127e44"
coeff_df = pd.DataFrame(model.coef_,X.columns,columns=['Coefficient'])
coeff_df

# %% id="0Y0J3fG27kmx"
# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# %% colab={"base_uri": "https://localhost:8080/", "height": 524} id="pZo2y21N7jtX" outputId="6b6bbf2d-e0c2-4d87-874b-d41f52d1ba98"
# Evaluate the model
from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Visualize the predictions vs. actual values
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs. Predicted House Prices")
plt.show()


# %% colab={"base_uri": "https://localhost:8080/", "height": 472} id="bvj0jLYz8s_k" outputId="05219677-7123-4835-94ad-47ca57111ad8"
plt.scatter(y_test,y_pred, color='red')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')
plt.show()

# %% colab={"base_uri": "https://localhost:8080/"} id="SGb3WV669Tfi" outputId="8275f889-d07e-4d38-fd46-46bc504a5493"
# prompt: print r2 matrix

from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)
print(f"R-squared (R2): {r2}")


# %% colab={"base_uri": "https://localhost:8080/"} id="sN6rKuvi_bm4" outputId="e2e54b90-671b-4a66-d5b0-c11bb61aab13"

# Scale y_test and y_pred
scaler_y = MinMaxScaler()
y_test_scaled = scaler_y.fit_transform(y_test.values.reshape(-1, 1))
y_pred_scaled = scaler_y.transform(y_pred.reshape(-1, 1))

# Calculate scaled MSE, MAE and R-squared
mae_scaled = mean_absolute_error(y_test_scaled, y_pred_scaled)
mse_scaled = mean_squared_error(y_test_scaled, y_pred_scaled)
rmse_scaled = np.sqrt(mse_scaled)
r2_scaled = r2_score(y_test_scaled, y_pred_scaled)

print(f"Scaled Mean Absolute Error (MAE): {mae_scaled}")
print(f"Scaled Mean Squared Error (MSE): {mse_scaled}")
print(f"Scaled Root Mean Squared Error (RMSE): {rmse_scaled}")
print(f"Scaled R-squared (R2): {r2_scaled}")

