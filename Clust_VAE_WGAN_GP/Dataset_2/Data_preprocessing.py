# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# %%
# Load the dataset
df = pd.read_csv("corporate_stress_dataset.csv")

# Drop the ID column if it exists
if 'ID' in df.columns:
    df.drop(columns=['ID'], inplace=True)
    print("ID column removed.")


# %%
print("Dataset Shape:", df.shape)
print("\nDataset Head:")
print(df.head())
print("\nDataset Info:")
df.info()

# %%
# Checking for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Descriptive Statistics
print("\nDescriptive Statistics (Numerical Features):")
print(df.describe())

print("\nDescriptive Statistics (Categorical Features):")
print(df.describe(include=['object']))


# %%
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
for col in numerical_features:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()

# %%
categorical_features = df.select_dtypes(include=['object']).columns
for col in categorical_features:
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, x=col, order=df[col].value_counts().index)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.xticks(rotation=45)
    plt.show()

# %%
# 3. Boxplots for outlier detection
for col in numerical_features:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()

# %%
# Handling Missing Values
# Fill numerical missing values with median and categorical with mode
for col in numerical_features:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

for col in categorical_features:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)

print("\nMissing Values After Imputation:")
print(df.isnull().sum())

# %%
df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)
boolean_features = df.select_dtypes(include=['bool']).columns
for col in boolean_features:
    df_encoded[col] = df_encoded[col].astype(int)
# Feature Scaling with Min-Max Scaler
scaler = MinMaxScaler()
df_scaled = df_encoded.copy()
df_scaled[numerical_features] = scaler.fit_transform(df_encoded[numerical_features])


# %%
for col in numerical_features:
    Q1 = df_scaled[col].quantile(0.25)
    Q3 = df_scaled[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Replace outliers with boundaries
    df_scaled[col] = np.where(df_scaled[col] < lower_bound, lower_bound,
                              np.where(df_scaled[col] > upper_bound, upper_bound, df_scaled[col]))


# %%
df_sampled = df_scaled.sample(n=3000, random_state=42)

# Save preprocessed data
df_sampled.to_csv("corporate_stress_dataset_preprocessed.csv", index=False)
print("\nPreprocessed data with 3000 samples saved as 'corporate_stress_dataset_preprocessed.csv'")

# %%
df_sampled

# %%
# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
df = pd.read_csv("corporate_stress_dataset.csv")

# Drop the ID column if it exists
if 'ID' in df.columns:
    df.drop(columns=['ID'], inplace=True)
    print("ID column removed.")

# Data Overview
print("Dataset Shape:", df.shape)
print("\nDataset Head:")
print(df.head())
print("\nDataset Info:")
df.info()

# Checking for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Descriptive Statistics
print("\nDescriptive Statistics (Numerical Features):")
print(df.describe())

print("\nDescriptive Statistics (Categorical and Boolean Features):")
print(df.describe(include=['object', 'bool']))

# Handling Missing Values
# Fill numerical missing values with median and categorical/boolean with mode
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
categorical_features = df.select_dtypes(include=['object']).columns
boolean_features = df.select_dtypes(include=['bool']).columns

for col in numerical_features:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

for col in categorical_features:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)

for col in boolean_features:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)

print("\nMissing Values After Imputation:")
print(df.isnull().sum())

# Encoding Categorical and Boolean Variables
# Encode categorical features using one-hot encoding
df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)

# Encode boolean features (convert True/False to 1/0)
for col in boolean_features:
    df_encoded[col] = df_encoded[col].astype(int)

# Feature Scaling with Min-Max Scaler
scaler = MinMaxScaler()
df_scaled = df_encoded.copy()
df_scaled[numerical_features] = scaler.fit_transform(df_encoded[numerical_features])

# Confirm all boolean columns are now integers
for col in boolean_features:
    assert df_scaled[col].dtype == np.int64 or df_scaled[col].dtype == np.int32, f"{col} is not encoded properly"

print("\nData after Scaling and Boolean Encoding:")
print(df_scaled.head())

# Outlier Detection and Treatment
# Using the IQR method
for col in numerical_features:
    Q1 = df_scaled[col].quantile(0.25)
    Q3 = df_scaled[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Replace outliers with boundaries
    df_scaled[col] = np.where(df_scaled[col] < lower_bound, lower_bound,
                              np.where(df_scaled[col] > upper_bound, upper_bound, df_scaled[col]))

# Selecting a subset of 3000 samples
df_sampled = df_scaled.sample(n=3000, random_state=42)

# Save preprocessed data
df_sampled.to_csv("corporate_stress_dataset_preprocessed.csv", index=False)
print("\nPreprocessed data with 3000 samples saved as 'corporate_stress_dataset_preprocessed.csv'")


# %%
# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
df = pd.read_csv("corporate_stress_dataset.csv")

# Drop the ID column if it exists
if 'ID' in df.columns:
    df.drop(columns=['ID'], inplace=True)
    print("ID column removed.")

# Data Overview
print("Dataset Shape:", df.shape)
print("\nDataset Head:")
print(df.head())
print("\nDataset Info:")
df.info()

# Checking for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Descriptive Statistics
print("\nDescriptive Statistics (Numerical Features):")
print(df.describe())

print("\nDescriptive Statistics (Categorical and Boolean Features):")
print(df.describe(include=['object', 'bool']))

# Handling Missing Values
# Fill numerical missing values with median and categorical/boolean with mode
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
categorical_features = df.select_dtypes(include=['object']).columns
boolean_features = df.select_dtypes(include=['bool']).columns

for col in numerical_features:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

for col in categorical_features:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)

for col in boolean_features:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)

print("\nMissing Values After Imputation:")
print(df.isnull().sum())

# Encoding Categorical and Boolean Variables
# Encode categorical features using one-hot encoding
df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)

# Encode boolean features (convert True/False to 1/0)
for col in boolean_features:
    df_encoded[col] = df_encoded[col].astype(int)

# Feature Scaling with Min-Max Scaler
scaler = MinMaxScaler()
df_scaled = df_encoded.copy()
df_scaled[numerical_features] = scaler.fit_transform(df_encoded[numerical_features])

# Confirm all boolean columns are now integers
for col in boolean_features:
    assert df_scaled[col].dtype == np.int64 or df_scaled[col].dtype == np.int32, f"{col} is not encoded properly"

print("\nData after Scaling and Boolean Encoding:")
print(df_scaled.head())

# Outlier Detection and Treatment
# Using the IQR method
for col in numerical_features:
    Q1 = df_scaled[col].quantile(0.25)
    Q3 = df_scaled[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Replace outliers with boundaries
    df_scaled[col] = np.where(df_scaled[col] < lower_bound, lower_bound,
                              np.where(df_scaled[col] > upper_bound, upper_bound, df_scaled[col]))

# Selecting a subset of 3000 samples
df_sampled = df_scaled.sample(n=3000, random_state=42)

# Save preprocessed data
df_sampled.to_csv("corporate_stress_dataset_preprocessed.csv", index=False)
print("\nPreprocessed data with 3000 samples saved as 'corporate_stress_dataset_preprocessed.csv'")



