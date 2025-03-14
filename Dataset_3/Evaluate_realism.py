# %%
import numpy as np 
import pandas as pd 
import time 
import random 
import os 
import matplotlib.pyplot as plt  
import torch 
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch
import math
import sys 
import sklearn.model_selection as skl

# %%
import numpy as np
import pandas as pd
import os
import time
from scipy.stats import ks_2samp, wasserstein_distance, norm
from sklearn.metrics.pairwise import rbf_kernel
from scipy import stats
# Define dataset directory


DATASET_DIRECTORY = "C:/Users/Mega Pc/Desktop/SF-GAN unsupervised/Dataset_3/"
REAL_DATA_PATH = os.path.join(DATASET_DIRECTORY, "preprocessed_obese_data.csv")
SYNTHETIC_DATA_PATH = os.path.join(DATASET_DIRECTORY, 'gen_samp_rho84.90_epsilon147.43.csv')

# Load datasets
df_real = pd.read_csv(REAL_DATA_PATH)
df_synthetic = pd.read_csv(SYNTHETIC_DATA_PATH) 

# Ensure column consistency
assert list(df_real.columns) == list(df_synthetic.columns), "Columns of real and synthetic datasets do not match."

# Compute Dimension-wise Probability (Fairness Evaluation)
def compute_dimensionwise_probability(real_data, synthetic_data):
    results = {}
    for column in real_data.columns:
        ks_stat, p_value = ks_2samp(real_data[column], synthetic_data[column])
        results[column] = {'ks_stat': ks_stat, 'p_value': p_value}
    return results

# Compute MMD (Maximum Mean Discrepancy)
def compute_mmd(real_data, synthetic_data, gamma=0.5):
    K_real = rbf_kernel(real_data, real_data, gamma=gamma)
    K_synth = rbf_kernel(synthetic_data, synthetic_data, gamma=gamma)
    K_cross = rbf_kernel(real_data, synthetic_data, gamma=gamma)
    return K_real.mean() + K_synth.mean() - 2 * K_cross.mean()

# Compute Wasserstein Distance

def compute_wasserstein_distance(real_data, synthetic_data):
    distances = []
    for column in real_data.columns:
        real_col = real_data[column].dropna()
        synth_col = synthetic_data[column].dropna()
        
        if not real_col.empty and not synth_col.empty:
            distances.append(wasserstein_distance(real_col, synth_col))
    
    return np.mean(distances) if distances else np.nan  # Returning NaN for consistency



# Compute Dimension-wise Probability Score
def dimension_wise_probability(real_data, synthetic_data, epsilon=1e-10):
    probabilities = []
    for column in real_data.columns:
        real_mean, real_std = norm.fit(real_data[column])
        real_std = max(real_std, epsilon)
        prob = norm.cdf(synthetic_data[column], loc=real_mean, scale=real_std)
        probabilities.append(np.mean(prob))
    return np.mean(probabilities)

# Run evaluations
start_time = time.time()
dwp_results = compute_dimensionwise_probability(df_real, df_synthetic)
mmd_value = compute_mmd(df_real.to_numpy(), df_synthetic.to_numpy())
wasserstein_value = compute_wasserstein_distance(df_real, df_synthetic)
dwp_score = dimension_wise_probability(df_real, df_synthetic)
end_time = time.time()

pvalue_acum = 0
for i in range(df_real.shape[1]):  # Iterating through each feature/column
    real_column = df_real.iloc[:, i]
    synthetic_column = df_synthetic.iloc[:, i]
    stat, pvalue = stats.ks_2samp(real_column, synthetic_column)
    pvalue_acum += pvalue

average_pvalue = pvalue_acum / float(df_real.shape[1])  # Averaging p-values across features


# Print results
print("\nEvaluation Results:")
print(f"MMD Value: {mmd_value:.4f}")
print(f"Dimension-wise Probability Score: {dwp_score:.4f}")
print('KS test average p-value:', average_pvalue)
print('Wasserstein distance: ',wasserstein_value)
#(f"Execution Time: {end_time - start_time:.2f} seconds\n")



# %%


# %%
from scipy.stats import ks_2samp
from sklearn.metrics.pairwise import rbf_kernel
def compute_dimensionwise_probability(real_data, synthetic_data, sensitive_attributes):
    """
    Compute Dimension-wise Probability to assess fairness in synthetic data clustering.
    
    Parameters:
    - real_data (pd.DataFrame): Real-world dataset.
    - synthetic_data (pd.DataFrame): Synthetic dataset.
    - sensitive_attributes (list): List of sensitive attributes (e.g., ['Gender', 'Ethnicity']).
    
    Returns:
    - dwp_results (dict): KS statistic and p-value for each sensitive attribute.
    """
    dwp_results = {}
    
    for attr in sensitive_attributes:
        # Extract sensitive attribute columns
        real_attr = real_data[attr].values
        synthetic_attr = synthetic_data[attr].values
        
        # Perform KS test
        ks_stat, p_value = ks_2samp(real_attr, synthetic_attr)
        dwp_results[attr] = {
            'ks_stat': ks_stat,
            'p_value': p_value
        }
        
    return dwp_results






# %%
"""import pandas as pd

# Load datasets
dataset_directory = "C:/Users/Mega Pc/Desktop/SF-GAN unsupervised/Dataset_1/"
df_real = pd.read_csv(dataset_directory + "preprocessed_HIV.csv")
df_synthetic = pd.read_csv(dataset_directory + 'gen_samp4.csv').drop("Unnamed: 0", axis=1)

# Convert to numpy arrays for MMD calculation
real_data_latent = df_real.to_numpy()
synthetic_data_latent = df_synthetic.to_numpy()


# Compute Dimension-wise Probability (Fairness Evaluation)
sensitive_attributes = ['Gender', 'Ethnic_2.0', 'Ethnic_3.0', 'Ethnic_4.0']
 # Update based on your dataset columns
dwp_results = compute_dimensionwise_probability(df_real, df_synthetic, sensitive_attributes)

print("\nDimension-wise Probability Results:")
for attr, results in dwp_results.items():
    print(f"{attr}: KS Statistic = {results['ks_stat']:.4f}, p-value = {results['p_value']:.4f}")
"""

# %%
import pandas as pd
from scipy.stats import ks_2samp

# Load datasets
dataset_directory = "C:/Users/Mega Pc/Desktop/SF-GAN unsupervised/Dataset_3/"
df_real = pd.read_csv(dataset_directory + "preprocessed_obese_data.csv")
df_synthetic = pd.read_csv(dataset_directory + 'gen_samp_rho2089.79_epsilon2400.01.csv')

# Ensure both datasets have the same columns
assert list(df_real.columns) == list(df_synthetic.columns), "Columns of real and synthetic datasets do not match."

# Compute Dimension-wise Probability (Fairness Evaluation) for all features
def compute_dimensionwise_probability(real_data, synthetic_data):
    results = {}
    for column in real_data.columns:
        real_col = real_data[column]
        synthetic_col = synthetic_data[column]

        # Apply Kolmogorov-Smirnov test
        ks_stat, p_value = ks_2samp(real_col, synthetic_col)

        results[column] = {
            'ks_stat': ks_stat,
            'p_value': p_value
        }
    return results

# Run the computation
dwp_results = compute_dimensionwise_probability(df_real, df_synthetic)

# Print results
print("\nDimension-wise Probability Results:")
for attr, results in dwp_results.items():
    print(f"{attr}: KS Statistic = {results['ks_stat']:.4f}, p-value = {results['p_value']:.4f}")


# %%
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel

df = pd.read_csv(dataset_directory + "preprocessed_obese_data.csv")
dfr = pd.read_csv(dataset_directory + 'gen_samp_rho2089.79_epsilon2400.01.csv')

# Convert data to numpy arrays
real_data_latent = df.to_numpy()
synthetic_data_latent = dfr.to_numpy()

# Function for MMD computation
def compute_mmd(real_data, synthetic_data, gamma=1.0):
    """
    Compute Maximum Mean Discrepancy (MMD) between real and synthetic data.
    """
    K_real = rbf_kernel(real_data, real_data, gamma=gamma)
    K_synth = rbf_kernel(synthetic_data, synthetic_data, gamma=gamma)
    K_cross = rbf_kernel(real_data, synthetic_data, gamma=gamma)
    
    # Compute MMD
    mmd = K_real.mean() + K_synth.mean() - 2 * K_cross.mean()
    return mmd

# Compute MMD
gamma = 0.5  # Adjust gamma for the RBF kernel as needed
mmd_value = compute_mmd(real_data_latent, synthetic_data_latent, gamma=gamma)
print(f"MMD Value: {mmd_value}")


# %%
import pandas as pd
from scipy import stats

# Dataset paths
dataset_directory = "C:/Users/Mega Pc/Desktop/SF-GAN unsupervised/Dataset_3/"
df_real = pd.read_csv(dataset_directory + "preprocessed_obese_data.csv")
df_synthetic = pd.read_csv(dataset_directory + 'gen_samp_rho2089.79_epsilon2400.01.csv')
# Kolmogorov-Smirnov test
# Assuming both datasets have the same number of features and are aligned
pvalue_acum = 0
for i in range(df_real.shape[1]):  # Iterating through each feature/column
    real_column = df_real.iloc[:, i]
    synthetic_column = df_synthetic.iloc[:, i]
    stat, pvalue = stats.ks_2samp(real_column, synthetic_column)
    pvalue_acum += pvalue

average_pvalue = pvalue_acum / float(df_real.shape[1])  # Averaging p-values across features
print('KS test average p-value:', average_pvalue)


# %%
import pandas as pd
from scipy.stats import wasserstein_distance

# Example dataframes
# Gen_df = pd.DataFrame(...)  # Your generated data
# df = pd.DataFrame(...)  # Your real data
dataset_directory = "C:/Users/Mega Pc/Desktop/SF-GAN unsupervised/Dataset_3/"
real_data_path = dataset_directory + "preprocessed_obese_data.csv"
synthetic_data_path = dataset_directory + 'gen_samp_rho2089.79_epsilon2400.01.csv'

# Load datasets
df_real = pd.read_csv(real_data_path)
df_synthetic = pd.read_csv(synthetic_data_path)
# Ensure the dataframes have the same columns
assert df_synthetic.columns.equals(df.columns), "Dataframes must have the same columns"

# Perform Wasserstein Distance calculation for each column
wasserstein_results = {}
distances = []

for column in df_synthetic.columns:
    # Convert columns to numeric, coercing errors to NaN and dropping them
    gen_col = pd.to_numeric(df_synthetic[column], errors='coerce').dropna()
    real_col = pd.to_numeric(df[column], errors='coerce').dropna()
    
    # Check if columns are not empty
    if len(gen_col) > 0 and len(real_col) > 0:
        # Calculate Wasserstein Distance
        distance = wasserstein_distance(gen_col, real_col)
        wasserstein_results[column] = distance
        distances.append(distance)
    else:
        wasserstein_results[column] = None
        print(f"Skipping Wasserstein Distance calculation for {column} due to empty data after conversion and dropping NaNs.")

# Calculate overall Wasserstein Distance
overall_distance = np.mean(distances) if distances else None

# Print the results
for column, distance in wasserstein_results.items():
    if distance is not None:
        #print(f"Wasserstein Distance for {column}: {distance:.3f}")
        distance1=distance
    else:
        print(f"Wasserstein Distance for {column}: Skipped due to empty data.")

if overall_distance is not None:
    print(f"\nOverall Wasserstein Distance: {overall_distance:.3f}")
else:
    print("\nOverall Wasserstein Distance: Not enough data to calculate overall distance.")


# %%
import numpy as np
from scipy.stats import norm
dataset_directory = "C:/Users/Mega Pc/Desktop/SF-GAN unsupervised/Dataset_3/"
real_data_path = dataset_directory + "preprocessed_obese_data.csv"
synthetic_data_path = dataset_directory + 'gen_samp_rho2089.79_epsilon2400.01.csv'

# Load datasets
df_real = pd.read_csv(real_data_path)
df_synthetic = pd.read_csv(synthetic_data_path)
def dimension_wise_probability(real_data, synthetic_data, epsilon=1e-10):
    probabilities = []
    for column in real_data.columns:
        real_mean, real_std = norm.fit(real_data[column])
        synthetic_mean, synthetic_std = norm.fit(synthetic_data[column])
        real_std = max(real_std, epsilon)  # Avoid division by zero
        prob = norm.cdf(synthetic_data[column], loc=real_mean, scale=real_std)
        probabilities.append(np.mean(prob))
    return np.mean(probabilities)

# Example usage
dwp_result = dimension_wise_probability(df_real, df_synthetic)
print("Dimension-wise Probability:", dwp_result)


# %%
df_synthetic.shape


