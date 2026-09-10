# %%
import numpy as np
import pandas as pd
import os
import time
from scipy.stats import ks_2samp, wasserstein_distance, norm
from sklearn.metrics.pairwise import rbf_kernel
from scipy import stats
# Define dataset directory
DATASET_DIRECTORY = "C:/Users/Mega Pc/Desktop/SF-GAN unsupervised/Dataset_5/"
REAL_DATA_PATH = os.path.join(DATASET_DIRECTORY, "Regensburg_Pediatric_Preprocessed.csv")
SYNTHETIC_DATA_PATH = os.path.join(DATASET_DIRECTORY, "Artificial_samples/gen_samp_rho0.37_epsilon4.51.csv")

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
print('wasserstein_distance', wasserstein_value)
#(f"Execution Time: {end_time - start_time:.2f} seconds\n")



# %%
import matplotlib.pyplot as plt
import numpy as np

# Privacy budgets
privacy_budgets = [10, 100, 1000, 100000]

# Example MMD scores for each method under the corresponding budgets
mmd_rdpcgan = [0.095, 0.070, 0.055, 0.048]
mmd_dpgan   = [0.090, 0.065, 0.052, 0.045]
mmd_ours    = [0.085, 0.060, 0.050, 0.042]

plt.figure(figsize=(6,4))

plt.plot(privacy_budgets, mmd_rdpcgan, marker='o', label='RDP-CGAN')
plt.plot(privacy_budgets, mmd_dpgan,   marker='s', label='DPGAN')
plt.plot(privacy_budgets, mmd_ours,    marker='^', label='Our Method')

# Convert 'inf' to a string for the x-axis label
x_labels = ['10', '100', '1000', 'inf']
plt.xticks(privacy_budgets, x_labels)

plt.xlabel('Privacy Budget (ε)')
plt.ylabel('MMD')
plt.title('MMD under Different Privacy Budgets')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



