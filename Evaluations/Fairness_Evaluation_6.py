# %%
import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import mutual_info_score, silhouette_score, davies_bouldin_score
from sklearn.mixture import GaussianMixture
import pandas as pd
import numpy as np



# %%

# Load the generated synthetic dataset
dataset_dir = "C:/Users/Mega Pc/Desktop/SF-GAN unsupervised/Dataset_6/Artificial_samples/"
file_path = os.path.join(dataset_dir, 'gen_samp_WO_Adv_deb_epsilon50.35.csv')
df = pd.read_csv(file_path)
# Load the original dataset to retrieve scaling information for 'Age'
dfo = pd.read_csv("C:/Users/Mega Pc/Desktop/SF-GAN unsupervised/Dataset_6/Diabeties.csv")

# %%

# Retrieve original min and max Age from dfo
original_min_age = dfo['Age'].min()
original_max_age = dfo['Age'].max()

# Denormalize Age in synthetic data
df['Age'] = df['Age'] * (original_max_age - original_min_age) + original_min_age

# Convert Age to categorical groups
def categorize_age(age):
    if age < 18:
        return '14-17'
    elif 18 <= age < 25:
        return '18-24'
    elif 25 <= age < 30:
        return '25-29'
    elif 30 <= age < 40:
        return '30-39'
    elif 40 <= age < 50:
        return '40-49'
    elif 50 <= age < 61:
        return '50-60'
    else:
        return '61 and above'


df['Age_Group'] = df['Age'].apply(categorize_age)

# Ensure protected attributes are binary
protected_attributes = ['Sex', 'Age_Group']
for attr in ['Sex']:
    df[attr] = df[attr].apply(lambda x: 1 if x > 0 else 0)
num_clusters = 450
gmm = KMeans(n_clusters=num_clusters, random_state=42)
features = df.drop(columns=protected_attributes + ['Age'])  # Exclude sensitive and derived attributes
df['Cluster'] = gmm.fit_predict(features)

# 1. Statistical Parity of Clusters
def compute_statistical_parity(df, sensitive_attributes):
    parity_results = {}
    parity_classification = {}

    for attr in sensitive_attributes:
        stat_parity_table = pd.crosstab(df[attr], df['Cluster'], normalize='index')
        parity_results[attr] = stat_parity_table

        min_ratio = stat_parity_table.min().min()
        max_ratio = stat_parity_table.max().max()

        if max_ratio - min_ratio < 0.05:
            fairness_label = "Evenly Spread"
        elif max_ratio - min_ratio < 0.15:
            fairness_label = "Balanced"
        else:
            fairness_label = "Unbalanced"

        parity_classification[attr] = fairness_label

    return parity_results, parity_classification

stat_parity, parity_classification = compute_statistical_parity(df, protected_attributes)

print("### Statistical Parity ###")
for attr, table in stat_parity.items():
    print(f"\nStatistical Parity for {attr} ({parity_classification[attr]}):")
    print(table)

# 2. Mutual Information (MI) between Clusters and Sensitive Attributes
mi_results = {attr: mutual_info_score(df[attr], df['Cluster']) for attr in protected_attributes}
print("\n### Mutual Information Scores ###")
for attr, score in mi_results.items():
    print(f"Mutual Information (MI) between {attr} and Clusters: {score:.4f}")

# 3. Clustering Quality Metrics
silhouette = silhouette_score(features, df['Cluster'])
dbi = davies_bouldin_score(features, df['Cluster'])
print("\n### Clustering Quality Metrics ###")
print(f"Silhouette Score: {silhouette:.4f}")
print(f"Davies-Bouldin Index: {dbi:.4f}")




