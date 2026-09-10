# Unsupervised Cluster-Based VAE–WGAN-GP for Fair and Privacy-Preserving Synthetic Data

This repository contains the research implementation of a hybrid generative framework for producing synthetic tabular data in unsupervised settings. The framework combines:

- a **Variational Autoencoder (VAE)** to learn a compact latent representation;
- **K-Means clustering** in the latent space to preserve unsupervised data structure;
- a **zero-Concentrated Differential Privacy (zCDP)** accounting component and noisy, clipped optimization in the VAE stage;
- a **Wasserstein GAN with Gradient Penalty (WGAN-GP)** to improve synthetic-data fidelity; and
- an adversarial **Fairness Critic** intended to reduce dependence between generated cluster structure and protected attributes.

The repository also includes an ablation without adversarial debiasing and a separate script for post-generation fairness and clustering evaluation.

## Motivation

Fairness-aware generators commonly require class or outcome labels, which restricts their use in unlabeled settings. Privacy-preserving generators can also reproduce latent associations with sensitive attributes. This project investigates a joint approach in which unsupervised clusters guide generation, zCDP is applied during latent-representation learning, and a fairness adversary penalizes sensitive-attribute predictability.

The intended objective is to balance three properties:

1. **Utility:** retain useful multivariate and cluster structure;
2. **Fairness:** limit associations between generated structure and protected attributes;
3. **Privacy:** bound information leakage through the stated zCDP mechanism and privacy accountant.

## Proposed architecture

![Overview of the proposed Clust-VAE-WGAN-GP architecture](architecture.png)

Place the architecture figure at `architecture.png`. If another filename or format is used, update the Markdown path above.

The pipeline consists of the following stages:

1. **Input and preprocessing**
   - Load a preprocessed tabular CSV file.
   - Scale continuous features to the range expected by the output activation.
   - Encode categorical and protected attributes numerically, typically through binary or one-hot encoding.

2. **Private cluster-based VAE**
   - Encoder: `input -> 512 -> 256 -> (mu, log-variance)`.
   - Latent dimension in the supplied Dataset 1 script: `20`.
   - Decoder: `latent -> 256 -> 512 -> reconstructed input` with a sigmoid output.
   - Training objective: reconstruction loss + KL-divergence term + distance-to-cluster-centroid term.
   - K-Means centroids are updated periodically from the learned latent representations.
   - The optimizer clips gradients and adds Gaussian noise; `zcdp_accountant.py` reports rho and converts it to an `(epsilon, delta)` value.

3. **WGAN-GP generation**
   - The reconstructed VAE output is passed to the adversarial stage.
   - Generator: `feature_dim -> 128 -> 256 -> 128 -> feature_dim`.
   - Data critic: `feature_dim -> 256 -> 128 -> 1`.
   - A gradient penalty with default weight `lambda_gp = 10` is used during critic training.

4. **Adversarial debiasing**
   - Generated samples are assigned to K-Means clusters.
   - One-hot cluster assignments are passed to the Fairness Critic.
   - The supplied Fairness Critic maps `number_of_clusters -> 128 -> 64 -> 2` and attempts to predict configured protected attributes.
   - The full generator objective combines adversarial and fairness terms. The ablation script omits this Fairness Critic and its loss.

5. **Synthetic-data postprocessing and evaluation**
   - Binary columns are thresholded at `0.5`; numerical columns retain continuous values.
   - The evaluation script clusters the synthetic records after excluding protected attributes.
   - It reports protected-group cluster distributions, mutual information, silhouette score, and Davies–Bouldin index.


## 🗂 Datasets

This work leverages multiple datasets, each with a dedicated tailored architecture:

- **Diabetes Health Indicators Dataset** – [Kaggle](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)
- **HIV Dataset** – [HealthGymAi](https://healthgym.ai/antiviral-hiv/)
- **Heart Failure Clinical Records** – [UCI Machine Learning Repository](https://doi.org/10.24432/C5Z89R)
- **Obesity Estimation Dataset** – [UCI Machine Learning Repository](https://doi.org/10.24432/C5H31Z)
- **Regensburg Pediatric Appendicitis Dataset** – [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/938/regensburg+pediatric+appendicitis)
- **Corporate Stress Dataset** – [Kaggle](https://www.kaggle.com/datasets/ankitpatel2100/corporate-stress-dataset-insights-into-workplace)





## Repository structure

```text
Clust_VAE_WGAN_GP/
├── README.md
├── requirements.txt
├── Figures/
│   └── architecture.png
├── Dataset_1/
│   ├── Architecture.py
│   ├── Architecture_WO_Adv_deb.py
│   ├── Fairness_evaluation.py
│   ├── zcdp_accountant.py
│   ├── preprocessed_HIV.csv
│   ├── models/                 # created/used during training
│   └── README.md               # optional dataset-specific documentation
├── Dataset_2/
├── Dataset_3/
├── Dataset_4/
├── Dataset_5/
└── Dataset_6/
```

Each `Dataset_n` directory should follow the same organization. Dataset-specific filenames, protected attributes, continuous columns, sample counts, and hyperparameters must be documented in its local `README.md` or in the dataset table below.

## Main files

| File | Purpose |
|---|---|
| `Architecture.py` | Trains the complete cluster-based VAE + zCDP + WGAN-GP + Fairness Critic pipeline and generates synthetic samples. |
| `Architecture_WO_Adv_deb.py` | Ablation that keeps the cluster-based VAE, privacy component, and WGAN-GP but removes adversarial debiasing. |
| `Fairness_evaluation.py` | Computes protected-group cluster distributions, mutual information, silhouette score, and Davies–Bouldin index on a generated CSV file. |
| `zcdp_accountant.py` | Computes the zCDP privacy parameter and converts it to an `(epsilon, delta)` report. |
| `preprocessed_*.csv` | Dataset-specific model input. Replace the wildcard with the exact documented filename. |

## Data requirements and organization

The training scripts expect a numeric CSV matrix with one row per record and one column per model feature.

Before training:

- remove exported index columns such as `Unnamed: 0`;
- impute or otherwise handle missing values consistently;
- scale numerical features consistently with the sigmoid output used by the current models, normally to `[0, 1]`;
- encode categorical features numerically;
- retain the protected attributes required by the fairness experiment;
- ensure that protected-attribute names in the code match the CSV columns exactly; and
- record every preprocessing step to make the experiment reproducible.

Complete this table with the exact information used in the associated article before public release:

| Folder | Dataset | Input CSV | Protected attributes | Numerical columns | Generated sample count | Data access / licence |
|---|---|---|---|---|---:|---|
| `Dataset_1` | HIV dataset | `preprocessed_HIV.csv` | `Gender`, `Ethnic_2.0`, `Ethnic_3.0`, `Ethnic_4.0` | `VL`, `CD4`, `Rel CD4` | 8,916 in the supplied script | Add source and licence |
| `Dataset_2` | Add dataset name | Add filename | Add attributes | Add columns | Add value | Add source and licence |
| `Dataset_3` | Add dataset name | Add filename | Add attributes | Add columns | Add value | Add source and licence |
| `Dataset_4` | Add dataset name | Add filename | Add attributes | Add columns | Add value | Add source and licence |
| `Dataset_5` | Add dataset name | Add filename | Add attributes | Add columns | Add value | Add source and licence |
| `Dataset_6` | Add dataset name | Add filename | Add attributes | Add columns | Add value | Add source and licence |

Do not commit restricted, identifiable, or otherwise non-redistributable health data. When a dataset cannot be shared, provide its official access page, eligibility conditions, preprocessing description, expected schema, and a small non-sensitive example if permitted.

## Requirements

- Python 3.10 or later is recommended.
- A CUDA-capable GPU is optional; PyTorch falls back to CPU when CUDA is unavailable.
- Install a PyTorch build compatible with the local operating system and CUDA version.

The Python dependencies imported by the supplied scripts are:

```text
numpy
pandas
torch
scikit-learn
scipy
matplotlib
```

A minimal `requirements.txt` can therefore contain:

```text
numpy>=1.24
pandas>=2.0
torch>=2.1
scikit-learn>=1.3
scipy>=1.10
matplotlib>=3.7
```

For exact reproduction, replace broad lower bounds with the versions used in the published experiments and provide the Python, CUDA, and operating-system versions.

## Installation

```bash
git clone <REPOSITORY_URL>
cd Clust_VAE_WGAN_GP

python -m venv .venv
```

Activate the environment on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Activate it on Linux or macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Configuration before execution

The current scripts contain dataset-specific constants and hard-coded absolute Windows paths. For each `Dataset_n` folder, verify or update:

- `dataset_directory` or `dataset_dir`;
- the input CSV filename;
- `protected_attributes`;
- `numerical_columns` and binary-column postprocessing;
- `num_clusters`;
- `latent_dim`;
- VAE and WGAN epoch counts;
- batch size and learning rates;
- gradient-clipping norm and noise multiplier;
- fairness-loss weight and gradient-penalty weight; and
- `num_fake_samples`.

For portability, the scripts should preferably derive their directory from the file location:

```python
from pathlib import Path

dataset_directory = Path(__file__).resolve().parent
df = pd.read_csv(dataset_directory / "preprocessed_HIV.csv")
model_directory = dataset_directory / "models"
model_directory.mkdir(parents=True, exist_ok=True)
```

If GPU execution is used, correct `torch.devise("cuda:0")` to `torch.device("cuda:0")` in scripts that still contain that typographical error.

## Running an experiment

The scripts are currently executed independently for each dataset.

### 1. Run the complete architecture

```bash

python Architecture.py
```

For each Dataset, the supplied script writes a generated file following this pattern:

```text
gen_samp_rho<RHO>_epsilon<EPSILON>.csv
```

It also saves the trained VAE, generator, discriminator, and Fairness Critic weights in the configured `models/` directory.

### 2. Run the ablation without adversarial debiasing

```bash
python Architecture_WO_Adv_deb.py
```

The supplied ablation writes a file following this pattern:

```text
gen_samp_WO_Adv_Deb_Epsilon_<EPSILON>.csv
```

Use the same preprocessing, split, random seeds, generation count, and privacy configuration as the complete model when making an ablation comparison.

### 3. Evaluate fairness and clustering quality

In `Fairness_evaluation.py`, set:

```python
dataset_dir = "/path/to/Clust_VAE_WGAN_GP/Dataset_1"
file_path = os.path.join(dataset_dir, "<GENERATED_FILE>.csv")
protected_attributes = ["Determined in each script according to its corresponding demographic attributes"]
num_clusters = K
```

Then run:

```bash
python Fairness_evaluation.py
```

The evaluation script reports:

- cluster proportions conditional on each protected attribute;
- mutual information between each protected attribute and cluster membership;
- silhouette score; and
- Davies–Bouldin index.

### Metric interpretation

| Metric | Interpretation |
|---|---|
| Protected-group cluster distribution | Compare the cluster distribution across groups for each protected attribute. Similar distributions indicate less group-dependent cluster allocation. |
| Mutual information | `0` indicates statistical independence in the empirical sample; larger values indicate stronger dependence. Values are not directly comparable across all attributes unless normalization and support sizes are considered. |
| Silhouette score | Higher is better; values near `1` indicate compact, well-separated clusters, values near `0` indicate overlap, and negative values can indicate poor assignments. |
| Davies–Bouldin index | Lower is better; it summarizes within-cluster dispersion relative to between-cluster separation. |

The labels `Evenly Spread`, `Balanced`, and `Unbalanced` in the supplied evaluation script are based on project-specific thresholds (`0.05` and `0.15`). They are heuristic reporting categories, not generally accepted fairness thresholds. The script uses `KMeans`, despite a comment that refers to a Gaussian mixture model.

## Default Dataset configurations

| Component | Parameter | Full model | Without adversarial debiasing |
|---|---|---:|---:|
| VAE | Latent dimension | 20 | 20 |
| VAE | Number of clusters | 15 | 15 |
| VAE | Batch size | 86 | 86 |
| VAE | Learning rate | `1e-4` | `1e-4` |
| VAE | Epochs | 20 | 20 |
| VAE | Cluster update interval | 10 epochs | 10 epochs |
| Privacy | Gradient clipping norm | 0.5 | 0.5 |
| Privacy | Noise multiplier variable | 0.0031 | 0.002 |
| Privacy | Target delta | `1e-5` | `1e-5` |
| WGAN-GP | Generator learning rate | `1e-4` | `1e-4` |
| WGAN-GP | Critic learning rate | `1e-2` | `1e-2` |
| WGAN-GP | Epochs | 15 | 15 |
| WGAN-GP | Gradient-penalty weight | 10 | 10 |
| Fairness | Fairness Critic learning rate | `1e-2` | Not applicable |
| Fairness | Fairness weight in generator loss | 0.1 | Not applicable |

These values document the supplied Dataset 1 scripts; they should not be assumed to apply to Datasets 2–6.

## Reproducibility checklist

Before publishing results or comparing the full and ablated models:

- report exact package and hardware versions;
- use explicit random seeds for NumPy, PyTorch, CUDA, and K-Means;
- keep train/test partitions fixed across models;
- fit preprocessing only on the training data;
- report the exact number of optimizer steps used by the privacy accountant;
- ensure the privacy accountant matches the implemented sampling and noise mechanism;
- keep privacy budgets equal in controlled ablations unless privacy is the ablated component;
- report mean, dispersion, and number of repeated runs;
- evaluate utility, fairness, privacy, and statistical fidelity on held-out data where applicable; and
- document all dataset-specific protected-attribute mappings.

## Known implementation notes

- The scripts are notebook-style Python files with `# %%` cells and dataset-specific constants.
- Output CSV files are written to the current working directory unless an explicit output path is added.
- `Fairness_evaluation.py` loads the original dataset into `dfo`, but the supplied version does not subsequently use that object.
- `chi2_contingency` is imported by the supplied evaluation example but is not used in the reported metrics.
- `binary_cross_entropy_with_logits` expects raw logits; if the Fairness Critic retains a final sigmoid layer, use binary cross-entropy on probabilities instead, or remove the sigmoid and keep the logits-based loss.
- The WGAN critic conventionally returns an unconstrained scalar. The current discriminator ends with a sigmoid; this should be reviewed against the intended WGAN-GP formulation before exact reproduction claims are made.

## References

- D. P. Kingma and M. Welling, [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114), 2013.
- I. Gulrajani, F. Ahmed, M. Arjovsky, V. Dumoulin, and A. Courville, [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028), 2017.
- M. Bun and T. Steinke, [Concentrated Differential Privacy: Simplifications, Extensions, and Lower Bounds](https://arxiv.org/abs/1605.02065), 2016.
- [scikit-learn K-Means documentation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html).
- [scikit-learn silhouette score documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html).
- [scikit-learn Davies–Bouldin score documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.davies_bouldin_score.html).

## Citation

If you use this code, please cite the associated paper. Replace the placeholder below with the final bibliographic record and DOI:

```bibtex
@InProceedings{clust_vae_wgan_gp,
author="Adouani, Malek
and Chelly Dagdia, Zaineb",

title="Fair and Privacy-Preserving Synthetic Data Generation via Clustering-Based Variational Autoencoder and Adversarially Debiased Wasserstein Generative Adversarial Networks with Gradient Penalty",
booktitle="Machine Learning and Knowledge Discovery in Databases. Research Track",
year="2026",
publisher="Springer Nature Switzerland",

pages="195--212",
}
```

## Acknowledgement

This work was supported by the France 2030 grants RHU RECORDS (ANR-18-RHUS-0004) and IHU PROMETHEUS (ANR-23-IAHU-0004) and the iRECORDS project, funded by ERA PerMed (JTC_2021).

