# Clust_VAE_WGAN_GP - Research Code
This repository contains the source code for our architecture ** Clust-VAE-WGAN-GP**, with different datasets. 
The implementation focuses on **synthetic data generation, fairness evaluation, and differential privacy constraints**.

 ## Abstract

The increasing reliance on machine learning in sensitive domains, such as healthcare, has amplified concerns about bias and privacy in data-driven decision-making. While fairness-aware generative models aim to mitigate bias, they often depend on labeled data, limiting their applicability in unsupervised settings. Conversely, differentially private generative models ensure privacy but may still encode hidden biases. Existing methods fail to jointly optimize fairness and privacy without explicit supervision. To address this gap, we propose a hybrid generative framework that integrates clustering-based Variational Autoencoders (VAEs) with Wasserstein Generative Adversarial Networks with Gradient Penalty (WGAN-GP) to generate fair and privacy-preserving synthetic data. The VAE structures latent representations under zeroConcentrated Differential Privacy (zCDP) while incorporating K-Means clustering directly in the latent space. The clustering serves as a factor to influence the generative process into producing samples that resemble real data in unsupervised settings. These structured representations along with cluster labels then guide WGAN-GP’s generator toward sample generation and enhance adversarial debiasing through the Fairness Critic, which penalizes correlations between synthetic data and sensitive attributes to ensure fairness. By integrating clustering-based VAEs with WGAN-GP, our framework enforces fairness while maintaining strong privacy guarantees. Experimental results demonstrate that it outperforms existing generative models by effectively reducing bias, preserving privacy, and ensuring high data utility across multiple fairness and privacy metrics.

## 🗂 Datasets

This work leverages multiple datasets, each with a dedicated tailored architecture:

- **Diabetes Health Indicators Dataset** – [Kaggle](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)
- **HIV Dataset** – [HealthGymAi](https://healthgym.ai/antiviral-hiv/)
- **Heart Failure Clinical Records** – [UCI Machine Learning Repository](https://doi.org/10.24432/C5Z89R)
- **Obesity Estimation Dataset** – [UCI Machine Learning Repository](https://doi.org/10.24432/C5H31Z)
- **Regensburg Pediatric Appendicitis Dataset** – [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/938/regensburg+pediatric+appendicitis)
- **Corporate Stress Dataset** – [Kaggle](https://www.kaggle.com/datasets/ankitpatel2100/corporate-stress-dataset-insights-into-workplace)


