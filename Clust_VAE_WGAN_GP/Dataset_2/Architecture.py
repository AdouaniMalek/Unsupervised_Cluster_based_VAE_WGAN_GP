# %%
import numpy as np 
import pandas as pd
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim
import sklearn.model_selection as skl
import torch.nn.functional as F
import torch.autograd as autograd
import matplotlib.pyplot as plt  
import os 
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.cluster import KMeans
from zcdp_accountant import compute_zcdp,get_privacy_spent


# Dataset References:
# Corporate Stress Dataset: Insights into Workplace
# Source: Ankit Patel, Kaggle, Updated 2 months ago
# Available at: https://www.kaggle.com/datasets/ankitpatel2100/corporate-stress-dataset-insights-into-workplace


# %%
if torch.cuda.is_available():
    print('Cuda is available')
    device = torch.devise("cuda:0")
else:
    device = torch.device('cpu')
print(device)

# %%
dataset_directory = "C:/Users/Mega Pc/Desktop/SF-GAN unsupervised/Dataset_2/"
df = pd.read_csv(dataset_directory + "corporate_stress_dataset_preprocessed.csv")
#df.head()

trainData = df.to_numpy(dtype="float32")
trainData = torch.from_numpy(trainData).float().to(device)

# %%
class Dataset:
    def __init__(self, data, transform=None):
        self.transform = transform
        self.data = data
        self.sampleSize = data.shape[0]
        self.featureSize = data.shape[1]

    def return_data(self):
        return self.data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.data[idx]
        return sample

# %%
class VAEWithClusters(nn.Module):
    def __init__(self, feature_dim, latent_dim, num_clusters, dropout_prob=0.3, l2_reg=1e-5):
        super(VAEWithClusters, self).__init__()
        self.latent_dim = latent_dim
        self.num_clusters = num_clusters
        self.dropout_prob = dropout_prob
        self.l2_reg = l2_reg

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, 512),  # Reduced number of neurons
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=self.dropout_prob),  # Add dropout to prevent overfitting
            nn.Linear(512, 256),  # Reduced number of neurons
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=self.dropout_prob),  # Add dropout to prevent overfitting
            nn.Linear(256, 2 * latent_dim)  # Output both mu and logvar
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),  # Reduced number of neurons
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=self.dropout_prob),  # Add dropout to prevent overfitting
            nn.Linear(256, 512),  # Reduced number of neurons
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=self.dropout_prob),  # Add dropout to prevent overfitting
            nn.Linear(512, feature_dim),
            nn.Sigmoid(),  # Use ReLU instead of Sigmoid
        )

        # Initialize cluster centroids
        self.cluster_centroids = torch.randn(num_clusters, latent_dim).to(device)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=1)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar, z


# %%
hyp_lr = 1e-5
hyp_batch_size = 64
hyp_b1 = 0.9
hyp_b2 = 0.999
micro_batch_size = 20
hyp_noise_multiplier = 0.01
latent_dim = 20
#n_epochs = 150
#Dataloaders
#TrainDataloader
dataset_train_object = Dataset(data=trainData, transform=False)
dataloader_train = DataLoader(dataset_train_object, batch_size=hyp_batch_size, shuffle=True, num_workers=0, drop_last=True)
#Dataset parameters
feature_s = dataset_train_object.featureSize
total_samples = len(dataset_train_object)
num_batches = len(dataloader_train)
iterations = total_samples * num_batches

# %%
import torch.nn as nn

def weights_init(m):
    """
    Custom weight initialization function.
    :param m: Module to initialize
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # Normal initialization for Conv layers
        nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)  # Use 0.0 bias for consistency
    elif classname.find('BatchNorm') != -1:
        # Constant initialization for BatchNorm layers
        nn.init.constant_(m.weight.data, 1.0)
        nn.init.constant_(m.bias.data, 0.0)
    elif isinstance(m, nn.Linear):
        # Xavier initialization for Linear layers
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)  # Use 0.0 bias for linear layers


# %%
CondVautoencoderModel = VAEWithClusters(feature_s ,latent_dim,num_clusters=10)
CondVautoencoderModel.apply(weights_init)
hyper = torch.FloatTensor
one = torch.FloatTensor([1])
mone = one * -1
CondVautoencoderModel.to(device)

# %%
#hyp_batch_size= 5
q = hyp_batch_size / total_samples
# Compute zCDP (rho)
rho = compute_zcdp(q, noise_multiplier=hyp_noise_multiplier, steps=10)
print(rho)
# Convert zCDP (rho) to (epsilon, delta) DP parameters
epsilon, delta, _ = get_privacy_spent(rho, target_delta=1e-5)
print(f"Achieves ({epsilon:.3f}, {delta:.1e})-DP")

# %%
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_optimizer(cls, epsilon_value, delta_value):
    class DPOptimizer(cls):
        def __init__(self, params, lr, betas, max_per_sample_grad_norm, noise_multiplier, batch_size, *args, **kwargs):
            super(DPOptimizer, self).__init__(params, lr=lr, betas=betas, *args, **kwargs)
            self.max_per_sample_grad_norm = max_per_sample_grad_norm
            self.noise_multiplier = noise_multiplier
            self.batch_size = batch_size
            self.epsilon_value = epsilon_value  # Store epsilon value
            self.delta_value = delta_value      # Store delta value

            for group in self.param_groups:
                group['aggregate_grads'] = [torch.zeros_like(param.data) if param.requires_grad else None for param in group['params']]

        def clip_grads_(self):
            params = self.param_groups[0]['params']
            clip_grad_norm_(params, max_norm=self.max_per_sample_grad_norm, norm_type=2)

            for group in self.param_groups:
                for param, accum_grad in zip(group['params'], group['aggregate_grads']):
                    if param.requires_grad:
                        accum_grad.add_(param.grad.data)

        def add_noise_(self):
            for group in self.param_groups:
                for param, accum_grad in zip(group['params'], group['aggregate_grads']):
                    if param.requires_grad:
                        param.grad.data = accum_grad.clone()

                        # Compute the standard deviation for the Gaussian noise
                        std = self.max_per_sample_grad_norm * np.sqrt(2 * np.log(1.25 / self.delta_value)) / self.epsilon_value
                        
                        # Generate noise
                        noise = torch.normal(mean=0, std=std, size=param.grad.data.size(), device=device, dtype=param.grad.data.dtype)
                        
                        # Add noise to gradients
                        param.grad += noise / self.batch_size



        def step(self, *args, **kwargs):
            self.clip_grads_()
            self.add_noise_()
            super(DPOptimizer, self).step(*args, **kwargs)

    return DPOptimizer

AdamZCDP = create_optimizer(Adam,epsilon,delta)

optimizer_CVAE = AdamZCDP(
    CondVautoencoderModel.parameters(),
    lr=hyp_lr,
    betas=(hyp_b1, hyp_b2),
    max_per_sample_grad_norm=0.5,
    noise_multiplier=hyp_noise_multiplier,
    batch_size=hyp_batch_size,
   
)


# %%
def ConVAE_cluster_loss(x_recon, x_orig, mu, logvar, z, cluster_centroids, alpha=0.5,beta =0.5):
    # Reconstruction loss
    recon_loss = nn.functional.mse_loss(x_recon, x_orig, reduction='mean')

    # KL divergence loss
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    # Clustering loss
    if cluster_centroids is not None:
        # Compute pairwise distances (batch_size x num_clusters)
        distances = torch.cdist(z, cluster_centroids, p=2)  # Pairwise distances
        # Compute the minimum distance for each latent vector
        min_distances = torch.min(distances, dim=1).values
        # Cluster loss as the mean of minimum distances
        cluster_loss = torch.mean(min_distances)
    else:
        cluster_loss = 0.0

    #print("Reconstruction Loss:", recon_loss.item())
    #print("KL Loss:", kl_loss.item())
    #print("Cluster Loss:", cluster_loss.item())

    return recon_loss +  alpha *kl_loss + beta * cluster_loss


# %%
# Number of clusters
num_clusters = 20  # Define the number of clusters for K-means
cluster_update_interval = 10  # Update interval for K-means centroids
n_epochs = 50

# Training loop
for epoch in range(n_epochs):
    epoch_loss = 0
    recon_loss_sum = 0
    kl_loss_sum = 0
    cluster_loss_sum = 0
    num_batches = 0  # Counter for the number of batches

    # Collect latent representations for clustering
    if epoch % cluster_update_interval == 0:
        all_latents = []

    for data_batch in dataloader_train:
        data_batch = data_batch.to(device)
        num_batches += 1  # Increment batch counter

        # Clear previous gradients
        optimizer_CVAE.zero_grad()

        # Forward pass
        x_recon, mu, logvar, z = CondVautoencoderModel(data_batch)

        # Collect latent representations for clustering
        if epoch % cluster_update_interval == 0:
            all_latents.append(z.detach().cpu())

        # Total loss using the custom loss function
        loss = ConVAE_cluster_loss(x_recon, data_batch, mu, logvar, z, CondVautoencoderModel.cluster_centroids)

        # Accumulate losses for epoch statistics
        epoch_loss += loss.item()

        # Backpropagation and optimization
        loss.backward()
        optimizer_CVAE.step()

    # Update cluster centroids using K-means
    if epoch % cluster_update_interval == 0:
        all_latents = torch.cat(all_latents, dim=0).numpy()
        kmeans = KMeans(n_clusters=num_clusters,max_iter=500)
        kmeans.fit(all_latents)
        CondVautoencoderModel.cluster_centroids = torch.tensor(
            kmeans.cluster_centers_, device=device, dtype=torch.float
        )
        print(f"Updated cluster centroids at epoch {epoch + 1}")

    # Compute and print average losses
    avg_epoch_loss = epoch_loss / num_batches

    print(f"Epoch {epoch + 1}/{n_epochs}")
    print(f"  Avg Total Loss: {avg_epoch_loss:.4f}")
    #print("reconstructed_data",x_recon)




# %%
model_file_path = dataset_directory + "models/"
# Define the model save path, incorporating the rho value in the filename
model_save_path = model_file_path + f"Cluster_based_VAE_rho_{rho:.2f}.pth"
# Save the model
torch.save(CondVautoencoderModel.state_dict(), model_save_path)
print(f"Model saved as {model_save_path}")

# %%
# Load the CondVAE model
loaded_model = VAEWithClusters(feature_s, latent_dim, num_clusters).to(device)

# Load the state dictionary into the model (securely)
save_path = model_save_path
loaded_model.load_state_dict(torch.load(save_path, weights_only=True))  # Explicitly set weights_only=True

# Set the model to evaluation mode
loaded_model.eval()

# Initialize lists to store reconstructed samples, original samples, and cluster labels
all_recons_samples = []
all_original_samples = []
all_cluster_labels = []

# Iterate over the entire dataloader
for data_batch in dataloader_train:
    data_batch = data_batch.to(device)

    # Forward pass through the model
    with torch.no_grad():
        x_recon, _, _, z = loaded_model(data_batch)  # z: latent representation

        # Assign clusters based on learned centroids
        distances = torch.cdist(z, loaded_model.cluster_centroids)  # Compute distances to centroids
        cluster_labels = torch.argmin(distances, dim=1)  # Assign to nearest centroid

    # Store reconstructed samples, original samples, and cluster labels
    all_recons_samples.append(x_recon.cpu())
    all_original_samples.append(data_batch.cpu())
    all_cluster_labels.append(cluster_labels.cpu())

# Concatenate all batches into single tensors
all_recons_samples = torch.cat(all_recons_samples, dim=0)
all_original_samples = torch.cat(all_original_samples, dim=0)
all_cluster_labels = torch.cat(all_cluster_labels, dim=0)

# Convert to numpy arrays if needed
reconstructed_data = all_recons_samples.numpy()
original_data = all_original_samples.numpy()
cluster_labels = all_cluster_labels.numpy()


# %%
# Define the column names of the new dataset
column_names = df.columns.to_list()

# Assuming `reconstructed_data` is your generated or processed data
# Create a DataFrame for your dataset
reconstructed_data_df = pd.DataFrame(reconstructed_data, columns=column_names)

# Select sensitive attributes: Age and Gender-related columns
train_protected_attributes = reconstructed_data_df[
    ['Age', 'Gender_Male', 'Gender_Non-Binary']
].to_numpy()


# %%
numerical_columns = [
    'Age', 'Experience_Years', 'Monthly_Salary_INR', 'Working_Hours_per_Week', 'Commute_Time_Hours',
    'Stress_Level', 'Sleep_Hours', 'Physical_Activity_Hours_per_Week', 'Manager_Support_Level',
    'Work_Pressure_Level', 'Annual_Leaves_Taken', 'Work_Life_Balance', 'Family_Support_Level',
    'Job_Satisfaction', 'Performance_Rating', 'Team_Size'
]
binary_columns = [col for col in column_names if col not in numerical_columns]
for col in binary_columns:
    reconstructed_data_df[col] = (reconstructed_data_df[col] >= 0.5).astype(int)

# %%
reconstructed_data_df.to_csv(dataset_directory+"reconst_D2.csv")

# %%
class DatasetWGAN:
    def __init__(self, data, protected_attributes, transform=None):
        # Transform
        self.transform = transform

        # load data here
        self.data = data
        
        self.protected_attributes = protected_attributes
        self.sampleSize = data.shape[0]
        self.featureSize = data.shape[1]
        self.label_size = 1

    def return_data(self):
        return self.data

    
    def return_protected_attributes(self):
        return self.protected_attributes
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data[idx]
        
        protected_attribute = self.protected_attributes[idx]

        if self.transform:
            pass

        return sample, protected_attribute

# %%
# Assuming protected_attributes is a dictionary or tensor with both Ethnicity and Gender
# We will pass these attributes directly to DatasetWGAN
WGAN_GP_train_tensor = torch.Tensor(reconstructed_data)
dataset_train_object_wgan = DatasetWGAN(
    data=WGAN_GP_train_tensor,  # Input data
    protected_attributes=train_protected_attributes,  # Contains Ethnicity and Gender
    transform=False  # No additional transformation applied
)

dataloader_train_wgan = DataLoader(
    dataset_train_object_wgan, 
    batch_size=hyp_batch_size, 
    shuffle=True, 
    num_workers=0, 
    drop_last=True
)


# %%
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, feature_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Linear(128, feature_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, feature_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),

            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

class FairnessCritic(nn.Module):
    def __init__(self, num_clusters):  # Pass number of clusters dynamically
        super(FairnessCritic, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(num_clusters, 128),  # Change input size to num_clusters
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


# %%
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans

def compute_cluster_labels(data, num_clusters=15):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(data.cpu().detach().numpy())
    cluster_labels = torch.tensor(cluster_labels, dtype=torch.long, device=data.device)
    cluster_labels_one_hot = F.one_hot(cluster_labels, num_classes=num_clusters).float()
    return cluster_labels_one_hot


# %%

hidden_dim = dataset_train_object_wgan.return_protected_attributes().shape[1]
wgan_input_size = dataset_train_object_wgan.featureSize 
generatorModel = Generator(wgan_input_size)
discriminatorModel = Discriminator(wgan_input_size)
fairnessCriticModel = FairnessCritic(num_clusters)


# %%
def fairness_critic_loss(critic_predictions, protected_attributes):
    """
    Computes the fairness critic loss for both numerical and categorical protected attributes.
    
    Parameters:
    - critic_predictions (torch.Tensor): Fairness critic's predictions for generated data.
      (assumes numerical predictions come first, followed by categorical predictions).
    - protected_attributes (dict): Dictionary containing 'Age' (numerical),
      'Gender' (binary categorical), and optional 'Ethnicity' (categorical) attributes.
    
    Returns:
    - combined_loss (torch.Tensor): Combined fairness critic loss.
    """
    # Numerical (Age) regression loss
    age_preds = critic_predictions[:, 0].squeeze(-1)  # Remove singleton dimension
    age_targets = protected_attributes['Age'].float()
    age_loss = F.mse_loss(age_preds, age_targets)

    # Categorical (Gender) classification loss
    gender_preds = critic_predictions[:, 1].squeeze(-1)
    gender_targets = protected_attributes['Gender'].float()
    gender_loss = F.binary_cross_entropy_with_logits(gender_preds, gender_targets)

    # Combine losses
    combined_loss = age_loss + gender_loss
    return combined_loss
def generator_loss(discriminator_predictions, critic_predictions, protected_attributes, alpha=0.1):
    """
    Compute the generator loss combining adversarial loss and fairness loss.

    Parameters:
    - discriminator_predictions (torch.Tensor): Discriminator's predictions for generated data.
    - critic_predictions (torch.Tensor): Fairness critic's predictions for generated data.
    - protected_attributes (dict): Dictionary with 'Age', 'Gender', and other sensitive attributes.
    - alpha (float): Weight for the fairness loss term.

    Returns:
    - loss (torch.Tensor): Combined generator loss.
    """
    # Adversarial loss (WGAN-GP)
    adversarial_loss = -discriminator_predictions.mean()

    # Fairness loss
    fairness_loss = fairness_critic_loss(
        critic_predictions, 
        protected_attributes
    )

    # Combine losses
    combined_loss = adversarial_loss + alpha * fairness_loss
    return combined_loss



def calc_gradient_penalty(netD, real_data, fake_data, device, lambda_gp=10):
    alpha = torch.rand(real_data.size(0), 1, device=device).expand_as(real_data)
    interpolates = alpha * real_data + (1 - alpha) * fake_data
    interpolates = interpolates.requires_grad_(True)
    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(disc_interpolates.size(), device=device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp
    return gradient_penalty

def discriminator_loss_GP(netD, real_data, fake_data, real_predictions, fake_predictions, device, lambda_gp=10):
    """
    Compute the discriminator loss with gradient penalty.

    Parameters:
    - netD (torch.nn.Module): Discriminator model.
    - real_data (torch.Tensor): Real data samples.
    - fake_data (torch.Tensor): Fake data samples generated by the generator.
    - real_predictions (torch.Tensor): Discriminator's predictions for real data.
    - fake_predictions (torch.Tensor): Discriminator's predictions for fake data.
    - device (torch.device): Computation device (CPU/GPU).
    - lambda_gp (float): Weight for the gradient penalty.

    Returns:
    - total_loss (torch.Tensor): Total discriminator loss.
    """
    # Wasserstein loss
    d_loss = fake_predictions.mean() - real_predictions.mean()

    # Gradient penalty
    gradient_penalty = calc_gradient_penalty(netD, real_data, fake_data, device, lambda_gp)

    # Combine losses
    total_loss = d_loss + gradient_penalty
    return total_loss



# %%
b1 = 0.5
b2 = 0.999
sample_interval = 100
weight_decay = 0.0001
hyp_lr = 1e-4
beta = 0.5 # Weight for feature matching loss
optimizer_G = torch.optim.Adam(generatorModel.parameters(), lr=1e-4, betas=(b1, b2), weight_decay=weight_decay)
optimizer_FC = torch.optim.Adam(fairnessCriticModel.parameters(), lr=hyp_lr, betas=(b1, b2),weight_decay=weight_decay)
optimizer_D = torch.optim.Adam(discriminatorModel.parameters(), lr=hyp_lr, betas=(b1, b2), weight_decay=weight_decay)

# %%
generatorModel.apply(weights_init)
discriminatorModel.apply(weights_init)
fairnessCriticModel.apply(weights_init)

# %%
Tensor = torch.FloatTensor
one = torch.FloatTensor([1])
mone = one * -1

# %%
print(f"Cluster labels shape: {cluster_labels.shape}")  # Should be (batch_size, num_clusters)
print(f"Fairness Critic Input shape: {cluster_labels.shape}")  # Matches input_dim

# %%

# Ensure cluster_labels is a tensor
if isinstance(cluster_labels, np.ndarray):
    cluster_labels = torch.tensor(cluster_labels, dtype=torch.long, device=device)

# Detach and convert to long type
cluster_labels = cluster_labels.detach().long()

# Apply one-hot encoding
cluster_labels_one_hot = F.one_hot(cluster_labels, num_classes=num_clusters).float()


# %%
batch_size = hyp_batch_size
num_epochs = 20
feature_dim = feature_s

lambda_gp = 10  # Gradient penalty weight
lmbda = 0.001  # Regularization weight for L1 regularization
sample_interval = 64

for epoch in range(num_epochs):
    for i_batch, (real_data, protected_attributes) in enumerate(DataLoader(dataset_train_object_wgan, batch_size=batch_size, shuffle=True)):
        # Move data to device
        real_data = real_data.to(device)
        protected_attributes = {
            'Ethnicity': protected_attributes[:, 0].to(device),
            'Gender': protected_attributes[:, 1].to(device),
            'Age': protected_attributes[:, 2].to(device)  # Age as numerical attribute
        }

        # --- Train Generator ---
        optimizer_G.zero_grad()

        # Generate synthetic data
        noise = torch.randn(batch_size, feature_dim, device=device)
        generated_data = generatorModel(noise)

        # Get discriminator predictions for generated data
        fake_predictions = discriminatorModel(generated_data)

        # Compute cluster labels for generated data
        cluster_labels = compute_cluster_labels(generated_data, num_clusters=num_clusters)

        # Get fairness critic predictions
        fairness_predictions = fairnessCriticModel(cluster_labels.detach())

        # Compute generator loss
        gen_loss = generator_loss(
            fake_predictions,
            fairness_predictions,
            protected_attributes
        )

        # L1 regularization for generator
        l1_penalty_gen = sum(torch.norm(param, 1) for param in generatorModel.parameters())
        total_gen_loss = gen_loss + lmbda * l1_penalty_gen

        total_gen_loss.backward()
        optimizer_G.step()

        # --- Train Discriminator ---
        optimizer_D.zero_grad()

        # Get discriminator predictions for real data
        real_predictions = discriminatorModel(real_data)

        # Calculate discriminator loss with gradient penalty
        d_loss = discriminator_loss_GP(
            discriminatorModel,
            real_data,
            generated_data.detach(),
            real_predictions,
            fake_predictions.detach(),
            device,
            lambda_gp=lambda_gp
        )

        d_loss.backward()
        optimizer_D.step()

        # --- Train Fairness Critic ---
        optimizer_FC.zero_grad()
        fairness_predictions = fairnessCriticModel(cluster_labels)

        fc_loss = fairness_critic_loss(fairness_predictions, protected_attributes)

        # L1 regularization for fairness critic
        l1_penalty_fc = sum(torch.norm(param, 1) for param in fairnessCriticModel.parameters())
        total_fc_loss = fc_loss + lmbda * l1_penalty_fc

        total_fc_loss.backward()
        optimizer_FC.step()

        # Logging and Monitoring
        batches_done = epoch * len(dataset_train_object_wgan) // batch_size + i_batch
        if batches_done % sample_interval == 0:
            # Wasserstein distance for monitoring
            wasserstein_distance = real_predictions.mean().item() - fake_predictions.mean().item()

            print(
                f"[Epoch {epoch + 1}/{num_epochs}] [Batch {i_batch + 1}/{len(dataset_train_object_wgan) // batch_size}] "
                f"[Loss G: {gen_loss.item():.3f}] [Loss D: {d_loss.item():.3f}] "
                f"[Wasserstein Dist: {wasserstein_distance:.3f}] [Loss F: {fc_loss.item():.3f}]",
                flush=True
            )


# %%
# Save models after training loop
generator_filename = f'generator_epoch_{epoch + 1}.pth'
discriminator_filename = f'discriminator_epoch_{epoch + 1}.pth'
fairness_critic_filename = f'fairness_critic_epoch_{epoch + 1}.pth'

torch.save(generatorModel.state_dict(), os.path.join(model_file_path, generator_filename))
torch.save(discriminatorModel.state_dict(), os.path.join(model_file_path, discriminator_filename))
torch.save(fairnessCriticModel.state_dict(), os.path.join(model_file_path, fairness_critic_filename))

print(f"Models saved: {generator_filename}, {discriminator_filename}, {fairness_critic_filename}")


# %%
# Load the saved generator model
generatorModel = Generator(feature_dim)
generatorModel.load_state_dict(torch.load(os.path.join(model_file_path, generator_filename), weights_only=True))
generatorModel.to(device)
generatorModel.eval()

# Define the number of synthetic samples to generate
num_fake_samples = 3000

# Define column names (match the generator's output dimensions)
column_names = df.columns.to_list()

# Ensure column count matches feature size
if len(column_names) != feature_dim:
    raise ValueError(f"Mismatch: Expected {feature_dim} columns, but got {len(column_names)}.")

# Identify numerical and binary columns
numerical_columns = [
    'Age', 'Experience_Years', 'Monthly_Salary_INR', 'Working_Hours_per_Week', 'Commute_Time_Hours',
    'Stress_Level', 'Sleep_Hours', 'Physical_Activity_Hours_per_Week', 'Manager_Support_Level',
    'Work_Pressure_Level', 'Annual_Leaves_Taken', 'Work_Life_Balance', 'Family_Support_Level',
    'Job_Satisfaction', 'Performance_Rating', 'Team_Size'
]
binary_columns = [col for col in column_names if col not in numerical_columns]

# Calculate number of complete batches and remainder
n_batches = num_fake_samples // batch_size
remainder = num_fake_samples % batch_size

# Initialize array to store generated data
gen_samples = np.zeros((num_fake_samples, feature_dim), dtype=np.float32)

# Generate complete batches
for i in range(n_batches):
    noise = torch.randn(batch_size, feature_dim, device=device)
    with torch.no_grad():
        gen_sample_tensor = generatorModel(noise)
    gen_samples[i * batch_size:(i + 1) * batch_size, :] = gen_sample_tensor.cpu().numpy()

# Handle the final incomplete batch (if any)
if remainder > 0:
    noise = torch.randn(remainder, feature_dim, device=device)
    with torch.no_grad():
        gen_sample_tensor = generatorModel(noise)
    gen_samples[n_batches * batch_size:, :] = gen_sample_tensor.cpu().numpy()

print(f"Generated samples shape: {gen_samples.shape}")

# Convert to DataFrame
gen_samples_df = pd.DataFrame(gen_samples, columns=column_names)

# Process binary columns: Threshold at 0.5 for binary transformation
for col in binary_columns:
    gen_samples_df[col] = (gen_samples_df[col] >= 0.5).astype(int)

# **Save synthetic data to CSV**
csv_save_path = os.path.join(dataset_directory + "Artificial_samples", f'gen_samp_rho{rho:.2f}_epsilon{epsilon:.2f}.csv')
gen_samples_df.to_csv(csv_save_path, index=False)

print(f"Saved synthetic data to CSV: {csv_save_path}")



