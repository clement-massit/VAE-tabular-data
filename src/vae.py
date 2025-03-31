import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
# from sklearn.model_selection import train_test_split
# from sklearn import preprocessing
import seaborn as sns
from scipy.stats import ks_2samp, wasserstein_distance

class VEncoder(nn.Module):
    def __init__(self, input_dim:int, hidden_dim: int, latent_dim: int, processor_dim:int):
        super(VEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(processor_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, processor_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(processor_dim, latent_dim)
        self.fc_logvar = nn.Linear(processor_dim, latent_dim)

    def forward(self, x):
        z_mu = self.fc_mu(x)
        z_logvar = self.fc_logvar(x)
        z = self.encoder(x)
        return z, z_mu, z_logvar


class VDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, input_dim: int, processor_dim: int):
        super(VDecoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.Linear(processor_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, processor_dim)
        
        )

    def forward(self, x):
        reconstruction = self.decoder(x)
        return reconstruction
    
class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    
    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        _, z_mu, z_logvar = self.encoder(x)
        z = self.reparametrize(z_mu, z_logvar)
        reconstruction = self.decoder(z)
        return reconstruction, z_mu, z_logvar
    
    @property
    def metrics(self):
        return {
            'total_loss': self.total_loss_tracker,
            'reconstruction_loss': self.reconstruction_loss_tracker,
            'kl_loss': self.kl_loss_tracker,
        }

    def calculate_reconstruction_loss(self, data, reconstruction):
        """
        In case of computer vision tasks, you can use binary cross entropy loss:
            reconstruction_loss = F.binary_cross_entropy(reconstruction, data, reduction='sum')
        For simplicity, we use mean absolute error here.
        """
        return F.mse_loss(reconstruction, data, reduction='sum') # or you can use F.l1_loss

    def calculate_kl_loss(self, z_mean, z_log_var):
        kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp(), dim=1)
        return kl_loss.mean()

    def calculate_total_loss(self, reconstruction_loss, kl_loss):
        return reconstruction_loss + kl_loss  # Adjust this multiplier as needed
    
    def train_step(self, data, optimizer):
        # Set model to training mode
        self.train()
        
        optimizer.zero_grad()
        # Forward pass
        z, z_mu, z_logvar = self.encoder(data)
        reconstruction = self.decoder(z)
        
        # Compute losses
        reconstruction_loss = self.calculate_reconstruction_loss(data, reconstruction)
        kl_loss = self.calculate_kl_loss(z_mu, z_logvar)
        total_loss = self.calculate_total_loss(reconstruction_loss, kl_loss)
        total_loss += 100*torch.mean(torch.relu(-reconstruction))   
        
        # Backward pass
        total_loss.backward()
        optimizer.step()

        # Update loss trackers
        self.total_loss_tracker = total_loss.item()
        self.reconstruction_loss_tracker = reconstruction_loss.item()
        self.kl_loss_tracker = kl_loss.item()

        return {
            "loss": self.total_loss_tracker,
            "reconstruction_loss": self.reconstruction_loss_tracker,
            "kl_loss": self.kl_loss_tracker,
        }

    def test_step(self, data):
        # Set model to evaluation mode
        self.eval()

        # Forward pass (no gradient computation)
        with torch.no_grad():
            z, z_mu, z_logvar = self.encoder(data)
            reconstruction = self.decoder(z)
            
            # Compute losses
            reconstruction_loss = self.calculate_reconstruction_loss(data, reconstruction)
            kl_loss = self.calculate_kl_loss(z_mu, z_logvar)
            total_loss = self.calculate_total_loss(reconstruction_loss, kl_loss)

        # Update loss trackers
        self.total_loss_tracker = total_loss.item()
        self.reconstruction_loss_tracker = reconstruction_loss.item()
        self.kl_loss_tracker = kl_loss.item()

        return {
            "z_recon": z,
            "loss": self.total_loss_tracker,
            "reconstruction_loss": self.reconstruction_loss_tracker,
            "kl_loss": self.kl_loss_tracker,
        }
    


def evaluate_model(self, dataloader, device):
    self.eval()
    all_real, all_recon = [], []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch[0].to(device)
            # x, = batch
            # x = x.to(torch.float32)
            
            z, z_mu, z_logvar = self.encoder(batch)
            reconstruction = self.decoder(z)
            # recon_loss, KLD = vae_loss(x_recon, batch, mu, logvar)
            # loss = recon_loss + KLD

            all_real.append(batch.cpu().numpy())
            all_recon.append(reconstruction.cpu().numpy())

    all_real = np.vstack(all_real)
    all_recon = np.vstack(all_recon)

    mse = np.mean((all_real - all_recon) ** 2)
    mae = np.mean(np.abs(all_real - all_recon))
    # perplexity = np.exp(-np.sum(np.bincount(all_indices) / len(all_indices) * np.log(np.bincount(all_indices) / len(all_indices) + 1e-10)))
    
    ks_stat, _ = ks_2samp(all_real.flatten(), all_recon.flatten())
    wasserstein_dist = wasserstein_distance(all_real.flatten(), all_recon.flatten())


    print(f"MSE: {mse:.6f}, MAE: {mae:.6f}")
    print(f"MSE: {mse:.6f}, MAE: {mae:.6f}, KS Test: {ks_stat:.6f}, Wasserstein Distance: {wasserstein_dist:.6f}")
    
    plt.figure(figsize=(10, 4))
    sns.histplot(all_real.flatten(), bins=50, kde=True, label="Real Data")
    sns.histplot(all_recon.flatten(), bins=50, kde=True, label="Reconstructed Data")
    plt.legend()
    plt.title("Distribution of Real vs. Reconstructed Data")
    plt.show()
    plt.savefig('./output/distribution_generated_real.png')
    return True
