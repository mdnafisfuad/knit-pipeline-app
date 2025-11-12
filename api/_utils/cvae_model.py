# /api/_utils/cvae_model.py
import torch
import torch.nn as nn

class CVAE(nn.Module):
    def __init__(self, feature_dim, label_dim, latent_dim=16):
        super(CVAE, self).__init__(); self.encoder = nn.Sequential(nn.Linear(feature_dim + label_dim, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU()); self.fc_mu, self.fc_log_var = nn.Linear(64, latent_dim), nn.Linear(64, latent_dim); self.decoder = nn.Sequential(nn.Linear(latent_dim + label_dim, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, feature_dim), nn.Sigmoid())
    def reparameterize(self, mu, log_var): std = torch.exp(0.5 * log_var); eps = torch.randn_like(std); return mu + eps * std
    def forward(self, x, c): combined_input = torch.cat([x, c], dim=1); h = self.encoder(combined_input); mu, log_var = self.fc_mu(h), self.fc_log_var(h); z = self.reparameterize(mu, log_var); combined_output = torch.cat([z, c], dim=1); return self.decoder(combined_output)