import os
import random
import numpy as np
import open3d as o3d

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

# =======================================
# 1. PointNet Encoder (VAE 用)
# =======================================
class PointNetEncoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        # PointNet の MLP 部分 (Conv1d)
        # 入力: (B, N, 3) → 転置して (B, 3, N) を Conv1d
        self.mlp = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )
        
        # グローバルプーリング後に mu / logvar を出す全結合層
        self.fc_mu = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        self.fc_logvar = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )

    def forward(self, x):
        """
        x: (B, N, 3) の点群
        戻り値: mu, logvar (各 (B, latent_dim))
        """
        # Conv1d に合わせて (B, 3, N) に転置
        x = x.permute(0, 2, 1)  # (B, 3, N)
        features = self.mlp(x)  # (B, 512, N)
        # グローバル max プーリング
        global_feat = torch.max(features, dim=2)[0]  # (B, 512)

        mu = self.fc_mu(global_feat)         # (B, latent_dim)
        logvar = self.fc_logvar(global_feat) # (B, latent_dim)
        return mu, logvar

# =======================================
# 2. デコーダ (最も単純な MLP 例)
# =======================================
class SimpleDecoder(nn.Module):
    def __init__(self, latent_dim=128, num_points=6890):
        super().__init__()
        self.num_points = num_points
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_points * 3)  # x, y, z * num_points
        )

    def forward(self, z):
        """
        z: (B, latent_dim)
        戻り値: (B, num_points, 3)
        """
        x = self.fc(z)  # (B, num_points*3)
        x = x.view(-1, self.num_points, 3)
        return x

# =======================================
# 3. VAE 本体 (PointNet + SimpleDecoder)
# =======================================
class PointNetVAE(nn.Module):
    def __init__(self, latent_dim=128, num_points=6890):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_points = num_points
        
        self.encoder = PointNetEncoder(latent_dim=latent_dim)
        self.decoder = SimpleDecoder(latent_dim=latent_dim, num_points=num_points)

    def reparameterize(self, mu, logvar):
        # VAE の Reparameterization Trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """
        x: (B, 6890, 3)
        戻り値: x_recon, mu, logvar
        """
        mu, logvar = self.encoder(x)         
        z = self.reparameterize(mu, logvar)  
        x_recon = self.decoder(z)            
        return x_recon, mu, logvar