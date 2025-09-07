import os
import random
import numpy as np
import open3d as o3d

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

from human_vae_model import PointNetVAE




# =======================================
# 4. 損失関数 (VAE Loss)
# =======================================
def vae_loss(x, x_recon, mu, logvar, use_chamfer=False):
    """
    x:       (B, N=6890, 3)
    x_recon: (B, N=6890, 3)
    mu, logvar: (B, latent_dim)
    
    use_chamfer=True なら Chamfer Distance を使う (ここでは省略)
    デフォルトは単純な MSE
    """
    # --- 1) 再構成ロス ---
    if not use_chamfer:
        # 頂点対応が固定されているので MSE で良い (B, N, 3) 同士
        recon_loss = F.mse_loss(x_recon, x, reduction='mean')
    else:
        # 点群を計算する Chamfer Distance (省略)
        recon_loss = None  # 実装例：open3d など
    
    # --- 2) KL ロス ---
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)  # (B,)
    kl_loss = kl_loss.mean()
    
    return recon_loss, kl_loss

# =======================================
# 5. データセット: 100個の PLY (training/registrations)
# =======================================
class FaustRegistrationDataset(Dataset):
    """
    指定ディレクトリ内の PLY をすべて読み込み、(6890,3) のテンソルを返す
    """
    def __init__(self, directory):
        super().__init__()
        files = [f for f in os.listdir(directory) if f.endswith('.ply')]
        self.filepaths = sorted(os.path.join(directory, f) for f in files)
    
    def __len__(self):
        return len(self.filepaths)
    
    def __getitem__(self, idx):
        fpath = self.filepaths[idx]
        mesh = o3d.io.read_triangle_mesh(fpath)
        verts = np.asarray(mesh.vertices, dtype=np.float32)  # (6890, 3)想定

        # 座標変換 (a, b, c) → (a, -c, b)
        verts = verts[:, [0, 2, 1]]  # (a, c, b)
        verts[:, 1] *= -1           # (a, -c, b)

        # 正規化処理
        verts -= verts.mean(axis=0)  # 重心を原点へ移動（中心化）
        scale = np.linalg.norm(verts, axis=1).max()  # 最遠点までの距離でスケーリング
        verts /= scale  # 最大距離が1になるように正規化

        return torch.from_numpy(verts)  # shape: (6890, 3)

# =======================================
# 6. 学習 & 評価 ループ
# =======================================
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        # batch: (B, 6890, 3)
        batch = batch.to(device)
        x_recon, mu, logvar = model(batch)
        
        recon_loss, kl_loss = vae_loss(batch, x_recon, mu, logvar)
        loss = recon_loss + kl_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * batch.size(0)
    
    return total_loss / len(loader.dataset)

def evaluate(model, loader, device):
    model.eval()
    recon_sum, kl_sum = 0.0, 0.0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            x_recon, mu, logvar = model(batch)
            
            recon_loss, kl_loss = vae_loss(batch, x_recon, mu, logvar)
            recon_sum += recon_loss.item() * batch.size(0)
            kl_sum += kl_loss.item() * batch.size(0)
    recon_avg = recon_sum / len(loader.dataset)
    kl_avg = kl_sum / len(loader.dataset)
    return recon_avg, kl_avg

# =======================================
# 7. メイン実行 (学習＆評価)
# =======================================
if __name__ == "__main__":
    # (1) データ読み込み
    base_dir = "MPI-FAUST/training/registrations"
    dataset = FaustRegistrationDataset(base_dir)
    
    # 全100ファイルを想定、80:20 で Train/Val に分割
    n_data = len(dataset)  # 例: 100
    indices = list(range(n_data))
    random.shuffle(indices)
    split = int(0.8 * n_data)
    train_idx = indices[:split]
    test_idx  = indices[split:]
    
    train_set = Subset(dataset, train_idx)
    test_set  = Subset(dataset, test_idx)
    
    train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
    test_loader  = DataLoader(test_set, batch_size=4, shuffle=False)
    
    # (2) モデル準備
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PointNetVAE(latent_dim=64, num_points=6890).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # (3) 学習ループ
    EPOCHS = 50
    for epoch in range(1, EPOCHS+1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        recon_val, kl_val = evaluate(model, test_loader, device)
        print(f"[Epoch {epoch}/{EPOCHS}] "
              f"TrainLoss: {train_loss:.4f} | "
              f"ValRecon: {recon_val:.4f}, ValKL: {kl_val:.4f}")
        
    # (4) モデル保存
    save_path = "pointnet_vae.pth"
    torch.save(model.state_dict(), save_path)
    print(f"モデルを {save_path} に保存しました")

