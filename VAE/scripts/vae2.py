#!/usr/bin/env python3
"""
PointNet‑VAE + VPoser + SMPL  
--------------------------------------------------------
verts(6890×3) → PointNetEncoder → (pose63̂ , β10̂ )
pose63̂ → VPoser.encode → (μ32, σ32) → reparam → z32
z32 → VPoser.decode → pose63̃
(pose63̃ , β10̂ ) → SMPL → verts̃

Loss =
    L_rec   : verts 再構成 MSE
  + λ_kl    : KL( q(z32|verts) ∥ N(0,1) )
  + λ_beta  : β MSE
  + λ_pose  : pose63 MSE (Enc 出力と GT)

"""
import os, glob, random, inspect
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import torch.multiprocessing as mp
from typing import Optional

# ---------- NumPy 互換 (NumPy 1.24+) ----------
np.bool   = bool   # type: ignore
np.int    = int    # type: ignore
np.float  = float  # type: ignore
np.complex = complex  # type: ignore
np.object = object # type: ignore
np.unicode = str   # type: ignore
np.str    = str   # type: ignore
if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec  # type: ignore

from smplx.body_models import SMPL
from human_body_prior.tools.model_loader import load_model


# ---------- NumPy 互換 (NumPy 1.24+) ----------
np.bool   = bool   # type: ignore
np.int    = int    # type: ignore
np.float  = float  # type: ignore
np.complex = complex  # type: ignore
np.object = object # type: ignore
np.unicode = str   # type: ignore
np.str    = str   # type: ignore
if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec  # type: ignore
# ------------------------------------------------

from os import path as osp

#Loading VPoser Body Pose Prior
from human_body_prior.tools.model_loader import load_model
from human_body_prior.models.vposer_model import VPoser


# ---------- Dataset ----------
class MotionDataset(Dataset):
    """*vertices.npz files → verts, β, pose63 (GT) per frame"""

    def __init__(self, data_dir: str, sample_files: Optional[int] = None):
        all_files = glob.glob(os.path.join(data_dir, '*_verts.npz'))
        self.files = (
            random.sample(all_files, sample_files)
            if isinstance(sample_files, int) and sample_files < len(all_files)
            else all_files
        )
        print("Selected files:")
        for f in self.files:
            print(f)
        self.index: list[tuple[str, int]] = []
        for f in tqdm(self.files, desc='Scanning frames'):
            num = np.load(f, mmap_mode='r')['vertices'].shape[0]
            self.index.extend([(f, i) for i in range(num)])

    def __len__(self):  # type: ignore[override]
        return len(self.index)

    def __getitem__(self, idx):  # type: ignore[override]
        fpath, frame_id = self.index[idx]
        if not hasattr(self, '_cache'):
            self._cache = {}
        if fpath not in self._cache:
            self._cache[fpath] = np.load(fpath, mmap_mode='r')
        data = self._cache[fpath]

        verts  = data['vertices'][frame_id].astype(np.float32)  # (6890,3)

        pose_full = data['poses'][frame_id].astype(np.float32)  # (72,)
        betas     = data['betas'].astype(np.float32)            # (10,)

        pose63 = pose_full[3:66]   # exclude hands (9 joints * 3)

        return (
            torch.from_numpy(verts).float(),     # [N,3]
            torch.from_numpy(betas).float(),     # [10]
            torch.from_numpy(pose63).float()     # [63]
        )


def collate_fn(batch):
    verts, betas, pose63 = zip(*batch)
    return (
        torch.stack(verts, 0),   # [B,N,3]
        torch.stack(betas, 0),   # [B,10]
        torch.stack(pose63, 0)   # [B,63]
    )

# ---------- PointNet Encoder ----------
class PointNetEncoder(nn.Module):
    """PointNet → pose63 & β10"""

    def __init__(self, global_feat_dim: int = 1024):
        super().__init__()
        self.mlp1 = nn.Sequential(nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU())
        self.mlp2 = nn.Sequential(nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU())
        self.mlp3 = nn.Sequential(nn.Conv1d(128, global_feat_dim, 1), nn.BatchNorm1d(global_feat_dim), nn.ReLU())
        self.fc_pose = nn.Linear(global_feat_dim, 63)
        self.fc_beta = nn.Linear(global_feat_dim, 10)

    def forward(self, x):  # x: [B,N,3]
        x = self.mlp1(x.transpose(1, 2))   # [B,64,N]
        x = self.mlp2(x)                   # [B,128,N]
        x = self.mlp3(x)                   # [B,1024,N]
        x = torch.max(x, dim=2)[0]         # global max pool → [B,1024]
        pose63 = self.fc_pose(x)
        betas  = self.fc_beta(x)
        return pose63, betas

# ---------- VAE Model ----------
class VAE_PointNet(nn.Module):
    def __init__(self, smpl_dir: str, vposer):
        super().__init__()
        self.encoder = PointNetEncoder()
        self.vposer  = vposer.eval()
        for p in self.vposer.parameters():
            p.requires_grad_(False)
        self.smpl = SMPL(model_path=smpl_dir, gender='NEUTRAL', batch_size=1)
        for p in self.smpl.parameters():
            p.requires_grad_(False)

    @staticmethod
    def reparameterize(mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def forward(self, verts_in):  # verts_in: [B,N,3]
        pose_hat, beta_hat = self.encoder(verts_in)  # (B,63), (B,10)

        # VPoser encode → latent distribution
        dist = self.vposer.encode(pose_hat)
        mu   = dist.loc
        logvar = 2 * torch.log(dist.scale + 1e-8)
        z    = dist.rsample()    # ← reparameterize one-liner

        # VPoser decode → pose63
        pose_dec = self.vposer.decode(z)['pose_body']  # (B,63)or(B,21,3)

        if pose_dec.dim() == 3:                         # safety
            pose_dec = pose_dec.reshape(pose_dec.size(0), -1)

        # ★ 6D ゼロ（r_hand, l_hand）を末尾に付与 → 69D
        zeros6 = torch.zeros(pose_dec.size(0), 6, device=pose_dec.device)
        pose_dec69 = torch.cat([pose_dec, zeros6], dim=1)   # (B,69)

        out = self.smpl(
            global_orient=torch.zeros(len(z), 3, device=z.device),
            body_pose=pose_dec69,
            betas=beta_hat,
            transl=torch.zeros(len(z), 3, device=z.device),
        )
        verts_out = out.vertices.flatten(1)  # [B, 6890*3]

        return verts_out, mu, logvar, beta_hat, pose_hat

# ---------- Loss ----------

def kld_standard(mu, logvar):
    return -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))


def loss_function(recon_v, gt_v,
                  mu, logvar,
                  beta_hat, beta_gt,
                  pose_hat, pose_gt,
                  w_kl=1e-4, w_beta=1.0, w_pose=1.0):
    L_rec  = F.mse_loss(recon_v, gt_v, reduction='mean')
    L_kl   = kld_standard(mu, logvar)
    L_beta = F.mse_loss(beta_hat, beta_gt, reduction='mean')
    L_pose = F.mse_loss(pose_hat, pose_gt, reduction='mean')
    total  = L_rec + w_kl*L_kl + w_beta*L_beta + w_pose*L_pose
    return total, {'rec': L_rec, 'kl': L_kl, 'beta': L_beta, 'pose': L_pose}

# ---------- Quick 3D scatter ----------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def quick_scatter(path, vin, vre):
    fig = plt.figure(figsize=(8, 4))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    ax1.scatter(vin[:,0], vin[:,1], vin[:,2], s=1)
    ax2.scatter(vre[:,0], vre[:,1], vre[:,2], s=1, c='r')
    for ax, ttl in zip((ax1, ax2), ('input', 'recon')):
        ax.set_title(ttl)
        ax.set_axis_off()
        ax.view_init(elev=-95, azim=90)
        ax.set_box_aspect([1,1,1])
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

# ---------- Train ----------

def train(args):
    device = torch.device(args.device)

    # VPoser をロード
    support_dir = '/root/development/project4010/VAE/human_body_prior/support_data/downloads'
    expr_dir = osp.join(support_dir,'vposer_v2_05') #'TRAINED_MODEL_DIRECTORY'  in this directory the trained model along with the model code exist
    vp, ps = load_model(expr_dir, model_code=VPoser,
                              remove_words_in_model_weights='vp_model.',
                              disable_grad=True)
    vposer = vp.to(device)
    vposer = vposer.to(device)

    ds = MotionDataset(args.data_dir, args.sample_files)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, persistent_workers=True,
                    num_workers=args.num_workers, pin_memory=args.pin_memory,
                    collate_fn=collate_fn, multiprocessing_context=mp.get_context('spawn'))

    model = VAE_PointNet(args.smpl_dir, vposer).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler() if args.use_amp else None

    for epoch in range(1, args.epochs+1):
        model.train()
        tot = 0.0
        pbar = tqdm(dl, desc=f'Epoch {epoch}')
        for verts, beta_gt, pose_gt in pbar:
            verts    = verts.to(device)
            beta_gt  = beta_gt.to(device)
            pose_gt  = pose_gt.to(device)
            verts_flat = verts.flatten(1)

            opt.zero_grad(set_to_none=True)
            if args.use_amp:
                with autocast():
                    recon, mu, logvar, beta_hat, pose_hat = model(verts)
                    loss, comp = loss_function(
                        recon, verts_flat, mu, logvar,
                        beta_hat, beta_gt,
                        pose_hat, pose_gt,
                        w_kl=args.kl_weight, w_beta=args.beta_weight, w_pose=args.pose_weight)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                recon, mu, logvar, beta_hat, pose_hat = model(verts)
                loss, comp = loss_function(
                    recon, verts_flat, mu, logvar,
                    beta_hat, beta_gt,
                    pose_hat, pose_gt,
                    w_kl=args.kl_weight, w_beta=args.beta_weight, w_pose=args.pose_weight)
                loss.backward()
                opt.step()

            tot += loss.item()
            pbar.set_postfix({k: f'{v:.3e}' for k, v in comp.items()}, total=f'{loss.item():.3e}')

        print(f'[{epoch}/{args.epochs}] mean total = {tot/len(dl):.6f}')

        # quick visual
        # model.eval()
        # with torch.no_grad():
        #     v_in, _, _ = next(iter(dl))
        #     v_in = v_in.to(device)
        #     recon, *_ = model(v_in[:1])
        #     quick_scatter(f'quickview_epoch{epoch:02d}.png',
        #                   v_in[0].cpu().numpy(),
        #                   recon[0].reshape(-1,3).cpu().numpy())

    torch.save(model.state_dict(), 'vae_pointnet_vposer.pth')
    print('✔ Training finished — model saved to "vae_pointnet_vposer.pth"')

# ---------- CLI ----------

def parse_args():
    import argparse, torch
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir',   default="/root/development/project4010/VAE/all_vertices")
    p.add_argument('--smpl_dir',   default="/root/development/project4010/VAE/models")
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--epochs',     type=int, default=30)
    p.add_argument('--lr',         type=float, default=1e-3)
    p.add_argument('--kl_weight',  type=float, default=1e-4)
    p.add_argument('--beta_weight',type=float, default=1.0)
    p.add_argument('--pose_weight',type=float, default=1.0)
    p.add_argument('--sample_files', type=int, default=1, help='random subset of files')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', choices=['cuda','cpu'])
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--pin_memory', action='store_true', default=True)
    p.add_argument('--use_amp',    action='store_true', default=False)
    return p.parse_args()

# ---------- Main ----------
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    torch.manual_seed(42)
    args = parse_args()
    train(args)
