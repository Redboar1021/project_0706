#!/usr/bin/env python3
# train_vae_pointnet_smpl.py
# -------------------------------------------------------------
# PointNet‑VAE + SMPL  ― 教師あり L2 と標準 KL を分離
# ・NumPy 1.24+ 互換ハック
# ・学習過程を quickview_epochXX.png に可視化保存
# ・CPU / GPU / AMP どれでも動作確認済み
# ・*New* 使われたデータセットのファイル名を txt に保存
# -------------------------------------------------------------

import os, glob, random, inspect
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from smplx.body_models import SMPL
import torch.multiprocessing as mp


# ---------- NumPy 互換 ----------
np.bool, np.int, np.float, np.complex = bool, int, float, complex
np.object, np.unicode, np.str = object, str, str
if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec
# --------------------------------

# ---------- Dataset ----------
class MotionDataset(Dataset):
    """AMASS などの *_verts.npz 群をフレーム展開して返すデータセット"""

    def __init__(self, data_dir: str, sample_files: int | None = None):
        all_files = glob.glob(os.path.join(data_dir, "*_verts.npz"))
        self.files = (
            random.sample(all_files, sample_files)
            if isinstance(sample_files, int) and sample_files < len(all_files)
            else all_files
        )
        # index: [(filepath, frame_id), ...]
        self.index = []
        for f in tqdm(self.files, desc="Counting frames"):
            self.index.extend(
                [
                    (f, i)
                    for i in range(np.load(f, mmap_mode="r")["vertices"].shape[0])
                ]
            )

    def __len__(self):  # type: ignore[override]
        return len(self.index)

    def __getitem__(self, idx):  # type: ignore[override]
        fpath, frame_id = self.index[idx]
        if not hasattr(self, "_cache"):
            self._cache = {}
        if fpath not in self._cache:
            self._cache[fpath] = np.load(fpath, mmap_mode="r")
        data = self._cache[fpath]

        verts = data["vertices"][frame_id]
        transl = data.get("trans", np.zeros_like(verts))[frame_id]
        verts = verts - transl[None, :]  # 平行移動除去

        pose = data["poses"][frame_id].astype(np.float32)  # (72,)
        betas = data["betas"].astype(np.float32)  # (10,)
        prior = np.concatenate([pose, betas])  # (82,)

        return torch.from_numpy(verts).float(), torch.from_numpy(prior).float()


def collate_fn(batch):
    xs, ps = zip(*batch)
    return torch.stack(xs, 0), torch.stack(ps, 0)


# ---------- PointNet Encoder ----------
class PointNetEncoder(nn.Module):
    def __init__(self, latent_dim: int, global_feat_dim: int = 1024):
        super().__init__()
        self.mlp1 = nn.Sequential(nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU())
        self.mlp2 = nn.Sequential(nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU())
        self.mlp3 = nn.Sequential(
            nn.Conv1d(128, global_feat_dim, 1), nn.BatchNorm1d(global_feat_dim), nn.ReLU()
        )
        self.fc_mu = nn.Linear(global_feat_dim, latent_dim)
        self.fc_logvar = nn.Linear(global_feat_dim, latent_dim)

    def forward(self, x):  # x: [B,N,3]
        x = self.mlp1(x.transpose(1, 2))  # [B,64,N]
        x = self.mlp2(x)  # [B,128,N]
        x = self.mlp3(x)  # [B,1024,N]
        x = torch.max(x, dim=2)[0]  # global pooling
        return self.fc_mu(x), self.fc_logvar(x)


# ---------- VAE ----------
class VAE_PointNet(nn.Module):
    def __init__(self, latent_dim: int, smpl_dir: str):
        super().__init__()
        self.encoder = PointNetEncoder(latent_dim)
        self.smpl = SMPL(model_path=smpl_dir, gender="NEUTRAL", batch_size=1)  # ← 固定 1
        for p in self.smpl.parameters():
            p.requires_grad_(False)

    @staticmethod
    def reparameterize(mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, z):  # z: [B,82]
        pose, betas = z[:, :72], z[:, 72:]
        out = self.smpl(
            global_orient=pose[:, :3],
            body_pose=pose[:, 3:],
            betas=betas,
            transl=torch.zeros(len(z), 3, device=z.device),
        )
        return out.vertices.flatten(1)  # [B, 6890*3]

    def forward(self, x_pts):  # x_pts: [B,N,3]
        mu, logvar = self.encoder(x_pts)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


# ---------- Loss ----------

def kld_standard(mu, logvar):
    return -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - torch.exp(logvar), dim=1))


def loss_function(
    recon_x,
    x_flat,
    mu,
    logvar,
    prior_mu,
    kl_weight=1e-4,  # 初期は小さく
    sup_weight=1.0,
):
    recon_loss = F.mse_loss(recon_x, x_flat, reduction="mean")
    kl_loss = kld_standard(mu, logvar)
    sup_loss = F.mse_loss(mu, prior_mu, reduction="mean")
    total = recon_loss + kl_weight * kl_loss + sup_weight * sup_loss
    return total, recon_loss, kl_loss, sup_loss


# ---------- quick 3D scatter ----------
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def quick_scatter(path, vin, vre):
    fig = plt.figure(figsize=(8, 4))
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")
    ax1.scatter(vin[:, 0], vin[:, 1], vin[:, 2], s=1)
    ax1.set_title("input")
    ax1.set_axis_off()
    ax2.scatter(vre[:, 0], vre[:, 1], vre[:, 2], s=1, c="r")
    ax2.set_title("recon")
    ax2.set_axis_off()
    for ax in (ax1, ax2):
        ax.set_axis_off()
        ax.view_init(elev=5, azim=90)
        ax.set_box_aspect([1, 1, 1])
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


# ---------- Train ----------

def train(args):
    device = torch.device(args.device)
    ds = MotionDataset(args.data_dir, args.sample_files)

    # --- NEW: save list of dataset files used in this run ---
    if args.filelist:
        with open(args.filelist, "w", encoding="utf-8") as fout:
            for fpath in ds.files:
                fout.write(f"{fpath}\n")
        print(f"✔ Saved list of {len(ds.files)} training files to '{args.filelist}'")
    # --------------------------------------------------------

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        collate_fn=collate_fn,
        multiprocessing_context=mp.get_context("spawn"),
    )

    model = VAE_PointNet(args.latent_dim, args.model_path).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler() if args.use_amp else None

    for epoch in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        pbar = tqdm(dl, desc=f"Epoch {epoch}")
        for x_pts, prior in pbar:
            x_pts, prior = x_pts.to(device), prior.to(device)
            x_flat = x_pts.flatten(1)
            opt.zero_grad(set_to_none=True)

            if args.use_amp:
                with autocast():
                    recon, mu, logvar = model(x_pts)
                    loss, r, k, s = loss_function(
                        recon,
                        x_flat,
                        mu,
                        logvar,
                        prior,
                        kl_weight=args.kl_weight,
                        sup_weight=args.sup_weight,
                    )
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                recon, mu, logvar = model(x_pts)
                loss, r, k, s = loss_function(
                    recon,
                    x_flat,
                    mu,
                    logvar,
                    prior,
                    kl_weight=args.kl_weight,
                    sup_weight=args.sup_weight,
                )
                loss.backward()
                opt.step()

            tot += loss.item()
            pbar.set_postfix(r=f"{r:.3e}", kl=f"{k:.3e}", sup=f"{s:.3e}", total=f"{loss:.3e}")

        print(f"[{epoch}/{args.epochs}] mean total = {tot / len(dl):.6f}")

        # ---------- quick visual ----------
        model.eval()
        with torch.no_grad():
            x_pts, _ = next(iter(dl))
            x_pts = x_pts.to(device)
            recon, *_ = model(x_pts[:1])
            quick_scatter(
                f"quickview_epoch{epoch:02d}.png",
                x_pts[0].cpu().numpy(),
                recon[0].reshape(-1, 3).cpu().numpy(),
            )

    torch.save(model.state_dict(), "vae_pointnet.pth")
    print("✔ Training finished — model saved to 'vae_pointnet.pth'")


# ---------- CLI ----------

def parse_args():
    import argparse, torch

    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="/root/development/project4010/VAE/all_vertices")
    p.add_argument("--model_path", default="models")  # SMPL_NEUTRAL.npz を含むフォルダ
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--latent_dim", type=int, default=82)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--kl_weight", type=float, default=1e-4)
    p.add_argument("--sup_weight", type=float, default=1.0)
    p.add_argument("--sample_files", type=int, default=1)
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
    )
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--pin_memory", action="store_true", default=True)
    p.add_argument("--use_amp", action="store_true", default=False)
    # NEW: output filename for the dataset list
    p.add_argument("--filelist", default="train_files.txt", help="Path to save training file names")
    return p.parse_args()


# ---------- Main ----------
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    torch.manual_seed(42)
    args = parse_args()
    train(args)
