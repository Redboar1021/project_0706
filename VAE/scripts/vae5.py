#!/usr/bin/env python3
"""vposerのエンコーダーも学習
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import os, glob, random, inspect

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

# ----------------------------------------------------------------------------
# 3rd‑party (SMPL & VPoser)
# ----------------------------------------------------------------------------
from smplx.body_models import SMPL
from human_body_prior.models.vposer_model import VPoser
from human_body_prior.tools.model_loader import load_model

###############################################################################
# Dataset
###############################################################################


class MotionDataset(Dataset):
    """
    Return (verts, betas, pose63) for each frame.

    ▸ 起動時に壊れた .npz をスキップ  
    ▸ `ram_cache=True` を渡すと “全データを一括で RAM( torch.Tensor ) に取り込み”
       以降の __getitem__ はインメモリ高速アクセスになる
    -----------------------------------------------------------------------
    Parameters
    ----------
    data_dir : str
        *_verts.npz が入ったディレクトリ
    sample_files : int | None, optional
        ランダムにファイル数を間引く（デバッグ用）
    ram_cache : bool, default False
        True ならロード時に全フレームを Tensor 化して保持する
    """

    def __init__(
        self,
        data_dir: str,
        sample_files: int | None = None,
        *,
        ram_cache: bool = True,
    ):
        self.ram_cache = ram_cache

        # ── 1. ファイルスキャン & 壊れたファイル排除 ──────────────────
        all_files = list(Path(data_dir).glob("*_verts.npz"))
        candidate = random.sample(all_files, sample_files) if sample_files else all_files

        self.files: list[Path] = []
        skipped: list[Path] = []
        frame_index: list[tuple[int, int]] = []  # (file_id, frame_id)

        for f in tqdm(candidate, desc="scan", unit="file"):
            try:
                n_frames = np.load(f, mmap_mode="r")["vertices"].shape[0]
            except Exception as e:
                skipped.append(f)
                tqdm.write(f"⚠️  skip {f.name} ({e})")
                continue
            fid = len(self.files)
            self.files.append(f)
            frame_index.extend([(fid, i) for i in range(n_frames)])

        if not self.files:
            raise RuntimeError("No readable *_verts.npz files found – aborting.")

        print(f"→ dataset scan done: {len(self.files)} good / {len(skipped)} skipped")

        # ── 2. 必要なら RAM へプリロード ────────────────────────────
        if self.ram_cache:
            verts_l, betas_l, pose_l = [], [], []
            for f in tqdm(self.files, desc="preload", unit="file"):
                npz = np.load(f)  # memmap=無しで一括読み
                verts_l.append(torch.from_numpy(npz["vertices"]).float())         # (F,6890,3)
                poses = torch.from_numpy(npz["poses"]).float()[:, 3:66]           # (F,63)
                pose_l.append(poses)
                betas = torch.from_numpy(npz["betas"]).float()                   # (10,)
                betas_l.append(betas.repeat(len(poses), 1))                      # (F,10)
            # 連結して大きな Tensor を保持
            self.verts  = torch.cat(verts_l)      # (N,6890,3)
            self.pose63 = torch.cat(pose_l)       # (N,63)
            self.betas  = torch.cat(betas_l)      # (N,10)
            self.index  = list(range(len(self.verts)))
        else:
            # memmap 動作用の index (file_path, frame_id)
            self.index = frame_index

    # ------------------------------------------------------------------
    # PyTorch Dataset API
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        if self.ram_cache:  # ------ インメモリ高速パス
            return self.verts[idx], self.betas[idx], self.pose63[idx]

        # ------ ディスク memmap パス ------
        fpath, frame = self.index[idx]
        data = np.load(self.files[fpath], mmap_mode="r")
        verts = data["vertices"][frame].astype(np.float32)
        pose_full = data["poses"][frame].astype(np.float32)
        betas = data["betas"].astype(np.float32)
        pose63 = pose_full[3:66]
        return (
            torch.from_numpy(verts),
            torch.from_numpy(betas),
            torch.from_numpy(pose63),
        )
class MotionDatasetCached(Dataset):
    """
    ◎ .npz → Tensor 化した結果を <data_dir>/cache.pt に保存し、
      次回から torch.load で即復元。
    """

    CACHE_NAME = "cache.pt"

    def __init__(self, data_dir: str, sample_files: int | None = None):
        data_dir = Path(data_dir)
        cache_fp = data_dir / self.CACHE_NAME

        if cache_fp.exists():
            # ---------- ❶ キャッシュを読むだけ ----------
            t0 = time.perf_counter()
            blob = torch.load(cache_fp, map_location="cpu")
            self.verts  = blob["verts"]
            self.betas  = blob["betas"]
            self.pose63 = blob["pose63"]
            print(f"✓ cache loaded in {time.perf_counter()-t0:.2f}s")
        else:
            # ---------- ❷ 通常ロード → キャッシュ保存 ----------
            raw = MotionDataset(data_dir, sample_files)   # 既存クラスを再利用

            verts_l, betas_l, pose_l = [], [], []
            for f in tqdm(raw.files, desc="preload", unit="file"):
                npz = np.load(f)
                verts_l.append(torch.from_numpy(npz["vertices"]).float())
                pose = torch.from_numpy(npz["poses"]).float()[:, 3:66]
                pose_l.append(pose)
                betas = torch.from_numpy(npz["betas"]).float()
                betas_l.append(betas.repeat(len(pose), 1))

            self.verts  = torch.cat(verts_l)
            self.betas  = torch.cat(betas_l)
            self.pose63 = torch.cat(pose_l)

            # 保存（半精度に落とすと容量½）
            torch.save(
                {"verts": self.verts,  
                 "betas": self.betas,
                 "pose63": self.pose63},
                cache_fp
            )
            print(f"✓ cache saved to {cache_fp}  "
                  f"({cache_fp.stat().st_size/1e6:.1f} MB)")

        self.n = len(self.verts)

    # -------- PyTorch Dataset API --------
    def __len__(self): return self.n

    def __getitem__(self, idx):
        return (self.verts[idx], self.betas[idx], self.pose63[idx])



class PointNetEncoder(nn.Module):
    def __init__(self, feat: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(3, 64, 1), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.ReLU(),
            nn.Conv1d(128, feat, 1), nn.ReLU(),
        )
        self.pose_fc = nn.Linear(feat, 63)
        self.beta_fc = nn.Linear(feat, 10)

    def forward(self, x):
        x = self.net(x.transpose(1, 2))
        x = torch.max(x, 2).values
        return self.pose_fc(x), self.beta_fc(x)


class VAE_PointNet(nn.Module):
    def __init__(self, smpl_dir: str, vposer: VPoser):
        super().__init__()
        self.enc = PointNetEncoder()
        self.vposer = vposer
        for p in self.vposer.encoder_net.parameters():
            p.requires_grad_(True)    # ←学習する
        for p in self.vposer.decoder_net.parameters():
            p.requires_grad_(False)   # ←固定
        self.smpl = SMPL(model_path=smpl_dir, gender="NEUTRAL", batch_size=1)
        for p in self.smpl.parameters():
            p.requires_grad_(False)

    def forward(self, verts):
        pose_hat, beta_hat = self.enc(verts)
        dist = self.vposer.encode(pose_hat)
        z = dist.rsample()
        pose_dec = self.vposer.decode(z)["pose_body"].reshape(z.size(0), -1)
        pose69 = torch.cat([pose_dec, torch.zeros(verts.size(0), 6, device=verts.device)], 1)
        out = self.smpl(
            global_orient=torch.zeros(verts.size(0), 3, device=verts.device),
            body_pose=pose69,
            betas=beta_hat,
            transl=torch.zeros(verts.size(0), 3, device=verts.device),
        )
        verts_out = out.vertices.flatten(1)
        return verts_out, dist.loc, torch.log(dist.scale ** 2 + 1e-8), beta_hat, pose_hat

###############################################################################
# Loss util (unchanged)
###############################################################################

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

###############################################################################
# Train (progress bar unchanged)
###############################################################################

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- VPoser load ---------- #
    t0 = time.perf_counter()
    print("→ loading VPoser …", flush=True, end="")
    vposer, _ = load_model(
        args.vposer_dir,
        model_code=VPoser,
        remove_words_in_model_weights="vp_model.",
        disable_grad=True,
    )
    vposer = vposer.to(device)
    print(f" done ({time.perf_counter()-t0:.1f}s)")

    # ---------- data ---------- #
    ds = MotionDatasetCached(args.data_dir, args.sample_files)
    print(f"→ dataset: {len(ds)} frames ")

    print("DataLoader: creating batches …", flush=True, end="")

    def collate_gpu(batch):
        v, b, p = zip(*batch)                     # すでに Tensor
        return (torch.stack(v).cuda(non_blocking=True),
                torch.stack(b).cuda(non_blocking=True),
                torch.stack(p).cuda(non_blocking=True))

    dl = DataLoader(ds,
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=0,      # GPU を触るので 0 が安全
                    pin_memory=False,   # 直接 .cuda するため不要
                    collate_fn=collate_gpu)
    
    # ---------- model ---------- #
    model = VAE_PointNet(args.smpl_dir, vposer).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ---------- loop ---------- #
    for ep in range(1, args.epochs + 1):
        print(f"★ epoch {ep}/{args.epochs} (batch_size={args.batch_size}, lr={args.lr})")
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(dl, desc=f"E{ep}/{args.epochs}", unit="batch")
        for verts, betas, pose63 in pbar:            #← ここで既に stacked
            verts, betas, pose63 = (t.to(device, non_blocking=True)
                                    for t in (verts, betas, pose63))

            rec, mu, logvar, beta_h, pose_h = model(verts)
            loss, comp = loss_function(
                rec,
                verts.flatten(1),
                mu,
                logvar,
                beta_h,
                betas,
                pose_h,
                pose63,
                args.kl_w,
                args.beta_w,
                args.pose_w,
            )

            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_loss += loss.item()
            pbar.set_postfix({k: f'{v:.3e}' for k, v in comp.items()}, total=f'{loss.item():.3e}')

        print(f"◎ epoch {ep}: mean_loss={epoch_loss/len(dl):.4f}\n", flush=True)

        # quick visual
        model.eval()
        with torch.no_grad():
            v_in, _, _ = next(iter(dl))
            v_in = v_in.to(device)
            recon, *_ = model(v_in[:1])
            quick_scatter(f'quickview_epoch{ep:02d}.png',
                          v_in[0].cpu().numpy(),
                          recon[0].reshape(-1,3).cpu().numpy())

    torch.save(model.state_dict(), "vae_pointnet_vposer.pth")
    print("✔ saved → vae_pointnet_vposer.pth")

###############################################################################
# CLI
###############################################################################
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Minimal PointNet‑VPoser VAE with progress bars",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument('--data_dir',   default="/root/development/project4010/VAE/all_vertices_run_walk")
    ap.add_argument('--smpl_dir',   default="/root/development/project4010/VAE/models")
    ap.add_argument("--vposer_dir", default="/root/development/project4010/VAE/human_body_prior/support_data/downloads/vposer_v2_05")
    ap.add_argument("--batch_size", type=int, default=128) #デフォは32
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-4) #デフォは1e-3
    ap.add_argument("--kl_w", type=float, default=1e-2) #デフォは1e-4
    ap.add_argument("--beta_w", type=float, default=1.0)
    ap.add_argument("--pose_w", type=float, default=1.0)
    ap.add_argument("--sample_files", type=int, default=None)
    args = ap.parse_args()

    torch.manual_seed(0)
    train(args)
