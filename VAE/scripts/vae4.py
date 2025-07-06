#!/usr/bin/env python3
"""vae_pointnet_vposer_min.py – progress friendly & robust
===========================================================
* **最小構成**の PointNet‑VAE + VPoser + SMPL。
* tqdm 進捗付きで学習の様子を可視化。
* **壊れた .npz** を自動スキップして途中で止まらないよう改良。

変更点 (2025‑06‑09)
-------------------
1. `MotionDataset` スキャン時に **try/except** でファイルを検査し、読み込み失敗は警告＋スキップ。
2. スキャン完了後に `OK / skipped` 件数を表示。読み込めるファイルがゼロならエラーを出して終了。
3. そのほかのロジック・ログは前回と同じ。
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
    """Return (verts, betas, pose63) for each frame.

    壊れたファイルを検出してスキップできるようにした。
    """

    def __init__(self, data_dir: str, sample_files: int | None = None):
        all_files = list(Path(data_dir).glob("*_verts.npz"))
        candidate_files = (
            random.sample(all_files, sample_files) if sample_files else all_files
        )

        self.files: List[Path] = []
        skipped: List[Path] = []
        self.index: List[Tuple[Path, int]] = []

        for f in tqdm(candidate_files, desc="scan", unit="file"):
            try:
                n_frames = np.load(f, mmap_mode="r")["vertices"].shape[0]
            except Exception as e:  # noqa: BLE001 – broad but safe: corrupt/io error
                skipped.append(f)
                tqdm.write(f"⚠️  skip {f.name} ({e})")
                continue
            self.files.append(f)
            self.index.extend([(f, i) for i in range(n_frames)])

        print(
            f"→ dataset scan done: {len(self.files)} good / {len(skipped)} skipped"
        )
        if not self.files:
            raise RuntimeError("No readable *_verts.npz files found – aborting.")

    # ------------------------------------------------------------------
    # PyTorch Dataset API
    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        f, i = self.index[idx]
        data = np.load(f, mmap_mode="r")  # 正常なファイルのみが残っている
        verts = data["vertices"][i].astype(np.float32)
        pose_full = data["poses"][i].astype(np.float32)
        betas = data["betas"].astype(np.float32)
        pose63 = pose_full[3:66]
        return (
            torch.from_numpy(verts),
            torch.from_numpy(betas),
            torch.from_numpy(pose63),
        )

###############################################################################
# Model (unchanged)
###############################################################################


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
        self.vposer = vposer.eval()
        for p in self.vposer.parameters():
            p.requires_grad_(False)
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

def kld(mu, logvar):
    return -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))


def loss_fn(rec, tgt, mu, logvar, beta_h, beta_t, pose_h, pose_t, w_kl, w_beta, w_pose):
    return (
        F.mse_loss(rec, tgt)
        + w_kl * kld(mu, logvar)
        + w_beta * F.mse_loss(beta_h, beta_t)
        + w_pose * F.mse_loss(pose_h, pose_t)
    )

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
    ds = MotionDataset(args.data_dir, args.sample_files)
    print(f"→ dataset: {len(ds)} frames from {len(ds.files)} usable files")

    print("DataLoader: creating batches …", flush=True, end="")

    def collate(batch):
        v, b, p = map(torch.stack, zip(*batch))
        return v, b, p

    dl = DataLoader(ds,
                    batch_size=32,
                    shuffle=True,
                    num_workers=8,
                    pin_memory=True,
                    persistent_workers=True,
                    prefetch_factor=8, 
                    collate_fn=collate)
    
    # dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: tuple(zip(*b)))

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

            print("b")
            rec, mu, logvar, beta_h, pose_h = model(verts)
            print("c")
            loss = loss_fn(
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

            print("d")
            epoch_loss += loss.item()
            print("e")
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            print("f")

        print(f"◎ epoch {ep}: mean_loss={epoch_loss/len(dl):.4f}\n", flush=True)

    torch.save(model.state_dict(), "vae_pointnet_vposer_min.pth")
    print("✔ saved → vae_pointnet_vposer_min.pth")

###############################################################################
# CLI
###############################################################################
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Minimal PointNet‑VPoser VAE with progress bars",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument('--data_dir',   default="/root/development/project4010/VAE/all_vertices")
    ap.add_argument('--smpl_dir',   default="/root/development/project4010/VAE/models")
    ap.add_argument("--vposer_dir", default="/root/development/project4010/VAE/human_body_prior/support_data/downloads/vposer_v2_05")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--kl_w", type=float, default=1e-4)
    ap.add_argument("--beta_w", type=float, default=1.0)
    ap.add_argument("--pose_w", type=float, default=1.0)
    ap.add_argument("--sample_files", type=int, default=None)
    args = ap.parse_args()

    torch.manual_seed(0)
    train(args)
