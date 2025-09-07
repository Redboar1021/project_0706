import torch, smplx, numpy as np
from pytorch3d.loss import chamfer_distance

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

dev = "cuda"
pc  = torch.tensor(np.loadtxt("/root/development/project4010/gaussian-splatting/pointcloud_xyz/out_iter_30000.xyz"), dtype=torch.float32, device=dev)  # (N,3)

smpl = SMPL(model_path="/root/development/project4010/VAE/models", gender="NEUTRAL", batch_size=1)

# ── optim variables ───────────────────────────────
betas        = torch.zeros(10, device=dev, requires_grad=True)
body_pose    = torch.zeros(69, device=dev, requires_grad=True)   # 23×3 axis‑angle
global_orient= torch.zeros(3,  device=dev, requires_grad=True)
transl       = torch.zeros(3,  device=dev, requires_grad=True)
scale        = torch.ones(1,   device=dev, requires_grad=True)   # 等方スケール

optim = torch.optim.Adam([betas, body_pose, global_orient, transl, scale], lr=1e-2)

# --- 1) 学習ループ --------------------------------------
for it in range(400):
    out = smpl(global_orient=global_orient.unsqueeze(0),
               body_pose   =body_pose   .unsqueeze(0),
               betas       =betas       .unsqueeze(0),
               transl      =transl      .unsqueeze(0))
    verts = out.vertices[0] * scale                # (6890,3)
    loss, _ = chamfer_distance(verts.unsqueeze(0), pc.unsqueeze(0))
    optim.zero_grad(); loss.backward(); optim.step()

    if it % 50 == 0:
        print(f"{it:03d}: {loss.item():.4f}")

# --- 2) フィッティング後の点群を取得 --------------------------------
with torch.no_grad():
    out_final  = smpl(global_orient=global_orient.unsqueeze(0),
                      body_pose   =body_pose   .unsqueeze(0),
                      betas       =betas       .unsqueeze(0),
                      transl      =transl      .unsqueeze(0))
    verts_fit  = (out_final.vertices[0] * scale).cpu().numpy()  # (6890,3)

# --- 3) .xyz で保存 --------------------------------------------------
save_path = "/root/development/project4010/fitted_smpl_vertices.xyz"
np.savetxt(save_path, verts_fit, fmt="%.6f")   # x y z 形式で保存
print(f"Saved fitted vertices to: {save_path}")

# --- 4) 3D 散布図で確認 ---------------------------------------------
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D         # noqa: F401 (触るだけで 3D 有効化)

fig = plt.figure(figsize=(6, 6))
ax  = fig.add_subplot(111, projection="3d")
ax.scatter(verts_fit[:,0], verts_fit[:,1], verts_fit[:,2], s=1)
ax.set_title("Fitted SMPL Vertices")
ax.set_box_aspect([1,1,1])                      # 等方スケール表示
plt.show()

