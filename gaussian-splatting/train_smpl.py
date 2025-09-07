import math
import os
import sys
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui, render_affine, render_pca, render_silhouette
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import numpy as np
from pathlib import Path
import pickle
from typing import Union, Optional, Tuple, List, Dict
from torch_scatter import scatter_add
import time, torch
def tic(): torch.cuda.synchronize(); return time.perf_counter()
def toc(t0, label): 
    torch.cuda.synchronize()
    print(f"{label}: {(time.perf_counter()-t0)*1e3:.1f} ms")
    return tic()

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from VAE.scripts.vae6 import VAE_PointNet
import torch.nn.functional as F
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

# ---------- NumPy 互換 (NumPy 1.24+) ----------
import inspect
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
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from mesh_neigh import MeshNeighborhoodLoss, compute_covariances_loop
from torchvision.utils import save_image

#!/usr/bin/env python3
"""
Fit SMPL parameters to a point cloud (XYZ) with tqdm progress bars.

- loss_curve.png     : ロス推移グラフ
- smpl_fitted.obj    : 最適化後メッシュ
- smpl_fitted.png    : レンダリング結果
"""
# -----------------------------------------------------------
import os, sys, argparse, pathlib, mmap, math
import numpy as np
import torch
import matplotlib.pyplot as plt
import trimesh
from tqdm import tqdm

from smplx import SMPL
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.loss import point_mesh_face_distance
from pytorch3d.renderer import (
    look_at_view_transform, FoVPerspectiveCameras,
    MeshRenderer, MeshRasterizer, RasterizationSettings,
    SoftPhongShader, PointLights, TexturesVertex
)

# ------------------------- 引数 -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--pointcloud",
    default="/mnt/c/Users/Yoshi/development/gaussian_splatting_20240729/"
            "gaussian-splatting/pointcloud_xyz/out_iter_30000.xyz",
    help="Windows 側 .xyz の WSL パス"
)
parser.add_argument('--smpl_dir',   default="/root/development/project4010/VAE/models")
parser.add_argument("--gender", default="neutral",
                    choices=["male", "female", "neutral"])
parser.add_argument("--iters", type=int, default=400)
parser.add_argument("--lr", type=float, default=5e-3)
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()
device = torch.device(args.device)

# ------------------- tqdm 付き XYZ 読込 ---------------------
def load_xyz_tqdm(path: pathlib.Path) -> torch.Tensor:
    """大量ポイントでも進捗を出しつつ XYZ 読み込み"""
    # 総行数を先に数えて tqdm の total に使う
    with open(path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        total_lines = 0
        while mm.readline():
            total_lines += 1

    pts = np.zeros((total_lines, 3), dtype=np.float32)
    with open(path, "r") as f:
        for i, line in enumerate(tqdm(f, total=total_lines, desc="Loading XYZ")):
            try:
                x, y, z = map(float, line.strip().split()[:3])
                pts[i] = (x, y, z)
            except ValueError:
                raise RuntimeError(f"Line {i+1} is malformed: {line}")

    return torch.tensor(pts, dtype=torch.float32, device=device)

def kld_standard(mu, logvar):
    return -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

xyz_path = pathlib.Path(args.pointcloud)
if not xyz_path.exists():
    sys.exit(f"[ERROR] Point cloud file not found: {xyz_path}")

points = load_xyz_tqdm(xyz_path)
pointcloud = Pointclouds(points=[points])  # (1, N, 3)

print(f"[INFO] Loaded {points.shape[0]:,} points from {xyz_path}")
# 必要ならここで中心化・スケール合わせを行う
# points -= points.mean(0, keepdim=True)

#アフィン変換用
R = torch.tensor(  # +X 軸回りに +90°:  y→z
    [[1, 0,  0],
     [0, 0, -1],
     [0, 1,  0]], dtype=torch.float32, device=device)        # (3,3)

t = torch.tensor([0.0, 0.0, 1.1618], dtype=torch.float32,    # (3,)
                 device=device).view(1, 1, 3)               # ← (1,1,3) にブロードキャスト用 reshape

# -------------------------- SMPL ----------------------------
smpl = SMPL(model_path=args.smpl_dir, gender=args.gender,
            create_global_orient=True, create_body_pose=True,
            create_betas=True).to(device)

betas         = torch.zeros(1, 10,  device=device, requires_grad=True)
global_orient = torch.zeros(1, 3,   device=device, requires_grad=False)
body_pose     = torch.zeros(1, 69,  device=device, requires_grad=True)
transl        = torch.zeros(1, 3,   device=device, requires_grad=False)

optim = torch.optim.Adam([betas, body_pose], lr=args.lr)

faces = torch.tensor(smpl.faces.astype(np.int64), device=device)[None]

# --- VPoser & VAE (読込のみ／勾配停止) --------------------------
vposer, _ = load_model(
        "/root/development/project4010/VAE/human_body_prior/support_data/downloads/vposer_v2_05",
        model_code=VPoser,
        remove_words_in_model_weights="vp_model.",
        disable_grad=True,          # ←ここで VPoser 側は完全停止
    )

vposer = vposer.to(device)

# --------------------- 最適化ループ --------------------------
loss_pmf_log, loss_rec_log, loss_kl_log = [], [], []
pbar = tqdm(range(1, args.iters + 1), desc="Fitting SMPL", unit="iter")

for it in pbar:
    out   = smpl(betas=betas, body_pose=body_pose,
                 global_orient=global_orient, transl=transl)
    verts = out.vertices            # (1, 6890, 3)
    verts_affine = torch.matmul(verts, R.T) + t    # 回転→並進（バッチ対応）
    mesh  = Meshes(verts=verts_affine, faces=faces)

    dist = vposer.encode(body_pose[0:1, :63])
    z = dist.rsample()
    pose_dec = vposer.decode(z)["pose_body"]
    L_rec  = F.l1_loss(pose_dec.reshape(1, 63), body_pose[0:1, :63], reduction='mean')
    mu, logvar = dist.loc, torch.log(dist.scale ** 2 + 1e-8)
    L_kl   = kld_standard(mu, logvar)
    λ_rec = 4.0*0.1
    λ_kl  = 0.005*0.1

    loss_pmf = point_mesh_face_distance(mesh, pointcloud).mean()
    loss = loss_pmf + λ_rec * L_rec + λ_kl * L_kl
    optim.zero_grad()
    loss.backward()
    optim.step()

    # --------- ログと tqdm 表示 --------------
    loss_pmf_log.append(loss_pmf.item())
    loss_rec_log.append(λ_rec * L_rec.item())
    loss_kl_log.append(λ_kl * L_kl.item())
    pbar.set_postfix(geom=f"{loss_pmf.item():.3e}",
                     rec=f"{L_rec.item():.3e}",
                     kl=f"{L_kl.item():.3e}")

# ------------------ グラフ描画 -----------------
import matplotlib.pyplot as plt

iters = range(1, len(loss_pmf_log) + 1)
fig, ax_left = plt.subplots(figsize=(7,4))

# ---------- 左軸：pmf ----------
ax_left.plot(iters, loss_pmf_log, label="pmf", color="tab:blue")
ax_left.set_xlabel("Iteration")
ax_left.set_ylabel("pmf", color="tab:blue")
ax_left.tick_params(axis="y", labelcolor="tab:blue")
ax_left.set_yscale("log")

# ---------- 右軸：recon & kl ----------
ax_right = ax_left.twinx()
ax_right.plot(iters, loss_rec_log, label=f"{λ_rec:g}*recon", color="tab:green")
ax_right.plot(iters, loss_kl_log,  label=f"{λ_kl:g}*kl",    color="tab:red")
ax_right.set_ylabel("recon / kl", color="tab:red")
ax_right.tick_params(axis="y", labelcolor="tab:red")
ax_right.set_yscale("log")

# ---------- 凡例 ----------
lines  = ax_left.get_lines() + ax_right.get_lines()
labels = [line.get_label() for line in lines]
ax_left.legend(lines, labels, loc="upper right")

plt.tight_layout()
plt.savefig("loss_curve.png", dpi=200)
print("[INFO] loss_curve.png saved")


fitted_mesh = trimesh.Trimesh(vertices=out.vertices[0].detach().cpu().numpy(),
                              faces=smpl.faces, process=False)
fitted_mesh.export("smpl_fitted.obj")
print("[INFO] smpl_fitted.obj saved")

textures = TexturesVertex(verts_features=torch.ones_like(verts))
R_temp, T_temp = look_at_view_transform(dist=3.0, elev=90.0, azim=180.0, device=device)
cameras = FoVPerspectiveCameras(R=R_temp, T=T_temp, device=device)
renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras,
        raster_settings=RasterizationSettings(image_size=512, faces_per_pixel=1)),
    shader=SoftPhongShader(device=device, cameras=cameras,
                           lights=PointLights(device=device, location=[[0,0,3]]))
)

# --- 1) 最適化が終わった後にもう一度 verts_affine を作成 ---
with torch.no_grad():
    verts_final = torch.matmul(out.vertices, R.T) + t        # (1, 6890, 3)

textures = TexturesVertex(verts_features=torch.ones_like(verts_final))
mesh_tex = Meshes(verts=verts_final, faces=faces, textures=textures)

fitted_mesh_affine = trimesh.Trimesh(vertices=verts_final[0].detach().cpu().numpy(),
                              faces=smpl.faces, process=False)
fitted_mesh_affine.export("smpl_fitted_affine.obj")
print("[INFO] smpl_fitted_affine.obj saved")

# --- 描画して保存 (1 回だけ) ---
img = renderer(mesh_tex.to(device))[0, ..., :3].cpu().numpy()
plt.imsave("smpl_fitted.png", img)
print("[INFO] smpl_fitted.png saved")
print("=== DONE ===")
