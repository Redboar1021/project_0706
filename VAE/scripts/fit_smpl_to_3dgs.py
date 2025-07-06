# fit_tpose_to_3dgs.py – outer‑R only + visual check
"""
Fits an SMPL **T‑pose** to a *3‑D Gaussian Splatting* model (`Graph‑deco` PLY) using a single
global affine `(s,R,t)` implemented with **LearnableAffine**.  After optimisation it can *visualise*
the original Gaussian centres and the fitted SMPL vertices in **two colours** (blue vs red) so you
can eyeball how well they overlap.

Usage
-----
```bash
python fit_tpose_to_3dgs.py --ply gaussians.ply --smpl SMPL_NEUTRAL.pkl --show
```
Requires `open3d` if `--show` is given.
"""

from pathlib import Path
import math, argparse, json, time, numpy as np
import torch
from torch import nn
from tqdm import tqdm
from plyfile import PlyData
from smplx.body_models import SMPL
from typing import List, Tuple, Union   
from human_body_prior.models.vposer_model import VPoser
from human_body_prior.tools.model_loader import load_model
try:
    import open3d as o3d
    HAS_O3D = True
except ImportError:
    HAS_O3D = False

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

# ---------------------------------------------------------------------------
# Learnable affine -----------------------------------------------------------
# ---------------------------------------------------------------------------

class LearnableAffine(nn.Module):
    """p' = s·R·p + t  (isotropic scale + rotation + translation)"""
    def __init__(self, init_scale: float = 1.0):
        super().__init__()
        self.r = nn.Parameter(torch.zeros(3))
        self.log_s = nn.Parameter(torch.tensor([math.log(init_scale)], dtype=torch.float32))
        self.t = nn.Parameter(torch.zeros(3))

    @staticmethod
    def _rodrigues(r: torch.Tensor) -> torch.Tensor:
        eps = 1e-8; th = torch.linalg.norm(r) + eps; n = r / th
        nx, ny, nz = n; z = torch.zeros_like(nx)
        K = torch.stack([torch.stack([z, -nz, ny]),
                         torch.stack([nz, z, -nx]),
                         torch.stack([-ny, nx, z])])
        I = torch.eye(3, device=r.device, dtype=r.dtype)
        return I + torch.sin(th)*K + (1-torch.cos(th))*(K@K)

    def forward(self, pts: torch.Tensor):
        R = self._rodrigues(self.r); s = self.log_s.exp()
        return (s * (pts @ R.T)) + self.t

# ---------------------------------------------------------------------------
# helper functions -----------------------------------------------------------
# ---------------------------------------------------------------------------

def weighted_umeyama(src, dst, w):
    """
    src : (Ns,3)  ガウシアン中心
    dst : (Nd,3)  T-pose 頂点
    w   : (Ns,)   α で重み付け（0-1）
    """
    # ----- 1) src → dst の最近傍対応を 1 回だけ取る -----
    idx = torch.cdist(src, dst).argmin(dim=1)   # (Ns,)
    dst_corr = dst[idx]                         # (Ns,3)  ← 同じ長さに揃う

    # ----- 2) 従来の重み付き Umeyama -----
    w = w / w.sum()                             # 正規化
    mus = (w[:,None] * src      ).sum(0)
    mud = (w[:,None] * dst_corr ).sum(0)

    Sc, Dc = src - mus, dst_corr - mud
    cov = (w[:,None] * Sc).T @ Dc               # 3×3

    U,S,Vt = torch.linalg.svd(cov, full_matrices=False)
    R = Vt.T @ U.T
    if torch.linalg.det(R) < 0:                 # 反射を防ぐ
        Vt[-1] *= -1;  R = Vt.T @ U.T

    var = (w * (Sc**2).sum(1)).sum()
    s = S.sum() / var
    t = mud - s * (R @ mus)
    return s, R, t



def robust_chamfer(A,B,w=None,beta=3e-2):
    d1 = torch.cdist(A,B).min(1).values; d2 = torch.cdist(B,A).min(1).values
    if w is not None: d1*=w
    huber = lambda d: torch.where(d<beta,0.5*d**2/beta,d-0.5*beta)
    return huber(d1).mean()+huber(d2).mean()


def R_to_axis_angle(R,device):
    th = torch.arccos(((torch.trace(R)-1)/2).clamp(-1+1e-6,1-1e-6))
    axis = torch.tensor([R[2,1]-R[1,2],R[0,2]-R[2,0],R[1,0]-R[0,1]],device=device)/(2*torch.sin(th))
    return axis, th

# ---------------------------------------------------------------------------
# PLY loader -----------------------------------------------------------------
# ---------------------------------------------------------------------------

def load_gaussians(ply_path:str):
    ply = PlyData.read(ply_path); v = ply['vertex']
    xyz = np.stack([v['x'],v['y'],v['z']],1).astype(np.float32)
    alpha = np.asarray(v['opacity'],dtype=np.float32)
    return torch.from_numpy(xyz), torch.from_numpy(alpha)

# ---------------------------------------------------------------------------
# main optimisation routine --------------------------------------------------
# ---------------------------------------------------------------------------

def fit(xyz,alpha,smpl_model,vposer,device='cuda'):
    P = xyz.to(device); W = (alpha/alpha.max()).to(device)
    V0 = smpl_model().vertices.squeeze(0).to(device)
    s0,R0,t0 = weighted_umeyama(P,V0,W)

    affine = LearnableAffine(init_scale=s0).to(device)
    axis,th = R_to_axis_angle(R0,device); affine.r.data.copy_(axis*th); affine.t.data.copy_(t0)

    betas  = torch.zeros(10, device=device, requires_grad=True)
    pose63 = torch.zeros(63, device=device, requires_grad=True)
    pose69 = torch.cat([pose63.unsqueeze(1), torch.zeros(xyz.size(0), 6, device=pose63.device)], 1)


    for p in vposer.encoder_net.parameters():
        p.requires_grad_(False)  
    for p in vposer.decoder_net.parameters():
        p.requires_grad_(False)   # ←固定

    opt = torch.optim.Adam([
        {'params': betas, 'lr':5e-3},
        {'params': pose63,'lr':5e-3},
        {'params': affine.parameters(),'lr':5e-3}
    ])

    pbar = tqdm(range(10), unit="iter")
    for it in pbar:
        dist = vposer.encode(pose63[None])
        z = dist.rsample()
        pose_dec = vposer.decode(z)["pose_body"].reshape(z.size(0), -1)
        pose69 = torch.cat([pose_dec, torch.zeros(xyz.size(0), 6, device=pose_dec.device)], 1)
        V = smpl_model(betas=betas[None], body_pose=pose69[None]).vertices.squeeze(0)
        Vg = affine(V)
        loss = robust_chamfer(P,Vg,W) + 1e-3*pose69.pow(2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 50 == 0:                       # ←更新頻度は好みで
            deg = affine.r.norm().item()*180/math.pi
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                theta=f"{deg:5.1f}°",
                s=f"{affine.log_s.exp().item():.3f}"
            )

    return affine, betas.detach().cpu(), pose69.detach().cpu()

# ---------------------------------------------------------------------------
# visualisation --------------------------------------------------------------
# ---------------------------------------------------------------------------

# ---------------------------------------------
# 4 方向から静止画を描くオーバーレイ表示関数
# （Open3D ウィンドウが開かない環境向け）
# ---------------------------------------------
def show_overlay(
        gauss_xyz: torch.Tensor,
        smpl_xyz:  torch.Tensor,
        angles: List[Tuple[int, int]] = [  # ★ 修正
            (10, 0),    # 正面
            (10, 90),   # 右側面
            (10, 180),  # 背面
            (80, 0)     # 上面
        ],
        out_dir: Union[str, Path, None] = None
    ):
    """
    2 点群を 4 方向から描画して 2×2 グリッドで表示。
    `out_dir` を指定すると PNG も保存する。

    Parameters
    ----------
    gauss_xyz : (N,3) torch.Tensor
    smpl_xyz  : (M,3) torch.Tensor
    angles    : list[(elev, azim)]
        matplotlib `view_init` 用の (仰角, 方位角) の一覧
    out_dir   : str | Path | None
        保存先ディレクトリ。None なら保存しない
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from pathlib import Path

    ga = gauss_xyz
    sm = smpl_xyz.cpu().numpy()

    fig = plt.figure(figsize=(8, 8))
    for i, (elev, azim) in enumerate(angles, start=1):
        ax = fig.add_subplot(2, 2, i, projection="3d")
        ax.scatter(ga[:, 0], ga[:, 1], ga[:, 2], s=1, c="b")
        ax.scatter(sm[:, 0], sm[:, 1], sm[:, 2], s=1, c="r")
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        ax.set_title(f"elev={elev}, azim={azim}")

    plt.tight_layout()
    plt.show()

    # -------- ファイル保存 --------
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, (elev, azim) in enumerate(angles, start=1):
            fig = plt.figure(figsize=(4, 4))
            ax  = fig.add_subplot(111, projection="3d")
            ax.scatter(ga[:, 0], ga[:, 1], ga[:, 2], s=1, c="b")
            ax.scatter(sm[:, 0], sm[:, 1], sm[:, 2], s=1, c="r")
            ax.view_init(elev=elev, azim=azim)
            ax.set_axis_off()
            fig.savefig(out_dir / f"overlay_{i:02d}.png", dpi=300, bbox_inches="tight")
            plt.close(fig)


# ---------------------------------------------------------------------------
if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ply', default = "/root/development/project4010/gaussian-splatting/output/80f38318-e/point_cloud/iteration_30000/point_cloud.ply")
    ap.add_argument('--smpl', default="/root/development/project4010/VAE/models")
    ap.add_argument("--vposer_dir", default="/root/development/project4010/VAE/human_body_prior/support_data/downloads/vposer_v2_05")
    ap.add_argument('--show', action='store_true', help='open3d overlay view')
    args = ap.parse_args()

    xyz, alpha = load_gaussians(args.ply)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    smpl = SMPL(model_path=args.smpl, gender="NEUTRAL", batch_size=1).to(device)
    vposer, _ = load_model(
        args.vposer_dir,
        model_code=VPoser,
        remove_words_in_model_weights="vp_model.",
        disable_grad=True,
    )
    vposer = vposer.to(device)

    start = time.time()
    affine, betas, pose69 = fit(xyz, alpha, smpl, vposer, device)
    elapsed = time.time()-start
    print('optim finished in', round(elapsed,2), 's')

    # save json
    out = Path(args.ply).with_suffix('.fit.json')
    R_final = LearnableAffine._rodrigues(affine.r.detach()).cpu()
    res = dict(betas=betas.tolist(), pose=pose69.tolist(), scale=affine.log_s.exp().item(),
               R=R_final.tolist(), t=affine.t.detach().cpu().tolist())
    out.write_text(json.dumps(res, indent=2)); print('saved', out)

    # save transformed SMPL vertices as XYZ for quick inspection
    with torch.no_grad():
        V_final = smpl(betas=betas[None].to(device), body_pose=pose69[None].to(device)).vertices.squeeze(0)
        Vg = affine(V_final).cpu().numpy()
    xyz_path = Path(args.ply).with_suffix('.fit_smpl.xyz')
    np.savetxt(xyz_path, Vg, fmt='%.6f')
    print('saved', xyz_path)

    # optional visual check
    with torch.no_grad():
        V = smpl(betas=betas[None].to(device), body_pose=pose69[None].to(device)).vertices.squeeze(0)
        Vg = affine(V).cpu().numpy()
    show_overlay(xyz.numpy(), Vg, out_dir="views")
