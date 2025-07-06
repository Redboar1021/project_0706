#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
"loss2に1e-4かけてる"
"密度化止めてる"
"初期値Tpose"
import math
import os
import sys
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui, render_affine, render_pca, render_silhouette, render_parents, render_silhouette_parents
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
from VAE.scripts.vae5 import VAE_PointNet
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

def to_uint8(t: torch.Tensor) -> torch.Tensor:
    """
    0-1 float / 0-255 float / uint8 のいずれでも受け取り、
    CHW 形式の uint8 (0-255) にして返す。
    """
    # ① CHW or HWC どちらでも受け付ける → CHW にそろえる
    if t.ndim == 3 and t.shape[0] not in (1, 3):      # HWC → CHW
        t = t.permute(2, 0, 1)
    # ② CPU に移し、float → uint8 へ
    t = t.detach().cpu()
    if t.dtype != torch.uint8:
        t = (t.clamp(0, 1) * 255).to(torch.uint8)
    return t

class LearnableAffine(nn.Module):
    "shapeは(B,N,3)でも(N,3)でも可。"
    def __init__(self, init_scale=1.0):
        super().__init__()
        self.r = nn.Parameter(torch.zeros(3))                 # 回転
        self.log_s = nn.Parameter(                            # log_s に対数を保持
            torch.tensor([np.log(init_scale)], dtype=torch.float32)
        )
        self.t = nn.Parameter(torch.zeros(3))                 # 並進

    @staticmethod
    def _rodrigues(r: torch.Tensor) -> torch.Tensor:
        eps = 1e-8
        θ   = torch.linalg.norm(r) + eps         # 常に >0
        n   = r / θ                              # (3,)
        nx, ny, nz = n                           # スカラー Tensor

        zero = torch.zeros_like(nx)              # Tensor 0 として扱う

        K = torch.stack([
            torch.stack([ zero, -nz,  ny]),
            torch.stack([  nz,  zero, -nx]),
            torch.stack([-ny,   nx,  zero])
        ])                                        # (3,3)

        I = torch.eye(3, device=r.device, dtype=r.dtype)
        return I + torch.sin(θ) * K + (1 - torch.cos(θ)) * (K @ K)

    def forward(self, pts):      # pts: (...,3)
        R = self._rodrigues(self.r)
        s = torch.exp(self.log_s)             # スケール (正値)
        return (s * (pts @ R.T)) + self.t


def freeze_gaussians(model: GaussianModel, freeze: bool = True): #これかえるとアフィン変換うまくいかん
    """freeze=True で Gaussians を凍結（勾配+更新を停止）"""
    # ① requires_grad を切り替え
    for tensor in (
        model._xyz, model._scaling, model._rotation):
        tensor.requires_grad_(not freeze)
    # for tensor in (
    #     model._xyz, model._scaling, model._rotation,
    #     model._opacity, model._features_dc, model._features_rest):
    #     tensor.requires_grad_(not freeze)

    # ② SparseGaussianAdam の param_groups.lr を 0↔元値 で切替
    # for g in model.optimizer.param_groups:
    #     if freeze:
    #         # 保存して後で戻せるようにする
    #         g.setdefault("backup_lr", g["lr"])
    #         g["lr"] = 0.0
    #     else:
    #         if "backup_lr" in g:
    #             g["lr"] = g.pop("backup_lr")

def freeze_gaussians_except_xyz(model: GaussianModel, freeze: bool = True):
    """freeze=True で Gaussians を凍結（勾配+更新を停止）"""
    # ① requires_grad を切り替え
    for tensor in (
        model._xyz, model._scaling, model._rotation):
        tensor.requires_grad_(not freeze)

# ---------- ヘルパ関数 ----------
def _freeze_first_rows(param: torch.Tensor, n_rows: int):
    """
    param の先頭 n_rows 分の勾配をゼロにする hook を登録する。
    戻り値は hook handle。不要になったら handle.remove() で解除。
    """
    def hook(grad):
        grad[:n_rows] = 0
        return grad

    return param.register_hook(hook)


# ---------- 使い方 ----------
def freeze_gaussians_var_6890(model: GaussianModel, freeze: bool = True):
    """
    freeze=True なら _scaling / _rotation の先頭 6890 個を凍結する。
    freeze=False なら以前付けた hook を解除する。
    """
    if not hasattr(model, "_freeze_hooks"):
        model._freeze_hooks = {}

    targets = {"scaling": model._scaling, "rotation": model._rotation}

    if freeze:
        for name, p in targets.items():
            if name not in model._freeze_hooks:           # 二重登録を防ぐ
                model._freeze_hooks[name] = _freeze_first_rows(p, 6890)
    else:
        # 既存 hook を解除
        for h in model._freeze_hooks.values():
            h.remove()
        model._freeze_hooks.clear()

def add_upper_bound_hook(param: torch.Tensor, max_val: float, n_fixed: int):
    """
    param[:n_fixed] は対象外（固定）。
    param[n_fixed:] が max_val を超えないよう勾配を操作する hook を登録。
    返り値: hook handle
    """
    def hook(grad):
        # 現在値が max_val 以上 & 上方向の勾配 (>0) → 勾配を 0 に
        mask = (param.data[n_fixed:] >= max_val) & (grad[n_fixed:] > 0)
        grad[n_fixed:][mask] = 0
        return grad

    return param.register_hook(hook)

from collections import deque
from typing import Deque, Tuple



# --------------------------------------------------------------------------
# 1) 法線計算ユーティリティ
# --------------------------------------------------------------------------
@torch.no_grad()
def compute_vertex_normals(v: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """
    v      : (N,3) float32 cuda
    faces  : (F,3) long cuda
    return : (N,3) 単位外向き法線
    """
    v0, v1, v2 = v[faces[:, 0]], v[faces[:, 1]], v[faces[:, 2]]
    fn = torch.cross(v1 - v0, v2 - v0)                           # (F,3)
    fn = fn / (fn.norm(dim=1, keepdim=True) + 1e-12)

    n = torch.zeros_like(v)
    for k in range(3):
        n.index_add_(0, faces[:, k], fn)                         # 面→頂点
    n = n / (n.norm(dim=1, keepdim=True) + 1e-12)
    return n


# --------------------------------------------------------------------------
# 2) Δ·n_loss + K-step 管理クラス
# --------------------------------------------------------------------------
class LossState:
    """
    ・prev_g, prev_n : 直近 1 step の参照
    ・hist           : K step 前の (g, n)
    """
    def __init__(
        self,
        gaussians,
        faces: torch.Tensor,
        *,
        K: int = 25,
        margin: float = 0.0,
        weight_1step: float = 5e-4,
        weight_Kstep: float = 5e-4,
    ):
        self.faces = faces                                            # (F,3) long cuda
        self.margin = margin
        self.w1 = weight_1step
        self.wK = weight_Kstep
        self.K = K

        with torch.no_grad():
            g0 = gaussians.get_xyz[:6890].clone().detach()       # (6890,3)
            n0 = compute_vertex_normals(g0, faces)

        self.prev_g = g0            # (6890,3)
        self.prev_n = n0            # (6890,3)
        self.hist: Deque[Tuple[torch.Tensor, torch.Tensor]] = deque(
            [(g0, n0)], maxlen=K + 1
        )

    # --------------------------------------------------------------
    # 呼び出し側は optimizer.step() のあとに state.step(new_g) を
    # --------------------------------------------------------------
    @torch.no_grad()
    def step(self, g_new: torch.Tensor):
        """
        g_new : (6890,3) float32 cuda  (detach された最新ガウシアン中心)
        """
        n_new = compute_vertex_normals(g_new, self.faces)
        self.hist.append((g_new, n_new))
        self.prev_g, self.prev_n = g_new, n_new

    # --------------------------------------------------------------
    # K step 前の参照を取得
    # --------------------------------------------------------------
    def get_Kstep(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # deque[0] が最古、[-1] が最新
        return self.hist[0] if len(self.hist) > 1 else (self.prev_g, self.prev_n)


# --------------------------------------------------------------------------
# 3) Δ·n ペナルティ本体
# --------------------------------------------------------------------------
def _delta_penalty(g: torch.Tensor, g_ref: torch.Tensor, n_ref: torch.Tensor,
                   margin: float, weight: float) -> torch.Tensor:
    """
    g      : (N,3) requires_grad=True
    g_ref  : (N,3) no_grad
    n_ref  : (N,3) no_grad
    """
    delta = g - g_ref
    d = (delta * n_ref).sum(dim=1)                  # signed step along normal
    penal = F.relu(margin - d).pow(2).mean()
    return weight * penal


def delta_n_penalty(g: torch.Tensor, state: LossState) -> torch.Tensor:
    """
    g     : (6890,3) requires_grad=True
    state : LossState
    戻り値: scalar loss
    """
    # ① 1-step 基準
    loss = _delta_penalty(
        g, state.prev_g, state.prev_n,
        margin=state.margin, weight=state.w1
    )

    # ② K-step 基準
    gK, nK = state.get_Kstep()
    loss += _delta_penalty(
        g, gK, nK,
        margin=state.margin, weight=state.wK
    )
    return loss

# ------------------------------------------------------------
# 1. k-NN グラフから edge_idx を作るユーティリティ
# ------------------------------------------------------------
def knn_edges(mu0: torch.Tensor,
              k: int = 6,
              radius: Optional[float] = None,   # ←ここを修正
              device: Union[str, torch.device] = "cuda") -> torch.LongTensor:
    """
    mu0    : (N,3)  – カノニカル (T pose) 位置
    k      : 1 頂点あたり近傍数
    radius : (m) 上限距離。None なら無制限
    return : (2,E)  – edge_idx  (i→j 方向, 片向き)
    """
    with torch.no_grad():
        N = mu0.shape[0]
        # batched cdist に切り替えたい場合はここを分割
        dist = torch.cdist(mu0, mu0)                    # (N,N)
        knn_idx = dist.topk(k + 1, largest=False).indices[:, 1:]  # 自分以外

        src = torch.arange(N, device=mu0.device).repeat_interleave(k)
        dst = knn_idx.reshape(-1)

        if radius is not None:
            mask = dist[src, dst] < radius
            src, dst = src[mask], dst[mask]

        edges = torch.stack([src, dst], 0)              # 片向き
        edges = torch.cat([edges, edges.flip(0)], 1)    # 対称化
        edges = torch.unique(edges, dim=1)              # 重複除去
        return edges.to(device)


# ------------------------------------------------------------
# 2. ARAP 正則化モジュール
# ------------------------------------------------------------
class ARAPLoss(nn.Module):
    def __init__(self,
                 edge_idx: torch.LongTensor,
                 weight: float = 1e-2,
                 eps: float = 1e-8):
        """
        edge_idx : (2,E)
        weight   : λ_arap
        """
        super().__init__()
        self.register_buffer("edge_idx", edge_idx)   # (2,E)
        self.weight = weight
        self.eps = eps
        # 単位行列 (det<0 修正用)
        self.register_buffer("diag_fix",
                             torch.tensor([-1.0, 1.0, 1.0]).view(1, 3))

    def forward(self, mu0: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """
        mu0 : (N,3)  – カノニカル位置 (requires_grad=False)
        mu  : (N,3)  – 現在位置 (requires_grad=True)
        """
        i, j = self.edge_idx        # (E,)
        rest   = mu0[j] - mu0[i]    # (E,3)
        deform = mu[j] - mu[i]      # (E,3)

        w = 1.0 / (rest.norm(dim=-1) + self.eps)      # (E,)
        outer = torch.einsum('e,ei,ej->eij', w, deform, rest)

        # S_i = Σ w (d ⊗ r) を index_add でアキュムレート
        S = torch.zeros(mu0.size(0), 3, 3, device=mu0.device)
        S.index_add_(0, i, outer)                     # (N,3,3)

        # batched SVD, full_matrices=False → U,Vh:(N,3,3)
        U, _, Vh = torch.linalg.svd(S, full_matrices=False)
        R = U @ Vh

        det_mask = (torch.det(R) < 0).unsqueeze(-1).unsqueeze(-1)
        Vh = torch.where(det_mask, Vh * self.diag_fix, Vh)
        R = U @ Vh                                    # det>0 保証

        Ri   = R[i]                                   # (E,3,3)
        pred = torch.einsum('eij,ej->ei', Ri, rest)   # 回転後 r
        err  = deform - pred                          # (E,3)

        loss = (w * err.square().sum(dim=-1)).sum() / w.sum()
        return self.weight * loss
    
import cv2
import scipy.ndimage as ndi

import gc, torch, cv2, numpy as np, scipy.ndimage as ndi



def make_clean_silhouette(img: torch.Tensor, thr: int = 1,
                    kernel: int = 3, device: Optional[torch.device] = None):
    """
    背景が (0,0,0) の画像から人物シルエットを作る高速版。
    * img    : (3,H,W) または (B,3,H,W)  float[0-1] / float[0-255] / uint8 いずれでも OK
    * thr    : 各チャネル > thr が 1 つでもあれば前景とみなす (0-255 基準)
    * kernel : 3 or 5 など; モルフォロジ閉処理のカーネルサイズ
    * 戻り値 : 前景=255, 背景=0 の uint8 (同次元)
    """

    dev = device or img.device
    x = img.to(dev)

    # ── 値域と dtype を統一 ────────────────────────────────
    if x.dtype != torch.uint8:
        scale = 255.0 if x.max() <= 1.0 else 1.0
        x = (x * scale).to(torch.uint8)

    # ── 3→1 チャンネル bool マスク: (B,1,H,W) ──────────────
    if x.ndim == 3:        # (3,H,W) → (1,3,H,W)
        x = x.unsqueeze(0)
    m = (x > thr).any(dim=1, keepdim=True)          # bool

    # ── モルフォロジ閉処理 (膨張→収縮) ※パディング same ─────
    pad = kernel // 2
    m = F.max_pool2d(m.float(), kernel, stride=1, padding=pad)     # 膨張
    m = -F.max_pool2d(-m, kernel, stride=1, padding=pad)           # 収縮
    m = m.bool()

    # ── 3チャンネル 0/255 に整形 ─────────────────────────
    mask = m.repeat(1, 3, 1, 1).to(torch.uint8) * 255
    if img.ndim == 3:                      # 入力が (3,H,W) なら squeeze
        mask = mask.squeeze(0)
    return mask

import faiss
def build_knn_once(x, k):
    cpu_x = x.detach().cpu().numpy().astype("float32")
    index = faiss.IndexFlatL2(3)
    index.add(cpu_x)
    _, idx = index.search(cpu_x, k+1)
    return torch.from_numpy(idx[:, 1:])  # 自身を除く

def spring_loss_knn(x, x0, edge_idx, w):
    d_now = (x.unsqueeze(1) - x[edge_idx]).pow(2).sum(-1).sqrt()
    d_ini = (x0.unsqueeze(1) - x0[edge_idx]).pow(2).sum(-1).sqrt()
    return w * (d_now - d_ini).pow(2).mean()






def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    pca = True
    global edge_idx_knn, x_snap
    edge_idx_knn  = None       # (N,k) の近傍インデックス
    x_snap    = None       # spring 基準点群

    first_iter = -5000
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    


    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- VPoser & VAE (読込のみ／勾配停止) --------------------------
    vposer, _ = load_model(
        "/root/development/project4010/VAE/human_body_prior/support_data/downloads/vposer_v2_05",
        model_code=VPoser,
        remove_words_in_model_weights="vp_model.",
        disable_grad=True,          # ←ここで VPoser 側は完全停止
    )

    vposer = vposer.to(device)

    vae = VAE_PointNet("/root/development/project4010/VAE/models", vposer).to(device)
    vae.load_state_dict(torch.load("/root/development/project4010/vae_pointnet_vposer_temp.pth",
                                   map_location=device))
    vae.eval()                      # ① 推論モード
    for p in vae.parameters():      # ② requires_grad=False で凍結
        p.requires_grad_(False)

    # --- 前処理用アフィン -------------------------------
    pre_affine = LearnableAffine().to(device)
    # pre_affine.r.data = torch.tensor([math.pi/2 - 0.02, 0, math.pi - 0.02], device=device) #temp2用
    pre_affine.r.data = torch.tensor([math.pi/2 - 0.02, 0, 0], device=device) #temp用

    rot_params  = [pre_affine.r]          # 回転だけ
    main_params = [pre_affine.log_s, pre_affine.t]

    opt_rot  = torch.optim.Adam(rot_params,  lr=5e-2)
    opt_main = torch.optim.Adam(main_params, lr=1e-2)

    sched_rot  = torch.optim.lr_scheduler.ExponentialLR(opt_rot,  gamma=0.995)
    sched_main = torch.optim.lr_scheduler.StepLR(opt_main, step_size=2000, gamma=0.5)
    cosine_rot = torch.optim.lr_scheduler.CosineAnnealingLR(opt_rot, T_max=8000, eta_min=1e-4)

    # --- 学習可能アフィンだけを作成 -------------------------------
    affine = LearnableAffine().to(device)


    # --- MeshNeighborhoodLoss ---------------------
    loss_helper = MeshNeighborhoodLoss(
        verts_init = gaussians.get_xyz,           # 初期 6890×3
        smpl_pkl   = '/root/development/project4010/VAE/models/SMPL_NEUTRAL.pkl',
        device     = device
    )


    smpl_pkl = Path("/root/development/project4010/VAE/models/SMPL_NEUTRAL.pkl")
    with open(smpl_pkl, "rb") as f:
        faces = pickle.load(f, encoding="latin1")["f"].astype("int64")   # (13776, 3)
    
    faces_t = torch.as_tensor(faces, device=device)                # Tensor 化


    # MAX_SCALE = -4.5
    MAX_SCALE = -3.912
    hook_handle = add_upper_bound_hook(
        gaussians._scaling,
        MAX_SCALE,
        n_fixed=6890
    )

    state = LossState(
        gaussians=gaussians,
        faces=faces_t,
        K=25,              # ← “じわじわ侵入” に 25 step 幅で対応
        margin=0.0,        # ← 1 mm 許容なら 1e-3
        weight_1step=1e-1,
        weight_Kstep=1e-1,
    )
    means3D_iteration_0 = None
    edge_idx = None
    arap_loss = None

    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        if iteration < 1:
            if iteration == first_iter:
                freeze_gaussians(gaussians, freeze=True)   # Gaussians を凍結
            # Pick a random Camera
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
            # Render
            bg = torch.rand((3), device="cuda") if opt.random_background else background

            render_pkg = render_affine(viewpoint_cam, gaussians, pre_affine, pipe, bg)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            #ガウシアンの座標を取得
            means3D = gaussians.get_xyz
            means3D_6890 = means3D[:6890]   
            #ガウシアンの透明度を取得
            opacity = gaussians.get_opacity   #0になるほど透明

            # Loss
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            loss.backward()
            # print(
            #     f"|grad_r|={pre_affine.r.grad.norm().item():.3e}, "
            #     f"|grad_s|={pre_affine.log_s.grad.abs().item():.3e}, "
            #     f"|grad_t|={pre_affine.t.grad.norm().item():.3e}"
            # )
            with torch.no_grad():
                opt_rot.step();   opt_rot.zero_grad(set_to_none=True)
                opt_main.step();  opt_main.zero_grad(set_to_none=True)
                gaussians.optimizer.step(); gaussians.optimizer.zero_grad(set_to_none=True)
                if iteration < 2000:
                    sched_rot.step();  sched_main.step()
                elif iteration == 2000:
                    # reset LR for cosine
                    for g in opt_rot.param_groups: g["lr"] = 1e-3
                else:
                    cosine_rot.step(); sched_main.step()
 
            # outputを保存
            out_dir = Path("pointcloud_xyz")
            out_dir.mkdir(exist_ok=True)
            verts = pre_affine(gaussians.get_xyz)  # (N, 3)


            if iteration % 100 == 0 or iteration == -4999 or iteration == -4998 or iteration == -4997 or iteration == -4996:
                fname = out_dir / f"pre_iter_{iteration:05d}.xyz"
                np.savetxt(fname, verts.cpu().detach().numpy(), fmt="%.6f")
                print(f"saved → {fname}")
            
            

            if iteration == 0:
                freeze_gaussians(gaussians, freeze=False)  # Gaussians を解凍
                # ガウシアンにアフィン変換かける
                gaussians.apply_affine(pre_affine)
                #MeshNeighborhoodLossを更新
                loss_helper.reset_rest_pose(gaussians.get_xyz.clone().detach())

                for p in pre_affine.parameters():
                    p.requires_grad_(False)

                with torch.no_grad():
                    affine.r.copy_(-pre_affine.r)
                    affine.log_s.copy_(-pre_affine.log_s)
                    s = pre_affine.log_s.exp()
                    R_temp = pre_affine._rodrigues(pre_affine.r)
                    affine.t.copy_(-(R_temp.T @ pre_affine.t) / s)

                # ③ ここから affine を trainable にして Stage-2 を続行
                affine_opt = torch.optim.Adam(affine.parameters(), lr=1e-3)
                # affine_opt = torch.optim.Adam(affine.parameters(), lr=1e-2)

                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                ckpt_path = "learnable_affine.pth"
                torch.save(affine.state_dict(), ckpt_path)   # GPU で持っていてもそのまま保存可
                print(f"saved → {ckpt_path}")

                if pca:
                    freeze_gaussians_var_6890(gaussians, freeze=True)  # Gaussians を凍結（勾配+更新を停止）

                means3D_iteration_0 = gaussians.get_xyz.clone().detach()  # 初期値を保存
                edge_idx = knn_edges(means3D_iteration_0, device=device)
                arap_loss = ARAPLoss(edge_idx).to(device)

                gaussian_centers = gaussians.get_xyz[:6890] # (6890,3)
                k = 8

                with torch.no_grad():
                    edge_idx_knn = build_knn_once(gaussian_centers, k).to(device)
                    x_snap   = gaussian_centers.detach().clone()   # spring の初期長さを固定

            continue


        iter_start.record()
        # if iteration == 1:
        #     freeze_gaussians_except_xyz(gaussians, freeze=True)  # Gaussians を凍結（xyz以外）
        # if iteration == 2500:
        #     freeze_gaussians_except_xyz(gaussians, freeze=False) # Gaussians を解凍（xyz以外）
        t = tic()
        # t = toc(t, "Training iteration start")
        #共分散を平均から計算
        cov, quat, sigma = loss_helper.covariances(means3D_6890, ite=iteration)
        # t = toc(t, "Training iteration covariances")
        # --- gaussians 内部も同期（勾配は要らない） ---
        if pca:
            with torch.no_grad():
                gaussians._rotation.data[:6890].copy_(quat)
                gaussians._scaling.data[:6890].copy_(sigma)   # (N,3)


        gaussians.update_learning_rate(iteration)

        

        # Every 1000 its we increase the levels of SH up to a maximum degree 
        if iteration % 1000 == 0:
        # if iteration % 100 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        if pca:
            # render_pkg = render_pca(viewpoint_cam, gaussians, quat, sigma, pipe, bg)
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        else:
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg)

        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        render_white = render_silhouette(viewpoint_cam, gaussians, pipe, bg)
        image_white, viewspace_point_tensor_white, visibility_filter_white, radii_white = render_white["render"], render_white["viewspace_points"], render_white["visibility_filter"], render_white["radii"]
        render_par = render_parents(viewpoint_cam, gaussians, pipe, bg)
        image_parents, viewspace_point_tensor_parents, visibility_filter_parents, radii_parents = render_par["render"], render_par["viewspace_points"], render_par["visibility_filter"], render_par["radii"]
        # render_par_white = render_silhouette_parents(viewpoint_cam, gaussians, pipe, bg)
        # image_parents_white, viewspace_point_tensor_parents_white, visibility_filter_parents_white, radii_parents_white = render_par_white["render"], render_par_white["viewspace_points"], render_par_white["visibility_filter"], render_par_white["radii"]
        #ガウシアンの座標を取得
        means3D = gaussians.get_xyz
        #ガウシアンの透明度を取得
        opacity = gaussians.get_opacity   #0になるほど透明
        # print("gaussians.get_xyz", means3D.shape)

        
        means3D_6890 = means3D[:6890]   
        input = means3D_6890.unsqueeze(0) # (1, 6890, 3)
        input = affine(input)   
        input = input.to(device)
        all_input = affine(means3D.unsqueeze(0)).to(device).flatten(1)  # (1, N, 3)  バッチ復元

        # 推論
        with torch.no_grad():
            output, mu, logvar, beta_h, pose_h = vae(input)

        # outputを保存
        out_dir = Path("pointcloud_xyz")
        out_dir.mkdir(exist_ok=True)
        verts_out = output.reshape(-1, 3)  # (N, 3)
        verts_all_in = means3D.reshape(-1, 3)  # (N, 3)
        verts_in = means3D_6890.reshape(-1, 3)  # (N, 3)

        if iteration % 1000 == 0 or iteration == 0 or iteration == 1 or iteration == 2 or iteration == 3 or iteration == 100:
            fname = out_dir / f"out_iter_{iteration:05d}.xyz"
            np.savetxt(fname, verts_out.cpu().numpy(), fmt="%.6f")
            print(f"saved → {fname}")

        if iteration % 1000 == 0 or iteration == 0 or iteration == 1 or iteration == 2 or iteration == 3 or iteration == 100:
            fname = out_dir / f"in_iter_{iteration:05d}.xyz"
            np.savetxt(fname, verts_in.cpu().detach().numpy(), fmt="%.6f")
            print(f"saved → {fname}")

        if iteration % 1000 == 0 or iteration == 0 or iteration == 1 or iteration == 2 or iteration == 3 or iteration == 100:
            fname = out_dir / f"in_all_iter_{iteration:05d}.xyz"
            np.savetxt(fname, verts_all_in.cpu().detach().numpy(), fmt="%.6f")
            print(f"saved → {fname}")

        output_temp = output.view(1, 6890, 3)
        # 1) 各面の重心 (13776, 3)
        face_centers = output_temp[0][faces_t].mean(dim=1)   # (13776, 3)

        # 2) バッチ次元を戻す  → (1, 13776, 3)
        face_centers = face_centers.unsqueeze(0)

        # 3) dim=1 で連結   → (1, 20666, 3)
        all_output = torch.cat([output_temp, face_centers], dim=1).flatten(1)  

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        white_gt_image = viewpoint_cam.original_image.clone().cuda()
        # # ---------------------- (H, W, 3) 形式 ----------------------
        # if white_gt_image.shape[-1] == 3:                 # チャンネル最後
        #     is_black = (white_gt_image[..., 0] == 0) & (white_gt_image[..., 1] == 0) & (white_gt_image[..., 2] == 0)
        #     white_gt_image[~is_black] = 255               # 黒以外 → 白

        # # ---------------------- (3, H, W) 形式 ----------------------
        # elif white_gt_image.shape[0] == 3:                # チャンネル先頭
        #     is_black = (white_gt_image[0] == 0) & (white_gt_image[1] == 0) & (white_gt_image[2] == 0)
        #     white_gt_image[:, ~is_black] = 255            # 黒以外 → 白

        white_gt_image = make_clean_silhouette(white_gt_image)
        
        Ll1 = l1_loss(image, gt_image)
        li_white = l1_loss(image_white, white_gt_image)
        li_white_parents = l1_loss(image_parents, gt_image)  # 親の画像とのL1損失
        if iteration % 100 == 0:
            white_out_dir = Path("/root/development/project4010/gaussian-splatting/output_temp")
            white_out_dir.mkdir(parents=True, exist_ok=True)
            img_pred = to_uint8(image_white).float() / 255.0 
            img_gt   = to_uint8(white_gt_image).float() / 255.0 

            save_image(img_pred, white_out_dir / f"iter_{iteration:06d}_pred.png")
            save_image(img_gt,   white_out_dir / f"iter_{iteration:06d}_gt.png")
        loss2 = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        loss2 = loss2.mean()
        loss2 = loss2 * 1e-3
        loss3 = F.mse_loss(all_output, all_input, reduction='mean')
        # loss3 = F.mse_loss(output, input.flatten(1), reduction='mean')
        loss3 = loss3 
        eps = 1e-6
        lambda_opacity = 1e-4
        opacity_penalty = torch.mean(1.0 / (opacity + eps))  # epsは0除算防止
        loss4 = lambda_opacity * opacity_penalty
        λ_lap, λ_spr = 1e-4, 1e-2 
        lap  = loss_helper.laplacian_loss(means3D_6890, weight=λ_lap)
        spr  = loss_helper.spring_loss(means3D_6890,  weight=λ_spr)     
        nor = delta_n_penalty(means3D_6890, state)
        loss_arap = arap_loss(means3D_iteration_0, means3D)
        loss_spring = spring_loss_knn(means3D_6890, x_snap, edge_idx_knn, w=1.0)
        # t = toc(t, "Training iteration losses")
        # loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + loss4
        # loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + loss2 + loss3 + loss4 + lap + spr + li_white * 0.1
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + loss2 + loss3  + li_white_parents + loss4 + loss_spring
        # loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + loss2 + loss3 + loss4 + spr + li_white + nor + loss_arap
        loss.backward()
        # t = toc(t, "Training iteration backward")
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()
            # t = toc(t, "Training iteration end")
            # Log and save
            training_report(tb_writer, iteration, Ll1, loss2, loss3, loss4, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), 1.0 - opt.lambda_dssim)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)


            # Densification
            # if iteration < opt.densify_until_iter:  #通常は 15000
            if iteration < 15000:  
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if not pca:
                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        # size_threshold = 20 if iteration > opt.opacity_reset_interval else None   #3000iteration 
                        size_threshold = 10 if iteration > 300 else None   
                        # gaussians.reset(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                        gaussians.reset(opt.densify_grad_threshold, 0.05, scene.cameras_extent, size_threshold, radii)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                affine_opt.step() 
                loss_helper.clamp_gaussians(gaussians,        # ② 射影で制約
                            margin=0.02)
                gaussians.optimizer.zero_grad(set_to_none = True)
                affine_opt.zero_grad(set_to_none=True)


            g_new = gaussians.get_xyz[:6890]  # (6890,3) – after update
            state.step(g_new.detach().clone())               # ← ここを更新

            #copy rotation and scaling
            if pca:
                gaussians._rotation.data[:6890].copy_(quat)   # (N,4)
                gaussians._scaling.data[:6890].copy_(sigma)   # (N,3)
            # t = toc(t, "Training iteration optimizer step")
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss2, loss3, loss4, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, alpha):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).double()
                    # l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} L1*alpha {} loss2 {} loss3 {} loss4 {}".format(iteration, config['name'], l1_test, psnr_test, l1_test * alpha, loss2, loss3, loss4))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[100, 1000, 3000, 7000, 10000, 15000, 20000, 30000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[100, 1000, 3000, 7000, 10000, 15000, 20000, 30000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
