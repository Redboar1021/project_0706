# mesh_neigh.py
import numpy as np, torch, pickle
from pathlib import Path
from typing import List, Union, Optional, Tuple
import time, torch
from scene import Scene, GaussianModel
def tic(): torch.cuda.synchronize(); return time.perf_counter()
def toc(t0, label): 
    torch.cuda.synchronize()
    print(f"{label}: {(time.perf_counter()-t0)*1e3:.1f} ms")
    return tic()


# ----------- util：単体 3×3 / バッチ ...×3×3 → quat (x,y,z,w) ----------
def mat3_to_quat(R: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    single = False
    if R.ndim == 2:
        R, single = R.unsqueeze(0), True                       # (1,3,3)

    B  = R.shape[0]
    m  = R.view(B, 9)                                          # flatten view
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = m.T

    t0 = 1.0 + m00 + m11 + m22                                 # trace
    t1 = 1.0 + m00 - m11 - m22
    t2 = 1.0 - m00 + m11 - m22
    t3 = 1.0 - m00 - m11 + m22

    qx = torch.sqrt(torch.clamp(t1, min=eps)) * 0.5
    qy = torch.sqrt(torch.clamp(t2, min=eps)) * 0.5
    qz = torch.sqrt(torch.clamp(t3, min=eps)) * 0.5
    qw = torch.sqrt(torch.clamp(t0, min=eps)) * 0.5

    qx = torch.where((m21 - m12) < 0, -qx,  qx)
    qy = torch.where((m02 - m20) < 0, -qy,  qy)
    qz = torch.where((m10 - m01) < 0, -qz,  qz)

    quat = torch.stack((qx, qy, qz, qw), dim=-1)
    quat = quat / quat.norm(dim=-1, keepdim=True)
    return quat.squeeze(0) if single else quat                 # (...,4)
# ------------------------------------------------------------------------



def _build_edges(faces: np.ndarray) -> np.ndarray:
    """三角形インデックス → 一意な無向エッジ (E,2)"""
    return np.vstack({
        tuple(sorted(e))
        for tri in faces
        for e in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0]))
    })

def compute_covariances_loop(mu, adj, eps=1e-4):
    """
    mu  : (N,3)  learnable
    adj : list[LongTensor]  近傍 idx
    return (N,3,3)  各頂点の共分散
    """
    N   = mu.size(0)
    dev = mu.device
    cov = torch.zeros((N, 3, 3), device=dev, dtype=mu.dtype)

    for i, neigh in enumerate(adj):
        diff = mu[neigh] - mu[i]               # (k,3)
        c    = (diff.T @ diff) / diff.size(0)  # (3,3)
        cov[i] = c + torch.eye(3, device=dev, dtype=mu.dtype) * eps

    return cov


class MeshNeighborhoodLoss:
    """
    - laplacian_loss / spring_loss で形状正則化
    - covariances() で Σ・quat・σ を毎 iteration 生成
    """

    # ---------------------------- build ---------------------------------
    def __init__(self,
                 verts_init: torch.Tensor,
                 *,
                 faces: Optional[np.ndarray] = None,
                 smpl_pkl: Optional[Union[str, Path]] = None,
                 device="cuda", dtype=torch.float32):

        if faces is None:
            with open(smpl_pkl, "rb") as f:
                faces = pickle.load(f, encoding="latin1")["f"]
        faces = faces.astype(np.int64)

        # 辺リスト & 近傍リスト
        edges_np = _build_edges(faces)
        self.edges = torch.as_tensor(edges_np, device=device)
        N = verts_init.shape[0]

        adj = [set() for _ in range(N)]
        for a, b in edges_np:
            adj[a].add(b)
            adj[b].add(a)
        self.adj: List[torch.Tensor] = [
            torch.as_tensor(sorted(n), device=device) for n in adj
        ]
        self.N = N

        # rest-pose 辺長
        self.verts0 = verts_init.to(device=device, dtype=dtype).detach()
        vi0, vj0 = self._pair(self.verts0)
        self.rest_len = (vi0 - vj0).norm(dim=1)                # (E,)
        self.max_rest_len = float(self.rest_len.max())   # L_max  (scalar)


        src = torch.cat([self.edges[:,0], self.edges[:,1]])        # (2E,)
        dst = torch.cat([self.edges[:,1], self.edges[:,0]])        # (2E,)
        self.edge_src = src.to(device)
        self.edge_dst = dst.to(device)

    def reset_rest_pose(self, new_verts: torch.Tensor):
        """
        new_verts : (N,3) torch.Tensor  新しい rest-pose
        トポロジは不変前提（頂点数・faces 同じ）
        """
        assert new_verts.shape[0] == self.N, "頂点数が変わっています"
        self.verts0 = new_verts.detach()
        vi0, vj0 = self._pair(self.verts0)
        self.rest_len = (vi0 - vj0).norm(dim=1)   # (E,)
        self.max_rest_len = float(self.rest_len.max())   # L_max  (scalar)


    # ------------------- public losses ----------------------------------
    def laplacian_loss(self, v: torch.Tensor, weight=1e-3):
        vi, vj = self._pair(v)
        return weight * (vi - vj).pow(2).sum(-1).mean()

    def spring_loss(self, v: torch.Tensor, weight=1e-3):
        vi, vj = self._pair(v)
        return weight * ((vi - vj).norm(dim=1) - self.rest_len).pow(2).mean()
    
    def _fast_cov(self, v, eps=1e-4):
        diff   = v[self.edge_dst] - v[self.edge_src]      # (2E,3)
        outer  = diff[:, :, None] * diff[:, None, :]      # (2E,3,3)
        sig    = torch.zeros((v.size(0),3,3), device=v.device)
        sig.index_add_(0, self.edge_src, outer)           # Σ_j diff diffᵀ
        deg    = torch.bincount(self.edge_src,
                                minlength=v.size(0)).clamp_min_(1).view(-1,1,1)
        return sig / deg + eps * torch.eye(3, device=v.device)

    # ----------------- main: Σ, quat, σ ---------------------------------
    def covariances(self, v: torch.Tensor, *,
                    eps=1e-4, return_quat=True, log_scale=True, ite=1.0):
        """v: (N,3) → (Σ, quat/R, σ)"""
        # --- Σ ----------------------------------------------------------
        t = tic()
        sig = self._fast_cov(v, eps)   
        # t = toc(t, "covariances: fast_cov")
        # --- 固有分解 ---------------------------------------------------
        evals, evecs = torch.linalg.eigh(sig)
        # t = toc(t, "covariances: eig")
        idx = torch.flip(torch.arange(3, device=v.device), dims=[0])
        evals, evecs = evals[:, idx], evecs[:, :, idx]
        # sigma = torch.sqrt(torch.clamp(evals, min=eps))*min((0.1 + 0.5 * ite / 15000), 0.6)
        sigma = torch.sqrt(torch.clamp(evals, min=eps))*0.4  #ガウシアンのサイズ
        if log_scale:
            sigma = sigma.log()

        if return_quat:
            quat = mat3_to_quat(evecs)              # (N,4)
            return sig, quat, sigma
        else:
            return sig, evecs, sigma

    # ------------------ helpers -----------------------------------------
    def _pair(self, verts):
        return verts[self.edges[:, 0]], verts[self.edges[:, 1]]
    
    # ───────────────────────────────────────────────
    @torch.no_grad()
    def clamp_gaussians(self, g: GaussianModel,
                        margin: float = 0.0):
        """
        g : GaussianModel
        * 既存の _xyz の順序・サイズはそのまま
        * self.max_rest_len は __init__ で保存した初期最長エッジ長
        * margin を付ければ少しの超過を許容 (例: 0.002 = 2 mm)
        """
        v = g._xyz                                # (N,3) Parameter
        vi, vj = self._pair(v)                    # 隣接頂点 (E,3)

        vec   = vi - vj
        len_e = vec.norm(dim=1)                   # 各エッジ長 (E,)

        limit = self.max_rest_len + margin
        mask  = len_e > limit                     # 超過エッジだけ

        if not mask.any():                        # 1 つも無ければ早期終了
            return

        # ── 超過分を 2 点で半分ずつ縮める ──────────────
        dir_    = vec[mask] / len_e[mask].unsqueeze(1)     # 単位方向 (E_ex,3)
        excess  = (len_e[mask] - limit) / 2.0              # (E_ex,)
        disp    = dir_ * excess.unsqueeze(1)               # (E_ex,3)

        src = self.edges[mask, 0]
        dst = self.edges[mask, 1]

        delta = torch.zeros_like(v)              # (N,3)
        delta.index_add_(0, src, -disp)          # 片方を内側へ
        delta.index_add_(0, dst,  disp)          # もう片方を内側へ

        v.add_(delta)                            # in-place 更新 (順序不変)

