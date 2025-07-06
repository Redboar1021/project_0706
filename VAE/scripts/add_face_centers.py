#!/usr/bin/env python3
"""
Add one point at the center of every SMPL face.

入力:
    /root/development/project4010/VAE/scripts/t_pose_vertices.xyz   (6890 × 3)
    /root/development/project4010/VAE/models/SMPL_NEUTRAL.pkl       ("f": 13776 × 3)

出力:
    /root/development/project4010/VAE/scripts/t_pose_vertices_add_13776.xyz
        - 先頭 6890 行: 元の頂点
        - 末尾 13776 行: 各面の重心
"""

from pathlib import Path
import pickle
import numpy as np

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

# ---------- パス設定 ----------
xyz_path   = Path("/root/development/project4010/VAE/scripts/t_pose_vertices.xyz")
smpl_pkl   = Path("/root/development/project4010/VAE/models/SMPL_NEUTRAL.pkl")
out_path   = Path("/root/development/project4010/VAE/scripts/t_pose_vertices_add_13776.xyz")

# ---------- 6890 頂点を読み込む ----------
# 行: "x y z" 形式を想定
verts = np.loadtxt(xyz_path, dtype=np.float32)            # (6890, 3)
assert verts.shape == (6890, 3), "頂点数が 6890 ではありません"

# ---------- 面インデックスを読み込む ----------
with open(smpl_pkl, "rb") as f:
    faces = pickle.load(f, encoding="latin1")["f"].astype(np.int64)   # (13776, 3)

# ---------- 各面の重心を計算 ----------
face_centers = verts[faces].mean(axis=1)                  # (13776, 3)

# ---------- 結合して書き出し ----------
all_pts = np.vstack([verts, face_centers])                # (20666, 3)
np.savetxt(out_path, all_pts, fmt="%.10f")                # 小数点 10 桁で保存

print(f"Saved {all_pts.shape[0]} points to {out_path}")
