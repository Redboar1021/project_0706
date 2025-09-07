import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from sklearn.cluster import DBSCAN
import pathlib

# ----------------------------- 入力ファイル -----------------------------
in_xyz  = pathlib.Path("/root/development/project4010/gaussian-splatting/pointcloud_xyz/out_iter_30000.xyz")
out_xyz = in_xyz.with_name("filtered.xyz")

# ----------------------------- 1) 読み込み ------------------------------
pc = np.loadtxt(in_xyz)                          # (N,3)

# ----------------------------- 2) DBSCAN で人物抽出 ----------------------
diag = np.linalg.norm(pc.ptp(axis=0))
eps  = 0.05 * diag
db   = DBSCAN(eps=eps, min_samples=30).fit(pc)

labels = db.labels_
uniq   = np.unique(labels[labels != -1])
mean_z = [(l, pc[labels == l][:, 2].mean()) for l in uniq]
best   = max(mean_z, key=lambda t: t[1])[0]
mask   = (labels == best)
pc_human = pc[mask]

# ----------------------------- 3) 保存 ----------------------------------
np.savetxt(out_xyz, pc_human, fmt="%.6f")
print(f"Saved filtered point cloud to: {out_xyz}")

# ----------------------------- 4) 可視化 -------------------------------
fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(121, projection="3d")
ax2 = fig.add_subplot(122, projection="3d")

ax1.scatter(pc[:,0],     pc[:,1],     pc[:,2],     s=1)
ax2.scatter(pc_human[:,0], pc_human[:,1], pc_human[:,2], s=1, c='tab:red')

ax1.set_title("Original")
ax2.set_title("Filtered (Human only)")

# 🟡─── ここでスケールを合わせる ──────────────────────────
all_pts = np.concatenate([pc, pc_human], axis=0)   # または pc でも可
x_min, y_min, z_min = all_pts.min(axis=0)
x_max, y_max, z_max = all_pts.max(axis=0)

for ax in (ax1, ax2):
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.set_box_aspect([1, 1, 1])   # 等方スケール

plt.tight_layout()
plt.show()
