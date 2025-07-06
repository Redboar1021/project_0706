import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 出力ディレクトリ
out_dir = '/root/development/project4010/VAE/all_vertices'
files = glob.glob(os.path.join(out_dir, '*_verts.npz'))

# ファイル数とサンプルリスト
print(f"Found {len(files)} files. Sample:\n", files[:5])

# 最初のファイルをロード
data = np.load(files[0])
verts = data['vertices']  # (T,6890,3)
poses = data['poses']
betas = data['betas']
trans = data['trans']

# 各データ形状を表示
print("vertices shape:", verts.shape)
print("poses shape:", poses.shape)
print("betas shape:", betas.shape)
print("trans shape:", trans.shape)

# 最初のフレーム頂点をランダムサンプリングして3Dプロット
frame0 = verts[0]
idx = np.random.choice(frame0.shape[0], size=500, replace=False)
sample_pts = frame0[idx]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(sample_pts[:,0], sample_pts[:,1], sample_pts[:,2])
ax.set_title('Sample vertices from first frame')
plt.show()
