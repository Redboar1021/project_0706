#!/usr/bin/env python3
import numpy as np
# NumPy 1.24+ で削除されたエイリアスを復活させる
np.bool    = bool
np.int     = int
np.float   = float
np.complex = complex
np.object  = object
np.unicode = str
np.str     = str
import inspect
inspect.getargspec = inspect.getfullargspec
import os
import sys
import glob
import numpy as np
import torch
from tqdm import tqdm

# ── プロジェクトルートを通す ───────────────────
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# ── SMPL モデル読み込み ─────────────────────────
from smplx.body_models import SMPL

device = 'cuda' if torch.cuda.is_available() else 'cpu'
smpl = SMPL(
    model_path=os.path.join(project_root, 'models'),
    gender='NEUTRAL',
    model_type='smpl',
    batch_size=1
).to(device)
smpl.eval()

# ── 入出力パス ───────────────────────────────
data_root    = os.path.join(project_root, 'data')
out_flat_dir = os.path.join(project_root, 'all_vertices')  # ← ここに全部まとめる
os.makedirs(out_flat_dir, exist_ok=True)

# ── motion ファイル読み込み関数 ────────────────
def load_motion(npz_path):
    arr = np.load(npz_path)
    # betas の取得（shape.npz から読み込むパターンも同様）
    if 'betas' in arr:
        betas = arr['betas']
    else:
        shape_file = os.path.join(os.path.dirname(npz_path), 'shape.npz')
        betas = np.load(shape_file)['betas']
    # SMPL-body: 先頭 10 次元だけ使う
    if betas.ndim == 1 and betas.shape[0] >= 10:
        betas = betas[:10]
    else:
        raise ValueError(f"Unexpected betas shape: {betas.shape}")

    # pose は常に 72 次元に切り出し
    poses = arr['poses'][:, :72]   # (T,72)
    trans = arr['trans']           # (T,3)
    return poses, betas, trans

# ── 全ファイルを再帰検索＆処理 ─────────────────
motion_paths = glob.glob(os.path.join(data_root, '**', '*.npz'), recursive=True)
# shape.npz は除外
motion_paths = [p for p in motion_paths if os.path.basename(p) != 'shape.npz']

success = 0
errors  = []

for idx, path in enumerate(tqdm(motion_paths, desc='Converting')):

    try:
        poses, betas, trans = load_motion(path)
        T = poses.shape[0]

        # テンソル化
        poses_t = torch.from_numpy(poses).float().to(device)
        betas_t = torch.from_numpy(betas).float().unsqueeze(0).to(device)
        trans_t = torch.from_numpy(trans).float().to(device)

        # 頂点をフレーム毎に計算
        verts = []
        with torch.no_grad():
            for i in range(T):
                out = smpl(
                    global_orient=torch.zeros_like(poses_t[i:i+1, :3]),
                    body_pose=   poses_t[i:i+1, 3:],
                    betas=       betas_t,
                    transl=      torch.zeros_like(trans_t[i:i+1])
                )
                verts.append(out.vertices[0].cpu().numpy())
        verts_all = np.stack(verts, axis=0)  # (T, 6890, 3)

        # 出力ファイル名（同名があれば上書き）
        base = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(out_flat_dir, base + '_verts.npz')
        np.savez_compressed(
            out_path,
            vertices=verts_all,
            poses=poses,
            betas=betas,
            trans=trans
        )

        # 確認用に読み戻しチェック
        chk = np.load(out_path)
        vchk = chk['vertices']
        assert vchk.ndim == 3 and vchk.shape[1] == 6890, f"bad shape {vchk.shape}"

        success += 1

    except Exception as e:
        errors.append((path, str(e)))

    # 節々で状況を表示（100件ごと）
    if (idx + 1) % 100 == 0:
        print(f"[INFO] {idx+1} / {len(motion_paths)} processed, success = {success}, errors = {len(errors)}")

# ── 最終サマリー ─────────────────────────────
print(f"\n=== 完了 ===\nTotal files: {len(motion_paths)}\nSucceeded: {success}\nFailed: {len(errors)}")
if errors:
    print("Failed list:")
    for fp, msg in errors:
        print(f" - {fp}: {msg}")
