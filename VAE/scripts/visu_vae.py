#!/usr/bin/env python3
# eval_vae_pointnet_smpl.py  (random sampling 対応版)
# -------------------------------------------------------------
# 例)
#   # 未学習フォルダから 20 ファイルだけランダム抽出し，各ファイル 100 フレームまで評価
#   python eval_vae_pointnet_smpl.py \
#       --data_dir  /path/to/unseen_vertices \
#       --smpl_dir  models \
#       --ckpt      vae_pointnet.pth \
#       --sample_files 20 --sample_frames 100 \
#       --num_vis 8 --device cuda
# -------------------------------------------------------------

import os, glob, random, inspect, argparse
import numpy as np, torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from smplx.body_models import SMPL
from tqdm import tqdm
from typing import Optional
from os import path as osp
from human_body_prior.tools.model_loader import load_model
from human_body_prior.models.vposer_model import VPoser

# ---------- NumPy 旧エイリアス ----------
np.bool, np.int, np.float, np.complex = bool, int, float, complex
np.object, np.unicode, np.str = object, str, str
if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec
# ----------------------------------------

# ---------- 可視化 ----------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
def scatter_cmp(path, vin, vre):
    fig = plt.figure(figsize=(8,4))
    ax1 = fig.add_subplot(121, projection="3d"); ax2 = fig.add_subplot(122, projection="3d")
    ax1.scatter(vin[:,0], vin[:,1], vin[:,2], s=1); ax1.set_title("input");  ax1.set_axis_off()
    ax2.scatter(vre[:,0], vre[:,1], vre[:,2], s=1, c='r'); ax2.set_title("recon"); ax2.set_axis_off()
    for ax in (ax1, ax2): 
        ax.set_axis_off()
        ax.view_init(elev=-95, azim=90)
        ax.set_box_aspect([1, 1, 1])
    plt.tight_layout(); plt.savefig(path, dpi=300); plt.close()

# ---------- Dataset ----------
class MotionDataset(Dataset):
    def __init__(self, data_dir: str,
                 sample_files: Optional[int] = None,
                 sample_frames: Optional[int] = None,
                 seed: int = 0):
        rng = random.Random(seed)

        # ① ファイルを無作為抽出
        all_files = glob.glob(os.path.join(data_dir, "*_verts.npz"))
        if sample_files and sample_files < len(all_files):
            self.files = rng.sample(all_files, sample_files)
        else:
            self.files = all_files

        # self.files = ["/root/development/project4010/VAE/all_vertices/General_A5_-_Pick_Up_Box_stageii_verts.npz"]
        print("Selected files:")
        for f in self.files:
            print(f)
        # ② フレームを無作為抽出
        self.index = []
        for f in self.files:
            n_frames = np.load(f, mmap_mode="r")["vertices"].shape[0]
            ids = list(range(n_frames))
            if sample_frames and sample_frames < n_frames:
                ids = rng.sample(ids, sample_frames)
            self.index.extend([(f, i) for i in ids])

    def __len__(self): return len(self.index)

    def __getitem__(self, idx):
        f, t = self.index[idx]
        cache = getattr(self, "_cache", {})
        if f not in cache:
            cache[f] = np.load(f, mmap_mode="r")
            self._cache = cache
        d = cache[f]
        v  = d["vertices"][t]
        return torch.from_numpy(v).float()

def collate_fn(b): return torch.stack(b,0)

# ---------- Model ----------
from vae2 import VAE_PointNet
# ---------- Main ----------
def main(opt):
    dev = torch.device(opt.device)
    ds  = MotionDataset(opt.data_dir,
                        sample_files=opt.sample_files,
                        sample_frames=opt.sample_frames,
                        seed=opt.seed)
    dl  = DataLoader(ds,batch_size=opt.batch,shuffle=False,
                     num_workers=opt.num_workers,collate_fn=collate_fn,
                     pin_memory=True)
    print(f"★ evaluating {len(ds)} frames "
          f"(files={opt.sample_files or 'all'}, frames/file={opt.sample_frames or 'all'})")
    
    # VPoser をロード
    support_dir = '/root/development/project4010/VAE/human_body_prior/support_data/downloads'
    expr_dir = osp.join(support_dir,'vposer_v2_05') #'TRAINED_MODEL_DIRECTORY'  in this directory the trained model along with the model code exist
    vp, ps = load_model(expr_dir, model_code=VPoser,
                              remove_words_in_model_weights='vp_model.',
                              disable_grad=True)
    vposer = vp.to(dev)

    model = VAE_PointNet("/root/development/project4010/VAE/models", vposer).to(dev)
    model.load_state_dict(torch.load(opt.ckpt, map_location=dev))
    model.eval()

    errs=[]; vis_cnt=0
    with torch.no_grad():
        for x in tqdm(dl):
            x=x.to(dev)
            out = model(x)
            recon = out[0] if isinstance(out, tuple) else out     # ← ここで 1 つ目だけ取り出す
            rec = recon.reshape(x.size(0), -1)
            mse=F.mse_loss(rec, x.flatten(1), reduction='none').mean(1)
            errs.extend(mse.cpu().tolist())

            for i in range(x.size(0)):
                if vis_cnt < opt.num_vis:
                    scatter_cmp(f"eval_vis_{vis_cnt:03d}.png",
                                x[i].cpu().numpy(),
                                rec[i].reshape(-1,3).cpu().numpy())
                    vis_cnt+=1
        print(f"平均再構成 MSE : {np.mean(errs):.4e}  （{len(errs)} サンプル）")

if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("--data_dir",     default="/root/development/project4010/VAE/all_vertices")
    a.add_argument("--smpl_dir",     default="/root/development/project4010/VAE/models")
    a.add_argument("--ckpt",         default="/root/development/project4010/vae_pointnet_vposer.pth")
    a.add_argument("--latent_dim",   type=int, default=82)
    a.add_argument("--batch",        type=int, default=32)
    a.add_argument("--num_vis",      type=int, default=5)
    a.add_argument("--num_workers",  type=int, default=0)
    a.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    # ★ 追加オプション
    a.add_argument("--sample_files",  type=int, default=3,
                   help="評価に使うファイル数 (None=全ファイル)")
    a.add_argument("--sample_frames", type=int, default=5,
                   help="各ファイルからランダム抽出するフレーム数 (None=全フレーム)")
    a.add_argument("--seed",         type=int, default=3, help="乱数シード")
    opt=a.parse_args()
    torch.manual_seed(opt.seed); np.random.seed(opt.seed); random.seed(opt.seed)
    main(opt)
