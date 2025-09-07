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
from gaussian_renderer import render, network_gui, render_affine, render_pca
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


def freeze_gaussians(model: GaussianModel, freeze: bool = True):
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
    # for tensor in (
    #     model._scaling, model._rotation,
    #     model._opacity, model._features_dc, model._features_rest):
    #     tensor.requires_grad_(not freeze)


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
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
    pre_affine.r.data = torch.tensor([math.pi/2 - 0.02, 0, 0], device=device)

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
                #共分散を凍結
                gaussians._scaling.requires_grad_(False)
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

            continue


        iter_start.record()
        # if iteration == 1:
        #     freeze_gaussians_except_xyz(gaussians, freeze=True)  # Gaussians を凍結（xyz以外）
        # if iteration == 2500:
        #     freeze_gaussians_except_xyz(gaussians, freeze=False) # Gaussians を解凍（xyz以外）
        t = tic()
        # t = toc(t, "Training iteration start")
        #共分散を平均から計算
        cov, quat, sigma = loss_helper.covariances(means3D, ite=iteration)
        # t = toc(t, "Training iteration covariances")
        # --- gaussians 内部も同期（勾配は要らない） ---
        with torch.no_grad():
            gaussians._rotation.copy_(quat)
            gaussians._scaling.data.copy_(sigma)   # (N,3)


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

        render_pkg = render_pca(viewpoint_cam, gaussians, quat, sigma, pipe, bg)
        # render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        #ガウシアンの座標を取得
        means3D = gaussians.get_xyz
        #ガウシアンの透明度を取得
        opacity = gaussians.get_opacity   #0になるほど透明
        # print("gaussians.get_xyz", means3D.shape)

        

        input = means3D.unsqueeze(0) # (1, 6890, 3)
        input = affine(input)   
        input = input.to(device)

        # 推論
        with torch.no_grad():
            output, mu, logvar, beta_h, pose_h = vae(input)

        # outputを保存
        out_dir = Path("pointcloud_xyz")
        out_dir.mkdir(exist_ok=True)
        verts_out = output.reshape(-1, 3)  # (N, 3)
        verts_in = means3D.reshape(-1, 3)  # (N, 3)

        if iteration % 1000 == 0 or iteration == 0 or iteration == 1 or iteration == 2 or iteration == 3 or iteration == 100:
            fname = out_dir / f"out_iter_{iteration:05d}.xyz"
            np.savetxt(fname, verts_out.cpu().numpy(), fmt="%.6f")
            print(f"saved → {fname}")

        if iteration % 1000 == 0 or iteration == 0 or iteration == 1 or iteration == 2 or iteration == 3 or iteration == 100:
            fname = out_dir / f"in_iter_{iteration:05d}.xyz"
            np.savetxt(fname, verts_in.cpu().detach().numpy(), fmt="%.6f")
            print(f"saved → {fname}")

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss2 = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        loss2 = loss2.mean()
        loss2 = loss2 * 1e-4
        loss3 = F.mse_loss(output, input.flatten(1), reduction='mean')
        loss3 = loss3 * 1e-4
        eps = 1e-6
        lambda_opacity = 1e-4
        opacity_penalty = torch.mean(1.0 / (opacity + eps))  # epsは0除算防止
        loss4 = lambda_opacity * opacity_penalty
        λ_lap, λ_spr = 1e-4, 1e-4 
        lap  = loss_helper.laplacian_loss(means3D, weight=λ_lap)
        spr  = loss_helper.spring_loss(means3D,  weight=λ_spr)
        # t = toc(t, "Training iteration losses")
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + loss4
        # loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + loss2 + loss3 + loss4 + lap + spr
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

                # if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                #     # size_threshold = 20 if iteration > opt.opacity_reset_interval else None   #3000iteration 
                #     size_threshold = 20 if iteration > 300 else None   
                #     # gaussians.reset(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                #     gaussians.reset(opt.densify_grad_threshold, 0.05, scene.cameras_extent, size_threshold, radii)
                
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

            #copy rotation and scaling
            gaussians._rotation.data.copy_(quat)   # (N,4)
            gaussians._scaling.data.copy_(sigma)   # (N,3)
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
