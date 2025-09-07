import open3d as o3d
import numpy as np

# --- パスを指定 ---
obj_path = "/root/development/project4010/gaussian-splatting/smpl_fitted_affine.obj"
ply_path = "/mnt/c/Users/Yoshi/Downloads/human_circle_run_only_human_4.ply"

# --- ① SMPL メッシュ（OBJ） ---
mesh_obj = o3d.io.read_triangle_mesh(obj_path)
mesh_obj.compute_vertex_normals()
mesh_obj.paint_uniform_color([0.8, 0.8, 0.8])   # 明るいグレー

# --- ② 点群 or メッシュ（PLY） ---
ply = o3d.io.read_point_cloud(ply_path) if ply_path.lower().endswith(".ply") else \
      o3d.io.read_triangle_mesh(ply_path)
# 点群なら大きさ確認して球で表示／メッシュなら法線計算
if isinstance(ply, o3d.geometry.PointCloud):
    # 点群を見やすく 4 mm の球で表示
    ply.paint_uniform_color([1.0, 0.2, 0.2])  # 赤
    geom_list = [mesh_obj, ply]
else:
    ply.compute_vertex_normals()
    ply.paint_uniform_color([1.0, 0.2, 0.2])  # 赤
    geom_list = [mesh_obj, ply]

# --- ③ ビューアで表示 ---
o3d.visualization.draw_geometries(
    geom_list,
    window_name="SMPL & PLY",
    width=1000, height=800,
    mesh_show_back_face=True    # 裏面も描画
)
