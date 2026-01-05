import numpy as np
import open3d as o3d
import torch
from gsplat.rendering import rasterization

device = "cuda" if torch.cuda.is_available() else "cpu"
import time

import tyro

from remo_splat import configs as config
from remo_splat.loader import load_gsplat

gsplatargs = tyro.cli(config.gs.ExampleReal)

means, opacities, scales, quats, colors, sh = load_gsplat(gsplatargs.file, histogram = gsplatargs.hist)

colors = torch.cat((colors, sh),dim = 1)
Poses = gsplatargs.dataset_helper.get_transforms_cv2()
Ks = torch.from_numpy(gsplatargs.dataset_helper.get_camera_intrinsic()).to(device).float()[None, :, :]
#
#
# Ks = [fx, 0, cx,
#       0, fy, cy,
#       0, 0, 1]
#
#

pcd = o3d.geometry.PointCloud()
camera_lines = []
for T_WC in Poses:
    # T_C0W = np.linalg.inv(T_WC)
    T_CW = np.linalg.inv(T_WC)
    width, height = gsplatargs.dataset_helper.get_image_size()
    viewmats = torch.from_numpy(T_CW).to(device).float()[None, :, :]

    intrinsic = o3d.camera.PinholeCameraIntrinsic(width = width, height = height, intrinsic_matrix = Ks[0].cpu().numpy())

    camera_lines.append(o3d.geometry.LineSet.create_camera_visualization(intrinsic, T_CW, scale = 0.1))
    # render
    ti = time.time()
    render, alphas, meta = rasterization(
        means, quats, scales, opacities, colors, viewmats, Ks, width, height,sh_degree = 3, render_mode="RGB+ED"
    )

    r_colors = render[..., :3]
    depth = render[..., 3]
    torch.cuda.synchronize()

    rgb = o3d.geometry.Image(r_colors[0].cpu().numpy())
    d = o3d.geometry.Image(depth[0].cpu().numpy())
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb,d, depth_scale=1.0, depth_trunc=100.0)
    pcd_ = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    pcd_ = pcd_.transform(T_WC)
    pcd += pcd_

o3d.visualization.draw_geometries([pcd] + camera_lines)
