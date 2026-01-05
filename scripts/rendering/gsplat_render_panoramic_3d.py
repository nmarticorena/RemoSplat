import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import spatialmath as sm
import torch
from gsplat.rendering import rasterization

from remo_splat.o3d_visualizer import Visualizer

device = "cuda" if torch.cuda.is_available() else "cpu"
import os
import time

import tyro

from remo_splat import configs as config
from remo_splat.configs import ExampleRealGSplat, S12Example
from remo_splat.loader import load_gsplat, load_points_tensor
from remo_splat.rendering import normals_from_depth

gsplatargs = tyro.cli(config.GSplatLoader)

means, opacities, scales, quats, colors, sh = load_gsplat(
    gsplatargs.file, histogram=gsplatargs.hist
)

colors = torch.cat((colors, sh), dim=1)
Poses = gsplatargs.dataset_helper.get_transforms_cv2()
Ks = (
    torch.from_numpy(gsplatargs.dataset_helper.get_camera_intrinsic())
    .to(device)
    .float()[None, :, :]
)

j = 0
os.makedirs(f"results/renders_fov/{gsplatargs.scene}", exist_ok=True)
os.makedirs(f"results/renders_fov/{gsplatargs.scene}/depth", exist_ok=True)
os.makedirs(f"results/renders_fov/{gsplatargs.scene}/rgb", exist_ok=True)

width = 800
height = 400

fov = 360 / 6

fovy = 120

fx = width / (2 * (np.tan(np.radians(fov / 2))))
fy = height / (2 * (np.tan(np.radians(fovy / 2))))

Ks[0, 0, 0] = fx
Ks[0, 1, 1] = fy
Ks[0, 0, 2] = width / 2
Ks[0, 1, 2] = height / 2

n_cameras = int(360 / (fov)) + 2  # up and down
print("Number of cameras", n_cameras)
Ks = Ks.repeat(n_cameras, 1, 1)


camera_pinhole = o3d.io.read_pinhole_camera_parameters("configs/apt_2_nav.json")

gui = Visualizer()

mesh = o3d.io.read_triangle_mesh("results/meshes/apt_2_nav.ply")
# gui.add_geometry("mesh", mesh)

# gui.set_camera("configs/apt_2_nav.json")
# gui.app.run_in_thread()
while True:
    for T_WC in Poses:
        T_C0W = np.linalg.inv(T_WC)
        poses = []
        angles = np.linspace(0, 360, n_cameras - 2, endpoint=False)
        for n in angles:
            Ry = sm.SE3.Ry(n, unit="deg").A
            T_CW = np.linalg.inv(T_WC @ Ry)
            poses.append(T_CW)
        # up and down
        Rup = sm.SE3.Rx(90, unit="deg").A
        T_CW = np.linalg.inv(T_WC @ Rup)
        poses.append(T_CW)

        Rdown = sm.SE3.Rx(-90, unit="deg").A
        T_CW = np.linalg.inv(T_WC @ Rdown)
        poses.append(T_CW)

        T_CW = np.stack(poses, axis=0)

        viewmats = torch.from_numpy(T_CW).to(device).float()

        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=width, height=height, intrinsic_matrix=Ks[0].cpu().numpy()
        )

        camera_lines = o3d.geometry.LineSet()
        for i in range(n_cameras):
            camera_lines += o3d.geometry.LineSet.create_camera_visualization(
                intrinsic, T_CW[i], scale=0.1
            )

        # render
        ti = time.time()
        render, alphas, meta = rasterization(
            means,
            quats,
            scales,
            opacities,
            colors,
            viewmats,
            Ks,
            width,
            height,
            sh_degree=3,
            render_mode="RGB+ED",
        )

        r_colors = render[..., :3]
        depth = render[..., 3]
        depth[alphas[:, :, :, -1] < 0.9] = 10000
        torch.cuda.synchronize()
        print("Time", time.time() - ti, " using ", n_cameras, "cameras")

        pcd = o3d.geometry.PointCloud()
        for i in range(n_cameras):
            rgb = o3d.geometry.Image(r_colors[i].cpu().numpy())
            d = o3d.geometry.Image(depth[i].cpu().numpy())
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                rgb, d, depth_scale=1.0, depth_trunc=5.0
            )
            pcd_ = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
            pcd_ = pcd_.transform(np.linalg.inv(T_CW[i]))
            pcd += pcd_
        pcd.paint_uniform_color([0.5, 0.5, 0.5])

        gui.update_geometry("pcd", pcd)
        gui.update_geometry("camera_lines", camera_lines)
        for _ in range(100):
            gui.app.run_one_tick()
        # gui.step([pcd] + camera_lines)

        # gui.set_camera("configs/apt_2_nav.json")
        j += 1
    j = 0
