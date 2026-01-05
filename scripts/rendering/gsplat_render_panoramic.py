import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import spatialmath as sm
import torch
from gsplat.rendering import rasterization

device = "cuda" if torch.cuda.is_available() else "cpu"
import os
import time

import tyro

from remo_splat import configs as config
from remo_splat.configs.gs import ExampleRealGSplat, S12Example
from remo_splat.loader import load_gsplat, load_points_tensor
from remo_splat.rendering import normals_from_depth
from remo_splat.utils import PanoramicViewer

gsplatargs = tyro.cli(config.gs.GSplatLoader)

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

width = 80
height = 40

fov = 90

fx = width / (2 * (np.tan(np.radians(fov / 2))))

Ks[0, 0, 0] = fx
Ks[0, 1, 1] = fx
Ks[0, 0, 2] = width / 2
Ks[0, 1, 2] = height / 2

n_cameras = 6  # """ int(360 / (fov)) """
print("Number of cameras", n_cameras)
Ks = Ks.repeat(n_cameras, 1, 1)


viewer = PanoramicViewer(width, height, n_cameras)
camera_pinhole = o3d.io.read_pinhole_camera_parameters(
    "scripts/rendering/camera_printer.json"
)


def cylindrical_projection(image, K):
    """Efficient cylindrical warp of an image."""
    return image
    h, w = image.shape[:2]
    f = K[0, 0]  # focal length from the intrinsic matrix
    center_x, center_y = K[0, 2], K[1, 2]  # Principal point

    # Create a meshgrid for pixel coordinates
    x, y = np.meshgrid(np.arange(w), np.arange(h))

    # Convert pixel coordinates to normalized camera coordinates
    x_norm = (x - center_x) / f
    y_norm = (y - center_y) / f

    # Cylindrical coordinate transformations
    theta = np.arctan(x_norm)
    h_cyl = y_norm / np.sqrt(1 + x_norm**2)

    # Map cylindrical coordinates back to image coordinates
    x_cyl = f * theta + center_x
    y_cyl = f * h_cyl + center_y

    # Mask to keep valid coordinates within bounds
    mask = (x_cyl >= 0) & (x_cyl < w) & (y_cyl >= 0) & (y_cyl < h)

    # Output image
    cyl_img = np.zeros_like(image)
    cyl_img[y[mask].astype(int), x[mask].astype(int)] = image[
        y_cyl[mask].astype(int), x_cyl[mask].astype(int)
    ]

    return cyl_img


while True:
    for T_WC in Poses:
        T_C0W = np.linalg.inv(T_WC)
        poses = []
        angles = np.linspace(0, 360, n_cameras, endpoint=False)
        for n in angles:
            Ry = sm.SE3.Ry(n, unit="deg").A
            T_CW = np.linalg.inv(T_WC @ Ry)
            poses.append(T_CW)

        T_CW = np.stack(poses, axis=0)

        viewmats = torch.from_numpy(T_CW).to(device).float()

        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=width, height=height, intrinsic_matrix=Ks[0].cpu().numpy()
        )

        camera_lines = [
            o3d.geometry.LineSet.create_camera_visualization(intrinsic, t, scale=0.1)
            for t in poses
        ]

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

        viewer.render(r_colors, depth, None)
        torch.cuda.synchronize()
        print("Time", time.time() - ti, " using ", n_cameras, "cameras")

        #
        # print(colors.shape, alphas.shape)
        # r_colors = r_colors.clamp(0,1)
        # r_colors = (r_colors*255*alphas).to(torch.uint8).squeeze_()
        # depth[depth == 0] = 1000
        # pcd = o3d.geometry.PointCloud()
        # for i in range(n_cameras):
        #     rgb = o3d.geometry.Image(r_colors[i].cpu().numpy())
        #     d = o3d.geometry.Image(depth[i].cpu().numpy())
        #     rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb,d, depth_scale=1.0, depth_trunc=100.0)
        #     pcd_ = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
        #     pcd_ = pcd_.transform(T_CW[i])
        #     pcd += pcd_
        # #
        # #
        # o3d.visualization.draw_geometries([pcd] + camera_lines)

        # subplots
        # fig, axs = plt.subplots(1, render.shape[0], figsize=(15, 5))

        # Stich the images

        # Stitch the cylindrical projections

        # distance = dis.reshape(-1,1) # Flatten #
        # plt.imshow(r_colors.cpu().numpy())
        # import cv2
        #
        # plt.savefig(f"results/renders_fov/{gsplatargs.scene}/rgb/{j}_render_fx{f:04.1f}.png")
        # plt.cla()
        # plt.clf()
        #
        # plt.imshow(depth.cpu().numpy())
        # # cv2.imshow("Depth", depth.cpu().numpy())
        # # cv2.waitKey(0)
        # plt.savefig(f"results/renders_fov/{gsplatargs.scene}/depth/{j}_depth_fx{f:04.1f}.png")
        # plt.cla()
        # plt.clf()
        j += 1
    j = 0
