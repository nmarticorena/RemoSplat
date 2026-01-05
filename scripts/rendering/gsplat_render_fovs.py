import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
from gsplat.rendering import rasterization

device = "cuda" if torch.cuda.is_available() else "cpu"
import os
import time

import tyro

from remo_splat import configs as config
from remo_splat.configs import ExampleRealGSplat, S12Example
from remo_splat.loader import load_gsplat, load_points_tensor
from remo_splat.rendering import normals_from_depth

gsplatargs = tyro.cli(config.GSplatLoader)

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


j = 0
os.makedirs(f"results/renders_fov/{gsplatargs.scene}", exist_ok=True)
os.makedirs(f"results/renders_fov/{gsplatargs.scene}/depth", exist_ok=True)
os.makedirs(f"results/renders_fov/{gsplatargs.scene}/rgb", exist_ok=True)
fx = Ks[0,0,0]
f_min = 0.001*fx
f_max = 1*fx
focal_lengths = [f for f in torch.linspace(f_min, f_max, 100)]




# How accurate is the projection of the points
# Lets try tto render with open3d

camera_pinhole = o3d.io.read_pinhole_camera_parameters("scripts/rendering/camera_printer.json")

vis = o3d.visualization.Visualizer()

vis.create_window(width=1920, height=1080)
__import__('pdb').set_trace()
for T_WC in Poses:
    T_CW = np.linalg.inv(T_WC)
    viewmats = torch.from_numpy(T_CW).to(device).float()[None, :, :]
    for f in focal_lengths:
        Ks[0,0,0] = f
        Ks[0,1,1] = f
        
        intrinsic = o3d.camera.PinholeCameraIntrinsic(width = 1920, height = 1440, intrinsic_matrix = Ks[0].cpu().numpy())


        camera_lines = o3d.geometry.LineSet.create_camera_visualization(intrinsic, T_CW, scale = 0.1)
        width, height = gsplatargs.dataset_helper.get_image_size()
        # render
        ti = time.time()
        render, alphas, meta = rasterization(
            means, quats, scales, opacities, colors, viewmats, Ks, width, height,sh_degree = 3, render_mode="RGB+ED"
        )
        #
        r_colors = render[..., :3]
        depth = render[..., 3]
        depth.squeeze_(0)

        normals = normals_from_depth(depth, Ks[0])
        normals = normals.cpu().numpy()
        print("Time", time.time() - ti)

        print(colors.shape, alphas.shape)
        r_colors = r_colors.clamp(0,1)
        r_colors = (r_colors*255*alphas).to(torch.uint8).squeeze_()
        # depth[depth == 0] = 1000
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(r_colors.cpu().numpy()), o3d.geometry.Image(depth.cpu().numpy()), depth_scale=1.0, depth_trunc=100.0)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)


        vis.add_geometry(pcd)
        vis.add_geometry(camera_lines)
        ctr = vis.get_view_control()
        pinhole_camera_params = ctr.convert_to_pinhole_camera_parameters()
        pinhole_camera_params.extrinsic = camera_pinhole.extrinsic
        ctr.convert_from_pinhole_camera_parameters(pinhole_camera_params, True)
        vis.poll_events()
        vis.remove_geometry(pcd)
        vis.remove_geometry(camera_lines)

        plt.imshow(r_colors.cpu().numpy())
        import cv2
        
        plt.savefig(f"results/renders_fov/{gsplatargs.scene}/rgb/{j}_render_fx{f:04.1f}.png")
        plt.cla() 
        plt.clf()

        plt.imshow(depth.cpu().numpy())
        # cv2.imshow("Depth", depth.cpu().numpy())
        # cv2.waitKey(0)
        plt.savefig(f"results/renders_fov/{gsplatargs.scene}/depth/{j}_depth_fx{f:04.1f}.png")
        plt.cla()
        plt.clf()
    j += 1

