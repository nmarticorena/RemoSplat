import cv2
import numpy as np
import open3d as o3d
import swift
import torch
from neural_robot.unity_frankie import NeuralFrankie as Robot

from remo_splat import configs
from remo_splat.o3d_visualizer import Visualizer
from remo_splat.teleop import ReplayTeleop, Teleop
from remo_splat.utils import CameraParams, Render, get_camera_lines

env = swift.Swift()
env.launch(realtime=True, browser="chromium", headless=False)
gui = Visualizer()

# Load the scene
config = configs.experiments.Bookshelf2D()
env.add(config.mesh)


gui.add_geometry("mesh", o3d.io.read_triangle_mesh(config.mesh_file))

render = Render(config.gsplat)
robot = Robot(config.robot_name, spheres=True)
robot.q = robot.qr
robot.base = config.T_WB


cv2.namedWindow("depth_2", cv2.WINDOW_NORMAL)
cv2.namedWindow("depth", cv2.WINDOW_NORMAL)

env.add(robot)
# teleop = ReplayTeleop(env, config.gsplat.scene, "rendering")
teleop = Teleop(env, robot.fkine(robot.q))
mat = gui.get_mat()
mat.point_size = 4
mat.base_color = np.array([1.0, 0.0, 0.0, 1.0])


mat2 = gui.get_mat()
mat2.point_size = 12
mat2.base_color = np.array([1.0, 1.0, 0.0, 1.0])


line_mat = gui.get_mat()
line_mat.base_color = np.array([0.0, 1.0, 1.0, 1.0])
gui.add_geometry("camera", o3d.geometry.LineSet(), line_mat)
gui.add_geometry("points", o3d.geometry.PointCloud(), mat)
gui.add_geometry("points_2", o3d.geometry.PointCloud(), mat2)

camera = CameraParams(80, 80, fov=90)
camera_2 = CameraParams(80, 80, fov=60)

while True:
    if teleop.step():
        teleop.load_trajectory(0)
    T_WC = teleop.get_pose()
    T_GC = config.gsplat.T_WG.inv() * T_WC
    T_CW = T_GC.inv()
    T_CW = torch.from_numpy(T_CW.A).float().cuda()[None, ...]
    rgb, depth = render.render(T_CW, camera)
    rgb, depth_2 = render.render(T_CW, camera_2)
    p_w = render.get_points(T_CW, depth, camera)
    p2_w = render.get_points(T_CW, depth_2, camera_2)
    gui.update_geometry("points", p_w)
    gui.update_geometry("points_2", p2_w)
    camera_lines = get_camera_lines(camera, T_CW)
    camera_lines += get_camera_lines(camera_2, T_CW)
    gui.update_geometry("camera", camera_lines)
    for _ in range(1):
        gui.app.run_one_tick()
    rgb = rgb.squeeze()
    depth = depth.squeeze()
    depth_2 = depth_2.squeeze()
    max_depth = 1
    # mask = ~torch.isinf(depth) & ~torch.isnan(depth)
    # fdepth = depth[mask]
    # if fdepth.numel() == 0:
    #     max_depth = 1
    # else:
    # max_depth = fdepth.max()
    # normalize depth
    depth = (depth * (255 / max_depth)).clamp(0, 255).to(torch.uint8)
    depth = cv2.resize(
        depth.cpu().numpy(), (2000, 2000), interpolation=cv2.INTER_NEAREST
    )
    depth = cv2.applyColorMap(depth, cv2.COLORMAP_JET)

    depth_2 = (depth_2 * (255 / max_depth)).clamp(0, 255).to(torch.uint8)
    depth_2 = cv2.resize(
        depth_2.cpu().numpy(), (2000, 2000), interpolation=cv2.INTER_NEAREST
    )
    depth_2 = cv2.applyColorMap(depth_2, cv2.COLORMAP_JET)
    cv2.imshow("depth", depth)
    cv2.imshow("depth_2", depth_2)
    cv2.waitKey(1)
    env.step(0.03)
