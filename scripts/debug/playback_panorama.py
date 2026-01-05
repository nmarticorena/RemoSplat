import open3d as o3d
import swift
import torch
from neural_robot.unity_frankie import NeuralFrankie as Robot

from remo_splat import configs
from remo_splat.lidar import CameraParams, DepthSensor
from remo_splat.o3d_visualizer import Visualizer
from remo_splat.teleop import ReplayTeleop, Teleop

env = swift.Swift()
env.launch(realtime=True, browser="chromium", headless=False)
gui = Visualizer()

# Load the scene
config = configs.experiments.ReplicaCAD()
env.add(config.mesh)

gui.add_geometry("mesh", o3d.io.read_triangle_mesh(config.mesh_file))

camera = CameraParams(80, 80, fov=90)

sensor = DepthSensor(config.gsplat, 1, camera)

robot = Robot(config.robot_name, spheres=True)
robot.q = robot.qr
robot.base = config.T_WB
env.add(robot)

teleop = Teleop(env, robot.fkine(robot.q))
pcd_mat = gui.get_mat(6)
gui.add_geometry("pcd", o3d.geometry.PointCloud(), pcd_mat)
gui.add_geometry("lines", o3d.geometry.LineSet())
gui.add_geometry("camera_lines", o3d.geometry.LineSet())

while True:
    if teleop.step():
        teleop.load_trajectory(0)
    T_WS = teleop.get_pose()  # Center of the sensor
    T_GS = config.gsplat.T_WG.inv() * T_WS  # Centre of the sensor on the gaussian frame
    T_GS = torch.from_numpy(T_GS.A).float().cuda()[None, ...]
    grad, min_depth = sensor.get_distance(T_GS, torch.zeros(1).cuda())
    lines = sensor.draw_lines(None, grad, min_depth)
    pcd = sensor.debug_pcd()
    gui.update_geometry("lines", lines)
    gui.update_geometry(
        "camera_lines", sensor.panorama_sensor.draw_camera_lines(sensor.T_CWs, 0.1)
    )
    gui.update_geometry("pcd", pcd)
    gui.app.run_one_tick()
    env.step()
