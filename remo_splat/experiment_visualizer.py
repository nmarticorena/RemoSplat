import copy
import time
from enum import IntEnum
from typing import Type

import numpy as np
import cv2
import open3d as o3d
import spatialmath as sm
import spatialmath.base as smb
import swift
import torch
from neural_robot.unity_frankie import NeuralFrankie
from o3d_visualizer import Visualizer
from tqdm import tqdm

from remo_splat import logger
from remo_splat.benchmarks.compare_distance import Comparisson
from remo_splat.configs import experiments, visualizer, gs
from remo_splat.kernels import visuals
from remo_splat.lidar import DepthSensor, EuclideanDistanceGaussian, reshape_depth_sensor, reshape_euclidean_sensor, EuclideanDistance
from remo_splat.utils import CameraParams, PanoramicViewer


class SwiftReplayer:
    def __init__(self, exp_name, robot_name = "curobo"):
        env = swift.Swift()
        env.launch(True, browser="chromium", rate=1, headless=True)

        self.instances = logger.load_folder(exp_name)
        self.exp_name = exp_name
        self.gui = Visualizer()

        self.robot = NeuralFrankie(
            robot_name, spheres=True
        )  # Change this based on the experiment info
        env.add(self.robot)
        self.spheres = [
            o3d.geometry.TriangleMesh.create_sphere(resolution = 20,radius=r.item())
            for r in (self.robot.SpheresRadius)
        ]

        spheres_mat = self.gui.get_mat(1, color=[0.0, 1.0, 0, 1.0])

        spheres_mat = o3d.visualization.rendering.MaterialRecord()
        spheres_mat.shader = "defaultLit"
        spheres_mat.base_color = [0.0, 1.0, 0.0, 1.0]  # RGBA
        spheres_mat.base_roughness = 0.5
        spheres_mat.base_reflectance = 0.5
        for ix, sphere in enumerate(self.spheres):
            self.gui.add_geometry(f"{ix}", sphere, spheres_mat)

        # self.n_runs = len(self.instances)
        self.n_runs = len(self.instances)
        self.step_slider = self.gui.add_slider("steps", 0, 200, self.update_step)
        self.instance_slider = self.gui.add_slider(
            "Runs",
            0,
            self.n_runs - 1,
            self.update_instance,
        )

        self.show_spheres = True
        self.gui.add_checkbox("show spheres", self._show_spheres_togle, default = True)

        self.show_robot = True
        self.gui.add_checkbox("Show robot", self._show_robot_togle, self.show_robot)
        self.robot_meshes = self.robot.load_meshes()

        # Add filter by spheres
        self.filter_spheres = False
        self.gui.add_checkbox("filter", self._filter_spheres_togle)
        self.hide_rest = False
        self.gui.add_checkbox("hide rest", lambda x: setattr(self, 'hide_rest', x), self.hide_rest)
        self.filter_id = 0
        self.gui.add_slider(
            "spheres", 0, len(self.robot.SpheresRadius) - 1, self._filter_spheres_id_cb
        )

        self.gt_display = False
        self.gui.add_checkbox("GT", self.togle_gt)
        gt_lines = self.gui.get_mat(1, color=[1.0, 1.0, 0, 1.0])
        self.gui.add_geometry("gt_line", o3d.geometry.LineSet(), gt_lines)

        self.gui.add_geometry("gt_mesh", o3d.geometry.TriangleMesh())

        self.gt_geometry = False
        self.gui.add_checkbox("gt mesh ", self.togle_gt_geometry)

        self.eef_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(0.0001)  # TODO RETURN THIS
        self.gui.add_geometry("eef_pose", self.eef_axis)

        target_eef = copy.deepcopy(self.eef_axis)
        self.gui.add_geometry("target_eef", target_eef)

        self.env = env

    def _show_spheres_togle(self, x):
        self.show_spheres = x
        self.update_step(int(self.step_slider.int_value))

    def _show_robot_togle(self, x):
        self.show_robot = x
        self.update_step(self.step_slider.int_value)

    def _filter_spheres_togle(self, x):
        "Togle filter by sphere id"
        self.filter_spheres = x
        self.update_step(int(self.step_slider.int_value))

    def _filter_spheres_id_cb(self, id):
        "Callback to change the sphere that is going to be shown"
        self.filter_id = int(id)
        self.update_step(int(self.step_slider.int_value))



    def update_grad_lines(self, id, name, log):
        """
        Update the geometry of the predicted distances lines
        """
        distance, grad = (
            self.data.get_data("d_distance"),
            self.data.get_data("sensor_grad"),
        )
        breakpoint()
        distance, grad = distance[id], grad[id]
        lines = self.get_lines(distance, grad)
        self.gui.update_geometry(name, lines)

    def togle_gt_geometry(self, x):
        "Togle GT distance lines"
        self.gt_geometry = x
        print(x)
        print(self.data.gt_mesh)
        if x:
            self.gui.update_geometry("gt_mesh", self.data.gt_mesh)
            mesh_lines = o3d.geometry.LineSet.create_from_triangle_mesh(
                self.data.gt_mesh
            )
            self.gui.update_geometry("gt_mesh_line", mesh_lines)
        else:
            self.gui.update_geometry("gt_mesh", o3d.geometry.TriangleMesh())
            self.gui.update_geometry("gt_mesh_line", o3d.geometry.LineSet())

    def get_lines(self, distance, grad):
        """Function to obtain the o3d lines from
        distance and gradient information"""
        if self.filter_spheres:
            grad = grad[self.filter_id][None, ...]
            distance = torch.from_numpy(distance[self.filter_id])
            origins = self.positions[self.filter_id][None, ...]
            distance = distance + self.robot.SpheresRadius[self.filter_id].cpu()
        else:
            origins = self.positions  # [N_sensor, 3]
            distance = torch.from_numpy(distance)
            distance = distance + self.robot.SpheresRadius.cpu()[..., None]
        distance = distance.unsqueeze(-1)
        grad = torch.from_numpy(grad)
        lines = visuals.line_from_dist_grad(origins, grad, distance)
        return lines

    def update_gt_lines(self, id):
        "Update gt lines, based on the step"
        distance, grad = (
            self.data.get_data("gt_distance"),
            self.data.get_data("gt_distance_grad"),
        )
        distance, grad = distance[id], -grad[id]
        lines = self.get_lines(distance, grad)
        self.gui.update_geometry("gt_line", lines)

    def togle_gt(self, x):
        "callback to toggle the gt distance visualization"
        self.gt_display = x
        self.update_step(int(self.step_slider.int_value))

    def update_instance(self, id: int):
        "Update episode"
        id = int(id)
        self.data = logger.LoggerLoader(
            self.exp_name + f"/{id:04d}", "", ""
        )  # Need to improve here

        # Update target pose
        self.gui.update_geometry("target_eef", self.data.target)

        self.n_steps = len(self.data.get_data("q"))
        if self.n_steps > self.step_slider.get_maximum_value:
            self.step_slider.set_limits(0, self.n_steps)
        self.update_step(int(self.step_slider.int_value))
        self.togle_gt_geometry(self.gt_geometry)
        self.id = id

    def update_step(self, id: int):
        "Update step visualization"
        id = int(id)
        id = np.clip(id, 0, self.n_steps - 1)
        self.robot.base = self.data.get_data("T_WB")[id]
        self.robot.q = self.data.get_data("q")[id]
        poses = self.robot.get_point_poses()
        eef = self.robot.fkine(self.robot.q)
        for ix, pose in enumerate(poses):
            mesh = copy.deepcopy(self.spheres[ix])
            mesh.transform(pose.detach().cpu().numpy())
            lines = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
            if not self.show_spheres:
                self.gui.update_geometry(f"{ix}", o3d.geometry.LineSet())
            else:
                if self.hide_rest:
                    if ix == self.filter_id:
                        self.gui.update_geometry(f"{ix}", lines)
                    else:
                        self.gui.update_geometry(f"{ix}", o3d.geometry.LineSet())
                else:
                    self.gui.update_geometry(f"{ix}", lines)

        for k, v in self.robot_meshes.items():
            if self.show_robot:
                mesh = copy.deepcopy(v)
                mesh.transform(self.robot.fkine(self.robot.q, end = k).A)
                self.gui.update_geometry(k, mesh)
            else:
                self.gui.update_geometry(k, o3d.geometry.TriangleMesh())

        # Update eef pose visualization
        axis_mesh = copy.deepcopy(self.eef_axis)
        axis_mesh.transform(eef.A)
        self.gui.update_geometry("eef_pose", axis_mesh)

        if self.gt_display:
            self.update_gt_lines(id)
        else:
            self.gui.update_geometry("gt_line", o3d.geometry.LineSet())

        self.positions = self.robot.transform_points()

    def loop(self):
        "Gui loop"
        self.update_instance(0)
        self.update_step(0)

        while True:
            self.env.step()
            self.gui.app.run_one_tick()

class ExperimentComparisson(SwiftReplayer):
    def __init__(self, config: visualizer.ExperimentVisualizerConfig):
        """
        Replayer of rmmi experiments
        The structure of the folder is
        -exp_name
            -bookshelf
                -3d
                    -depth
                    -euclidean
                -2d
            -table
        """
        self._initialize_enums(config)
        self.show_pred = False
        self.show_euclidean = False
        self.show_gt = True
        self.filter_spheres = True

        self.id = 0
        self.step_id = 0

        self.r_extra = torch.eye(4)[None, ...].cuda() #[1,4,4]
        # self.r_to_camera = torch.from_numpy((sm.SE3.Ry(90, unit = "deg") @ sm.SE3.Rz(90, unit = "deg")).A).cuda().float()
        self.r_to_camera = torch.eye(4).cuda().float()

        # NOTE: The default values needs to be the first one of the radio
        self.scene = args.env_names[0]
        from remo_splat.configs import experiments
        self.sensor = args.sensors[0]
        self.dim = args.dimensions[0]
        self.active = args.active[0]
        if self.active == "null":
            self.active = ""

        self.meta_exp_name = args.folder_name
        demo_path = (
            self.meta_exp_name + f"/{self.scene}/{self.dim}/{self.sensor}"
        )  # to get how many runs we have

        super().__init__(demo_path, args.robot_name) # the robot name should be infered
        self.gui.add_radio_button(args.env_names, self.change_scene)
        self.gui.add_radio_button(args.sensors, self.change_sensor)
        self.gui.add_radio_button(args.dimensions, self.change_dim)
        self.gui.add_radio_button(args.active, self.change_active)

        self.data = logger.LoggerLoader(self.get_path(), "", "")


        gt_mat = self.gui.get_mat(1, color=[1.0, 1.0, 0, 1.0])
        gt_mat.shader = "unlitLine"
        gt_mat.base_color = np.array([1.0, 1.0, 1.0, 1.0])
        self.gui.add_geometry("ellipsoids_lines", o3d.geometry.LineSet(), gt_mat)
        self.gui.add_geometry("ellipsoids", o3d.geometry.TriangleMesh())
        self.gui.add_checkbox("ellipsoids", self.togle_ellipsoids)

        self.gui.add_checkbox("Pred", self.togle_pred, self.show_pred)
        pred_lines = self.gui.get_mat(1, color=[1.0, 0, 0, 1.0])
        self.gui.add_geometry("pred_line", o3d.geometry.LineSet(), pred_lines)

        self.gui.add_checkbox("euclidean_pred", self.togle_euclidean, self.show_euclidean)
        euclidean_lines = self.gui.get_mat(1, color=[0.0, 0.0, 1.0, 1.0])
        new_euclidean_lines = self.gui.get_mat(1, color=[1.0, 0.0, 0.0, 1.0])
        self.gui.add_geometry("euclidean_line", o3d.geometry.LineSet(), euclidean_lines)
        self.gui.add_geometry("new_euclidean_line", o3d.geometry.LineSet(), new_euclidean_lines)

        self.percentage_loaded = 100
        self.gui.add_slider("%Loaded of splats", 0 , 100 , self._set_percentage_loaded_cb, is_float = False)

        # Add stuff to re-render the depth cameras
        self.splats = self.get_splats(0)
        self.load_gui_sensor()

        # To check if is due to the orientations
        self.r_x, self.r_y, self.r_z = 0, 0, 0
        self.gui.add_slider("Extra_angle x", -180,180, lambda x: self.change_rot("x", x))
        self.gui.add_slider("Extra_angle y", -180,180, lambda x: self.change_rot("y", x))
        self.gui.add_slider("Extra_angle z", -180,180, lambda x: self.change_rot("z", x))


        # Control the pcs point size
        self.gui.add_slider("PCD Size", 1, 30, self.set_pcd_size, is_float = False)

        # Visualized the back projected depth map
        self.show_depth = self.Depth.OFF
        self.gui.add_radio_button(self.Depth._member_names_, self.togle_depth, self.show_depth)
        self.create_pcd_mat()

        # Further debuging og the Depth image by rendering the panorama
        # self.panorama_view = PanoramicViewer(80,80,6, upscale=20)
        cv2.namedWindow("Panorama", cv2.WINDOW_NORMAL)


        # Visualize the pose of the sensor we are debugging
        self.sensor_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(0.00001) # TODO RETURN THIS
        self.gui.add_geometry("sensor_pose", self.sensor_axis)

        self.gui.add_slider("Further_distance", 0.5, 10., self._further_distance_cb, is_float = True)
        self.gui.add_slider("max_radius", 0.0, 100.0, self._max_radius_cb, is_float = True)
        self.gui.add_slider("Alpha cutoff", 0.0, 1.0, self._alpha_cb, is_float = True)
        self.gui.add_slider("eps_2d", 0.0, 20.0, self._eps2d_cb, is_float = True)


        # Visualize the ellispoid meshes
        self.gui.add_button("Load Mesh", self.load_mesh)

        # Save image
        self.img_id = 0
        self.gui.add_button("Save Image", self.save_image)

        self.gif_id = 0
        self.gui.add_button("Save GIF", self.save_gif)

        self.gui.add_button("Save Traj", self.save_traj)

        self.depth_fixed_distance = 0
        self.gui.add_slider("Depth Fixed distance", 0, 1, self._set_fixed_distance_cb, is_float = True)

        self.show_trajectory = False
        self.gui.add_checkbox("Show trajectory", self._togle_traj, self.show_trajectory)

        self.show_all_trajectories = False
        self.gui.add_checkbox("Show all trajectories", self._togle_all_traj, self.show_all_trajectories)

        self.show_all_results = False
        self.gui.add_checkbox("Show all results", self._togle_all_results, self.show_all_results)

        self.random_colours = np.random.rand(self.n_runs, 3)

        traj_mat = self.gui.get_mat(10, line = True)
        self.gui.add_geometry("all_base_trajectory", o3d.geometry.LineSet(), traj_mat)
        self.gui.add_geometry("all_trajectory", o3d.geometry.LineSet(), traj_mat)

        self.gui.add_button("Record Rotations", self.record_rotations)
        self.gui.add_button("Remove Cameras", self.remove_cameras)

    def _set_percentage_loaded_cb(self, x):
        self.percentage_loaded = x
        self.reload_gui_sensor()

    def _togle_all_results(self, x):
        """
        Callback to draw all of the final configurations of the robots
        """
        if x:
            for i in tqdm(range(self.n_runs)):
                data = logger.LoggerLoader(
                    self.get_folder_path() + f"{i:04d}", "", ""
                )
                q = data.get_data("q")[-1]
                T_WB = data.get_data("T_WB")[-1]
                colour = self.random_colours[i].tolist()
                for k, v in self.robot_meshes.items():
                    mesh = copy.deepcopy(v)
                    mesh.paint_uniform_color(colour)
                    self.robot.base = T_WB
                    mesh.transform(self.robot.fkine(q, end = k).A)
                    self.gui.update_geometry(f"{k}_{i}", mesh)

    def _togle_traj(self, x):
        """
        Callback to toggle the trajectory of the robot
        """
        self.show_trajectory = x
        if x:
            base, eef = self.data.get_traj_o3d(self.robot)
            self.gui.update_geometry("base_trajectory", base)
            self.gui.update_geometry("trajectory", eef)

        else:
            self.gui.update_geometry("base_trajectory", o3d.geometry.LineSet())
            self.gui.update_geometry("trajectory", o3d.geometry.LineSet())

    def _togle_all_traj(self, x):
        """
        Callback to toggle the trajectory of all the robots
        """
        self.show_all_trajectories = x
        if x:
            t_base = o3d.geometry.LineSet()
            t_eef = o3d.geometry.LineSet()
            for i in tqdm(range(self.n_runs)):
                data = logger.LoggerLoader(
                    self.get_folder_path() + f"{i:04d}", "", ""
                )
                base, eef = data.get_traj_o3d(self.robot)
                t_base += base
                t_eef += eef
            self.gui.update_geometry("all_base_trajectory", t_base)
            self.gui.update_geometry("all_trajectory", t_eef)

        else:
            self.gui.update_geometry("all_base_trajectory", o3d.geometry.LineSet())
            self.gui.update_geometry("all_trajectory", o3d.geometry.LineSet())

    def _initialize_enums(self, config:visualizer.ExperimentVisualizerConfig):
        """
        Initialize the enums for the scene, sensor, dim and active
        to ease up the usage of sliders and radio buttons
        """
        self.Scene = IntEnum("Scene", {name: i for i, name in enumerate(config.env_names)})
        self.Sensor = IntEnum("Sensor", {name: i for i, name in enumerate(config.sensors)})
        self.Dim = IntEnum("Dim", {name: i for i, name in enumerate(config.dimensions)})
        self.Active = IntEnum("Active", {name: i for i, name in enumerate(config.active)})
        self.Depth = IntEnum("Depth", {"OFF": 0, "RGB": 1, "INDIVIDUAL": 2, "JET": 3})

    def _further_distance_cb(self, x):
        """
        Callback to change the further distance of the depth sensor
        """
        self.gui_sensor.render.far_plane = x
        self.update_depth()

    def _max_radius_cb(self, x):
        """
        Callback to change the max radius of the depth sensor
        """
        self.gui_sensor.render.set_max_radius(x)
        self.update_depth()

    def _alpha_cb(self, x):
        """
        Callback to change the alpha cutoff of the depth sensor
        """
        self.gui_sensor.render.set_alpha_cutoff(x)
        self.update_depth()

    def _eps2d_cb(self, x):
        """
        Callback to change the eps2d of the depth sensor
        """
        self.gui_sensor.render.set_eps2d(x)
        self.update_depth()

    def _set_fixed_distance_cb(self, x):
        """
        Callback to change the fixed distance of the depth sensor
        """
        self.depth_fixed_distance = x
        self.update_depth()

    def take_screenshot(self, filename):
        img = self.gui.app.render_to_image(self.gui.scene.scene, 1920, 1080)
        cv2.imwrite(filename, cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB))

    def remove_cameras(self):
        for i in range(10):
            self.gui.update_geometry(f"camera_{i}", o3d.geometry.LineSet())
        self.gui.update_geometry("camera", o3d.geometry.LineSet())

    def record_rotations(self):
        from remo_splat.utils import CameraParams
        import os
        os.makedirs("rotations", exist_ok=True)
        camera = CameraParams(80, 80, fov=90)
        scale = 0.01

        poses = self.robot.get_point_poses()
        T_ws = np.float64(poses[self.filter_id][None,...].cpu().numpy())[0,...]
        T_sw = sm.SE3.Ry(-90, unit = "deg").A @np.linalg.inv(T_ws)# @ sm.SE3.Rx(90, unit="deg").A
        cammera_geom = o3d.geometry.LineSet.create_camera_visualization(80,80, intrinsic = np.float64(camera.K.cpu().numpy()), extrinsic = T_sw, scale = 0.05)
        line_geometry = self.gui.get_mat(5, color=[.0, .0, .0, 1.0], line = True)
        self.gui.add_geometry("camera", cammera_geom, line_geometry)
        self.take_screenshot("rotations/rotation_000.png")
        self.gui.app.run_one_tick()

        def rotation(axis, angle):
            if axis == "x":
                return sm.SE3.Rx(angle, unit="deg").A
            elif axis == "y":
                return sm.SE3.Ry(angle, unit="deg").A
            elif axis == "z":
                return sm.SE3.Rz(angle, unit="deg").A
            else:
                raise ValueError("Axis must be x, y or z")

        def loop_rotation(start, end, step, axis, id):
            nonlocal name
            self.gui.add_geometry(f"camera_{id}", o3d.geometry.LineSet(), line_geometry)
            for i in range(start, end, step):
                R = rotation(axis, i)
                T = R @ T_sw
                cammera_geom = o3d.geometry.LineSet.create_camera_visualization(80,80, intrinsic = np.float64(camera.K.cpu().numpy()), extrinsic = T, scale = 0.05)
                self.gui.update_geometry(f"camera_{id}", cammera_geom)
                self.gui.app.run_one_tick()
                self.take_screenshot(f"rotations/rotation_{name:03d}.png")
                name += 1

        name = 1
        step = 5

        loop_rotation(0, 90 + step, step, "x", 1)
        loop_rotation(0, -90 -step, -step, "x", 2)
        loop_rotation(0, 90 + step, step, "y", 3)
        loop_rotation(90, 180+step, step, "y", 4)
        loop_rotation(180, 270 + step, step, "y", 5)

    def save_image(self):
        """
        Save both the current o3d viewer and the panorama viewer
        """
        import matplotlib.pyplot as plt
        image = self.gui.app.render_to_image(self.gui.scene.scene, 1920, 1080)

        plt.imsave(f"{self.img_id}_o3d.png", np.array(image))
        try:
            cv2.imwrite(f"{self.img_id}_panorama.png", self.rgb)
        except AttributeError:
            print("No panorama image to save, please render the panorama first")

        self.img_id += 1

    def save_gif(self):
        """
        Generate a gif of the trajectory
        """
        import imageio
        import os
        images = []
        panoramic = []
        together = []
        os.makedirs(f"temp/{self.gif_id}", exist_ok=True)
        folder_path = f"temp/{self.gif_id}"

        for i in tqdm(range(0, self.n_steps, 10)):
            self.update_step(i)
            self.gui.app.run_one_tick()
            o3d_render = self.gui.app.render_to_image(self.gui.scene.scene, 1920, 1080)

            cv2.imwrite(f"{folder_path}/{i:04d}.png", cv2.cvtColor(np.array(o3d_render), cv2.COLOR_BGR2RGB))


            images.append(o3d_render)
            panoramic.append(self.rgb.copy())
            cv2.imwrite(f"temp/panorama_{i:04d}.png", self.rgb)
            rgb = self.rgb.copy()
            rgb = cv2.resize(rgb, (1080, 1080))
            together.append(np.hstack((o3d_render, rgb)))


        imageio.mimsave(f"{self.gif_id}_trajectory.gif", images, duration=0.1, loop = 0)
        imageio.mimsave(f"{self.gif_id}_panorama.gif", panoramic, duration=0.1, loop = 0)
        imageio.mimsave(f"{self.gif_id}_together.gif", together, duration=1, loop = 0)

        self.gif_id += 1

        return

    def save_traj(self):
        """
        Generate a gif of the trajectory
        """
        import os
        import subprocess
        for scene in self.Scene:
            self.update_instance(0)

            self.n_runs = len(self.instances)
            self.change_scene(scene.value)
            if "table" in scene.name or "bookshelf" in scene.name:
                self.change_active(1)
                self._max_radius_cb(100)

            self.change_rot("x", 90)
            self.filter_id = 73
            self._further_distance_cb(2.)
            self.load_mesh()
            total = min(self.n_runs, 8)
            for j in tqdm(range(total)):
                folder_path = f"temp/{scene.name}/traj_{self.gif_id}"
                os.makedirs(folder_path, exist_ok=True)
                os.makedirs(f"{folder_path}/traj", exist_ok=True)
                os.makedirs(f"{folder_path}/panorama", exist_ok=True)

                self.update_instance(j)
                if ("table" in scene.name) or ("bookshelf" in scene.name):
                    self.load_mesh()

                self._further_distance_cb(2.)
                frames = 0
                for i in tqdm(range(0, self.n_steps + 5, 5)):
                    self.update_step(i)
                    self.gui.app.run_one_tick()
                    o3d_render = self.gui.app.render_to_image(self.gui.scene.scene, 1920, 1080)

                    cv2.imwrite(f"{folder_path}/traj/{frames:04d}.png", cv2.cvtColor(np.array(o3d_render), cv2.COLOR_BGR2RGB))
                    rgb = self.rgb.copy()
                    rgb = cv2.resize(rgb, (1080, 1080))

                    cv2.imwrite(f"{folder_path}/panorama/{frames:04d}.png", rgb)
                    frames += 1

                subprocess.call([f"ffmpeg -framerate 5 -i {folder_path}/traj/%04d.png -c:v libx264 -pix_fmt yuv420p temp/{scene.name}{self.gif_id}_traj.mp4"], shell = True)
                subprocess.call([f"ffmpeg -framerate 5 -i {folder_path}/panorama/%04d.png -c:v libx264 -pix_fmt yuv420p temp/{scene.name}{self.gif_id}_panorama.mp4"], shell = True)

                self.gif_id += 1

        return

    def togle_ellipsoids(self, x):
        if x:
            self.gui.update_geometry("ellipsoids", "")
        else:
            self.gui.update_geometry("ellipsoids", o3d.geometry.TriangleMesh())

    def load_mesh(self):
        from safer_splat.splat.gsplat_utils import GSplatLoader

        gsplat = GSplatLoader(
            None,
            "cuda",
            means = self.gui_sensor.render.means,
            scales = self.gui_sensor.render.scales,
            rots = self.gui_sensor.render.quats,
            colors = self.gui_sensor.render.colors,
        )
        mesh = gsplat.get_mesh(transform = self.splats.gsplat.T_WG.A)
        self.gui.update_geometry("elipsoids", mesh)
        # mesh_lines = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
        # self.gui.update_geometry("ellipsoids_lines", mesh_lines)


        return True

    def change_rot(self, axis = "x", new_angle = 0):
        if axis == "x":
            self.r_x = new_angle
        elif axis == "y":
            self.r_y = new_angle
        elif axis == "z":
            self.r_z = new_angle
        R = sm.SE3.Rx(self.r_x, unit="deg") @ sm.SE3.Ry(self.r_y, unit="deg") @ sm.SE3.Rz(self.r_z, unit="deg")
        self.r_extra[0,:,:] = torch.from_numpy(R.A).cuda()
        self.update_depth()

    def load_gui_sensor(self):
        self.gui_sensor = DepthSensor(self.splats.gsplat, n_cameras= 1, camera_params= CameraParams(80,80,fov = 90))

    def reload_gui_sensor(self):
        seq_id = self.id
        self.splats.load(seq_id)
        self.load_gui_sensor()
        self.togle_euclidean(self.show_euclidean)

    def update_instance(self, id: int):
        super().update_instance(id)
        self.reload_gui_sensor()

    def create_pcd_mat(self):
        front_pcd = self.gui.get_mat(4, color=[0.0, 1.0, 0.0, 1.0])
        right_pcd = self.gui.get_mat(4, color=[218 / 255, 44 / 55, 230 / 255, 1.])
        left_pcd = self.gui.get_mat(4, color=[75 / 255, 230 / 255, 44 / 255, 1.])
        down_pcd = self.gui.get_mat(4, color=[230 / 255, 156 / 255, 44 / 255, 1.])

        # List of point cloud names and their corresponding variables
        pcd_names = [
            "front_pcd",
            "right_pcd",
            "back_pcd",
            "left_pcd",
            "up_pcd",
            "down_pcd",
        ]
        pcd_values = [
            front_pcd,
            right_pcd,
            None,
            left_pcd,
            None,
            down_pcd,
        ]  # Ensure all are defined

        # Add each point cloud to the GUI
        for name, pcd in zip(pcd_names, pcd_values):
            self.gui.add_geometry(name, o3d.geometry.PointCloud(), pcd)

        # Add a point cloud for the all rgb view
        self.gui.add_geometry("rgb_pcd", o3d.geometry.PointCloud())

        # Store order mapping dynamically
        self.order = {i: name for i, name in enumerate(pcd_names)}

    def set_pcd_size(self, x):
        """
        Callback to change the point size of the point clouds
        """
        for k, v in self.order.items():
            pcd = self.gui.get_mat(point_size = x, color=[0.0, 1.0, 0.0, 1.0])
            self.gui.update_material(v, pcd)
        rgb_pcd = self.gui.get_mat(point_size = x, color=[1.0, 1.0, 1.0, 1.0])
        self.gui.update_material("rgb_pcd", rgb_pcd)
        if self.show_depth != self.Depth.OFF:
            self.update_depth()

    def update_grad_lines(self, id, name, log):
        distance, grad = (
            self.data.get_data("d_distance"),
            self.data.get_data("sensor_grad"),
        )
        breakpoint()
        if "euclidean" in self.sensor:
            distance, grad = reshape_euclidean_sensor(distance, grad)
        elif self.sensor == "depth":
            distance, grad = reshape_depth_sensor(distance, grad)

        # if "gt" not in self.sensor:
            # distance, grad = logger.get_mins(distance, grad)

        distance, grad = distance[id], grad[id]
        lines = self.get_lines(distance, grad)
        self.gui.update_geometry(name, lines)

    def compute_euclidean_sensor(self, name, euclidean_sensor: EuclideanDistanceGaussian):
        import warp as wp
        wp.init()
        p_w = self.robot.transform_points()
        euclidean_sensor.test_new_kernel(p_w, self.robot.SpheresRadius)

        # grad, d = euclidean_sensor.get_distance(p_w, self.robot.SpheresRadius)
        # grad , d = grad.detach().cpu().numpy(), d.detach().cpu().numpy()
        # lines = self.get_lines(d, grad)
        # self.gui.update_geometry(name, lines)

    def clear_depth(self):
        self.clear_individual_pcd()
        self.clear_rgb_pcd()

    def clear_individual_pcd(self):
        for _, v in self.order.items():
            self.gui.update_geometry(v, o3d.geometry.PointCloud())

    def clear_rgb_pcd(self):
        self.gui.update_geometry("rgb_pcd", o3d.geometry.PointCloud())

    def update_depth(self):
        T_GW = self.splats.gsplat.T_WG.inv()
        T_GW = torch.from_numpy(T_GW.A).cuda().float()
        poses = self.robot.get_point_poses()
        pose = T_GW @ poses[self.filter_id][None,...] @ self.r_to_camera  @ self.r_extra


        self.gui_sensor.get_distance_gui(pose, torch.zeros(1).cuda(), self.depth_fixed_distance)

        if self.show_depth == self.Depth.RGB:
            G_pcd = self.gui_sensor.debug_rgb_pcd()
            self.gui.update_geometry("rgb_pcd", G_pcd.transform(self.splats.gsplat.T_WG.A))
            self.clear_individual_pcd()
        elif self.show_depth == self.Depth.INDIVIDUAL:
            G_pcd = self.gui_sensor.debug_individual_pcd()
            for k,v in self.order.items():
                self.gui.update_geometry(v, G_pcd[int(k)].transform(self.splats.gsplat.T_WG.A))
            self.clear_rgb_pcd()
        elif self.show_depth == self.Depth.JET:
            G_pcd = self.gui_sensor.debug_rgb_pcd(depth = True)
            self.gui.update_geometry("rgb_pcd", G_pcd.transform(self.splats.gsplat.T_WG.A))
            self.clear_individual_pcd()
        else:
            return

        ti = time.perf_counter()
        # self.panorama_view.render(*self.gui_sensor._render(pose))
        variant = "jet" if self.show_depth == self.Depth.JET else "rgb"
        rgb = self.gui_sensor.stiching(pose, variant = variant )
        self.rgb = rgb.copy()
        # self.rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        cv2.imshow("Panorama", self.rgb)
        cv2.waitKey(1)
        print("Time to render panorama", time.perf_counter() - ti)
        sensor_axis = copy.deepcopy(self.sensor_axis)
        pose = torch.linalg.inv(T_GW) @ pose
        sensor_axis.transform(pose.cpu().numpy()[0,:,:])
        self.gui.update_geometry("sensor_pose", sensor_axis)

    def update_step(self, id: int):
        super().update_step(id)
        id = int(id)
        id = np.clip(id, 0, self.n_steps - 1)
        self.step_id = id
        if self.show_trajectory:
            self._togle_traj(self.show_trajectory)
        if self.show_pred:
            self.update_grad_lines(id, "pred_line", self.data)
        else:
            self.gui.update_geometry("pred_line", o3d.geometry.LineSet())
        if self.show_euclidean:
            # self.compute_euclidean_sensor("euclidean_line", self.euclidean_sensor)
            self.compute_euclidean_sensor("new_euclidean_line", self.new_euclidean_sensor)
        else:
            self.gui.update_geometry("euclidean_line", o3d.geometry.LineSet())
            self.gui.update_geometry("new_euclidean_line", o3d.geometry.LineSet())


        if self.show_depth:
            self.update_depth()
        else:
            self.clear_depth()

    def togle_pred(self, x):
        self.show_pred = x
        self.update_step(int(self.step_slider.int_value))

    def togle_euclidean(self, x):
        self.show_euclidean = x
        if self.show_euclidean:
            self.euclidean_sensor = EuclideanDistanceGaussian(self.splats.gsplat, max_steps = 10, min = False)
            self.new_euclidean_sensor = EuclideanDistanceGaussian(self.splats.gsplat, max_steps = 10, min = False ,distance_type = "sphere-to-disk")

        self.update_step(int(self.step_slider.int_value))

    def togle_depth(self, x):
        self.show_depth = x
        self.update_step(int(self.step_slider.int_value))

    def reload_data(self):
        print("reloading", self.get_path())
        self.data = logger.LoggerLoader(self.get_path(), "", "")
        print(self.data)
        self.exp_name = self.get_folder_path()
        print("Reloading splats")
        self.update_instance(self.id)
        self.reload_gui_sensor()
        print(self.data)
        print(self.sensor)
        self.update_step(self.step_id)

    def get_path(self):
        # if self.sensor == "gt":
        #     dim = ""
        #     if self.active:
        #         active= "_active"
        #     else:
        #         active= ""
        # else:
        dim = self.dim
        active = self.active
        return (
            self.meta_exp_name + f"/{self.scene}/{dim}/{self.sensor}{active}/{self.id:04d}"
        )

    def get_folder_path(self):
        # if self.sensor == "gt":
        #     dim = ""
        #     if self.active:
        #         active= "_active"
        #     else:
        #         active= ""
        # else:
        dim = self.dim
        active = self.active

        return f"{self.meta_exp_name}/{self.scene}/{dim}/{self.sensor}{active}/"

    def change_scene(self, id):
        self.scene = self.Scene(id).name
        self.splats = self.get_splats(id)

        self.reload_data()

    def get_splats(self, scene_id):
        scene = self.Scene(scene_id).name
        if "table_new" in scene:
            return experiments.ReachingTable(gsplat = gs.ReachingTable(scene = f"{scene}/{scene}_0000"))
            # return experiments.ReachingTable()
        elif "bookshelf_cage" in scene:
            # return experiments.ReachingBookshelf()
            return experiments.ReachingBookshelf(gsplat = gs.ReachingBookshelf(scene = f"{scene}/{scene}_0000"))
        else:
            return experiments.ReachingRealWorldConfig(gsplat = gs.ExampleReal(scene = scene))

    def change_sensor(self, id):
        self.sensor = self.Sensor(id).name
        self.reload_data()

    def change_dim(self, id):
        self.dim = self.Dim(id).name
        self.reload_data()

    def change_active(self, id):
        self.active = self.Active(id).name
        if self.active == "null":
            self.active = ""
        self.reload_data()


class GradComparisson(SwiftReplayer):
    def __init__(self, exp_name):
        self.show_euclidean = False
        self.show_euclidean_3d = False
        self.show_depth = False
        self.show_depth_3d = False
        self.filter_spheres = False
        super().__init__(exp_name)
        self.gt_display = False
        self._3d_geometry = False
        self._2d_geometry = False

        self.gui.add_checkbox("Euclidean Red", self.togle_euclidean)
        self.gui.add_checkbox("Euclidean 3D Cyan", self.togle_euclidean_3d)
        euclidean_lines = self.gui.get_mat(1, color=[1.0, 0, 0, 1.0], line=True)
        euclidean_lines_3d = self.gui.get_mat(1, color=[0.5, 1.0, 0.5, 1.0], line=True)
        self.gui.add_geometry("euclidean_line", o3d.geometry.LineSet(), euclidean_lines)
        self.gui.add_geometry(
            "euclidean_line_3D", o3d.geometry.LineSet(), euclidean_lines_3d
        )
        self.euclidean = Comparisson(
            self.exp_name,
            is_3D=False,
            sensor_variant="EuclideanDistanceGaussian",
            load_mesh=True,
        )
        self.euclidean_3D = Comparisson(
            self.exp_name,
            is_3D=True,
            sensor_variant="EuclideanDistanceGaussian",
            load_mesh=True,
        )

        self.gui.add_checkbox("Depth Green", self.togle_depth)
        self.gui.add_checkbox("Depth 3D", self.togle_depth_3d)
        depth_lines = self.gui.get_mat(1, color=[0.0, 1.0, 0, 1.0], line=True)
        depth_lines_3d = self.gui.get_mat(1, color=[0.0, 1.0, 1.0, 1.0], line=True)
        self.gui.add_geometry("depth_line", o3d.geometry.LineSet(), depth_lines)
        self.gui.add_geometry("depth_line_3D", o3d.geometry.LineSet(), depth_lines_3d)
        self.depth = Comparisson(
            self.exp_name, is_3D=False, sensor_variant="DepthSensor"
        )
        self.depth_3D = Comparisson(
            self.exp_name, is_3D=True, sensor_variant="DepthSensor"
        )

        gt_mat = self.gui.get_mat(1, color=[1.0, 1.0, 0, 1.0])
        gt_mat.shader = "unlitLine"
        gt_mat.base_color = np.array([1.0, 1.0, 1.0, 1.0])
        self.gui.add_geometry("gt_mesh_line", o3d.geometry.LineSet(), gt_mat)

        self.gui.add_geometry("ellipsoids", o3d.geometry.TriangleMesh())
        self.gui.add_checkbox("3D Ellipsoids", self.togle_3d_geometry)

        self.gui.add_geometry("disk", o3d.geometry.TriangleMesh())
        self.gui.add_checkbox("2D Ellipsoids", self.togle_disk_geometry)

    def update_grad_lines(self, id, name, log):
        distance, grad = log.get_data(id)
        lines = self.get_lines(distance, grad)
        self.gui.update_geometry(name, lines)

    def update_step(self, id: int):
        super().update_step(id)
        id = int(id)
        id = np.clip(id, 0, self.n_steps - 1)
        if self.show_euclidean:
            self.update_grad_lines(id, "euclidean_line", self.euclidean)
        else:
            self.gui.update_geometry("euclidean_line", o3d.geometry.LineSet())

        if self.show_euclidean_3d:
            self.update_grad_lines(id, "euclidean_line_3D", self.euclidean_3D)
        else:
            self.gui.update_geometry("euclidean_line_3D", o3d.geometry.LineSet())

        if self.show_depth:
            self.update_grad_lines(id, "depth_line", self.depth)
        else:
            self.gui.update_geometry("depth_line", o3d.geometry.LineSet())

        if self.show_depth_3d:
            self.update_grad_lines(id, "depth_line_3D", self.depth_3D)
        else:
            self.gui.update_geometry("depth_line_3D", o3d.geometry.LineSet())

    def update_instance(self, id: int):
        self.load_sequence(id)
        super().update_instance(id)
        self.togle_gt_geometry(self.gt_geometry)
        self.togle_3d_geometry(self._3d_geometry)
        self.togle_disk_geometry(self._2d_geometry)

    def load_sequence(self, id):
        self.euclidean.load_sequence(int(id))
        self.euclidean_3D.load_sequence(int(id))
        self.depth.load_sequence(int(id))
        self.depth_3D.load_sequence(int(id))

    def togle_3d_geometry(self, x):
        self._3d_geometry = x
        if x:
            self.gui.update_geometry("ellipsoids", self.euclidean_3D.mesh)
        else:
            self.gui.update_geometry("ellipsoids", o3d.geometry.TriangleMesh())

    def togle_disk_geometry(self, x):
        self._2d_geometry = x
        if x:
            self.gui.update_geometry("disk", self.euclidean.mesh)
        else:
            self.gui.update_geometry("disk", o3d.geometry.TriangleMesh())

    def togle_euclidean(self, x):
        self.show_euclidean = x
        self.update_step(int(self.step_slider.int_value))

    def togle_euclidean_3d(self, x):
        self.show_euclidean_3d = x
        self.update_step(int(self.step_slider.int_value))

    def togle_depth(self, x):
        self.show_depth = x
        self.update_step(int(self.step_slider.int_value))

    def togle_depth_3d(self, x):
        self.show_depth_3d = x
        self.update_step(int(self.step_slider.int_value))


if __name__ == "__main__":
    import tyro

    args = tyro.cli(visualizer.ExperimentVisualizerConfig)

    replayer = ExperimentComparisson(args)
    replayer.loop()
