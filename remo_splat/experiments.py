import json
import shutil

import numpy as np
import open3d as o3d
import spatialgeometry as sg
import swift
import torch
from tqdm import tqdm

import remo_splat
from remo_splat.configs.experiments import (ExperimentConfig,
                                                   ReachingExperimentConfig,
                                                   Sensor, load_mesh)
from remo_splat.controller import MMController
from remo_splat.lidar import DepthSensor, EuclideanDistanceGaussian
from remo_splat.logger import Logger
from remo_splat.o3d_visualizer import Visualizer
from remo_splat.teleop import ReplayTeleop


class Experiment:
    def __init__(self, config: ExperimentConfig, env: swift.Swift, robot_fn):
        self.env = env
        self.config = config
        self.robot_fn = robot_fn

        self.period = self.config.controller.step_time
        self.logger = Logger(self.config.exp_name, self.config.controller)
        self.load_experiment(config, env, robot_fn)
        self.T_WG = config.gsplat.T_WG
        self.T_GW = torch.tensor(self.T_WG.inv().A, dtype=torch.float32).to("cuda")

    def load_experiment(self, config: ExperimentConfig, env: swift.Swift, robot_fn):
        env.add(config.mesh)
        robot = robot_fn(config.robot_name, spheres=True)
        robot.q = robot.qr
        robot.base = config.T_WB

        env.add(robot)
        self.robot = robot
        self.teleop = ReplayTeleop(
            env,
            config.gsplat.scene,
            config.trajectory_file,
            query_rate=self.period,
        )
        self.sensor_func = (
            DepthSensor if config.sensor == "depth" else EuclideanDistanceGaussian
        )
        if self.sensor_func == DepthSensor:
            self.sensor = DepthSensor(
                config.gsplat, robot.transform_points().shape[0], config.camera
            )
        else:
            self.sensor = EuclideanDistanceGaussian(config.gsplat)

        self.controller = MMController(
            self.robot, self.sensor, config.controller, logger=self.logger
        )

    def step(self):
        """
        Step the controller, we return true when we run out of poses on the trajectory
        """
        done_poses = self.teleop.step()
        done, self.robot.qd, failed = self.controller.step(
            self.teleop.get_pose(), self.T_GW
        )
        self.env.step(self.period)
        base_new = self.robot.fkine(
            self.robot._q, end=self.robot.links[self.robot.base_dofs]
        ).A
        self.robot._T = base_new
        self.robot.q[: self.robot.base_dofs] = 0
        return done_poses or failed

    def run_traj(self, traj_index: int, n_repeat: int):
        """
        Run a full loop of the experiment with the given trajectory index
        We repeat the trajectory n_repeat times hopefully randomizing some aspect of the robot
        """
        self.teleop.load_trajectory(traj_index)
        for i in tqdm(range(n_repeat)):
            self.logger.initialize(f"{self.config.exp_name}/{traj_index}/{i}", None)
            while not self.step():
                pass
            self.logger.save()
            self.teleop.reset_traj()

class ReachingExperiment:
    def __init__(self, config: ReachingExperimentConfig, env: swift.Swift, robot_fn):
        self.env = env
        self.config = config
        self.robot_fn = robot_fn

        self.debug_target = sg.Axes(0.1)

        self.pose_index = 0
        self.period = self.config.controller.step_time
        self._initialize_logger(config, self.pose_index)
        self.data_folder = remo_splat.__path__[0] + "/../data"
        try:
            self.mesh, _ = load_mesh(f"{self.data_folder}/{config.gsplat.exp_name}.json")
        except FileNotFoundError:
            self.mesh = sg.Mesh()
        self._initialize_o3d_viewer(config)
        self.load_experiment(config, env, robot_fn)
        self.T_WG = config.gsplat.T_WG
        self.T_GW = torch.tensor(self.T_WG.inv().A, dtype=torch.float32).to("cuda")

    def load_experiment(
        self, config: ReachingExperimentConfig, env: swift.Swift, robot_fn
    ):
        self._launch_enviroment(env, config)
        self.debug_target.T = config.objective

        self._initialize_robot(config, robot_fn, env)
        self._initialize_sensor(config)


    def load_pose(self, pose_index: int):
        """
        Load a new pose to reach, in this dataset we have only one pose per experiment
        So also we need to load the next gsplat
        """
        self.pose_index = pose_index
        config = self.config
        config.load(pose_index)

        self._initialize_sensor(config)
        self._initialize_logger(config, pose_index)
        self._initialize_controller(config)
        self._update_swift(config)
        self._update_gui(config)
        self._reset_robot(config)
        print("Finished loading Pose")

    def run_pose(self):
        for _ in tqdm(range(self.config.steps)):
            done, self.robot.qd, failed = self.controller.step(
                self.config.objective, self.T_GW, gt_sdf=self.config.gt_sdf
            )
            self.env.step(self.period)
            base_new = self.robot.fkine(
                self.robot._q, end=self.robot.links[self.robot.base_dofs]
            ).A
            self.robot._T = base_new
            self.robot.q[: self.robot.base_dofs] = 0
            if done:
                break
            if self.vis:
                self.vis.app.run_one_tick()
        if self.logger:
            self.logger.save()

    def _initialize_o3d_viewer(self, config:ReachingExperimentConfig):
        if not config.o3d_vis:
            self.vis = None
            return
        self.vis = Visualizer()
        mesh_mat = self.vis.get_mat()
        self.vis.add_geometry("mesh", self.mesh)

        # Set up the line mat
        line_mat = self.vis.get_mat()
        line_mat.shader = "unlitLine"
        self.vis.add_geometry("lines", o3d.geometry.LineSet(), line_mat)

        gt_line_mat = self.vis.get_mat(color=[1, 1, 0, 1])
        gt_line_mat.shader = "unlitLine"
        self.vis.add_geometry("gt_lines", o3d.geometry.LineSet(), gt_line_mat)


    def _initialize_logger(self, config:ReachingExperimentConfig, pose_index:int):
        if not config.log:
            self.logger = None
            return
        self.logger = Logger(
            f"{config.exp_name}/{pose_index:04d}", config.controller
        )
        self.logger.initialize(f"{config.exp_name}/{pose_index:04d}", None)
        logger_path = self.logger.folder_name

        print(f"Copying {self.config.json_path} to {logger_path}/scene.json")
        try:
            shutil.copyfile(self.config.json_path, f"{logger_path}/scene.json")
        except FileNotFoundError:
            print("No scene file found")


    def _reset_robot(self, config:ReachingExperimentConfig):
        self.robot.q = self.robot.qr
        self.robot.base = self.config.T_WB
        self.robot.qd = np.zeros_like(self.robot.qd)

    def _update_gui(self, config:ReachingExperimentConfig):
        if not self.vis:
            return
        self.mesh, _ = load_mesh(
            f"{self.data_folder}/{config.gsplat.exp_name}.json"
        )
        self.vis.update_geometry("mesh", self.mesh)

    def _update_swift(self, config:ReachingExperimentConfig):
        # update the obs on swift:
        for ix, ob in enumerate(config.obstacles):
            self.swift_obs[ix].T = ob.T
            # if isinstance(ob, sg.Cylinder):
            #     self.swift_obs[ix].length = ob.length
            #     self.swift_obs[ix].radius = ob.radius
            # elif isinstance(ob, sg.Cuboid):
            #     self.swift_obs[ix].scale = ob.scale

    def _launch_enviroment(self, env:swift.Swift, config:ReachingExperimentConfig):
        env.launch(realtime=True, browser="chromium", headless=not config.gui)
        self.swift_obs = []
        for ob in config.obstacles:
            env.add(ob)
            self.swift_obs.append(ob)

    def _initialize_robot(self, config:ReachingExperimentConfig, robot_fn, env:swift.Swift):
        robot = robot_fn(config.robot_name, spheres= config.spheres)
        gt_robot = robot_fn("points_0", spheres=False)
        robot.q = robot.qr
        robot.base = config.T_WB

        env.add(robot)
        self.robot = robot
        self.gt_robot = gt_robot
        return


    def _initialize_controller(self, config:ReachingExperimentConfig):
        logger = self.logger if config.log else None
        self.controller = MMController(self.robot, self.sensor, config.controller, gui=self.vis, logger = logger)

    def _initialize_sensor(self, config: ReachingExperimentConfig):
        if config.sensor == Sensor.depth or config.sensor == Sensor.depth_min:
            sensor = DepthSensor(
                config.gsplat, self.robot.transform_points().shape[0], config.camera
            )
        elif config.sensor == Sensor.euclidean:
            sensor = EuclideanDistanceGaussian(config.gsplat, mesh =f"{self.logger.folder_name}/mesh.stl" )
        elif config.sensor == Sensor.euclidean_less:
            sensor = EuclideanDistanceGaussian(config.gsplat, max_steps=3)
            print("Using euclidean less")
        elif config.sensor == Sensor.euclidean_all:
            sensor = EuclideanDistanceGaussian(config.gsplat, max_steps=3, min = False)
        elif config.sensor in [Sensor.gt, Sensor.gt_active, Sensor.gt_active_faster, Sensor.gt_all]:
            sensor = None
            config.controller.ideal = True
            config.controller.ideal_all = False
            if config.sensor == Sensor.gt_active:
                config.controller.collision_cost = "w_avg"
                config.steps = 200
            elif config.sensor == Sensor.gt_all:
                config.controller.ideal_all = True
            elif config.sensor == Sensor.gt_active_faster: #TODO: Need to update this
                config.controller.collision_cost = "w_avg"
                config.steps = 1000
                config.step_time = 0.01 # 100 hz
                config.controller.step_time = config.step_time
                self.period = config.step_time
                self.logger = Logger(
                    f"{self.config.exp_name}/{self.pose_index:04d}", config.controller
                )
        elif config.sensor is None:
            sensor = None
        else:
            raise ValueError(f"Unknown sensor type: {config.sensor}")
        self.sensor = sensor





if __name__ == "__main__":
    env = swift.Swift()
    import tyro
    from neural_robot.unity_frankie import NeuralFrankie

    from remo_splat.configs import experiments

    config = tyro.cli(experiments.ReachingBookshelf)

    robot_fn = NeuralFrankie
    exp = ReachingExperiment(
        config,
        env,
        robot_fn,
    )
    for j in range(0, 500):
        # j = 3
        print("Loading pose", j)
        exp.load_pose(j)
        exp.run_pose()
