"""
experiment_visualizer.py

This file contain the code for replaying an experiment results

- [ ] Load an scene
- [ ] Get sliders working for swift robot
- [ ]

"""

import re

import numpy as np
import swift
import torch
from neural_robot.unity_frankie import NeuralFrankie

from remo_splat import logger
from remo_splat.configs import gs
from remo_splat.lidar import (DepthSensor, DistanceSensor,
                                     EuclideanDistanceGaussian)
from remo_splat.utils import CameraParams

VERBOSE = False


class ExperimentReplayer:
    def __init__(
        self,
        seq_name,
        gsplat_variant: gs.GSLoader,
        sensor_variant: type[DistanceSensor],
    ):
        env = swift.Swift()
        env.launch(False, browser="chromium", headless=True)

        self.instances = logger.load_folder(seq_name)

        self.n_runs = len(self.instances)
        self.run_id = 0

        self.data = logger.LoggerLoader(self.instances[self.run_id], "", "")
        self.step_id = 0
        self.n_steps = len(self.data.get_data("q"))
        self.robot = NeuralFrankie(
            "curobo", spheres=True
        )  # Change this based on the experiment info
        env.add(self.robot)
        self.exp_name = self.parse_exp_name(seq_name, gsplat_variant, sensor_variant)
        if VERBOSE:
            print("Experiment name", self.exp_name)

        self.logger = logger.Logger(self.exp_name + "/0000")
        self.env = env
        self.gsplat_variant = gsplat_variant
        if sensor_variant == DepthSensor:
            if VERBOSE:
                print("Depth sensor")
            camera_config = CameraParams(80, 80, fov=90)
            self.sensor = sensor_variant(
                gsplat_variant, self.robot.transform_points().shape[0], camera_config
            )
        if sensor_variant == EuclideanDistanceGaussian:
            if VERBOSE:
                print("Euclidean sensor")
            self.sensor = sensor_variant(
                gsplat_variant,
                f"logs/experiments/{self.exp_name}/{self.run_id:04d}/mesh.stl",
            )
        self.sensor_variant = sensor_variant
        self.robot.base = self.data.get_data("T_WB")[0]

        self.robot.q = self.data.get_data("q")[0]

        self.log()  # log first step

    def parse_exp_name(self, seq_name, gsplat_variant, sensor_variant):
        gsplat = "3d" if gsplat_variant.is_3D else "2d"
        sensor = sensor_variant.__name__
        return f"distance/{seq_name}/{gsplat}/{sensor}"

    def step(self):
        self.step_id += 1
        if self.step_id >= self.n_steps:
            return True
        self.robot.base = self.data.get_data("T_WB")[self.step_id]
        self.robot.q = self.data.get_data("q")[self.step_id]

        self.env.step()
        return False

    def step_run(self):
        self.logger.save()

        self.run_id += 1
        if self.run_id == 48:
            self.run_id += 1
        if self.run_id >= 100:  # TODO>= self.n_runs:
            return True
        self.step_id = 0
        if VERBOSE:
            print(f"Loading run {self.instances[self.run_id]}")

        self.data = logger.LoggerLoader(self.instances[self.run_id], "", "")
        self.robot.base = self.data.get_data("T_WB")[0]
        self.robot.q = self.data.get_data("q")[0]

        self.log()  # log first step

        self.n_steps = len(self.data.get_data("q"))
        next_scene = re.sub("\d{4}", f"{self.run_id:04d}", self.gsplat_variant.scene)
        self.gsplat_variant.load(next_scene)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        if self.sensor_variant == DepthSensor:
            if VERBOSE:
                print("Depth sensor")
            camera_config = CameraParams(80, 80, fov=90)
            self.sensor = self.sensor_variant(
                self.gsplat_variant,
                self.robot.transform_points().shape[0],
                camera_config,
            )
        if self.sensor_variant == EuclideanDistanceGaussian:
            if VERBOSE:
                print("Euclidean sensor")
                print(
                    "Path"
                    + f"logs/experiments/{self.exp_name}/{self.run_id:04d}/mesh.stl"
                )
            self.sensor = self.sensor_variant(
                self.gsplat_variant,
                f"logs/experiments/{self.exp_name}/{self.run_id:04d}/mesh.stl",
            )

        self.logger.initialize(f"{self.exp_name}/{self.run_id:04d}", None)
        return False

    def log(self):
        data = {"T_WB": self.robot.base, "q": self.robot.q}
        if isinstance(self.sensor, DepthSensor):
            g_w, distance = self.sensor.log_distances_grad(
                torch.eye(4).cuda().float(), torch.eye(4).cuda().float(), self.robot
            )

        elif isinstance(self.sensor, EuclideanDistanceGaussian):
            g_w, distance = self.sensor.log_distances_grad(
                torch.eye(4).cuda().float(), torch.eye(4).cuda().float(), self.robot
            )

        data["g_w"] = g_w.cpu().numpy()
        data["distance"] = distance.cpu().numpy()
        self.logger.log(data)

    def loop(self):
        while True:
            if self.step():
                if self.step_run():
                    break
            self.log()


if __name__ == "__main__":
    from tqdm import tqdm

    from remo_splat.configs import gs

    # loaders = [gs.ReachingBookshelf, gs.ReachingTable]
    loaders = [gs.ReachingTable]
    # sensors = [DepthSensor, EuclideanDistanceGaussian]
    sensors = [DepthSensor]
    # sensors = [EuclideanDistanceGaussian]
    # dimensions = [True, False]  # 3D, 2D
    dimensions = [False]

    for loader in tqdm(loaders, desc="Loaders", position=0):
        for sensor in tqdm(sensors, desc="Sensors", position=1, leave=False):
            for dim in tqdm(dimensions, desc="Dimensions", position=2, leave=False):
                gs_loader = loader(is_3D=dim)
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                if loader == gs.ReachingTable:
                    exp = ExperimentReplayer("sample_traj_table", gs_loader, sensor)
                else:
                    exp = ExperimentReplayer("sample_traj", gs_loader, sensor)
                exp.loop()
