from dataclasses import dataclass
from typing import List, Type

import swift
import tyro
from neural_robot.unity_frankie import NeuralFrankie
from tqdm import tqdm

from remo_splat.configs import experiments, gs
from remo_splat.experiments import ReachingExperiment
from remo_splat.lidar import DepthSensor, EuclideanDistanceGaussian


@dataclass
class Config:
    sensor_name:str = "euclidean_less"
    dimension:bool = False
    active: bool = False


args = tyro.cli(Config)

robot_fn = NeuralFrankie

sensor_names = {
    "depth": experiments.Sensor.depth,
    "euclidean": experiments.Sensor.euclidean,
    "euclidean_less": experiments.Sensor.euclidean_less
}

# scenes: List[Type[experiments.ReachingExperimentConfig]] = [experiments.ReachingTable, experiments.ReachingBookshelf]
scenes: List[Type[experiments.ReachingExperimentConfig]] = [experiments.ReachingBookshelf]
sensor = sensor_names[args.sensor_name]

dim = args.dimension
act = args.active

for scene in tqdm(scenes):
    env = swift.Swift()
    loader = scene() # type: ignore
    loader = type(loader.gsplat)(is_3D=dim)
    exp_name = (
        "final_active/"
        + scene.exp_name
        + ("/3D/" if dim else "/2D/")
        + (args.sensor_name + ("active" if act else ""))
    )
    config = scene(
        gsplat=loader, exp_name=exp_name, robot_name="curobo", sensor=sensor
    )
    config.gui = False
    if act: 
        config.controller.collision_cost = "w_avg"
    else:
        config.controller.collision_cost = ""
    print(config)

    exp = ReachingExperiment(
        config,
        env,
        robot_fn,
    )
    for j in range(0,500):
        print("Loading pose", j)
        exp.load_pose(j)
        exp.run_pose()
