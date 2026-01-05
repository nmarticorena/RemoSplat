"""
Benchmark script to run the results of rmmi scenes.

scenes define the enviroments,
sensors define the sensors to use,
dimensions define if the gsplat is 2D or 3D,
active defines if the controller is active or not,
frecuencies define the control frecuencies to test.

results are saved in the folder logs/experiments/{exp_name}
"""


from itertools import product
from typing import List, Type
from dataclasses import dataclass, field
import os
import psutil

import swift
from neural_robot.unity_frankie import NeuralFrankie
import spatialmath as sm
from tqdm import tqdm

from remo_splat.configs import controllers, experiments, gs
from remo_splat.experiments import ReachingExperiment

p = psutil.Process(os.getpid())
p.cpu_affinity([7,8,9,10])
robot_fn = NeuralFrankie

@dataclass
class ReachingBookshelf(experiments.ReachingExperimentConfig):
    gsplat: gs.GSplatLoader = field(default_factory=lambda: gs.ReachingBookshelf(scene = "bookshelf_cage/bookshelf_cage_0000"))
    T_WB: sm.SE3 = field(default_factory=lambda: sm.SE3(-0.5, 0, 0))
    robot_name: str = "curobo"
    exp_name: str = "bookshelf_cage"
    env_type: str = "bookshelf_cage"


@dataclass
class ReachingTable(experiments.ReachingExperimentConfig):
    gsplat: gs.GSplatLoader = field(default_factory=lambda: gs.ReachingTable(scene = "table_new/table_new_0000"))
    T_WB: sm.SE3 = field(default_factory=lambda: sm.SE3(-0.5, 0, 0))
    robot_name: str = "curobo"
    exp_name: str = "table_new"
    env_type: str = "table_new"


@dataclass
class ReachingBookshelfNoisy(experiments.ReachingExperimentConfig):
    gsplat: gs.GSplatLoader = field(default_factory=lambda: gs.ReachingBookshelf(scene = "bookshelf_cage_noisy/bookshelf_cage_noisy_0000"))
    T_WB: sm.SE3 = field(default_factory=lambda: sm.SE3(-0.5, 0, 0))
    robot_name: str = "curobo"
    exp_name: str = "bookshelf_cage_noisy"
    env_type: str = "bookshelf_cage_noisy"


@dataclass
class ReachingTableNoisy(experiments.ReachingExperimentConfig):
    gsplat: gs.GSplatLoader = field(default_factory=lambda: gs.ReachingTable(scene = "table_new_noisy/table_new_noisy_0000"))
    T_WB: sm.SE3 = field(default_factory=lambda: sm.SE3(-0.5, 0, 0))
    robot_name: str = "curobo"
    exp_name: str = "table_new_noisy"
    env_type: str = "table_new_noisy"

scenes: List[Type[experiments.ReachingExperimentConfig]] = [ReachingTable, ReachingBookshelf]
sensors = [
    experiments.Sensor.depth,
    experiments.Sensor.euclidean_less,
    experiments.Sensor.gt
]
dimensions = [False] # 2D
active = [False, True ] # active collision cost
frecuencies = [20]
step_times = [1/f for f in frecuencies] # seconds per step

def main(
    n_scenes: int = 500,
):
    """Run RMMI Benchmark over multiple variations"""
    for step_time, scene, dim, sensor, act in tqdm(product(step_times, scenes, dimensions, sensors, active)):
        env = swift.Swift()
        loader = scene() # type: ignore
        # loader = type(loader.gsplat)(is_3D=dim)
        dim_name = "/3D/" if dim else "/2D/"
        dim_name = dim_name if sensor != experiments.Sensor.gt else "/"
        exp_name = (
            f"test/{int(1/step_time)}/"
            + scene.exp_name
            + dim_name
            + (str(sensor) + ("active" if act else ""))
        )
        controller_config = controllers.NeoConfig()
        if act:
            controller_config.collision_cost = "w_avg"
        else:
            controller_config.collision_cost = ""
        if sensor == experiments.Sensor.depth_min:
            controller_config.only_min = True
        controller_config.step_time = step_time

        config = scene(
            gsplat=loader.gsplat,
            exp_name=exp_name,
            robot_name="curobo",
            sensor=sensor,
            controller = controller_config
        )
        config.log = True
        config.gui = False
        print(config)

        exp = ReachingExperiment(
            config,
            env,
            robot_fn,
        )
        print(f" Sensor {sensor}, dim {dim}, active {act}, step_time {step_time}, exp_name {exp_name}")
        for j in range(n_scenes): # to try
            print("Loading pose", j)
            exp.load_pose(j)
            exp.run_pose()

if __name__ == "__main__":
    import tyro
    tyro.cli(main)