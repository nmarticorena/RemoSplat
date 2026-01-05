from itertools import product
from typing import List, Type

import swift
from neural_robot.unity_frankie import NeuralFrankie
from tqdm import tqdm

from remo_splat.configs import controllers, experiments
from remo_splat.experiments import ReachingExperiment

robot_fn = NeuralFrankie

sensor_names = {
    experiments.Sensor.depth: "depth",
    experiments.Sensor.euclidean: "euclidean",
    experiments.Sensor.euclidean_less: "euclidean_less",
    experiments.Sensor.euclidean_all: "euclidean_all",
}

scenes: List[Type[experiments.ReachingExperimentConfig]] = [experiments.ReachingTable, experiments.ReachingBookshelf]
sensors = [
    experiments.Sensor.depth,
    # experiments.Sensor.euclidean_all,
    # experiments.Sensor.euclidean,
    # experiments.Sensor.euclidean_less,
]
dimensions = [False]#, True]  # 2D, 3D
active = [False, True]
step_times = [0.05] # 20 hz
for step_time, scene, dim, sensor, act in tqdm(product(step_times, scenes, dimensions, sensors, active)):
    env = swift.Swift()
    loader = scene() # type: ignore
    loader = type(loader.gsplat)(is_3D=dim)
    exp_name = (
        f"time_profiling_solvers/"
        + scene.exp_name
        + ("/3D/" if dim else "/2D/")
        + (sensor_names[sensor] + ("active" if act else ""))
    )
    controller_config = controllers.NeoConfig()
    if act:
        controller_config.collision_cost = "w_avg"
    else:
        controller_config.collision_cost = ""
    controller_config.step_time = step_time

    config = scene(
        gsplat=loader, exp_name=exp_name, robot_name="gsplat_rbf", sensor=sensor, controller = controller_config
    )
    config.log = True
    config.steps = 100
    config.gui = False
    print(config)

    exp = ReachingExperiment(
        config,
        env,
        robot_fn,
    )
    for j in range(0,10): # to try
        print("Loading pose", j)
        exp.load_pose(j)
        exp.run_pose()
