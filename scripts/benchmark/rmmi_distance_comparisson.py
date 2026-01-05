"""
Reaching benchmark using different frecuencies and sensors
"""


from itertools import product
from typing import List, Type

import swift
import tyro
import numpy as np
from neural_robot.unity_frankie import NeuralFrankie
from tqdm import tqdm

from remo_splat.configs import controllers, experiments, gs
from remo_splat.experiments import ReachingExperiment
from remo_splat.lidar import DepthSensor, EuclideanDistanceGaussian

robot_fn = NeuralFrankie
# robots = ["real_robot", "new_points_2", "curobo"]
robots = ["curobo","real_robot", "points_1"]
spheres = [True, False , False]
# spheres = [False, False, True]

np.random.seed(123)


# scenes: List[Type[experiments.ReachingExperimentConfig]] = [experiments.ReachingTable, experiments.ReachingBookshelf]
scenes: List[Type[experiments.ReachingExperimentConfig]] = [experiments.ReachingBookshelf]
sensors = [
    experiments.Sensor.depth,
    # experiments.Sensor.euclidean_all,
    experiments.Sensor.euclidean_less,
    experiments.Sensor.gt_all,
    experiments.Sensor.gt,
]
# dimensions = [False, True]  # 2D, 3D
dimensions = [False]
active = [True, False ] #, True]
# frecuencies = [5, 10,15,20,25,30,35,40,50] # hz
frecuencies = [2,3,4,5,6,7,8,9,10,11,12,14,15] # hz
step_times = [1/f for f in frecuencies] # seconds per step
#step_times = [0.5, 0.25, 0.1] # 2, 4, 10, 20 hz
# step_times = [0.1,0.33,0.01, 0.2,0.05] # 100 hz
# step_times = [1/30, 1/40, 1/50, 1/60] # 30, 40, 50, 60 hz
# step_times = [1/125, 1/150, 1/200] # 30, 40, 50, 60 hz
# step_times = [1/225, 1/250, 1/300] # 30, 40, 50, 60 hz
# active = [False]
for step_time, scene, dim, sensor, act in tqdm(product(step_times, scenes, dimensions, sensors, active)):
    if sensor in [experiments.Sensor.gt, experiments.Sensor.gt_all]:
        robots_ = robots
    else:
        robots_ = ["curobo"]
    for robot, s in zip(robots_,spheres):

        np.random.seed(100)
        env = swift.Swift()
        loader = scene() # type: ignore
        loader = type(loader.gsplat)(is_3D=dim)
        exp_name = (
            f"comparisson_frecuencies_v3/{int(1/step_time)}/"
            # f"check_euclidean_all/{int(1/step_time)}/"
            + scene.exp_name
            + ("/3D/" if dim else "/2D/")
            + (str(sensor) + ("active" if act else ""))
            + robot
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
            gsplat=loader, exp_name=exp_name, robot_name=robot, sensor=sensor, controller = controller_config, spheres=spheres[robots.index(robot)]
        )
        config.log = True
        config.gui = False
        print(config)

        exp = ReachingExperiment(
            config,
            env,
            robot_fn,
        )
        # exp_indexs = [i for i in range(500)]
        exp_indexs = [np.random.randint(0,500) for _ in range(20)]
        for j in exp_indexs: # to try
            print("Loading pose", j)
            exp.load_pose(j)
            exp.run_pose()
