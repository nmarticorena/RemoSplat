from itertools import product


import swift
import tyro
from neural_robot.unity_frankie import NeuralFrankie
from tqdm import tqdm

from remo_splat.configs import experiments, gs, controllers
from remo_splat.experiments import ReachingExperiment
from remo_splat.lidar import DepthSensor, EuclideanDistanceGaussian

robot_fn = NeuralFrankie

sensor_names = {
    experiments.Sensor.depth: "depth",
    experiments.Sensor.euclidean: "euclidean",
    experiments.Sensor.euclidean_less: "euclidean_less",
    experiments.Sensor.gt: "gt",
    experiments.Sensor.gt_active: "gt_active",
    experiments.Sensor.gt_active_faster: "gt_active_faster<Up>",
    experiments.Sensor.gt_all: "gt_all",
}

scenes = [experiments.ReachingTable, experiments.ReachingBookshelf]
# scenes = [experiments.ReachingBookshelf]
sensors = [
    # experiments.Sensor.gt_all,
    experiments.Sensor.gt
    # experiments.Sensor.gt_active_faster, experiments.Sensor.gt_active, experiments.Sensor.gt
    # experiments.Sensor.gt_active
]
dimensions = [False]  # 2D, 3D

active = ["", "w_avg"] # False, True
frecuencies = [20] # Hz
step_times = [1/f for f in frecuencies] # seconds per step


for step_time, scene, dim, sensor, act in tqdm(product(step_times, scenes, dimensions, sensors, active)):
    env = swift.Swift()
    loader = scene()
    # loader = type(loader.gsplat)(is_3D=dim)
    exp_name = (
        "gt_rmmi_test/"
        + scene.exp_name
        + "/"
        + (sensor_names[sensor])
    )
    controller_config = controllers.NeoConfig()
    controller_config.collision_cost = act
    controller_config.step_time = step_time
    config = scene(
        gsplat=loader.gsplat, exp_name=exp_name, robot_name="curobo", sensor=sensor, controller=controller_config
    )
    config.log = True
    config.gui = False
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
