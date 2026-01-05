import time

import swift
import tyro
from neural_robot.unity_frankie import NeuralFrankie

from remo_splat.swift_utils import draw_cameras, draw_target_poses
from remo_splat.configs.experiments import ReachingRealWorldConfig
from remo_splat.experiments import ReachingExperiment
from remo_splat.lidar import DepthSensor, EuclideanDistanceGaussian

args = tyro.cli(ReachingRealWorldConfig)
env = swift.Swift()

robot_fn = NeuralFrankie
# args.sensor = DepthSensor
args.log = True
# args.controller.collisions = False
exp_name = f"test_real_world/{args.env_type}/2D/{str(args.sensor)}"
args.exp_name = exp_name

exp = ReachingExperiment(args, env, robot_fn, )

# args.T_WEp = args.T_WC # Go to the cameras poses
draw_target_poses(env, args.T_WEp)
# draw_cameras(env, args.T_WC)

for i in range(len(args.T_WEp)):
    exp.load_pose(i)
    exp.run_pose()
