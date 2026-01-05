import swift
import tyro

from neural_robot.unity_frankie import NeuralFrankie

from remo_splat.configs.experiments import ReachingRealWorldConfig
from remo_splat.experiments import ReachingExperiment

args = tyro.cli(ReachingRealWorldConfig)
env = swift.Swift()

robot_fn = NeuralFrankie
args.sensor = None
args.gui = True
args.log = False
args.controller.collisions = False

exp = ReachingExperiment(args, env, robot_fn)


for i in range(4):
    exp.load_pose(i)
    exp.run_pose()
