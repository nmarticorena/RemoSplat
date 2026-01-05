from dataclasses import dataclass

import swift
import tyro
from neural_robot.unity_frankie import NeuralFrankie as Robot

from remo_splat import configs
from remo_splat.teleop import TeleopRecorder


@dataclass
class Config:
    name: str = "bookshelf"


arg = tyro.cli(Config)

env = swift.Swift()
env.launch(realtime=True, browser="chromium")

# Load the scene
config = configs.experiments.Bookshelf()
env.add(config.mesh)

robot = Robot(config.robot_name, spheres=True)
robot.q = robot.qr
robot.base = config.T_WB

env.add(robot)
teleop = TeleopRecorder(env, robot.fkine(robot.q), config.gsplat.scene, arg.name)

while True:
    teleop.step()
    env.step(0.03)  # 30ms
