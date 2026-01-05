import swift
from neural_robot.unity_frankie import NeuralFrankie as Robot

from remo_splat import configs
from remo_splat.teleop import ReplayTeleop

env = swift.Swift()
env.launch(realtime = True, browser = "chromium")

# Load the scene
config = configs.experiments.Bookshelf()
env.add(config.mesh)

robot = Robot(config.robot_name, spheres = True)
robot.q = robot.qr
robot.base = config.T_WB

env.add(robot)
teleop = ReplayTeleop(env, config.gsplat.scene, "rendering")

while True:
    teleop.step()
    env.step(0.001)

