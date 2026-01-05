"""
@author: Nicolas M

This script is a interactive script that:
    - Load a sdf of a captured scene
    - Load a pose starting by the robot end effector
    - Allows to move the target pose and save it
    - save all the poses in a json file
"""

import json

import mm_neo.configs.sdf as configs
import mm_neo.utils.swift_utils as swift_utils
import numpy as np
import spatialgeometry as sg
import spatialmath as sm
import swift
import tyro
from mm_neo.utils.swift_utils import (interative_mesh,
                                      interative_panda_gripper, load_mesh)
from neural_robot.unity_frankie import NeuralFrankie

args = tyro.cli(configs.MeshArgs)


poses = {"poses": []}


def add_pose(a):
    global interactive, poses
    poses["poses"].append(interactive.sphere.T.tolist())
    print(interactive.sphere)


def save_poses(a):
    global poses
    poses["scene_pose"] = scene_mesh.sphere.T.tolist()
    # poses["robot_pose"] = frankie.base.T.tolist()

    with open(f"configs/{args.env_name}.json", "w") as f:
        json.dump(poses, f, indent=4)


env = swift.Swift()
env.launch(realtime=True, headless=False, comms="rtc", browser="chromium")

ax_goal = sg.Axes(0.1)
env.add(ax_goal)

transform = sm.SE3(0, 0, 0)

filename = "/home/nmarticorena/Documents/phd_tools/results/meshes/printer.stl"
mesh = load_mesh(filename)
env.add(mesh)
frankie = NeuralFrankie("curobo")
frankie.q = frankie.qr


env.add(frankie)

swift_utils.set_camera_robot(env, frankie.base, [-3, 3, 3])

arrived = False
dt = 0.025

interactive = interative_panda_gripper(env, frankie)

add_button = swift.Button(add_pose, "add pose")
env.add(add_button)


save_button = swift.Button(save_poses, "save pose")
env.add(save_button)
scene_mesh = interative_mesh(env, [mesh], sm.SE3(mesh.T), [sm.SE3()], range=5)


wTep = sm.SE3(0.85, 0, 1.10)
wTep.A[:3, :3] = frankie.fkine(frankie.q).A[:3, :3]
# wTep = frankie.fkine(frankie.q) * sm.SE3.Rz(np.pi)
# # wTep.A[:3, :3] = np.diag([-1, 1, -1])
# wTep.A[0, -1] += 1.5
# wTep.A[1, -1] += 1.0
# wTep.A[2, -1] += 0.1
ax_goal.T = wTep
env.step()

if args.ros:
    while not rospy.is_shutdown():
        env.step()
else:
    while True:
        env.step()
