import spatialgeometry as sg
from swift import Swift
from spatialmath import SE3

def draw_poses(env: Swift, poses:SE3, length:float):
    for pose in poses:
        env.add(sg.Axes(length, pose = pose))

def draw_cameras(env: Swift, poses: SE3):
    draw_poses(env, poses, 0.1)

def draw_target_poses(env:Swift, poses:SE3):
    draw_poses(env, poses, 0.3)

