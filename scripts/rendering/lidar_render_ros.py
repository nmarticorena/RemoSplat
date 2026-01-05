# First toy example of rendering laser information from a 3DFGS
import matplotlib.pyplot as plt
import numpy as np
import rospy
import torch

from remo_splat.configs import ExampleBookshelf, ExampleReal
from remo_splat.ellipsoids.meshes import to_o3d
from remo_splat.gaussians_3d import Gaussians3D
from remo_splat.kernels.distance import draw_intersected
from remo_splat.kernels.visuals import line_to_o3d, lines
from remo_splat.loader import (load_points, meshify_top_points,
                                      test_meshify)
from remo_splat.ros_node.pub_utils import GSLidar, RVIZMesh

rospy.init_node("gs_lidar")

sensor = GSLidar()
mesh = RVIZMesh()
mesh.publish("/home/nmarticorena/Documents/phd_tools/results/meshes/bookshelf.ply")

# a = ExampleReal()
a = ExampleBookshelf()
xyz, opacities, scales, rots = load_points(a.file, 0)
print(xyz.shape)
g = Gaussians3D()
g.load(a.file, 0)

centroid = xyz.mean(axis=0)
origin = torch.tensor([centroid[0], centroid[1], centroid[2]])

directions = []
for i in range(0, 180, 10):
    for j in range(0,180, 10):
        cos = np.cos(np.deg2rad(i))
        sin = np.sin(np.deg2rad(i))
        cos2 = np.cos(np.deg2rad(j))
        sin2 = np.sin(np.deg2rad(j))
        x = cos * sin2
        y = sin * sin2
        z = cos2
        directions.append(torch.tensor([x, y, z]))
        directions.append(torch.tensor([-x, -y, -z]))

directions = torch.stack(directions)

for i in range(1000):
    # Clean the previous lines
    origin += torch.tensor([0, 0, 0.001])

    intersected_mesh, distance, dir_index = draw_intersected(g, origin.numpy(), directions.numpy())
    points, __ = lines(origin, directions, distance)
    sensor.publish(points)

