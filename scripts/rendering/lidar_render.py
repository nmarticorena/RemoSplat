# First toy example of rendering laser information from a 3DFGS
import json

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch

from remo_splat.configs import ExampleBookshelf, ExampleReal
from remo_splat.ellipsoids.meshes import to_o3d
from remo_splat.gaussians_3d import Gaussians3D
from remo_splat.kernels.distance import draw_intersected, get_distance
from remo_splat.kernels.visuals import line_to_o3d, lines
from remo_splat.loader import (load_points, meshify_top_points,
                                      test_meshify)
from remo_splat.utils import draw_line_o3d, uniform

camera_pinhole = o3d.io.read_pinhole_camera_parameters("o3dcamera.json")

a = ExampleBookshelf()
xyz, opacities, scales, rots = load_points(a.file, 0)
print(xyz.shape)
mesh = meshify_top_points(xyz, opacities, scales * 3, rots, .01)
mesh = to_o3d(mesh)
#
g = Gaussians3D()
g.load(a.file, 0)



centroid = xyz.mean(axis=0)
# origin = torch.tensor([[centroid[0], centroid[1], centroid[2]],
#                        [1, 1, 1],
#                        [2, 2, 2]]) # 3 Sensors
#

origin = uniform((200,3), [-1,1],device = "cuda")

directions = []
for i in range(0, 180, 60):
    for j in range(0,180, 60):
        cos = np.cos(np.deg2rad(i))
        sin = np.sin(np.deg2rad(i))
        cos2 = np.cos(np.deg2rad(j))
        sin2 = np.sin(np.deg2rad(j))
        x = cos * sin2
        y = sin * sin2
        z = cos2
        directions.append(torch.tensor([x, y, z]))
        directions.append(torch.tensor([-x, -y, -z]))

directions = torch.stack(directions).cuda().float()

# distance = get_distance(origin.numpy(), torch.tensor(
#     [0., 0., 1.]), torch.tensor(xyz))
o3d.visualization.draw_geometries([mesh])
# Initialize the visualizer and set the initial camera parameters
vis = o3d.visualization.Visualizer()
vis.create_window(width=1920, height=1080)
vis.add_geometry(mesh)
ctr = vis.get_view_control()

# Set initial camera parameters
pinhole_camera_params = ctr.convert_to_pinhole_camera_parameters()
pinhole_camera_params.extrinsic = camera_pinhole.extrinsic
ctr.convert_from_pinhole_camera_parameters(pinhole_camera_params)

for i in range(10000):
    # Clean the previous lines
    # vis.clear_geometries()
    # vis.add_geometry(mesh)
    origin += torch.tensor([0, 0, 0.001], device = "cuda")
    intersected_mesh, distance, dir_index = draw_intersected(g, origin, directions)
    # ray_lines = line_to_o3d(*lines(origin, directions, distance))
    # for l in ray_lines:
    # vis.add_geometry(ray_lines)

    # Update the camera parameters
    # ctr.convert_from_pinhole_camera_parameters(pinhole_camera_params, True)

    # vis.poll_events()
    # vis.update_renderer()   



# Now combined_line_set contains all line segments from the original LineSets
o3d.visualization.draw_geometries([ray_lines])


# intersected_mesh.export("intersected.ply")
# intersected_mesh = to_o3d(intersected_mesh)
# intersected_mesh.paint_uniform_color([0, 1, 0])

exit()

# import trimesh
# def clean_mesh(mesh):
#     # Identify faces that contain only zero vertices
#     valid_faces = []
#     for face in mesh.faces:
#         vertices = mesh.vertices[face]
#         # Check if the face vertices are all zeros
#         if not np.all(vertices == 0):
#             valid_faces.append(face)
    
#     # Create a new mesh with only the valid faces
#     cleaned_faces = np.array(valid_faces)
#     cleaned_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=cleaned_faces)
    
#     return cleaned_mesh


cleaned = clean_mesh(intersected_mesh)
cleaned.export("intersected_cleaned.ply")

d = distance.numpy()
plt.hist(d)
plt.show()

# get the points that are closer than 0.1
xyz = xyz[d < 0.1]
scales = scales[d < 0.1]
rots = rots[d < 0.1]

closer = to_o3d(test_meshify(scales, xyz, rots, color=[0, 0, 1]))
closer.paint_uniform_color([0, 0, 1])
mesh.paint_uniform_color([1, 0, 0])

o3d.visualization.draw_geometries([mesh, line, closer])

print(d)
