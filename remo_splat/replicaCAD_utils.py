# TODO list:

# [x] Open gt sdf from iSDF dataset
# [x] Create point cloud of collided for sanity check
# [x] Visualize point cloud and mesh
# [ ] Implement get_distance method
# [x] Implement is_collided method
# [ ] Test with robot


import os

import numpy as np
import torch
import trimesh
from neural_robot.utils.math import transform_points
from skimage import measure
from spatialmath import SE3


class CollisionChecker:
    def __init__(self, folder_path):
        self.voxel_size = 0.01  # 1cm
        sdf_path = os.path.join(folder_path, "1cm", "sdf.npy")
        transform = SE3(
            np.loadtxt(os.path.join(folder_path, "1cm", "transform.txt")), check=False
        ).norm()

        self.transform = torch.from_numpy(transform.A).float().cuda()

        voxel = np.load(sdf_path)

        self.aabb = (
            np.array([[0, 0, 0], [voxel.shape[0], voxel.shape[1], voxel.shape[2]]])
            * self.voxel_size
        )  # on local frame

        self.voxel = torch.from_numpy(voxel).float().cuda()

    def get_distance(self, P_W, radius):
        """
        Get the distance of the robot to the environment,
        Goin to assume that all the robot are within the environment

        Args:
            P_W (torch.tensor): [N,3] Location of the sensors
            radius (torch.tensor): [N] Radius of the sensors

        Returns:
            torch.tensor: [N] Distance to the environment
        """
        P_V = transform_points(
            P_W, self.transform.inverse()
        )  # Transform point to the voxel frame

        P_ind = (P_V / self.voxel_size).round().long()

        # Extract integer voxel indices and fractional offsets
        P_floor = P_ind.floor().long()
        P_ceil = P_floor + 1
        frac = P_ind - P_floor.float()

        # Ensure points are inside valid voxel range
        size = torch.tensor(self.voxel.shape).cuda()
        valid_mask = (P_floor >= 0).all(dim=1) & (P_ceil < size).all(dim=1)
        distances = torch.zeros(P_W.shape[0]).cuda()

        # Compute trilinear interpolation for valid points
        if valid_mask.any():
            P_floor_valid = P_floor[valid_mask]
            P_ceil_valid = P_ceil[valid_mask]
            frac_valid = frac[valid_mask]

            c000 = self.voxel[
                P_floor_valid[:, 0], P_floor_valid[:, 1], P_floor_valid[:, 2]
            ]
            c001 = self.voxel[
                P_floor_valid[:, 0], P_floor_valid[:, 1], P_ceil_valid[:, 2]
            ]
            c010 = self.voxel[
                P_floor_valid[:, 0], P_ceil_valid[:, 1], P_floor_valid[:, 2]
            ]
            c011 = self.voxel[
                P_floor_valid[:, 0], P_ceil_valid[:, 1], P_ceil_valid[:, 2]
            ]
            c100 = self.voxel[
                P_ceil_valid[:, 0], P_floor_valid[:, 1], P_floor_valid[:, 2]
            ]
            c101 = self.voxel[
                P_ceil_valid[:, 0], P_floor_valid[:, 1], P_ceil_valid[:, 2]
            ]
            c110 = self.voxel[
                P_ceil_valid[:, 0], P_ceil_valid[:, 1], P_floor_valid[:, 2]
            ]
            c111 = self.voxel[
                P_ceil_valid[:, 0], P_ceil_valid[:, 1], P_ceil_valid[:, 2]
            ]

            # Trilinear interpolation
            c00 = c000 * (1 - frac_valid[:, 0]) + c100 * frac_valid[:, 0]
            c01 = c001 * (1 - frac_valid[:, 0]) + c101 * frac_valid[:, 0]
            c10 = c010 * (1 - frac_valid[:, 0]) + c110 * frac_valid[:, 0]
            c11 = c011 * (1 - frac_valid[:, 0]) + c111 * frac_valid[:, 0]

            c0 = c00 * (1 - frac_valid[:, 1]) + c10 * frac_valid[:, 1]
            c1 = c01 * (1 - frac_valid[:, 1]) + c11 * frac_valid[:, 1]

            interp_distance = c0 * (1 - frac_valid[:, 2]) + c1 * frac_valid[:, 2]
            distances[valid_mask] = interp_distance

        return distances

    def is_collided(self, P_W, radius):
        """
        Check if the robot is collided with the environment

        Args:
            P_W (torch.tensor): [N,3] Location of the sensors
            radius (torch.tensor): [N] Radius of the sensors

        Returns:
            torch.tensor: [N] 1 if collided 0 if not
        """
        P_V = transform_points(
            P_W, self.transform.inverse()
        )  # Transform point to the voxel frame

        # Convert to voxel coordinates
        P_ind = (P_V / self.voxel_size).round().long()
        size = torch.tensor(self.voxel.shape).cuda()
        # Check if the point is inside the voxel
        mask = (P_ind >= 0).all(dim=1) & (P_ind < size).all(dim=1)
        collided = torch.zeros(P_W.shape[0]).bool().cuda()

        collided[~mask] = False

        collided[mask] = (
            self.voxel[P_ind[mask][:, 0], P_ind[mask][:, 1], P_ind[mask][:, 2]]
            < radius[mask]
        )
        return collided

    def meshify(self):
        """
        Create a mesh from the voxel,
        This function is a sanyty check to see how far away of the distance we can obtain values
        """
        voxel = self.voxel.cpu().numpy()
        verts, faces, normals, values = measure.marching_cubes(
            voxel, spacing=[self.voxel_size] * 3, level=0.05
        )
        mesh = trimesh.Trimesh(verts, faces)
        return mesh


if __name__ == "__main__":
    folder_path = "/media/nmarticorena/DATA/iSDF_data/data/gt_sdfs/apt_2/"
    collision_checker = CollisionChecker(folder_path)

    import open3d as o3d

    import remo_splat.o3d_visualizer as gui
    from remo_splat.utils import uniform

    # Load mesh

    w_aabb = (
        torch.from_numpy(collision_checker.aabb).cuda().float()
        @ collision_checker.transform[:3, :3].T
        + collision_checker.transform[:3, 3]
    )

    sample_points = uniform((1_100_000, 3), w_aabb, device="cuda")
    collided = collision_checker.is_collided(
        sample_points, torch.ones(sample_points.shape[0], device="cuda") * 0.01
    )
    sample_points = sample_points[collided]

    o3d_points = o3d.geometry.PointCloud()
    o3d_points.points = o3d.utility.Vector3dVector(sample_points.cpu().numpy())

    g = gui.Visualizer()
    other_mesh = o3d.io.read_triangle_mesh("results/meshes/apt_2_nav.ply")
    g.add_geometry("pcd", o3d_points)
    g.add_geometry("mesh", other_mesh)
    while True:
        g.app.run_one_tick()
