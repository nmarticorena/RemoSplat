import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
import trimesh
import warp as wp

from neural_robot.robot import NeuralRobot
from neural_robot.utils import math as helper_math
from neural_robot.utils.math import vectorized_rot_vec
from open3d.geometry import PointCloud as o3d_pcd
from safer_splat.splat import gsplat_utils
from spatialmath import SE3

from remo_splat.sdf import IdealSDF
from remo_splat.configs.gs import GSplatLoader
from remo_splat.ellipsoids.meshes import to_o3d
from remo_splat.gaussians_3d import Gaussians3D
from remo_splat.kernels.distance import draw_intersected
from remo_splat.kernels.visuals import (line_from_dist_grad,
                                               line_from_origin_destination,
                                               line_to_o3d, lines)
from remo_splat.utils import CameraParams, Render

torch.set_printoptions(precision=2, sci_mode=False)

"""
Todo:

- [x] Implement the panoramic sensor
- [ ] Implementh the depth sensor (interface using all the depth images)
- [x] Test the depth back preojection
- [ ] Impprove memory consumtion, do not repeat any tensor that is not necessary
"""


class DistanceSensor(ABC):
    @abstractmethod
    def get_distance(self, P_W, radius):
        """
        Interface for getting the distance depending on the sensor used

        Args:
            P_W (torch.tensor): [N,3] Location of the sensors

        Returns:
            tuple:
                Δd: (torch.tensor) [M,3] Spatial derivative of the distance
                d: (torch.tensor) [M] Distance from the sensors and the ellipsoids
        """
        raise (NotImplementedError)

    @abstractmethod
    def get_points(self, P_W, radius):
        """
        Interface for getting the closest point on the ellipsoids

        Args:
            P_W (torch.tensor): [N,3] Location of the sensors

        Returns:
            p_e (torch.tensor): [N,3] Closest point on the ellipsoids
        """
        raise (NotImplementedError)

    @abstractmethod
    def gui_debug(self, gradient, distance, gui, gt_sdf, error):
        """
        Interface for debugging the sensor

        Args:
            gradient (torch.tensor): [N,3] Gradient of the distance
            distance (torch.tensor): [N] Distance of the sensors
            gui (open3d.visualization.gui.Application): GUI to update the geometry
            gt_sdf (SDF): Ground truth SDF
            error (torch.tensor): [N] Error of the distance
        """
        raise (NotImplementedError)


class EuclideanDistance(DistanceSensor):
    def __init__(self, gaussians: Gaussians3D):
        self.max_steps = 50
        self.min = True
        self.gaussians = gaussians
        means = wp.to_torch(gaussians.gaussians.means)
        scales = wp.to_torch(gaussians.gaussians.scales)
        quats = wp.to_torch(gaussians.gaussians.angles)
        quats = torch.cat(
            (quats[:, 3, None], quats[:, :3]), dim=1
        )  # change the queaternions order, warp have it on (x,y,z,w) and we need it on (w,x,y,z)
        scales = torch.einsum("bii->bi", scales)  # I saved the diagonal matrix
        scales = torch.sqrt(scales)
        self.splats = gsplat_utils.GSplatLoader(
            None, "cuda", means=means, scales=scales, rots=quats
        )

    def test_new_kernel(self, P_W, radius):
        from kernels.distance import distance_disk
        distance, _ ,_ ,info = self.splats.query_distance_batched(
            P_W, n_steps=self.max_steps, distance_type = self.distance_type
        )
        d_2 = wp.zeros(distance.shape, device = "cuda")
        g_2 = wp.zeros((P_W.shape[0],self.splats.means.shape[0]), dtype=wp.vec3f)
        p_w = wp.from_torch(P_W, dtype=wp.vec3f)
        means = wp.from_torch(self.splats.means, dtype=wp.vec3f)
        rots = self.splats.rots
        rots = torch.cat((rots[:, 1:], rots[:, 0, None]), dim=1)
        # change the queaternions order, warp have it on (x,y,z,w) and we need it on (w,x,y,z)
        rots = wp.from_torch(rots, dtype=wp.quat)

        scales = wp.from_torch(self.splats.scales, dtype=wp.vec3f)
        wp.launch(distance_disk, dim =(P_W.shape[0], self.splats.means.shape[0]),
                  inputs=[p_w, means, rots, scales, d_2, g_2], device = "cuda")

        d = wp.to_torch(d_2)




    def get_points(self, P_W: torch.Tensor, radius: torch.Tensor):
        """
        Get the closest point on the ellipsoids

        Args:
            P_W (torch.tensor): [N,3] Location of the sensors
            radius (torch.tensor): [N] Radius of the sensors
        Returns:
            p_e (torch.tensor): [N,3] Closest point on the ellipsoids
        """
        distance, _, _, info = self.splats.query_distance_batched(
            P_W, n_steps=self.max_steps
        )  # dis[NxGx1] grad_h[NxGx3]
        closest = info["y"]
        return closest

    def get_distance(self, P_W, radius):
        self.radius = radius
        dis, grad_h, _, info = self.splats.query_distance_batched(
            P_W, distance_type = self.distance_type, n_steps=self.max_steps
        )  # dis[NxGX1] grad_h[NxGx3]
        # Trick to get it working on the meantime
        dis = torch.sqrt(dis)

        dis -= radius.reshape(-1, 1)
        if self.min:

            distance, indices = torch.min(dis, dim=1)
            grad_h = torch.gather(
                grad_h, 1, indices.reshape(-1, 1, 1).expand(-1, -1, 3)
            ).squeeze(1)

            # print(distance.min())
            self.origin = P_W
            self.closest = info["y"]

            self.closet_min = torch.gather(
                self.closest, 1, indices.reshape(-1, 1, 1).expand(-1, 1, 3)
            ).squeeze(1)

            # TODO need to implement them this way!!!
            # distance = dis.reshape(-1,1) # Flatten #
            # grad_h = grad_h.reshape(-1,3)
        else:
            # self.radius_filter(dis, grad_h, info["y"])
            # distance = dis.reshape(-1)
            # grad_h = grad_h.reshape(-1,3)
            distance = dis
            grad_h = grad_h
            pass
        grad_h = grad_h / torch.linalg.norm(grad_h, dim=-1, keepdim=True)

        return -1 * grad_h, distance


    def radius_filter(self, distance:torch.Tensor, grad:torch.Tensor, position:torch.Tensor ,n_sample:int = 3 ,radius_th: float = 0.5):

        breakpoint()
        closer_index = distance.argmin(dim = -1) # N
        skip = torch.zeros(distance.shape, dtype = torch.bool).cuda()
        closer = position[torch.arange(distance.shape[0]),closer_index]
        for _ in range(n_sample):
            radius = torch.linalg.norm(position - closer.unsqueeze(1), dim = -1)

            skip[radius<radius_th] = True

    def log_distances_grad(
        self, T_GW: torch.Tensor, T_WG: torch.Tensor, robot: NeuralRobot
    ):
        """
        Obtain the gradient and distance of the depth sensor approach direclty using the robot


        """
        X_WSp = robot.transform_points().detach()
        X_GSp = helper_math.transform_points(X_WSp, T_GW)
        self.radius = robot.SpheresRadius
        g_G, distance = self.get_distance(X_GSp, self.radius)
        g_W = helper_math.rotate_vec(g_G, T_WG[:3, :3])

        return g_W, distance

    def draw_lines(self, positions, gradient, distance):
        if positions is None:
            positions = self.origin
        distance = (distance + self.radius).unsqueeze(-1)
        return line_from_dist_grad(positions, gradient, distance)

    def gt_lines(self):  # TODO check how to get this from the minimum
        return line_from_origin_destination(self.origin, self.closet_min)

    def debug_pcd(self, error: Optional[torch.Tensor] = None):
        print("gui debug")
        pcd = o3d.geometry.PointCloud()
        closest = self.closest.reshape(-1, 3)
        pcd.points = o3d.utility.Vector3dVector(closest.cpu().numpy())
        if error is not None:
            error_normalized = (error - error.min()) / (error.max() - error.min())
            colors = np.zeros_like(self.closest.cpu().numpy())
            colormap = plt.cm.get_cmap("jet")
            colors = colormap(error_normalized.cpu().numpy())[:, :3]
            print(colors)
            pcd.colors = o3d.utility.Vector3dVector(colors)
        else:
            pcd.paint_uniform_color([0.0, 1.0, 0])

        return pcd

    def gui_debug(self, gradient, distance, gui, gt_sdf, error):
        print("gui debug")
        lines = self.draw_lines(None, gradient, distance)
        # gt_lines = self.gt_lines()
        gui.update_geometry("lines", lines)
        pcd = self.debug_pcd(error)

        gui.update_geometry("pcd", pcd)

    def get_pcd_error(self, sdf: IdealSDF):
        d_P_w = sdf.get_distance(self.closest[0, :, :])

        return d_P_w


class EuclideanDistanceGaussian(EuclideanDistance):
    def __init__(self, gaussian_config: GSplatLoader, mesh="", max_steps=50, min = True, distance_type = None):
        self.distance_type = distance_type
        self.render = Render(gaussian_config)
        # It migth be that the quats are not in the right order
        #
        # self.render.quats = torch.cat(
        #     (self.render.quats[:, 3, None], self.render.quats[:, :3]), dim=1
        # )
        self.max_steps = max_steps
        self.min = min

        self.splats = gsplat_utils.GSplatLoader(
            None,
            "cuda",
            means=self.render.means,
            scales=self.render.scales,
            rots=self.render.quats,
        )
        if mesh != "":
            print(f"writing mesh on the path {mesh}")
            folder = os.path.dirname(mesh)
            if not os.path.exists(folder):
                os.makedirs(folder)
                print("Creating the folder")
            self.splats.save_mesh(mesh)


class LidarSensor(DistanceSensor):
    def __init__(self, gaussians: Gaussians3D, n_sensors, step_angle: int):
        self.gaussians = gaussians
        self.get_directions(step_angle)
        self.origins = torch.zeros((n_sensors, 3), device="cuda")
        self.n_sensors = n_sensors

        self.directions = self.directions.contiguous()
        self.origins = self.origins.contiguous()

    def get_distance(self, P_W, radius):
        P_W = P_W.contiguous()
        _, distance, index, sorted = draw_intersected(
            self.gaussians, P_W, self.directions
        )
        sorted -= radius.reshape(-1, 1, 1)
        # I have the distance of all the intersections [S,G,D]
        final_distance = torch.empty((1), device="cuda")
        final_grad = torch.empty((1, 3), device="cuda")
        # the direction is the gradient in this case. I just need to put negatives
        # the direction is the gradient in this case. I just need to put negatives
        # Find all valid collisions
        # valid_mask = (sorted[:, 0, :] < 1000)  # Mask for valid collisions [I, L]
        # indices = valid_mask.nonzero(as_tuple=True)  # Get indices of valid collisions
        #
        # # Gather distances and gradients
        # final_distance = sorted[indices[0], 0, indices[1]]  # Collect distances
        # final_grad = self.directions[indices[1]]  # Collect corresponding gradients

        # Mask for valid collisions (distances < 1000)
        valid_mask = sorted < 1000  # Boolean mask of the same shape as `sorted`

        # Replace invalid distances with a large value to ignore them during `min`
        sorted_valid = torch.where(
            valid_mask, sorted, torch.full_like(sorted, float("inf"))
        )

        # Find the minimum distance and its indices for each sensor
        # We reduce across the Objects and directions dimensions
        min_distance, min_indices = sorted_valid.view(self.n_sensors, -1).min(
            dim=1
        )  # [N_sensors]

        # Convert the 1D indices back to original tensor dimensions
        min_object_indices = min_indices // sorted.size(-1)  # Object indices
        min_dir_indices = min_indices % sorted.size(-1)  # Direction indices

        # Gather the corresponding gradients
        final_grad = -self.directions[min_dir_indices]  # Shape: [N_sensors, 3]

        return final_grad, min_distance

    def get_points(self, P_W: torch.Tensor, radius: torch.Tensor):
        """
        Get the closest point on the ellipsoids

        Args:
            P_W (torch.tensor): [N,3] Location of the sensors
            radius (torch.tensor): [N] Radius of the sensors
        Returns:
            p_e (torch.tensor): [N,3] Closest point on the ellipsoids
        """
        P_W = P_W.contiguous()
        opacities = wp.to_torch(self.gaussians.gaussians.opacities)
        _, distance, index, sorted = draw_intersected(
            self.gaussians, P_W, self.directions
        )

        ours_opacity = opacities[index[0, :, :]]
        transmitance = 1 - ours_opacity
        transmitance = torch.cumprod(transmitance, 0)

        # Make all the high distance to zero
        sorted[sorted > 10] = 0
        # import matplotlib.pyplot as plt

        distance = torch.sum(ours_opacity * transmitance * sorted[0, :, :], dim=0)
        distance = distance / torch.sum(ours_opacity * transmitance, dim=0)
        # sorted -= radius.reshape(-1,1,1)

        # x = sorted[0,:100,0].cpu().numpy()
        # opacity = ours_opacity[:100,0].cpu().numpy()
        # plt.plot(x, opacity, label = "Opacity")
        # transmitance = transmitance[:100,0].cpu().numpy()
        # plt.plot(x, transmitance, label = "Transmitance")
        # plt.vlines(distance[0].cpu(), 0,1, label = f"Distance: {distance[0]:.2f}" , color = "red")
        # plt.legend()
        # plt.show()
        # # I have the distance of all the intersections [S,G,D]
        # Mask for valid collisions (distances < 1000)
        final_grad = -self.directions  # Shape: [N_dir, 3]

        # distance = sorted[:,0,:].squeeze()

        closest = P_W + final_grad[distance < 10] * distance[distance < 10].unsqueeze(
            -1
        )
        return closest

    def draw_lines(self, positions, distance):
        ray_lines = lines(positions, self.directions, distance)
        l = line_to_o3d(*ray_lines)
        return l

    def get_directions(self, step_angle: int):
        directions = []
        for i in range(0, 180, step_angle):
            for j in range(0, 180, step_angle):
                cos = np.cos(np.deg2rad(i))
                sin = np.sin(np.deg2rad(i))
                cos2 = np.cos(np.deg2rad(j))
                sin2 = np.sin(np.deg2rad(j))
                x = cos * sin2
                y = sin * sin2
                z = cos2
                directions.append(torch.tensor([x, y, z]))
                directions.append(torch.tensor([-x, -y, -z]))
        self.directions = torch.stack(directions).cuda().float()
        self.directions = self.directions / torch.linalg.norm(
            self.directions, dim=1, keepdim=True
        )


class PanoramaSensor:
    """
    Helper class to define the cameras

    Attributes:

    """

    def __init__(self, n_cameras, camera_params: CameraParams):
        self.n_cameras = n_cameras
        self.camera_params = camera_params

        # self.n_fov = int(360 / self.camera_params.fov) + 2
        self.n_fov = 6

        self.K = torch.zeros((n_cameras * self.n_fov, 3, 3), device="cuda")
        self.K[:, 0, 0] = self.camera_params.focal_length
        self.K[:, 1, 1] = self.camera_params.focal_length
        self.K[:, 0, 2] = self.camera_params.width / 2
        self.K[:, 1, 2] = self.camera_params.height / 2
        self.K[:, 2, 2] = 1
        self.w = self.camera_params.width
        self.h = self.camera_params.height

        self.K_inv = torch.inverse(self.K)

        self.R = self.get_rots()  # R_sc From sensor to camera
        self.pixel_grid = self.get_pixel_grid()

    def get_rots(self):
        """
        Compute the rotations to create the 360 degree cameras

        Returns:
            rots (torch.tensor): [N * n_fov, 3, 3] Rotation matrices
        """
        angles = np.linspace(0, 360, self.n_fov - 2, endpoint=False)
        rotations = []
        for angle in angles:
            Ry = SE3.Ry(angle, unit="deg").A
            rotations.append(Ry)
        rotations.append(SE3.Rx(90, unit="deg").A)
        rotations.append(SE3.Rx(-90, unit="deg").A)

        rots = torch.from_numpy(np.stack(rotations, axis=0)).to("cuda").float()
        return rots

    def get_poses(self, T_WP):
        """
        Get the poses of each of the cameras of the array

        Args:
            T_WP (torch.tensor): [N, 4, 4] World to 360 camera poses
        Returns:
            T_CWs (torch.tensor): [N * n_fov, 4, 4] Camera to world poses
            T_WCs (torch.tensor): [N * n_fov, 4, 4] World to camera poses
        """
        T_WPs = T_WP.unsqueeze(1).expand(-1, self.n_fov, -1, -1)  # [N, N_fov, 4,4]
        T_WCs = torch.einsum("nfij,fjk->nfik", T_WPs, self.R)
        T_CWs = torch.linalg.inv(T_WCs)
        T_WCs = T_WCs.reshape(-1, 4, 4)
        # Flatten them
        T_CWs = torch.flatten(T_CWs, start_dim=0, end_dim=1)
        return T_CWs, T_WCs

    def get_pixel_grid(self):
        """
        Get the pixel grid for the cameras

        Returns:
            homogeneous_coords (torch.tensor): [N, H, W, 3] Homogeneous pixel coordinates
        """
        # Create pixel coordinate grid
        u, v = torch.meshgrid(
            torch.arange(self.w, device="cuda"),
            torch.arange(self.h, device="cuda"),
            indexing="xy",  # Matches pixel indexing convention
        )

        # Convert to homogeneous image coordinates (u, v, 1)
        homogeneous_coords = torch.stack(
            (u, v, torch.ones_like(u, device="cuda")), dim=-1
        )
        # Stack them for each camera
        homogeneous_coords = (
            homogeneous_coords.unsqueeze(0)
            .expand(self.n_cameras * self.n_fov, -1, -1, -1)
            .float()
        )
        return homogeneous_coords

    def back_projection(self, depth_images: torch.Tensor, max_depth=4):
        """
        Project the depth images to the 3D space

        Args:
            depth_images (torch.tensor): [N, H, W] Depth images
        Returns:
            points (torch.tensor): [N, H, W, 3] 3D points in the camera frame
        """
        coords = torch.einsum(
            "nij,n...j->n...i", self.K_inv, self.pixel_grid
        )  # [N, H, W, 3]

        p_c = coords * depth_images.unsqueeze(-1)
        distance = torch.norm(p_c, dim=-1)
        distance[distance.isnan()] = torch.inf
        return p_c, distance

    def get_points_world(self, T_WCs, depth_images):
        """
        Get the points in the world frame

        Args:
            T_WCs (torch.tensor): [N*n_fov,4,4] Transformation from the world to each camera
            depth_images (torch.tensor): [N, H, W] Depth images
        Returns:
            points (torch.tensor): [P, 3] 3D points in the world frame
        """
        p_c, _ = self.back_projection(depth_images)
        # homogenize p_c
        p_c = torch.cat((p_c, torch.ones_like(p_c[:, :, :, 0:1])), dim=-1)
        p_w_h = torch.einsum("nij,nhwj->nhwi", T_WCs, p_c)

        # Convert back to Euclidean coordinates
        p_w = p_w_h[..., :3] / (p_w_h[..., 3:4] + 1e-8)

        return p_w.reshape(-1, 3)

    def _draw_lines(self, T_CWs):
        return self.draw_camera_lines(T_CWs)  # Just a wrapper

    def draw_camera_lines(self, T_CWs, scale=0.1):
        """
        Draw camera lines based on open3D

        Arguments
        ---------
            T_CWs (torch.tensor): [N*n_fov,4,4] Transformation from the world to each camera

        Returns
        -------
            Lines (o3d.geometry.LineSet): Set of lines of the fov and the cameras
        """
        camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=self.w, height=self.h, intrinsic_matrix=self.K[0].cpu().numpy()
        )
        poses = T_CWs.cpu().numpy()
        camera_lines = [
            o3d.geometry.LineSet.create_camera_visualization(
                camera_intrinsic, t, scale=scale
            )
            for t in poses
        ]
        all_lines = o3d.geometry.LineSet()
        for lines in camera_lines:
            all_lines += lines
        return all_lines


class DepthSensor(DistanceSensor):
    """
    Approach based on rendering 360 cameras from the robot to model the distance
    to the closest obstacles
    """

    def __init__(
        self,
        gaussian_config: GSplatLoader,
        n_cameras: int,
        camera_params: CameraParams = CameraParams(80, 80, fov=90),
    ):
        self.render = Render(gaussian_config)
        self.panorama_sensor = PanoramaSensor(n_cameras, camera_params)
        self.T_CWs = (
            torch.eye(4, device="cuda")
            .unsqueeze(0)
            .repeat(n_cameras * self.panorama_sensor.n_fov, 1, 1)
        )
        self.T_WCs = (
            torch.eye(4, device="cuda")
            .unsqueeze(0)
            .repeat(n_cameras * self.panorama_sensor.n_fov, 1, 1)
        )
        self.p_c = torch.zeros((1, 3), device="cuda")

    def get_pcd_error(self, sdf: IdealSDF):
        p_c = torch.cat((self.p_c, torch.ones_like(self.p_c[:, :, :, 0:1])), dim=-1)
        p_w_h = torch.einsum("nij,nhwj->nhwi", self.T_WCs, p_c)

        # Convert back to Euclidean coordinates
        p_w = p_w_h[..., :3]
        p_w = p_w.reshape(-1, 3)
        p_w = p_w[~torch.isnan(p_w).any(dim=1)]

        d_P_w = sdf.get_distance(p_w)

        return d_P_w

    def debug_pcd(self, error: Optional[torch.Tensor] = None):
        p_c = torch.cat((self.p_c, torch.ones_like(self.p_c[:, :, :, 0:1])), dim=-1)
        p_w_h = torch.einsum("nij,nhwj->nhwi", self.T_WCs, p_c)

        # Convert back to Euclidean coordinates
        p_w = p_w_h[..., :3] / (p_w_h[..., 3:4] + 1e-8)
        p_w = p_w.reshape(-1, 3)
        p_w = p_w[~torch.isnan(p_w).any(dim=1)]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(p_w.cpu().numpy())
        if error is not None:
            error_normalized = (error - error.min()) / (error.max() - error.min())
            colors = np.zeros_like(p_w.cpu().numpy())
            colormap = plt.cm.get_cmap("jet")
            colors = colormap(error_normalized.cpu().numpy())[:, :3]
            pcd.colors = o3d.utility.Vector3dVector(colors)

        return pcd

    def debug_rgb_pcd(self, depth = False) -> o3d.geometry.PointCloud:
        """
        Generates an Open3D point cloud from transformed 3D points with RGB colors.

        The points are transformed from camera coordinates to world coordinates and then
        converted to an Open3D point cloud with RGB colors.

        Returns:
            o3d.geometry.PointCloud: An Open3D point cloud with RGB colors.
        """
        pcd = o3d.geometry.PointCloud()

        p_c = torch.cat((self.p_c, torch.ones_like(self.p_c[:, :, :, 0:1])), dim=-1)
        p_w_h = torch.einsum("nij,nhwj->nhwi", self.T_WCs, p_c)

        # Convert back to Euclidean coordinates
        p_w = p_w_h[..., :3] / (p_w_h[..., 3:4] + 1e-8)
        p_w = p_w.reshape(-1, 3)
        if depth:
            depth_jet = self.depth_jet(self.depth_img)
            rgb = depth_jet.reshape(-1,3)
            rgb = rgb[~torch.isnan(p_w).any(dim=1).cpu()]

            pcd.colors = o3d.utility.Vector3dVector(rgb)
        else:
            rgb = self.color.reshape(-1, 3)
            rgb = rgb[~torch.isnan(p_w).any(dim=1)]

            pcd.colors = o3d.utility.Vector3dVector(rgb.cpu().numpy())

        p_w = p_w[~torch.isnan(p_w).any(dim=1)]

        pcd.points = o3d.utility.Vector3dVector(p_w.cpu().numpy())



        return pcd

    def debug_individual_pcd(
        self,
    ) -> Tuple[o3d.geometry.PointCloud, ...]:
        """
        Generates six Open3D point clouds from transformed 3D points.

        The points are transformed from camera coordinates to world coordinates and then
        split into six separate point clouds corresponding to different camera views.

        **Order of the returned point clouds:**
        1. Front
        2. Right
        3. Back
        4. Left
        5. Up
        6. Down

        Returns:
            Tuple[o3d.geometry.PointCloud, ...]: A tuple containing six Open3D point clouds,
            each representing points from a specific camera perspective.
        """
        p_c = torch.cat((self.p_c, torch.ones_like(self.p_c[:, :, :, 0:1])), dim=-1)
        p_w_h = torch.einsum("nij,nhwj->nhwi", self.T_WCs, p_c)

        # Convert back to Euclidean coordinates
        p_w = p_w_h[..., :3] / (p_w_h[..., 3:4] + 1e-8)
        p_w = p_w.reshape(-1, 3)
        p_w = p_w.reshape(6, -1, 3)
        p_w[torch.isnan(p_w)] = 0

        p_w = p_w.detach().cpu()

        # Separate from each camera
        # Order is front, right, back, left up and down
        pcds = []
        for p_c_w in p_w:
            pcd = o3d_pcd()
            pcd.points = o3d.utility.Vector3dVector(p_c_w.numpy())
            pcds.append(pcd)
        return tuple(pcds)

    def get_min(self, depth):
        _depth = depth.clone()
        _depth[depth == 0] = torch.inf
        # mask = depth==0
        # depth[mask] = torch.inf
        closest_depths, indices = torch.min(_depth.view(_depth.shape[0], -1), dim=1)
        return closest_depths, indices

    def set_pose(self, T_WS):
        """
        Set sensor pose

        Parameters
        ----------
        T_WS : torch.Tensor
            pose of the sensor [n,4,4]
        """
        self.T_CWs, self.T_WCs = self.panorama_sensor.get_poses(T_WS)


    def get_distance_gui(self, T_WS, radius, depth_fixed=0.):
        """
        Get the distance to the closest obstacle for GUI visualization

        Parameters
        ----------
        T_WS (torch.tensor): [N, 4, 4] World to sensor poses
        radius (torch.tensor): [N] Radius of the sensors
        depth_fixed (float): if provided, use this depth instead of rendering
        Returns
        -------
        g_grad (torch.tensor): [N, 3] Gradient of the closest point on the gaussian frame
        distance (torch.tensor): [N] Distance to the closest point
        """
        self.T_CWs, self.T_WCs = self.panorama_sensor.get_poses(T_WS)
        self.color, self.depth_img = self.render.render(
            self.T_CWs,
            Ks=self.panorama_sensor.K,
            width=self.panorama_sensor.w,
            height=self.panorama_sensor.h,
        )
        if depth_fixed > 0.1:
            valid_mask = (self.depth_img > 0.1) & torch.isfinite(self.depth_img)
            depth = torch.ones_like(self.depth_img) * depth_fixed
            depth[~valid_mask] = 0
        else:
            depth = self.depth_img

        self.p_c, depth = self.panorama_sensor.back_projection(depth)
        depth = depth - radius.repeat_interleave(6, 0).reshape(-1, 1, 1)
        self.radius = radius.repeat_interleave(6, 0)
        # First approach just take the closest depth of each image
        w = self.panorama_sensor.w

        closest_depths, indices = self.get_min(depth)

        h_indices = indices // w
        w_indices = indices % w
        location = torch.stack(
            (w_indices, h_indices, torch.ones_like(w_indices)), dim=1
        ).float()

        # Get the gradient of the closest point
        # Each location is in the pixel coordinate I just need to get them on the 3D space
        grad = torch.einsum("nij,nj->ni", self.panorama_sensor.K_inv, location)

        # normalize grad
        grad = grad / torch.linalg.norm(grad, dim=1, keepdim=True)
        w_grad = vectorized_rot_vec(grad, self.T_WCs[:, :3, :3])

        depth[depth == torch.inf] = 0
        return w_grad, closest_depths

    def get_distance(self, T_WS, radius):
        """
        Get the distance to the closest obstacle
        Parameters
        ----------
        T_WS (torch.tensor): [N, 4, 4] World to sensor poses
        radius (torch.tensor): [N] Radius of the sensors
        Returns
        -------
        g_grad (torch.tensor): [N * 6, 3] Gradient of the closest point on the gaussian frame
        distance (torch.tensor): [N * 6] Distance to the closest point
        """
        self.T_CWs, self.T_WCs = self.panorama_sensor.get_poses(T_WS)
        self.color, self.depth = self.render.render(
            self.T_CWs,
            Ks=self.panorama_sensor.K,
            width=self.panorama_sensor.w,
            height=self.panorama_sensor.h,
        )
        self.p_c, self.depth = self.panorama_sensor.back_projection(self.depth)
        depth = self.depth - radius.repeat_interleave(6, 0).reshape(-1, 1, 1)
        self.radius = radius.repeat_interleave(6, 0)
        # First approach just take the closest depth of each image
        w = self.panorama_sensor.w

        closest_depths, indices = self.get_min(depth)

        h_indices = indices // w
        w_indices = indices % w
        location = torch.stack(
            (w_indices, h_indices, torch.ones_like(w_indices)), dim=1
        ).float()

        # Get the gradient of the closest point
        # Each location is in the pixel coordinate I just need to get them on the 3D space
        grad = torch.einsum("nij,nj->ni", self.panorama_sensor.K_inv, location)

        # normalize grad
        grad = grad / torch.linalg.norm(grad, dim=1, keepdim=True)
        w_grad = vectorized_rot_vec(grad, self.T_WCs[:, :3, :3])

        depth[depth == torch.inf] = 0
        return w_grad, closest_depths

    def get_distance_sdf(self, T_WS, sdf):
        self.T_CWs, self.T_WCs = self.panorama_sensor.get_poses(T_WS)
        color, depth, meta = self.render.render(
            self.T_CWs,
            Ks=self.panorama_sensor.K,
            width=self.panorama_sensor.w,
            height=self.panorama_sensor.h,
        )
        self.p_c, _ = self.panorama_sensor.back_projection(depth)
        p_c = torch.cat((self.p_c, torch.ones_like(self.p_c[:, :, :, 0:1])), dim=-1)
        p_s = torch.einsum(
            "nij,nhwj->nhwi", self.panorama_sensor.R, p_c
        )  # Point with respect to the center of the sensor
        p_s = p_s[..., :3] / (p_s[..., 3:4] + 1e-8)
        # replace nans for 10000
        p_s[torch.isnan(p_s)] = 10000

        # We are lazy so to get the gradient through autodif
        p_s.requires_grad_(True)
        grad, distance = sdf.get_closest(p_s.reshape(-1, 3))

        # Drop the gradient
        grad = grad.detach()
        distance = distance.detach()

        # normalize grad
        grad = grad / torch.linalg.norm(grad, dim=1, keepdim=True)
        w_grad = vectorized_rot_vec(grad, T_WS[:, :3, :3])

        depth[depth == torch.inf] = 0
        return w_grad, distance, meta

    def _get_all_points(self, P_W, radius):
        T_CWs, T_WCs = self.panorama_sensor.get_poses(P_W)
        color, depth = self.render.render(
            T_CWs,
            self.panorama_sensor.K,
            self.panorama_sensor.w,
            self.panorama_sensor.h,
        )
        depth[depth == torch.inf] = 0
        pcd = self.panorama_sensor.get_points_world(T_WCs, depth)
        return pcd

    def get_points(self, P_W, radius):
        T_CWs, T_WCs = self.panorama_sensor.get_poses(P_W)
        grad, distance = self.get_distance(
            P_W, radius
        )  # this grad is in the camera frame
        o = T_WCs[:, :3, -1]
        points = o + grad * distance.unsqueeze(-1)
        return points

    def draw_lines(self, T_WCs, grad, distance):
        if T_WCs is None:
            T_WCs = self.T_WCs
        positions = T_WCs[:, :3, -1]
        distance = (distance + self.radius).unsqueeze(-1)
        return line_from_dist_grad(positions, grad, distance)

    def depth_to_cv2(self, depth_image, min=0, max=5):
        depth_image = depth_image.squeeze().unsqueeze(-1)
        depth_image = (depth_image - min) / (max - min)
        depth_image = (depth_image * 255).byte()
        depth_image = depth_image.cpu().numpy()

        depth_image = np.repeat(depth_image, 3, axis=-1)
        return depth_image

    def draw_closer(self, depth_image, min=0, max=5, resize=500):
        _, indices = self.get_min(depth_image)
        indices = torch.unravel_index(indices, depth_image.shape)
        h = indices[-2].cpu().numpy()
        w = indices[-1].cpu().numpy()
        depth_image = self.depth_to_cv2(depth_image)
        depth_image = cv2.circle(depth_image, (w[0], h[0]), 1, (125, 0, 0), -1)
        resized = cv2.resize(
            depth_image, (resize, resize), interpolation=cv2.INTER_NEAREST
        )
        return resized

    def log_images(self, exp_name, P_W):
        folder_name = f"logs/depth_sensor/{exp_name}"
        os.makedirs(folder_name, exist_ok=True)
        T_CWs, T_WCs = self.panorama_sensor.get_poses(P_W)
        color, depth = self.render.render(
            T_CWs,
            self.panorama_sensor.K,
            self.panorama_sensor.w,
            self.panorama_sensor.h,
        )
        # depth is of size [N_sensor * n_fov, H, W]
        n_sensor = self.panorama_sensor.n_cameras
        n_fov = self.panorama_sensor.n_fov
        _, euclidean_depth = self.panorama_sensor.back_projection(depth)

        for i in range(n_sensor):
            for j in range(n_fov):
                # Only save the depth images
                depth_image = depth[i * n_fov + j][None, ...]
                e_depth = euclidean_depth[i * n_fov + j][None, ...]
                original = self.draw_closer(depth_image)
                euclidean = self.draw_closer(e_depth)
                cv2.imshow("image", original)
                cv2.imshow("depth", euclidean)
                cv2.waitKey(5000)
                # cv2.imwrite(f"{folder_name}/{i:03d}_{j}.png", depth_image)

    def _render(self, T_WS:torch.Tensor):
        self.T_CWs, self.T_WCs = self.panorama_sensor.get_poses(T_WS)
        color, depth, extras, meta = self.render.render_all(
            self.T_CWs,
            Ks=self.panorama_sensor.K,
            width=self.panorama_sensor.w,
            height=self.panorama_sensor.h,
            extras_names=["alpha","normal","surf_normals"]
        )
        return color, depth, extras, meta


    def log_distances_grad(
        self, T_GW: torch.Tensor, T_WG: torch.Tensor, robot: NeuralRobot
    ):
        """
        Obtain the gradient and distance of the depth sensor approach direclty using the robot


        """
        X_WSp = robot.get_point_poses().detach()
        X_GSp = T_GW @ X_WSp
        g_G, distance = self.get_distance(X_GSp, robot.SpheresRadius)
        g_W = helper_math.rotate_vec(g_G, T_WG[:3, :3])

        return g_W, distance

    def gui_debug(self, gradient, distance, gui, gt_sdf, error):
        lines = self.draw_lines(None, gradient, distance)
        camera_lines = self.panorama_sensor.draw_camera_lines(self.T_CWs, 0.01)
        gui.update_geometry("lines", lines)
        gui.update_geometry("camera_lines", camera_lines)
        if gt_sdf is not None:
            pcd = self.debug_pcd(error)
        else:
            pcd = self.debug_pcd()

        gui.update_geometry("pcd", pcd)

    def draw_panoramic(self, P_W, out_shape=(80 * 4, 80 * 4), upscale = 6):
        """Only valid for one sensor"""

        T_CWs, T_WCs = self.panorama_sensor.get_poses(P_W)
        color, depth = self.render.render(
            T_CWs,
            Ks=self.panorama_sensor.K,
            width=self.panorama_sensor.w,
            height=self.panorama_sensor.h,
        )
        rots = self.panorama_sensor.R.cpu().numpy()  # Rotation matrices for each camera

        W, H = out_shape

        panorama = np.zeros((H, W, 3), dtype=np.float32)

        Ks = self.panorama_sensor.K.cpu().numpy()

        x = np.arange(self.panorama_sensor.w)
        y = np.arange(self.panorama_sensor.h)
        xx, yy = np.meshgrid(x, y)
        grid = np.stack((xx, yy, np.ones_like(xx)), axis=-1)
        dirs = np.einsum("ij,klj->kli", np.linalg.inv(Ks[0]), grid)

        for img, R in zip(color.cpu().numpy(), rots):
            R = R[:3, :3]  # R_cs
            dir = (R @ dirs.reshape(-1, 3).T).T

            norm = np.linalg.norm(dir, axis=1)
            theta = np.arccos(dir[:, 2] / norm)
            phi = np.arctan2(dir[:, 1], dir[:, 0])
            u = ((phi + np.pi) / (2 * np.pi) * W).astype(
                int
            )  # Azimuth mapped to [0, W)
            v = (theta / np.pi * H).astype(int)  # Inclination mapped to [0, H)
            # Clip indices to avoid out-of-bounds errors
            u = np.clip(u, 0, W - 1)
            v = np.clip(v, 0, H - 1)
            img_flat = img.reshape(-1, 3)  # Flatten the image to match u, v
            # Assign pixel values
            panorama[v, u] = img_flat

        panorama_uint8 = (panorama * 255).astype(np.uint8)

        # Create mask where all channels are zero (i.e., holes)
        mask = np.all(panorama_uint8 == 0, axis=-1).astype(np.uint8) * 255  # 0 or 255 mask

        # Inpaint using Navier-Stokes method or Telea method
        panorama_filled = cv2.inpaint(panorama_uint8, mask, inpaintRadius=2, flags=cv2.INPAINT_NS)

        # Resize the panorama to the desired output shape
        panorama_resized = cv2.resize(
            panorama_filled, (W * upscale, H * upscale), interpolation=cv2.INTER_LINEAR
        )
        return panorama_resized

    def depth_uint(self, depth):
        depth[depth == torch.inf] = 0

        depth_uint = (depth - depth.min()) / (depth.max() - depth.min())
        depth_uint = (depth_uint * 255).byte().cpu().numpy()
        depth_uint = depth_uint.astype(np.uint8)
        return depth_uint

    def depth_jet(self, depth):
        depth_uint = self.depth_uint(depth)
        jet = np.zeros((*depth_uint.shape,3))
        for ix, d in enumerate(depth_uint):
            jet[ix] = cv2.applyColorMap(d, cv2.COLORMAP_JET)
        return jet/255

    def stiching(self, P_W, out_shape=(80 * 3, 80 * 3), upscale=6, variant = "jet"):
        """
        Draw images as a cube
        """
        T_CWs, _ = self.panorama_sensor.get_poses(P_W)

        color, depth = self.render.render(
            T_CWs,
            Ks=self.panorama_sensor.K,
            width=self.panorama_sensor.w,
            height=self.panorama_sensor.h,
        )
        if variant == "rgb":
            color = color.cpu().numpy() * 255
            panorama = np.zeros((out_shape[0], out_shape[1], 3), dtype=np.uint8)
            panorama[80:160, 80:160, :] = color[1]
            panorama[0:80, 80:160, :] = np.rot90(color[4], k=-1)
            panorama[160:240, 80:160, :] = cv2.rotate(color[5], cv2.ROTATE_90_COUNTERCLOCKWISE)
            panorama[80:160, 0:80, :] = color[0]
            panorama[80:160, 160:240, :] = color[2]
            panorama_resized = cv2.resize(
                panorama, (out_shape[0] * upscale, out_shape[1] * upscale), interpolation=cv2.INTER_LINEAR
            )
        else:
            color = self.depth_uint(depth)
            panorama = np.zeros((out_shape[0], out_shape[1]), dtype=np.uint8)
            panorama[80:160, 80:160] = color[1]  # Front
            panorama[0:80, 80:160] = np.rot90(color[4], k=-1)  # top
            panorama[160:240, 80:160] = cv2.rotate(color[5], cv2.ROTATE_90_COUNTERCLOCKWISE) # Bottom
            panorama[80:160, 0:80] = color[0] # Left
            panorama[80:160, 160:240] = color[2] # Right
            panorama_resized = cv2.resize(
                panorama, (out_shape[0] * upscale, out_shape[1] * upscale), interpolation=cv2.INTER_LINEAR
            )



        # Resize the panorama to the desired output shape
        panorama_resized = panorama_resized.astype(np.uint8)
        if variant == "jet":

            invalid_mask = panorama_resized == 0
            panorama_resized = cv2.applyColorMap(panorama_resized, cv2.COLORMAP_JET)

            panorama_resized[invalid_mask] = [0, 0, 0]
        return (panorama_resized)

def reshape_depth_sensor(distance, gradient):
    """
    Reshape the depth sensor data to the correct shape

    the logs are saved as [N_steps, 6 * N_sensors, 1 or 3].
    To ease our work we reshape the data to [N_steps, N_sensors, 6, 1 or 3]
    """
    if len(distance.shape) == 2:
        N_Steps = distance.shape[0]

        return distance.reshape(N_Steps, -1, 6), gradient.reshape(
            N_Steps, -1, 6, gradient.shape[-1]
        )
    elif len(distance.shape) == 1:
        return distance.reshape(-1, 6), gradient.reshape(
            -1, 6, gradient.shape[-1]
        )


def reshape_euclidean_sensor(distance, gradient):
    """ """
    N_Steps = distance.shape[0]

    return distance.reshape(N_Steps, -1, 1), gradient.reshape(
        N_Steps, -1, 1, gradient.shape[-1]
    )


class GUI:
    def __init__(self, mesh, camera_params):
        if isinstance(mesh, trimesh.Trimesh):
            mesh = to_o3d(mesh)

        self.viz = o3d.visualization.Visualizer()
        self.viz.create_window(width=1920, height=1080)
        self.viz.add_geometry(mesh)

        self.mesh = mesh

    def update(self, lines):
        self.viz.clear_geometries()
        self.viz.add_geometry(self.mesh)
        self.viz.add_geometry(lines)
        self.viz.poll_events()
        self.viz.update_renderer()
