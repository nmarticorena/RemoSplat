import copy
import sys
import time
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import spatialmath as sm
import spatialmath.base as smb
import torch
import trimesh
import warp as wp
from matplotlib.cm import get_cmap
from scipy.stats import chi2

from remo_splat.ellipsoids.meshes import (ellipsoid_mesh, get_mesh,
                                                 to_o3d)
# from ctypes import c_float
from remo_splat.loader import load_gsplat, load_points

tape = wp.Tape()
step = 0.01

def uniform(shape, r, device="cuda"):
    sample = (r[0] - r[1]) * torch.rand(*shape) + r[1]
    return sample.to(device)

@wp.struct
class Gaussian:
    mean: wp.vec3f
    scale: wp.mat33f
    angles: wp.quat
    rot: wp.mat33f
    cov: wp.mat33f
    opacity: wp.float32

@wp.struct
class Gaussians:
   """ Struct to store the gaussians """
   means: wp.array(dtype=wp.vec3f) # means [n,3]  #type:ignore
   scales: wp.array(dtype=wp.mat33f) # scales [n,3,3] #type:ignore
   angles: wp.array(dtype=wp.quat) # angles [n,4] #type:ignore
   rots: wp.array(dtype=wp.mat33f) # rots [n,3,3] #type:ignore
   covs: wp.array(dtype=wp.mat33f) # covs [n,3,3] #type:ignore
   opacities: wp.array(dtype=wp.float32) # opacities [n] #type:ignore
   sigmas: wp.array(dtype= wp.vec3f) # sigmas [n,3] #type:ignore

@wp.func
def _mahalanobis_distance(gaussian: Gaussian, point: wp.vec3, chi: wp.float32):
    diff = point - gaussian.mean
    sigma_inv = wp.inverse(gaussian.cov)
    distance = diff @ sigma_inv
    return wp.sqrt(wp.dot(distance, diff)) / chi - 1.0

@wp.kernel
def set_values_quat(
    gaussians: Gaussians,
    means: wp.array(dtype=wp.vec3f), #type:ignore
    angles: wp.array(dtype=wp.vec4f), #type:ignore
    scales: wp.array(dtype=wp.vec3f), #type:ignore
):
    tid = wp.tid()

    # gaussians.means[tid] = means[tid]
    gaussians.scales[tid] = wp.mat33f(
        scales[tid][0] * scales[tid][0],
        0.0,
        0.0,
        0.0,
        scales[tid][1] * scales[tid][1],
        0.0,
        0.0,
        0.0,
        scales[tid][2] * scales[tid][2],
    ) 

    angle = wp.quat(angles[tid][1], angles[tid][2], angles[tid][3], angles[tid][0]) # [TODO] check here

    gaussians.angles[tid] = angle
    gaussians.rots[tid] = wp.quat_to_matrix(angle)

    gaussians.covs[tid] = (
        gaussians.rots[tid]
        @ gaussians.scales[tid]

        @ wp.transpose(gaussians.rots[tid])
    )

@wp.kernel
def set_values(
    gaussians: wp.array(dtype=Gaussian), #type:ignore
    means: wp.array(dtype=wp.vec3f), #type:ignore
    angles: wp.array(dtype=wp.vec3f), #type:ignore
    scales: wp.array(dtype=wp.vec3f), #type:ignore
):
    tid = wp.tid()
    gaussians[tid].mean = means[tid]

    gaussians[tid].scale = wp.mat33f(
        scales[tid][0],
        0.0,
        0.0,
        0.0,
        scales[tid][1],
        0.0,
        0.0,
        0.0,
        scales[tid][2],
    )

    quad_angle = wp.quat_rpy(angles[tid][0], angles[tid][1], angles[tid][2])

    gaussians[tid].angles = quad_angle
    gaussians[tid].rot = wp.quat_to_matrix(quad_angle)

    gaussians[tid].cov = (
        gaussians[tid].rot
        @ gaussians[tid].scale
        @ wp.transpose(gaussians[tid].scale)
        @ wp.transpose(gaussians[tid].rot)
    )

@wp.kernel
def update_rot(gaussians: wp.array(dtype=Gaussian), rot_delta: wp.quat):  #type:ignore
    tid = wp.tid()
    gaussians[tid].angles = gaussians[tid].angles * rot_delta
    gaussians[tid].rot = wp.quat_to_matrix(gaussians[tid].angles)

    gaussians[tid].cov = (
        gaussians[tid].rot
        @ gaussians[tid].scale
        @ gaussians[tid].scale
        @ wp.transpose(gaussians[tid].rot)
    )

@wp.kernel
def mahalanobis_distance_grid(
    gaussians: wp.array(dtype=Gaussian), #type:ignore
    points: wp.array(ndim=3, dtype=wp.vec3), #type:ignore
    distances: wp.array(ndim=3, dtype=float), #type:ignore
    chi: wp.float32,
):
    i, j, k, l = wp.tid() #type:ignore
    g = gaussians[i]
    wp.atomic_min(
        distances,
        j,
        k,
        l,
        _mahalanobis_distance(g, points[j, k, l], chi),
    )

@wp.kernel
def mahalanobis_distances(
    gaussians: wp.array(dtype=Gaussian), #type:ignore
    points: wp.array(dtype=wp.vec3), #type:ignore
    distances: wp.array(dtype=float, ndim=2), #type:ignore
    chi: wp.float32,
):
    i, j = wp.tid()
    g = gaussians[i]
    distances[i, j] = _mahalanobis_distance(g, points[j], chi)

@wp.kernel
def mahalanobis_distance(
    gaussians: wp.array(dtype=Gaussian), #type:ignore
    points: wp.array(dtype=wp.vec3), #type:ignore
    distances: wp.array(dtype=float), #type:ignore
    chi: wp.float32,
):
    i, j = wp.tid()
    g = gaussians[i]

    wp.atomic_min(
        distances,
        j,
        _mahalanobis_distance(g, points[j], chi),
    )

# @wp.kernel
# def apply_transform(
#     gaussians: Gaussians,
#     transform: wp.mat44f,
# ):
#     i = wp.tid()
#     gaussians.means[i] = wp.transform(transform, gaussians.means[i])
#     gaussians.angles[i] = wp.transform(transform, gaussians.angles[i])
#     gaussians.rots[i] = wp.transform(transform, gaussians.rots[i])
#     gaussians.covs[i] = wp.transform(transform, gaussians.covs[i])
#
@wp.kernel
def pdf(
    gaussians: wp.array(dtype=Gaussian), #type:ignore
    points: wp.array(dtype=wp.vec3), #type:ignore
    prob: wp.array(dtype=float), #type:ignore
):
    i, j = wp.tid()  # i for gaussians, j for points
    diff = points[j] - gaussians[i].mean

    normalization = 1.0 / (
        2.0 * wp.pi * wp.sqrt(wp.determinant(gaussians[i].cov))
    )  # ["n_gaussians"]

    distance = wp.inverse(gaussians[i].cov) * diff
    exponent = -0.5 * wp.dot(diff, distance)

    wp.atomic_max(prob, j, normalization * wp.exp(exponent))

@wp.kernel
def update_points(
    points: wp.array(dtype=wp.vec3), #type:ignore
    grad: wp.array(dtype=wp.vec3), #type:ignore
    distance: wp.array(dtype=float), #type:ignore
):
    i = wp.tid()
    # points[i] = points[i] - grad[i] * distance[i] * step
    grad_norm = wp.normalize(grad[i])
    points[i] = points[i] - grad_norm * step

@wp.kernel
def meshify_mask(
    gaussians: Gaussians,
    vertices: wp.array(dtype=wp.vec3f), #type:ignore
    faces: wp.array(dtype=wp.vec4i), #type:ignore
    x_s: wp.array(dtype=wp.float32, ndim=2), #type:ignore
    y_s: wp.array(dtype=wp.float32, ndim=2), #type:ignore
    z_s: wp.array(dtype=wp.float32, ndim=2), #type:ignore
    n_theta: wp.int32,
    n_sigma: wp.int32,
    mask : wp.array(dtype=wp.bool) #type:ignore
):
    i = wp.tid()
    if mask[i]:
        
        g = gaussians
        get_mesh(i, g.sigmas, g.means, g.angles, vertices, faces, x_s, y_s, z_s, n_theta, n_sigma)


@wp.kernel
def meshify(
    gaussians: Gaussians,
    vertices: wp.array(dtype=wp.vec3f), #type:ignore
    faces: wp.array(dtype=wp.vec4i), #type:ignore
    x_s: wp.array(dtype=wp.float32, ndim=2), #type:ignore
    y_s: wp.array(dtype=wp.float32, ndim=2), #type:ignore
    z_s: wp.array(dtype=wp.float32, ndim=2), #type:ignore
    n_theta: wp.int32,
    n_sigma: wp.int32,
):
    i = wp.tid()
    g = gaussians
    get_mesh(i, g.sigmas, g.means, g.angles, vertices, faces, x_s, y_s, z_s, n_theta, n_sigma)

    
@wp.kernel
def reduce_sum(lengths: wp.array(dtype=float), total_length: wp.array(dtype=float)): #type:ignore
    # Accumulate the sum of lengths in total_length[0]
    tid = wp.tid()
    wp.atomic_add(total_length, 0, lengths[tid] * lengths[tid])

@wp.kernel
def top_k_opacity(gaussians: wp.array(dtype=Gaussian), k: int, topk_gaussians: wp.array(dtype=Gaussian)): #type:ignore
    i = wp.tid()
    # topk_gaussians[i] = gaussians[i]


class Gaussians3D:
    def __init__(self):
        percentile = 0.6827
        self.chi = wp.float32(np.sqrt(chi2.ppf(percentile, 3))) #type:ignore
        return

    def toy(self, mu, sigmas, opacities:torch.Tensor = torch.tensor([1]),rot: Optional[sm.SO3] = None, repeat: int = 1):
        mu = mu.repeat(repeat,1)
        sigmas = sigmas.repeat(repeat,1)
        opacities = opacities.repeat(repeat)

        means = wp.array(mu, dtype = wp.vec3f)
        scales = wp.array(sigmas, dtype = wp.vec3f)
        if rot is None:
            print("no rot")
            rots = [sm.SO3().A for _ in range(mu.shape[0])]
        else:
            rots = rot.A 
        quad = np.array([smb.r2q(i, order = "sxyz") for i in rots]) #type:ignore
        quad=  quad.repeat(repeat,1)
        angles = wp.array(quad, dtype = wp.vec4f)
        self.quad = angles
        n_gaussians = mu.shape[0]
        self.gaussians = Gaussians()
        self.gaussians.means = means 
        self.gaussians.scales = wp.zeros((n_gaussians), dtype=wp.mat33f) #type:ignore
        self.gaussians.angles = wp.zeros((n_gaussians), dtype=wp.quat) #type:ignore
        # self.gaussians.angles = angles 
        self.gaussians.rots = wp.zeros((n_gaussians), dtype=wp.mat33f) #type:ignore
        self.gaussians.covs = wp.zeros((n_gaussians), dtype=wp.mat33f) #type:ignore
        if opacities.shape[0] == 1:
            opacities = torch.ones(n_gaussians)
        self.gaussians.opacities = wp.from_torch(opacities, dtype = wp.float32)
        self.gaussians.sigmas = scales


        wp.launch(
            kernel=set_values_quat,
            dim=(n_gaussians),
            inputs=[self.gaussians, means, angles, scales],
        )
        wp.synchronize()

    def load(self, filename, th=0, decil:int = 0, create_mesh = False):
        """
        Load gaussians from a file

        Parameters
        ----------
        filename : str
            path to the file
        th : float, optional
            minimum opacity threshold, by default 0
        decil : int, optional
            decil to load, by default 0
        """
        if filename.endswith(".ply"):
            loader = load_points
        else:
            loader = load_gsplat
        xyz, opacity, scales, rots, _, _ = loader(filename, th) #type:ignore
        
        # check if numpy
        if isinstance(xyz, np.ndarray):
            xyz = torch.from_numpy(xyz).float().cuda()
            opacity = torch.from_numpy(opacity).float().cuda()
            scales = torch.from_numpy(scales).float().cuda()
            rots = torch.from_numpy(rots).float().cuda()

        self.x_max = xyz[:, 0].max()
        self.x_min = xyz[:, 0].min()
        self.y_max = xyz[:, 1].max()
        self.y_min = xyz[:, 1].min()
        self.z_max = xyz[:, 2].max()
        self.z_min = xyz[:, 2].min()
        self.centroids = xyz
        
        if decil > 0:
            n_gaussians = int(xyz.shape[0]//10)
            sorted_indices = torch.argsort(opacity)

            xyz = xyz[sorted_indices[decil * n_gaussians: (decil + 1) * n_gaussians], :]
            scales = scales[sorted_indices[decil * n_gaussians: (decil + 1) * n_gaussians], :]
            rots = rots[sorted_indices[decil * n_gaussians: (decil + 1) * n_gaussians], :]
            opacity = opacity[sorted_indices[decil * n_gaussians: (decil + 1) * n_gaussians]]
        else: # Load all
            n_gaussians = xyz.shape[0]

        print(f"Loading {n_gaussians} Gaussians")

        means = wp.from_torch(xyz, dtype=wp.vec3f)
        angles = wp.from_torch(rots, dtype=wp.vec4f)
        scales = wp.from_torch(scales, dtype=wp.vec3f)
        opacity = wp.from_torch(opacity, dtype=wp.float32)
        self.gaussians = Gaussians()
        self.gaussians.means = means 
        self.gaussians.scales = wp.zeros((n_gaussians), dtype=wp.mat33f) #type:ignore
        self.gaussians.angles = wp.zeros((n_gaussians), dtype=wp.quat) #type:ignore
        # self.gaussians.angles = angles  #type:ignore
        self.gaussians.rots = wp.zeros((n_gaussians), dtype=wp.mat33f) #type:ignore
        self.gaussians.covs = wp.zeros((n_gaussians), dtype=wp.mat33f) #type:ignore
        self.gaussians.opacities = opacity
        self.gaussians.sigmas = scales


        wp.launch(
            kernel=set_values_quat,
            dim=(n_gaussians), #type:ignore
            inputs=[self.gaussians, means, angles, scales],
        )
        wp.synchronize()
        print(self.gaussians)
        if create_mesh:
            mesh = self.meshify(open3d=False)
            # mesh.paint_uniform_color([0,1,0])
        else:
            return None
        return mesh

    def sort_opacity(self, descending=True):
        opacity = wp.to_torch(self.gaussians.opacities)
        values, indices = opacity.sort(descending=descending)
        means = wp.to_torch(self.gaussians.means) 
        scales = wp.to_torch(self.gaussians.scales)
        angles = wp.to_torch(self.gaussians.angles)
        rots = wp.to_torch(self.gaussians.rots) 
        covs = wp.to_torch(self.gaussians.covs)
        sigmas = wp.to_torch(self.gaussians.sigmas)
        print("indices", indices.shape) 
        print("when torch",covs.shape)
        print("warp", self.gaussians.covs.numpy().shape)
        self.gaussians = Gaussians()
        self.gaussians.means = wp.from_torch(means[indices], dtype=wp.vec3f)   
        self.gaussians.scales = wp.from_torch(scales[indices], dtype=wp.mat33f)
        self.gaussians.angles = wp.from_torch(angles[indices], dtype=wp.quat)
        self.gaussians.rots = wp.from_torch(rots[indices],dtype = wp.mat33f)
        self.gaussians.covs = wp.from_torch(covs[indices], dtype=wp.mat33f)
        self.gaussians.sigmas = wp.from_torch(sigmas[indices], dtype=wp.vec3f)
        self.gaussians.opacities = wp.from_torch(opacity[indices], dtype=wp.float32)
        # check 
        x = self.gaussians.opacities.numpy()
        plt.plot(x)
        scales = self.gaussians.sigmas.numpy()
        plt.plot(scales)

    def meshify(self, n_theta =8, n_phi=8, percentage = 1, color = [1,0,0], mask = None, open3d = False):
        print("total gaussians", self.gaussians.means.shape[0])
        n_gaussians = int(self.gaussians.means.shape[0] * percentage)


        verts = np.zeros(((n_gaussians) * n_theta * n_phi, 3))
        faces = np.zeros((n_gaussians * ((n_theta - 1) * (n_phi - 1) + 1), 4))
        thetas = np.linspace(0, 2 * np.pi, n_theta)
        phis = np.linspace(0, np.pi, n_phi)
        xs = np.outer(np.cos(thetas), np.sin(phis))
        ys = np.outer(np.sin(thetas), np.sin(phis))
        zs = np.outer(np.ones(n_theta), np.cos(phis))
        verts = wp.array(verts, dtype=wp.vec3f)
        faces = wp.array(faces, dtype=wp.vec4i)
        xs = wp.array(xs, dtype=wp.float32, ndim=2)
        ys = wp.array(ys, dtype=wp.float32, ndim=2)
        zs = wp.array(zs, dtype=wp.float32, ndim=2)

        n_theta = wp.int32(n_theta)
        n_phi = wp.int32(n_phi)
        if mask is not None:
            mask = wp.array(mask, dtype=wp.bool)
            wp.launch(
                kernel=meshify_mask,
                dim=(n_gaussians), #type:ignore
                inputs=[self.gaussians, verts, faces, xs,ys,zs ,n_theta, n_phi, mask],
            )
        else:
            wp.launch(
                kernel=meshify,
                dim=(n_gaussians), #type:ignore
                inputs=[self.gaussians, verts, faces, xs,ys,zs ,n_theta, n_phi],
            )
        mesh = trimesh.Trimesh(vertices=verts.numpy(), faces=faces.numpy().reshape(-1,4))

        if open3d:
            mesh = to_o3d(mesh, color)
        return mesh



    def generate_random(
        self,
        n_gaussians,
        xyz_range=200,
        x_range=[-10, 10],
        y_range=[-10, 10],
        z_range=[-10, 10],
        angles=[-torch.pi, torch.pi],
        scale_x=[0.1, 1.0],
        scale_y=[0.1, 1.0],
        scale_z=[0.1, 1.0],
    ):
        scales_x = uniform((n_gaussians * 1,), scale_x, device="cuda")
        scales_y = uniform((n_gaussians * 1,), scale_y, device="cuda")
        scales_z = uniform((n_gaussians * 1,), scale_z, device="cuda")
        scales = torch.stack([scales_x, scales_y, scales_z], dim=1)

        rot_x = uniform((n_gaussians,), angles, device="cuda")
        rot_y = uniform((n_gaussians,), angles, device="cuda")
        rot_z = uniform((n_gaussians,), angles, device="cuda")
        rot_angles = torch.stack([rot_x, rot_y, rot_z], dim=1)

        mu_x = uniform((n_gaussians * 1,), x_range, device="cuda")
        mu_y = uniform((n_gaussians * 1,), y_range, device="cuda")
        mu_z = uniform((n_gaussians * 1,), z_range, device="cuda")
        mu = torch.stack([mu_x, mu_y, mu_z], dim=1)
        self.xyz_range = xyz_range
        # self.gaussians = wp.array(
            # [Gaussian() for i in range(n_gaussians)], dtype=Gaussian
        # )
        # print(mu.shape)
        means = wp.from_torch(mu, dtype=wp.vec3f)
        angles = wp.from_torch(rot_angles, dtype=wp.vec3f)
        scales = wp.from_torch(scales, dtype=wp.vec3f)

        wp.launch(
            kernel=set_values,
            dim=n_gaussians,
            inputs=[self.gaussians, means, angles, scales],
        )
        print(self.gaussians)
        self.x_min = mu_x.min()
        self.x_max = mu_x.max()

        self.y_min = mu_y.min()
        self.y_max = mu_y.max()

        self.z_min = mu_z.min()
        self.z_max = mu_z.max()

    def step(self, points):
        tape.zero()
        tape.reset()

        num_points = points.shape[0]
        points = wp.from_torch(points, dtype=wp.vec3)
        distance = wp.zeros(shape=num_points, requires_grad=True)
        distance.fill_(sys.float_info.max)
        loss = wp.zeros(1, dtype=float, requires_grad=True)
        with tape:
            wp.launch(
                kernel=mahalanobis_distance,
                dim=[len(self.gaussians), num_points],
                inputs=[self.gaussians, points, distance, self.chi],
            )
            wp.launch(kernel=reduce_sum, dim=num_points, inputs=[distance, loss])
        tape.backward(loss)
        # print(tape.gradients[points])
        return wp.to_torch(distance)
        # return wp.to_torch(points)

    def update_rot(self, delta, axis="x"):
        if axis == "x":
            q = wp.quat_rpy(delta, 0.0, 0.0)
        elif axis == "y":
            q = wp.quat_rpy(0.0, delta, 0.0)
        else:
            q = wp.quat_rpy(0.0, 0.0, delta)

        wp.launch(
            kernel=update_rot, dim=len(self.gaussians), inputs=[self.gaussians, q]
        )

    def get_pdfs(self, grid_size):
        x = torch.linspace(-self.xyz_range * 2, self.xyz_range * 2, grid_size)
        y = torch.linspace(-self.xyz_range * 2, self.xyz_range * 2, grid_size)
        z = torch.linspace(-self.xyz_range * 2, self.xyz_range * 2, grid_size)
        X, Y, Z = torch.meshgrid(x, y, z)
        grid_points = torch.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1).cuda()

        distance = wp.array(shape=grid_size**3, dtype=float)
        points = wp.from_torch(grid_points, dtype=wp.vec3)
        distance.fill_(-10000)
        for i in range(1):
            with wp.ScopedTimer(
                f"Demo pairwise with {len(self.gaussians)} x {grid_size ** 3}",
                synchronize=True,
            ):
                wp.launch(
                    kernel=pdf,
                    dim=[len(self.gaussians), grid_size * grid_size * grid_size],
                    inputs=[self.gaussians, points, distance],
                )
        d = wp.array.numpy(distance)
        d = d.reshape(X.shape)

        print(d.min())
        print(d.max())
        mask = (1e-2 > d) & (d > 1e-4)

        X = X[mask]
        Y = Y[mask]
        Z = Z[mask]
        d = d[mask]
        return X, Y, Z, d

    def plot_pdfs(self, grid_size):
        X, Y, Z, d = self.get_pdfs(grid_size)

        ax = plt.figure().add_subplot(projection="3d")

        ax.set_xlim((-20, 20))
        ax.set_ylim((-20, 20))
        ax.set_zlim((-20, 20))
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        cmap = get_cmap("Blues")  # Choose a colormap
        ax.scatter(X, Y, Z, c=1 / d, cmap=cmap, alpha=0.5)

    def plot_mahalanobis_distance(self, grid_size):
        x = torch.linspace(-self.xy_range * 2, self.xy_range * 2, grid_size)
        y = torch.linspace(-self.xy_range * 2, self.xy_range * 2, grid_size)
        X, Y = torch.meshgrid(x, y)
        grid_points = torch.stack([X.ravel(), Y.ravel()], axis=1).cuda()

        distance = wp.array(shape=grid_size**2, dtype=float)
        distance.fill_(10000)
        points = wp.from_torch(grid_points, dtype=wp.vec2)
        wp.launch(
            kernel=mahalanobis_distance,
            dim=[len(self.gaussians), grid_size * grid_size * grid_size],
            inputs=[self.gaussians, points, distance, self.chi],
        )
        d = wp.array.numpy(distance)
        d = d.reshape(X.shape)
        # d = 1 / d
        print(d.min())
        print(d.max())
        # Plot the gradient using contourf
        cmap = get_cmap("Blues")  # Choose a colormap

        # levels = np.linspace(0, 5, 100)
        plt.contourf(X, Y, d, levels=40, cmap=cmap, vmin=1e-5, vmax=10, alpha=0.1)

    def generate_grid(self, grid_size, xlim, ylim, zlim):
        x = torch.linspace(xlim[0], xlim[1], grid_size)
        y = torch.linspace(ylim[0], ylim[1], grid_size)
        z = torch.linspace(zlim[0], zlim[1], grid_size)
        X, Y, Z = torch.meshgrid(x, y, z)
        grid_points = torch.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1).cuda()
        return X, Y, Z, grid_points

    def generate_grid_2(self, grid_size, xlim, ylim, zlim):
        x = torch.linspace(xlim[0], xlim[1], grid_size)
        y = torch.linspace(ylim[0], ylim[1], grid_size)
        z = torch.linspace(zlim[0], zlim[1], grid_size)
        X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")
        grid_points = torch.stack([X, Y, Z], axis=-1).cuda()
        return X, Y, Z, grid_points

    def compute_distance_grid(self, grid_points):
        n = grid_points.shape[0]
        print(grid_points.shape)
        distance = wp.array(shape=(n, n, n), dtype=float)
        distance.fill_(10000)
        points = wp.from_torch(grid_points, dtype=wp.vec3)
        with wp.ScopedTimer(
            f"mahalanobis distance with {len(self.gaussians)} x {grid_points.shape[0]}",
            synchronize=True,
        ):
            wp.launch(
                kernel=mahalanobis_distance_grid,
                dim=[len(self.gaussians), n, n, n],
                inputs=[self.gaussians, points, distance, self.chi],
            )
        return distance

    def compute_distance(self, points):
        #
        distance = wp.array(shape=points.shape[0], dtype=float)
        distance.fill_(10000)
        points = wp.from_torch(points, dtype=wp.vec3)
        with wp.ScopedTimer(
            f"mahalanobis distance with {len(self.gaussians)} x {points.shape[0]}",
            synchronize=True,
        ):
            wp.launch(
                kernel=mahalanobis_distance,
                dim=[len(self.gaussians), points.shape[0]],
                inputs=[self.gaussians, points, distance, self.chi],
            )
        return distance


def benchmark_distances(gaussians: Gaussians3D, points: np.ndarray):
    ti = time.time()
    distance_1 = wp.array(shape=points.shape[0], dtype=float)
    distance_1.fill_(10000)
    p = wp.from_numpy(points, dtype=wp.vec3)
    wp.launch(
        kernel=mahalanobis_distance,
        dim=[len(gaussians.gaussians), points.shape[0]],
        inputs=[gaussians.gaussians, p, distance_1, gaussians.chi],
    )
    wp.synchronize()
    print("Time for mahalanobis_distance", time.time() - ti)

    ti = time.time()
    distance = wp.array(shape=(len(gaussians.gaussians), points.shape[0]), dtype=float)
    # distance.fill_(10000)
    p = wp.from_numpy(points, dtype=wp.vec3)
    wp.launch(
        kernel=mahalanobis_distances,
        dim=[len(gaussians.gaussians), points.shape[0]],
        inputs=[gaussians.gaussians, p, distance, gaussians.chi],
    )
    values, indices = wp.to_torch(distance).min(axis=0)
    wp.synchronize()
    print("Time for mahalanobis_distances", time.time() - ti)
    # Difference:
    error = (wp.to_torch(distance_1) - values).abs()
    print("Error", error.max())
    print("avg error", error.mean())

    print(distance)
    print(distance.shape)


if __name__ == "__main__":
    g = Gaussians3D()
    g.generate_random(
        1_000_000, x_range=[-1000, 1000], y_range=[-1000, 1000], z_range=[-1000, 1000]
    )
    p = np.random.rand(80, 3)
    benchmark_distances(g, p)

    gaussians = Gaussians3D(2, scale_x=[2.0, 2.0])
    gaussians.plot_pdfs(250)

    plt.show()

    for i in range(10):
        gaussians.update_rot(wp.pi / 6, "z")
        gaussians.plot_pdfs(250)
        plt.show()
        plt.show()
