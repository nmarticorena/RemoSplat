import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import seaborn as sns
import torch
import trimesh
from plyfile import PlyData
from torch.nn import Sigmoid

from remo_splat.ellipsoids.meshes import (ellipsoid_mesh,
                                                 test_disk_meshify,
                                                 test_disk_meshify_from_gsplat,
                                                 test_meshify, to_o3d)


def configure_matplotlib():
    sns.set()
    sns.set_style(style="whitegrid")
    sns.set_palette("colorblind", 6)
    sns.set_context("paper")
    matplotlib.rcParams["ps.useafm"] = True
    matplotlib.rcParams["pdf.use14corefonts"] = True
    matplotlib.rcParams["legend.columnspacing"] = 1.0
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["xtick.major.pad"] = "0"
    plt.rcParams["ytick.major.pad"] = "0"
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    plt.rcParams.update({"font.size": 8})
    plt.rcParams.update({"axes.labelsize": 10})
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams["svg.fonttype"] = "none"
    # default xlim[]
    plt.rcParams["axes.xmargin"] = 0
    plt.rcParams.update({"figure.figsize": (3.5, 3.5)})


configure_matplotlib()


def load_points_tensor(path, th=0):
    xyz, opacity, scales, rots, features, extra_f = load_points(path, th, False)
    return (
        torch.from_numpy(xyz).float().cuda(),
        torch.from_numpy(opacity).float().cuda(),
        torch.from_numpy(scales).float().cuda(),
        torch.from_numpy(rots).float().cuda(),
        torch.from_numpy(features).float().cuda().transpose(1, 2),
        torch.from_numpy(extra_f).float().cuda().transpose(1, 2),
    )


# from https://github.com/graphdeco-inria/gaussian-splatting/blob/8a70a8cd6f0d9c0a14f564844ead2d1147d5a7ac/scene/gaussian_model.py#L217
def load_points(
    path: str, th=0, histogram=False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the gaussian points from a ply file
    Parameters
    ----------
    path : str
        Path to the ply file
    th : float
        Threshold for the opacity
    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        A tuple containing:
        - xyz: np.ndarray
            the center of the gaussians [n, 3]
        - opacity: np.ndarray
            the opacity of the gaussians [n, 1]
        - scales: np.ndarray
            the scales of the gaussians [n, 3]
        - rots: np.ndarray
            the rotations of the gaussians [n, 4] [w, x, y, z]
        - features_dc: np.ndarray
            the features of the gaussians [n, 3, 1]
        - features_extra: np.ndarray
            the extra features of the gaussians [n, 3*(max_sh_degree + 1) ** 2 - 3]
    """
    data = PlyData.read(path)
    xyz = np.stack(
        (
            np.asarray(data.elements[0]["x"]),
            np.asarray(data.elements[0]["y"]),
            np.asarray(data.elements[0]["z"]),
        ),
        axis=1,
    )
    opacity = np.asarray(data.elements[0]["opacity"])
    print(opacity.min())
    print(opacity.max())
    opacity = Sigmoid()(torch.from_numpy(opacity)).numpy()
    if histogram:
        plt.hist(opacity, bins=20)
        plt.axvline(x=th, color="r", label="threshold")
        plt.legend()
        plt.title("opacity histogram")
        plt.xlabel("opacity")
        plt.ylabel("count")

        plt.show()

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(data.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(data.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(data.elements[0]["f_dc_2"])
    max_sh_degree = 3
    extra_f_names = [
        p.name for p in data.elements[0].properties if p.name.startswith("f_rest_")
    ]
    extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
    if len(extra_f_names) == 3 * (max_sh_degree + 1) ** 2 - 3:
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(data.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape(
            (features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1)
        )
        print("features_extra", extra_f_names)

    else:
        features_extra = np.zeros((xyz.shape[0], 1))
    scale_names = [
        p.name for p in data.elements[0].properties if p.name.startswith("scale_")
    ]
    scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(data.elements[0][attr_name])
    # pdb.set_trace()
    scales = np.exp(scales)
    print("scales:", scale_names)
    norm = np.linalg.norm(scales, axis=1)
    print("scales norm", norm)
    print("scales min", norm.min)
    if histogram:
        fig, axs = plt.subplots(1, 3)

        plt.legend()
        plt.title("scales norm")

        plt.hist(scales[:, 0], label="x", histtype="step", bins=10)
        plt.hist(scales[:, 1], label="y", histtype="step", bins=10)
        try:
            plt.hist(scales[:, 2], label="z", histtype="step", bins=100)
        except IndexError:
            pass
        plt.legend()
        plt.title("scales norm")
        plt.show()

    rot_names = [
        p.name for p in data.elements[0].properties if p.name.startswith("rot")
    ]
    rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))

    print("rots:", rot_names)

    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(data.elements[0][attr_name])
    rots = torch.nn.functional.normalize(torch.from_numpy(rots)).numpy()
    print(np.linalg.norm(rots, axis=1))
    print(xyz.shape, opacity.shape, scales.shape, rots.shape)
    mask = opacity > th
    return (
        xyz[mask, :],
        opacity[mask],
        scales[mask, :],
        rots[mask, :],
        features_dc[mask, :, :],
        features_extra[mask, :],
    )


def meshify_top_points(centroids, opacity, scales, rots, n=0.1) -> trimesh.Trimesh:
    """
    Create mesh of the top n% gaussians based on opacity
    Parameters
    ----------
    centroids : np.ndarray
        The centroids of the gaussians [n, 3]
    opacity : np.ndarray
        The opacity of the gaussians [n, 1]
    scales : np.ndarray
        The scales of the gaussians [n, 3]
    rots : np.ndarray
        The rotations of the gaussians [n, 4]
    n : float
        The percentage of the gaussians to use
    Returns
    -------
    trimesh.Trimesh
        The mesh of the top n% gaussians
    """

    idx = np.argsort(opacity)
    mask = idx[len(opacity) - int(n * len(opacity)) :]  # get the top n%
    plt.plot(opacity[mask], label="opacity_sorted_torch")
    mesh = test_meshify(
        scales[idx[mask], :],
        centroids[idx[mask], :],
        rots[idx[mask], :],
        n_theta=8,
        n_sigma=8,
    )

    return mesh


def meshify_points(centroids, scales, rots):
    """
    Create a mesh from the centroids, scales and rotations
    """

    # Create a list of spheres
    from tqdm import tqdm

    for i in tqdm(range(centroids.shape[0])):
        # Scale the sphere
        mesh = ellipsoid_mesh(scales[i, :], theta_points=5, sigma_points=5)
        rot = trimesh.transformations.quaternion_matrix(rots[i, :])[:3, :3]
        # print(rot)
        mesh.vertices = mesh.vertices @ rot.T
        mesh.vertices += centroids[i, :]
        if i == 0:
            vertices = mesh.vertices
            faces = mesh.faces
        else:
            vertices = np.vstack((vertices, mesh.vertices))
            faces = np.vstack((faces, mesh.faces + i * mesh.vertices.shape[0]))
    trimesh.export("results/mesh_example.ply", trimesh.Trimesh(vertices, faces))
    return vertices, faces


def load_colmap_points(path: str) -> o3d.geometry.PointCloud:
    """
    Load the 3D points resulting from colmap

    POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
    """
    data = np.loadtxt(path, usecols=(1, 2, 3, 4, 5, 6))
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    r = data[:, 3] / 255
    g = data[:, 4] / 255
    b = data[:, 5] / 255

    xyzrgb = np.stack((x, y, z, r, g, b), axis=1)
    return xyzrgb


def load_gsplat(
    path: str, th: float = 0.0, histogram=False
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """
    Load the gaussian points from a pt file
    Parameters
    ----------
    path: str
        Path to the pt file
    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        - xyz: np.ndarray
            the center of the gaussians [n, 3]
        - opacity: np.ndarray
            the opacity of the gaussians [n, 1]
        - scales: np.ndarray
            the scales of the gaussians [n, 3]
        - rots: np.ndarray
            the rotations of the gaussians [n, 4] [w, x, y, z]
        - features_dc: np.ndarray
            the features of the gaussians [n, 3, 1]
        - features_extra: np.ndarray
            the extra features of the gaussians [n, 3*(max_sh_degree + 1) ** 2 - 3]
    """
    saved = torch.load(path, weights_only=True, map_location="cuda:0")
    splats = saved["splats"]
    means = splats["means"]
    opacities = Sigmoid()(splats["opacities"])
    quat = torch.nn.functional.normalize(splats["quats"])
    scales = torch.exp(splats["scales"])
    sh0 = splats["sh0"]
    shn = splats["shN"]

    if histogram:
        plt.hist(opacities.cpu().numpy(), bins=20)
        plt.title("opacity histogram")
        plt.show()
        scales_norm = torch.norm(scales, dim=1)
        plt.hist(scales_norm.cpu().numpy(), bins=100, log=True)
        plt.xscale("log")
        plt.title("scales histogram")
        plt.show()

    mask = opacities > th
    return (
        means[mask, :],
        opacities[mask],
        scales[mask, :],
        quat[mask, :],
        sh0[mask, :],
        shn[mask, :],
    )


if __name__ == "__main__":
    import os
    from collections import defaultdict
    from dataclasses import dataclass
    from typing import Dict, List, Optional, Tuple, Union

    import tyro
    # from nerf_tools.utils.utils import load_from_json
    from ellipsoids.meshes import test_disk_meshify, test_meshify, to_o3d

    C0 = 0.28209479177387814

    def SH2RGB(sh):
        return sh * C0 + 0.5

    from remo_splat import configs

    # config = tyro.cli(configs.ExampleBookshelf)
    @dataclass
    class Config:
        file: str = ""
        hist: bool = True
        is_3D: bool = True
        gsplat: bool = True
        scene: str = ""

    import warp as wp

    wp.init()

    config = tyro.cli(Config)
    # Load the data
    if config.gsplat:
        xyz, opacity, scales, rots, sh0, _ = load_gsplat(config.file, config.hist)
        rgb = SH2RGB(sh0).squeeze()
        # xyz = xyz.cpu().numpy()
        # opacity = opacity.cpu().numpy()
        # scales = scales.cpu().numpy()
        # rots = rots.cpu().numpy()
        # idx = np.argsort(opacity)
        idx = torch.argsort(opacity)

    else:
        xyz, opacity, scales, rots, features, extra_f = load_points(
            config.file, th=0.0, histogram=config.hist
        )
        rgb = features[:, :, 0]

        idx = np.argsort(opacity)

        scales = scales.astype(np.float32)
        xyz = torch.from_numpy(xyz).float().cuda()
        opacity = torch.from_numpy(opacity).float().cuda()
        scales = torch.from_numpy(scales).float().cuda()
        rots = torch.from_numpy(rots).float().cuda()
        rgb = torch.from_numpy(rgb).float().cuda()

    if config.is_3D:
        meshify = test_meshify
        # save each 0.1 opacity increments in differnet meshes:
        for i in range(10):
            mask = (i * 0.1 < opacity) & (opacity < i + 1 * 0.1)
            mesh = meshify(
                scales[mask, :],
                xyz[mask, :],
                rots[mask, :],
                rgb[mask, :],
                n_theta=8,
                n_sigma=8,
            )
            mesh.export(f"results/{config.scene}_{i}.ply")
    else:
        # Save all the points
        mesh = test_disk_meshify_from_gsplat(scales, xyz, rots, color=rgb, n_theta=8)
        mesh.export(f"results/{config.scene}.ply")
    # if len(scales) < 500000:
    #     # Save all the points
    #     mesh = test_meshify(scales, xyz, rots, rgb, n_theta=8, n_sigma=8)
    #     mesh.export(f"results/{config.scene}.ply")
