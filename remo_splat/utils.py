import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Literal

import cv2
import einops
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
import torch.utils.dlpack
import trimesh
from gsplat.rendering import (rasterization, rasterization_2dgs,
                              rasterization_2dgs_inria_wrapper)
from kornia.filters import MedianBlur

from remo_splat.configs.gs import GSLoader, GSplatLoader
from remo_splat.loader import load_gsplat


def to_o3d(mesh: trimesh.Trimesh, color=None) -> o3d.geometry.TriangleMesh:
    o3d_mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(mesh.vertices),
        triangles=o3d.utility.Vector3iVector(mesh.faces),
    )
    if color is not None:
        o3d_mesh.paint_uniform_color(color)
    return o3d_mesh


@dataclass
class CameraParams:
    width: int
    height: int
    focal_length: float = field(default=0, repr=False)
    fov: float = field(default=0, repr=False)

    def __post_init__(self):
        # print(
        #     f"CameraParams created with: width={self.width}, height={self.height}, fov={self.fov}, focal_length={self.focal_length}"
        # )
        if self.focal_length and self.fov:
            raise ValueError("Provide either 'focal_length' or 'fov', not both.")
        if self.focal_length:
            self.fov = 2 * np.rad2deg(np.arctan(self.width / (2 * self.focal_length)))
        elif self.fov:
            self.focal_length = self.width / (2 * np.tan(np.deg2rad(self.fov) / 2))

        self.K = (
            torch.tensor(
                [
                    [self.focal_length, 0, self.width / 2],
                    [0, self.focal_length, self.height / 2],
                    [0, 0, 1],
                ]
            )
            .cuda()
            .float()
        )

    @property
    def aspect_ratio(self) -> float:
        """Compute the aspect ratio of the camera."""
        return self.width / self.height

    def set_from_focal_length(self, focal_length: float):
        """Update the camera parameters using the focal length."""
        self.focal_length = focal_length
        self.fov = 2 * np.rad2deg(np.arctan(self.width / (2 * focal_length)))

    def set_from_fov(self, fov: float):
        """Update the camera parameters using the field of view."""
        self.fov = fov
        self.focal_length = self.width / (2 * np.tan(np.deg2rad(fov) / 2))

    def set_width(self, width: int):
        """Update the camera parameters using the width."""
        self.width = width
        self.focal_length = 2 * self.width / np.tan(np.deg2rad(self.fov) / 2)
        self.fov = 2 * np.rad2deg(np.arctan(width / (2 * self.focal_length)))
        self.K[0, 2] = width / 2
        self.K[0, 0] = self.focal_length

    def set_fov(self, fov: float):
        """Update the camera parameters using the field of view."""
        self.fov = fov
        self.focal_length = self.width / (2 * np.tan(np.deg2rad(fov) / 2))
        self.K[0, 0] = self.focal_length
        self.K[1, 1] = self.focal_length

    def set_height(self, height: int):
        """Update the camera parameters using the height."""
        self.height = height
        self.focal_length = 2 * height / np.tan(np.deg2rad(self.fov) / 2)
        self.fov = 2 * np.rad2deg(np.arctan(height / (2 * self.focal_length)))
        self.K[1, 2] = height / 2
        self.K[1, 1] = self.focal_length


def get_camera_lines(params: CameraParams, T_CW: torch.Tensor) -> o3d.geometry.LineSet:
    """
    Get the camera lines for the visualization
    Arguments
    ---------
    params: CameraParams
        Camera parameters
    T_CW: torch.Tensor
        Camera pose [N,4,4]
    Returns
    -------
    camera_lines: o3d.geometry.LineSet
    """
    if T_CW.dim() == 2:
        T_CW = T_CW.unsqueeze(0)
    width = params.width
    height = params.height
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=width, height=height, intrinsic_matrix=params.K.cpu().numpy()
    )
    lines = o3d.geometry.LineSet()
    for pose in T_CW:
        lines += o3d.geometry.LineSet.create_camera_visualization(
            intrinsic, pose.cpu().numpy(), scale=0.1
        )

    return lines


def get_line(origin, direction):
    """
    Compute the line
    Arguments
    ---------
    origin: torch.tensor
        points origins [n,3]
    direction: torch.tensor
        gradient info [n,3]
    """
    dir = direction
    p = origin
    p = [p, p + dir]
    return p


def draw_line_o3d(origin, direction, length=1, color=[0, 1, 0]):
    """
    Draw a line in open3d
    Arguments
    ---------
    origin: torch.tensor
        points origins [n,3]
    direction: torch.tensor
        gradient info [n,3]
    length: float
        step default = 1
    color: list
        color of the line
    """
    p = origin.detach().cpu().numpy()
    dir = (direction * length).detach().cpu().numpy()

    p = [p, p + dir]
    line = o3d.geometry.LineSet()
    line.points = o3d.utility.Vector3dVector(p)
    line.lines = o3d.utility.Vector2iVector(np.array([[0, 1]]))
    line.colors = o3d.utility.Vector3dVector([color])

    return line


def uniform(shape, r, device="cuda"):
    sample = (r[0] - r[1]) * torch.rand(*shape, device=r.device) + r[1]
    return sample.to(device)


def draw_line(x, g, delta=0.1, label="direction", color="g"):
    """
    Arguments
    ---------
    x: torch.tensor
        points origins [n,2]
    g: torch.tensor
        gradient info [n,2]
    delta: float
        step default = 0.1
    """

    p = x.detach().cpu().numpy()
    dir = (g * delta).detach().cpu().numpy()
    plt.quiver(p[:, 0], p[:, 1], dir[:, 0], dir[:, 1], label=label, color=color)
    return


def draw_surface(x, g, d, label="", color="g"):
    """
    Draw a point in the surface after moving:
    x = x - g*d
    Arguments
    ---------
    x: torch.tensor
        point origins [n,2]
    g: torch.tensor
        gradient info [n,2]
    d: torch.tensor
        distance measured [n,1]
    """
    p_origin = x.detach().cpu().numpy()

    p_reached = (x - g * d).detach().cpu().numpy()

    x = np.array([p_origin[:, 0], p_reached[:, 0]])
    y = np.array([p_origin[:, 1], p_reached[:, 1]])

    plt.plot(x, y, label=label, color=color)


class PanoramicViewer:
    def __init__(self, width, height, num_frames, upscale=1, name="test"):
        self.n_images = 2
        self.width = width
        self.height = height
        self.upscale = upscale
        self.num_frames = num_frames
        self.set_image()
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        self.name = name

    def set_image(self):
        self.final_image = np.zeros(
            (self.height * self.n_images * self.upscale, self.width * self.num_frames * self.upscale, 3), dtype=np.uint8
        )

    def render(self, color, depth, extras,meta: Optional[Dict] = None, save_path:Optional[str] = None):
        """
        Arguments
        ---------
        color: torch.tensor
            color image [n,h,w,3]
        depth: torch.tensor
            depth image [n,h,w] in meters
        """
        self.n_images = 2 + len(extras.keys())
        self.set_image()
        n, h, w, _ = color.shape
        flattened_image = (
            torch.cat([color[i] for i in range(n)], dim=1).detach().cpu().numpy()
        )

        depth[depth > 10] = 0
        flattened_depth = torch.cat([depth[i] for i in range(n)], dim=1)

        flattened_image = cv2.cvtColor(flattened_image, cv2.COLOR_RGB2BGR)

        depth = (
            (flattened_depth - flattened_depth.min())
            / (flattened_depth.max() - flattened_depth.min())
            * 255
        ).clip(0, 255)
        depth = depth.detach().cpu().numpy().astype(np.uint8)
        depth = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
        depth = cv2.resize(depth, (0, 0), fx=self.upscale, fy=self.upscale)
        flattened_image = cv2.resize(
            flattened_image, (0, 0), fx=self.upscale, fy=self.upscale
        )

        self.final_image[: h * self.upscale, :] = flattened_image * 255
        self.final_image[h * self.upscale :h * self.upscale * 2 , :] = depth

        img_index = 2
        for keys in extras.keys():
            _img = extras.get(keys)
            self.final_image[h * self.upscale*img_index: h*self.upscale *(img_index + 1),:] = self.flat_img(_img)
            img_index+=1

        cv2.imshow(self.name, self.final_image)
        key = cv2.waitKey(100)
        return

    def save(self, path):
        """
        Save the final image to the path
        """
        cv2.imwrite(path, self.final_image)
        print(f"Image saved to {path}")

    def flat_img(self, tensor_img):
        n = tensor_img.shape[0]
        tensor_img *= 255
        tensor_img = tensor_img.to(torch.uint8)
        flattened_tensor_img = (
            torch.cat([tensor_img[i] for i in range(n)], dim = 1).detach().cpu().numpy()
        )
        flattened_tensor_img = cv2.resize(
            flattened_tensor_img, (0, 0), fx=self.upscale, fy=self.upscale
        )
        if len(flattened_tensor_img.shape) == 2:
            flattened_tensor_img = np.repeat(flattened_tensor_img[...,None], 3, axis = -1)
        return flattened_tensor_img

class Render:
    def __init__(
            self,
            config: GSLoader,
            far_plane:float = 0.9,
            depth_mode: Literal["expected", "median"] = "median",
            alpha_cutoff: float = 0.7,
            add_noise: int = 0,
            median_th: float = 0.5,
            eps2d: float = 0.0,
            max_radius: float = 100,
    ):
        if config is not None:
            self.load(config, add_noise=add_noise)
            self.is_3D = config.is_3D
            self.inria = config.inria
        if self.is_3D:
            self.rasterizer = rasterization
        else:
            print("using 2D")
            if self.inria:
                self.rasterizer = rasterization_2dgs_inria_wrapper
            else:
                self.rasterizer = rasterization_2dgs


        self.config = config
        self.far_plane = far_plane
        self.alpha_cutoff = alpha_cutoff
        self.depth_mode = depth_mode
        self.median_th = median_th
        self.eps2d = eps2d
        self.max_radius = max_radius

    def set_alpha_cutoff(self, alpha_cutoff: float):
        print(f"Setting alpha cutoff to {alpha_cutoff}")
        self.alpha_cutoff = alpha_cutoff

    def set_eps2d(self, eps2d: float):
        print(f"Setting eps2d to {eps2d}")
        self.eps2d = eps2d

    def set_max_radius(self, max_radius: float):
        print(f"Setting max_radius to {max_radius}")
        self.max_radius = max_radius

    def get_points(self, T_CW, depth_img, color, cam_params: Optional[CameraParams] = None):
        """
        Get pcd of the projected point cloud
        Arguments
        ---------
        T_CW: torch.tensor
            Camera pose [N,4,4]
        depth_img: torch.tensor
            Depth image [N,H,W]
        Ks: torch.tensor
            Camera intrinsics [N,3,3]
        """
        T_WC = T_CW.inverse()
        if cam_params is None:
            Ks = self.Ks.repeat(T_CW.shape[0], 1, 1)
        else:
            Ks = cam_params.K.repeat(T_CW.shape[0], 1, 1)
        if len(depth_img.shape) != 3:
            depth_img = depth_img.unsqueeze(0)
        width = depth_img.shape[-1]
        height = depth_img.shape[-2]
        pixel_grid = self.get_pixel_grid(T_CW.shape[0], width, height)
        coords = torch.einsum("nij,n...j->n...i", Ks.inverse(), pixel_grid)

        p_c = coords * depth_img.unsqueeze(-1)
        p_c = torch.cat((p_c, torch.ones_like(p_c[:, :, :, 0:1])), dim=-1)
        p_w_h = torch.einsum("nij,nhwj->nhwi", T_WC, p_c)

        p_w = p_w_h[..., :3] / (p_w_h[..., 3:4] + 1e-8)
        p_w = p_w[~torch.isnan(p_w)]
        p_w = p_w.reshape(-1, 3)
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(p_w.cpu().numpy())
        t_pcd = o3d.core.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(p_w))
        pcd = o3d.t.geometry.PointCloud(t_pcd)
        # color = np.random.rand(p_w.shape[0], 3)
        # pcd.point.colors = o3d.utility.Vector3dVector(np.asarray(color.cpu().numpy().reshape(-1, 3)))
        pcd_cpu = pcd.to(o3d.core.Device("CPU:0"))
        return pcd_cpu

    def get_pixel_grid(self, n_cameras, width, height):
        """
        Get the pixel grid for the cameras

        Returns:
            homogeneous_coords (torch.tensor): [N, H, W, 3] Homogeneous pixel coordinates
        """

        # Create pixel coordinate grid
        u, v = torch.meshgrid(
            torch.arange(width, device="cuda"),
            torch.arange(height, device="cuda"),
            indexing="xy",  # Matches pixel indexing convention
        )

        # Convert to homogeneous image coordinates (u, v, 1)
        homogeneous_coords = torch.stack(
            (u, v, torch.ones_like(u, device="cuda")), dim=-1
        )
        # Stack them for each camera
        homogeneous_coords = (
            homogeneous_coords.unsqueeze(0).expand(n_cameras, -1, -1, -1).float()
        )
        return homogeneous_coords

    def render_all(
        self,
        T_CW,
        cam_params: Optional[CameraParams] = None,
        Ks: Optional[torch.Tensor] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        extras_names: List[str] = [],
        median_filter = False
    ) -> tuple[torch.Tensor, torch.Tensor, Dict, Dict]:
        ti = time.time()
        if cam_params is not None:
            Ks = cam_params.K.repeat(T_CW.shape[0], 1, 1)
            width = cam_params.width
            height = cam_params.height
        elif Ks is None or width is None or height is None:
            Ks = self.Ks.repeat(T_CW.shape[0], 1, 1)
            width = self.W
            height = self.H
        extras = {}
        if self.is_3D:
            rendered = self.rasterizer(
                self.means,
                self.quats,
                self.scales,
                self.opacities,
                self.colors,
                T_CW,
                Ks,
                width,
                height,
                sh_degree=3,
                render_mode="RGB+ED",
                near_plane=0.02,
                far_plane=self.far_plane,  # Influence distance
                packed=False,
            )
            render, alphas, meta = rendered
            r_colors = render[..., :3]
            depth = render[..., 3]
            breakpoint()
            max_radius = meta
            if "alpha" in extras_names:
                extras["alpha"] = alphas.detach()
        else:
            rendered = self.rasterizer(
                self.means,
                self.quats,
                self.scales,
                self.opacities,
                self.colors,
                T_CW,
                Ks,
                width,
                height,
                sh_degree=3,
                render_mode="RGB+ED" if self.depth_mode == "expected" else "RGB+D",
                depth_mode=self.depth_mode,
                near_plane=0.01060660171,
                max_radius = self.max_radius,
                far_plane=self.far_plane,  # Influence distance
                packed=False, # Need to update the cuda kernel to also work in this mode
                eps2d = self.eps2d,
                # eps2d = 0.0,
                median_cutoff = self.median_th,
            )
            if self.inria:
                output, meta = rendered

                render = output[0]
                alphas = output[1]
                depth = render[..., 3]
            else:
                (
                    render,
                    depth,
                    alphas,
                    normals,
                    surf_normals,
                    distorsion,
                    median,
                    meta,
                ) = rendered
                # detach all
                normals.detach()
                # surf_normals.detach()
                distorsion.detach()
                median.detach()
                render.detach()

                # depth = depth.squeeze(-1)  # [B,H,W]
                depth = median.squeeze(-1)


                # depth[alphas[:, :, :, -1] < 0.8] = torch.inf
                if "alpha" in extras_names:
                    extras["alpha"] = alphas.detach()
                if "normal" in extras_names:
                    extras["normal"] = normals
                if "surf_normals" in extras_names:
                    extras["surf_normals"] = surf_normals.detach()

        depth[alphas[:, :, :, -1] < self.alpha_cutoff] = torch.inf
        # When doing filter
        if median_filter:
            pass # No longer required
            # depth = depth.unsqueeze(1).squeeze(-1).detach()  # [B,1,H,W]
            # # after_filter = self.blur(depth).squeeze().detach()  # [B,H,W]
            # after_filter[after_filter == 0] = torch.inf
            # after_filter[after_filter.isnan()] = torch.inf
            # depth = after_filter
        r_colors = render[...,:3]
        return r_colors.detach(), depth.detach(), extras, meta

    def render(
        self,
        T_CW,
        cam_params: Optional[CameraParams] = None,
        Ks: Optional[torch.Tensor] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ):
        rgb, depth, _, _ = self.render_all(T_CW, cam_params, Ks, width, height, extras_names= [], median_filter = False)
        return rgb, depth

    def render_debug(
        self,
        T_CW,
        cam_params: Optional[CameraParams] = None,
        Ks: Optional[torch.Tensor] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ):
        return self.render_all(T_CW, cam_params, Ks, width, height, [])

    def debug_depth(
        self,
        T_CW: Optional[torch.Tensor] = None,
        depth: Optional[torch.Tensor] = None,
        cam_params: Optional[CameraParams] = None,
        Ks: Optional[torch.Tensor] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ):
        """
        Render the depth image
        """
        if depth is None:
            _, _depth, _, _ = self.render_all(T_CW, cam_params, Ks, width, height, extras_names= ["depth"], median_filter = False)
        else:
            _depth = depth
        _depth = _depth.squeeze(0)
        _depth[_depth > self.far_plane] = 0  # Set far plane to 0
        _depth = (
            (_depth - _depth.min())
            / (_depth.max() - _depth.min())
            * 255
        ).clip(0, 255)
        _depth = _depth.detach().cpu().numpy().astype(np.uint8)
        _depth = cv2.applyColorMap(_depth, cv2.COLORMAP_JET)
        return _depth

    def debug_rgb(
        self,
        T_CW: Optional[torch.Tensor] = None,
        rgb: Optional[torch.Tensor] = None,
        cam_params: Optional[CameraParams] = None,
        Ks: Optional[torch.Tensor] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ):
        """
        Render the RGB image
        """
        if rgb is None:
            _rgb, _, _, _ = self.render_all(T_CW, cam_params, Ks, width, height, median_filter = False)
        else:
            _rgb = rgb.squeeze(0)
        _rgb = _rgb.detach().cpu().numpy()
        _rgb = (_rgb * 255).astype(np.uint8)
        return cv2.cvtColor(_rgb, cv2.COLOR_RGB2BGR)

    def debug_opacity(
        self,
        T_CW: Optional[torch.Tensor] = None,
        opacities: Optional[torch.Tensor] = None,
        cam_params: Optional[CameraParams] = None,
        Ks: Optional[torch.Tensor] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ):
        """
        Render the opacity image
        """
        if opacities is None:
            _, _, extras, _ = self.render_all(T_CW, cam_params, Ks, width, height, extras_names=["alpha"], median_filter=False)
            _opacities = extras["alpha"]
        else:
            _opacities = opacities
        _opacities = _opacities.squeeze(0)
        return (_opacities * 255).detach().cpu().numpy().astype(np.uint8)

    def sample_render(self, views):
        if views is int:
            views = [views]

        T_WC = np.array(self.config.dataset_helper.get_transforms_cv2(views))

        T_CW = torch.from_numpy(T_WC).inverse().cuda().float()  # [1,4,4]
        Ks = self.Ks.repeat(len(views), 1, 1)
        return self.render(T_CW, Ks, self.W, self.H)

    def load(
        self, config: Optional[GSLoader] = None, custom_data: Optional[dict] = None, add_noise: int = 0
    ):
        """
        Load the Gaussian splat data from the config or custom data.
        Arguments
        ---------
        config: GSLoader
            Configuration object containing the file path and other parameters.
        custom_data: dict
            Custom data dictionary containing the means, opacities, scales, quats, colors, and is_3D flag.
        add_noise: int
            Amount of extra low opacity elliposids added to the scene.
        """
        print("GSplat Loading")
        if not custom_data and config is not None:
            means, opacities, scales, quats, sh0, sh = load_gsplat(
                config.file, histogram=config.hist
            )

            colors = torch.cat((sh0, sh), dim=1)

            is_3D = config.is_3D
        elif custom_data is not None:
            means:torch.Tensor = custom_data.get("means")
            opacities = custom_data.get("opacities")
            scales = custom_data.get("scales")
            quats = custom_data.get("quats")
            colors = torch.cat(
                (custom_data.get("features_dc"), custom_data.get("features_extra")),
                dim=1,
            )
            is_3D = custom_data.get("is_3D")
        else:
            raise ValueError("Either config or custom_data must be provided")

        if add_noise > 0:
            aabb_min = means.min(dim=0).values
            aabb_max = means.max(dim=0).values

            extra_means = uniform((add_noise, 3), torch.stack((aabb_min, aabb_max)), device=means.device)
            means = torch.cat((
                    means,
                    uniform((add_noise, 3), torch.stack((aabb_min, aabb_max)), device=means.device),

                ),
                dim=0,
            )
            opacities = torch.cat((
                opacities,
                torch.ones((add_noise), device=opacities.device) * 0.06,
            ),
                dim=0,
            )
            scales = torch.cat(
                (scales, torch.ones((add_noise, 3), device=scales.device) * 0.1), dim=0
            )
            extra_rots = torch.randn((add_noise, 4), device=quats.device)
            extra_rots = torch.nn.functional.normalize(extra_rots, dim=-1)
            quats = torch.cat(
                (quats,extra_rots), dim=0
            )
            colors = torch.cat(
                (colors, torch.zeros((add_noise,colors.shape[-2], colors.shape[-1]), device=colors.device)),
                dim=0,
            )

        self.means = means.contiguous()
        self.opacities = opacities.contiguous()
        # self.opacities = torch.ones_like(self.opacities)
        self.scales = scales.contiguous()
        if not is_3D:
            self.scales[..., 2] = 0
        self.quats = quats.contiguous()
        self.colors = colors.contiguous()

        print("A total of ", self.means.shape[0], "gaussian splats loaded")

        if not custom_data:
            self.T_WC = config.dataset_helper.get_transforms_cv2()
            Ks = config.dataset_helper.get_camera_intrinsic()
            self.Ks = torch.from_numpy(Ks).float().cuda()[None, ...]
            self.W, self.H = config.dataset_helper.get_image_size()

    def get_mesh(self):
        from nerf_tools.utils.math import quaternion_to_rotation_matrix
        from nerf_tools.utils.meshes import create_gs_mesh


        rots = quaternion_to_rotation_matrix(self.quats)

        # sh = self.colors.mean(dim=1)  # Assuming colors are in the first 3 channels
        sh = self.colors[:,0,:]  # Use only the first 4 channels for SH0
        C0 = 0.28209479177387814

        def SH2RGB(sh):
            return sh * C0 + 0.5

        rgb = SH2RGB(sh).unsqueeze(1)  # Add a channel dimension

        print(f"Meshing {self.means.shape[0]} splats with")
        mesh = create_gs_mesh(self.means.cpu().numpy(),
                            rots.cpu().numpy(),
                            self.scales.cpu().numpy(),
                            # colors = np.ones_like(self.means.cpu().numpy()) * 0.5,
                            colors = rgb.squeeze().cpu().numpy(),
                            res = 4)
        return mesh


class ManualRender(Render):
    def __init__(
        self,
        means,
        opacities,
        scales,
        quats,
        features_dc,
        features_extra,
        is_3D=True,
        inria=False,
    ):
        custom_data = {
            "means": means.cuda(),
            "opacities": opacities.cuda(),
            "scales": scales.cuda(),
            "quats": quats.cuda(),
            "features_dc": features_dc.cuda(),
            "features_extra": features_extra.cuda(),
            "is_3D": is_3D,
        }
        self.is_3D = is_3D
        self.inria = inria
        super().__init__(config=None)
        self.load(config=None, custom_data=custom_data)
