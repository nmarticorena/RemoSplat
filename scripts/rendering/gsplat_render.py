import matplotlib.pyplot as plt
import numpy as np
import torch
from gsplat.rendering import rasterization

device = "cuda" if torch.cuda.is_available() else "cpu"
import os
import time

import tyro

from remo_splat import configs as config
from remo_splat.configs import ExampleRealGSplat, S12Example
from remo_splat.loader import load_gsplat, load_points_tensor
from remo_splat.rendering import normals_from_depth

# args = S12Example()
gsplatargs = tyro.cli(config.gs.GSplatLoader)

# means, opacities, scales, quats, colors , sh= load_points_tensor(args.file)
means, opacities, scales, quats, colors, sh = load_gsplat(
    gsplatargs.file, histogram=gsplatargs.hist
)
__import__("pdb").set_trace()


# colors.squeeze_(1)
colors = torch.cat((colors, sh), dim=1)
# colors = sh
# define cameras
Poses = gsplatargs.dataset_helper.get_transforms_cv2()
Ks = (
    torch.from_numpy(gsplatargs.dataset_helper.get_camera_intrinsic())
    .to(device)
    .float()[None, :, :]
)

j = 0
os.makedirs(f"results/renders/{gsplatargs.scene}", exist_ok=True)

for T_WC in Poses:
    T_CW = np.linalg.inv(T_WC)
    viewmats = torch.from_numpy(T_CW).to(device).float()[None, :, :]

    real_color = gsplatargs.dataset_helper.sample_rgb(j)
    real_depth = gsplatargs.dataset_helper.sample_metric_depth(j)

    depth_int = gsplatargs.dataset_helper.sample_depth(j)

    real_normals = (
        torch.from_numpy(gsplatargs.dataset_helper.depth_original_scale(real_depth))
        .cuda()
        .float()
    )

    width, height = gsplatargs.dataset_helper.get_image_size()
    # render
    ti = time.time()
    render, alphas, meta = rasterization(
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
        Ks,
        width,
        height,
        sh_degree=3,
        render_mode="RGB+D",
    )
    #
    r_colors = render[..., :3]
    depth = render[..., 3]
    depth.squeeze_(0)

    normals = normals_from_depth(depth, Ks[0])
    normals = normals.cpu().numpy()
    print("Time", time.time() - ti)
    # plt.imshow(normals)
    # plt.show()

    print(colors.shape, alphas.shape)
    r_colors = r_colors.clamp(0, 1)
    r_colors = (r_colors * 255 * alphas).to(torch.uint8)

    fix, ax = plt.subplots(2, 3, figsize=(15, 10))
    ax[0, 0].imshow(r_colors[0].cpu().numpy())
    ax[0, 0].set_title("Rendered Image")
    ax[0, 1].imshow(real_color)
    ax[0, 1].set_title("Real Image")
    ax[0, 2].imshow(np.abs(r_colors[0].cpu().numpy() - real_color))
    ax[0, 2].set_title("Image Error")

    ax[1, 0].imshow(depth.cpu().numpy())
    ax[1, 0].set_title("Rendered Depth")
    ax[1, 1].imshow(real_depth)
    ax[1, 1].set_title("Real Depth")
    ax[1, 2].imshow(np.abs(depth.cpu().numpy() - real_depth))
    ax[1, 2].set_title("Depth Error")

    plt.savefig(f"results/renders/{gsplatargs.scene}/render_{j}.png")
    plt.cla()
    plt.clf()
    j += 1
