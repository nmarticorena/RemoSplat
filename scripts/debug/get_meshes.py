# To test how different is the information stored
# using the 3d and 2d gaussian

import warp as wp
from gsplat.rendering import rasterization, rasterization_2dgs

from remo_splat.configs.gs import ExampleRealGSplat as Config
from remo_splat.ellipsoids.meshes import (test_disk_meshify_from_gsplat,
                                                 test_meshify_from_gsplat)
from remo_splat.gaussians_3d import Gaussians3D
from remo_splat.loader import load_gsplat

wp.init()

config_2D = Config(is_3D=False, step=300)

# mean, opacity, scales, rots, features, features_extra = load_gsplat(config_3D.file)
# mean_mcmc, opacity_mcmc, scales_mcmc, rots_mcmc, features_mcmc, features_extra_mcmc = (
#     load_gsplat(config_mcmc.file)
# )


mean_2D, opacity_2D, scales_2D, rots_2D, features_2D, features_extra_2D = load_gsplat(
    config_2D.file
)


# mesh_3d = test_meshify_from_gsplat(scales, mean, rots, n_theta=7, n_sigma=7)
# mesh_mcmc = test_meshify_from_gsplat(
#     scales_mcmc, mean_mcmc, rots_mcmc, n_theta=7, n_sigma=7
# )
mesh_2d = test_disk_meshify_from_gsplat(scales_2D, mean_2D, rots_2D)
#
# g = Gaussians3D()
# mesh_gaussians = g.load(config_3D.file, create_mesh=True)
# mesh_gaussians.export("results/bookshelf_gaussians.ply")

# mesh_3d.export("results/replica_3d.ply")
mesh_2d.export("results/printer_2.ply")
# mesh_mcmc.export("results/replica_mcmc.ply")
#
