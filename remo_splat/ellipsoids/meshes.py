import time

import numpy as np
import open3d as o3d
import torch
import trimesh
import warp as wp


@wp.kernel
def disk_meshify(
    sigmas: wp.array(dtype=wp.vec2f),  # type: ignore
    mu: wp.array(dtype=wp.vec3f),  # type:ignore
    rots: wp.array(dtype=wp.quat),  # type:ignore
    verts: wp.array(dtype=wp.vec3f),  # type:ignore
    faces: wp.array(dtype=wp.vec3i),  # type:ignore
    x_s: wp.array(dtype=wp.float32, ndim=1),  # type:ignore
    y_s: wp.array(dtype=wp.float32, ndim=1),  # type:ignore
    n_theta: wp.int32,
):
    i = wp.tid()  # iterate over ellipsoids

    transform = wp.transformation(pos=mu[i], rot=rots[i], dtype=wp.float32)
    face_ti = i * n_theta * 2
    vert_initial = i * (n_theta + 1)
    for ti in range(n_theta):
        x = x_s[ti] * sigmas[i][0]
        y = y_s[ti] * sigmas[i][1]
        z = 0.0
        verts[ti + vert_initial] = wp.transform_point(transform, wp.vec3f(x, y, z))
        next_id = vert_initial + (ti + 1) % n_theta
        faces[face_ti + ti * 2] = wp.vec3i(
            ti + vert_initial, next_id, n_theta + vert_initial
        )
        faces[face_ti + ti * 2 + 1] = wp.vec3i(
            n_theta + vert_initial, next_id, ti + vert_initial
        )
    verts[n_theta + vert_initial] = wp.transform_point(
        transform, wp.vec3f(0.0, 0.0, 0.0)
    )


@wp.func
def get_mesh(
    i: int,
    sigmas: wp.array(dtype=wp.vec3f),  # type: ignore
    mu: wp.array(dtype=wp.vec3f),  # type: ignore
    rots: wp.array(dtype=wp.quat),  # type: ignore
    verts: wp.array(dtype=wp.vec3f),  # type: ignore
    faces: wp.array(dtype=wp.vec4i),  # type: ignore
    x_s: wp.array(dtype=wp.float32, ndim=2),  # type: ignore
    y_s: wp.array(dtype=wp.float32, ndim=2),  # type: ignore
    z_s: wp.array(dtype=wp.float32, ndim=2),  # type: ignore
    n_theta: wp.int32,
    n_sigma: wp.int32,
):
    verts_i = i * (n_theta * n_sigma)
    faces_i = i * ((n_theta - 1) * (n_sigma - 1))
    face_counter = wp.int32(0)
    transform = wp.transformation(pos=mu[i], rot=rots[i], dtype=wp.float32)
    for ti in range(n_theta):
        for si in range(n_sigma):
            x = x_s[ti, si] * sigmas[i][0]
            y = y_s[ti, si] * sigmas[i][1]
            z = z_s[ti, si] * sigmas[i][2]

            verts[verts_i + ti * n_sigma + si] = wp.transform_point(
                transform, wp.vec3f(x, y, z)
            )

            if (ti < n_theta - 1) and (si < n_sigma - 1):
                faces[faces_i + ti * (n_theta - 1) + si] = wp.vec4i(
                    verts_i + ti * n_sigma + si,
                    verts_i + ti * n_sigma + si + 1,
                    verts_i + (ti + 1) * n_sigma + si + 1,
                    verts_i + (ti + 1) * n_sigma + si,
                )
                face_counter += 1


@wp.kernel
def meshify(
    sigmas: wp.array(dtype=wp.vec3f),  # type:ignore
    mu: wp.array(dtype=wp.vec3f),  # type:ignore
    rots: wp.array(dtype=wp.quat),  # type:ignore
    verts: wp.array(dtype=wp.vec3f),  # type:ignore
    faces: wp.array(dtype=wp.vec4i),  # type:ignore
    x_s: wp.array(dtype=wp.float32, ndim=2),  # type:ignore
    y_s: wp.array(dtype=wp.float32, ndim=2),  # type:ignore
    z_s: wp.array(dtype=wp.float32, ndim=2),  # type:ignore
    n_theta: wp.int32,
    n_sigma: wp.int32,
):
    i = wp.tid()
    get_mesh(i, sigmas, mu, rots, verts, faces, x_s, y_s, z_s, n_theta, n_sigma)


@wp.kernel
def meshify_3d(
    sigmas: wp.array(dtype=wp.vec3f),  # type:ignore
    mu: wp.array(dtype=wp.vec3f),  # type:ignore
    rots: wp.array(dtype=wp.quat),  # type:ignore
    e_color: wp.array(dtype=wp.vec3f),  # type:ignore
    verts: wp.array(dtype=wp.vec3f),  # type:ignore
    faces: wp.array(dtype=wp.vec3i),  # type:ignore
    colors: wp.array(dtype=wp.vec3f),  # type:ignore
    x_s: wp.array(dtype=wp.float32, ndim=2),  # type:ignore
    y_s: wp.array(dtype=wp.float32, ndim=2),  # type:ignore
    z_s: wp.array(dtype=wp.float32, ndim=2),  # type:ignore
    n_theta: wp.int32,
    n_sigma: wp.int32,
):
    i, ti, si = wp.tid()  # type:ignore
    verts_i = i * (n_theta * n_sigma)
    faces_i = i * ((n_theta - 1) * (n_sigma - 1)) * 2

    transform = wp.transformation(pos=mu[i], rot=rots[i], dtype=wp.float32)
    x = x_s[ti, si] * sigmas[i][0]
    y = y_s[ti, si] * sigmas[i][1]
    z = z_s[ti, si] * sigmas[i][2]

    verts[verts_i + ti * n_sigma + si] = wp.transform_point(
        transform, wp.vec3f(x, y, z)
    )

    if ti < n_theta - 1 and si < n_sigma - 1:
        faces[faces_i + (ti * (n_theta - 1) + si) * 2] = wp.vec3i(
            verts_i + ti * n_sigma + si,
            verts_i + ti * n_sigma + si + 1,
            verts_i + (ti + 1) * n_sigma + si + 1,
        )
        faces[faces_i + (ti * (n_theta - 1) + si) * 2 + 1] = wp.vec3i(
            verts_i + ti * n_sigma + si,
            verts_i + (ti + 1) * n_sigma + si + 1,
            verts_i + (ti + 1) * n_sigma + si,
        )

        colors[faces_i + (ti * (n_theta - 1) + si) * 2] = e_color[i]
        colors[faces_i + (ti * (n_theta - 1) + si) * 2 + 1] = e_color[i]


def disk_mesh(sigmas, n_theta=100):
    theta = np.linspace(0, 2 * np.pi, n_theta)
    x = np.cos(theta)
    y = np.sin(theta)

    vertices = np.zeros((n_theta + 1, 3))
    faces = []
    for i in range(n_theta):
        vertices[i, 0] = x[i] * sigmas[0]
        vertices[i, 1] = y[i] * sigmas[1]
        next_i = int((i + 1) % n_theta)

        faces.append([i, next_i, n_theta])
        faces.append([n_theta, next_i, i])
    return trimesh.Trimesh(vertices, faces)


def ellipsoid_mesh(
    sigmas: np.ndarray, theta_points=50, sigma_points=50
) -> trimesh.Trimesh:
    """
    Create a mesh of an ellipsoid
    :param sigmas: the sigmas of the ellipsoid
    :param theta_points: number of points in the theta direction
    :param sigma_points: number of points in the sigma direction
    :return: vertices, faces
    """
    # Create the mesh of an ellipsoid
    theta = np.linspace(0, 2 * np.pi, theta_points)
    sigma = np.linspace(0, np.pi, sigma_points)

    x = np.outer(np.cos(theta), np.sin(sigma))
    y = np.outer(np.sin(theta), np.sin(sigma))
    z = np.outer(np.ones(theta_points), np.cos(sigma))

    vertices = np.zeros((theta_points * sigma_points, 3))
    faces = []
    # Scale the ellipsoid
    for i in range(theta_points):
        for j in range(sigma_points):
            x[i, j] = x[i, j] * sigmas[0]
            y[i, j] = y[i, j] * sigmas[1]
            z[i, j] = z[i, j] * sigmas[2]

            # Create the mesh
            vertices[i * sigma_points + j, 0] = x[i, j]
            vertices[i * sigma_points + j, 1] = y[i, j]
            vertices[i * sigma_points + j, 2] = z[i, j]
            if i < theta_points - 1 and j < sigma_points - 1:
                faces.append(
                    [
                        i * sigma_points + j,
                        i * sigma_points + j + 1,
                        (i + 1) * sigma_points + j + 1,
                        (i + 1) * sigma_points + j,
                    ]
                )

    faces.append(
        [
            (theta_points - 1) * sigma_points,
            (theta_points - 1) * sigma_points + 1,
            sigma_points + 1,
            sigma_points,
        ]
    )
    print(faces)
    print(vertices)
    return trimesh.Trimesh(vertices, faces)


def to_o3d(mesh: trimesh.Trimesh, color=None) -> o3d.geometry.TriangleMesh:
    o3d_mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(mesh.vertices),
        triangles=o3d.utility.Vector3iVector(mesh.faces),
    )
    if color is not None:
        o3d_mesh.paint_uniform_color(color)
    return o3d_mesh


def test_disk_meshify(sigmas, mu, rots, n_theta=10):
    """
    Run the kernel that meshify the 2D gaussians
    Parameters
    ----------
    sigmas : np.ndarray
        The sigmas of the gaussians
    mu : np.ndarray
        The means of the gaussians
    rots : np.ndarray
        The rotations of the gaussians
    n_theta : int, optional
        The number of points in the theta direction, by default 10
    """
    verts = np.zeros((sigmas.shape[0] * (n_theta + 1), 3))
    faces = np.zeros((sigmas.shape[0] * n_theta * 2, 3))
    thetas = np.linspace(0, 2 * np.pi, n_theta)
    xs = np.cos(thetas)
    ys = np.sin(thetas)
    sigmas = wp.from_torch(sigmas, dtype=wp.vec2f)
    mu = wp.from_torch(mu, dtype=wp.vec3f)
    rots = wp.from_torch(rots, dtype=wp.quat)
    verts = wp.array(verts, dtype=wp.vec3f)
    faces = wp.array(faces, dtype=wp.vec3i)
    xs = wp.array(xs, dtype=wp.float32)
    ys = wp.array(ys, dtype=wp.float32)
    n_theta = wp.int32(n_theta)

    with wp.ScopedTimer(
        f"Disk Meshify {sigmas.shape[0]} ellispoids using theta x{n_theta.value}",
        synchronize=True,
    ):
        wp.launch(
            disk_meshify,
            dim=(sigmas.shape[0]),
            inputs=[sigmas, mu, rots, verts, faces, xs, ys, n_theta],
        )
        mesh = trimesh.Trimesh(verts.numpy(), faces.numpy())
        mesh.remove_degenerate_faces()
    return mesh


def test_meshify_from_gsplat(sigmas, mu, rots, n_theta=10, n_sigma=10, color=[1, 0, 0]):
    # Need to change rot from [w,x,y,z] to [x,y,z,w]
    rots = torch.roll(rots, -1, dims=-1)
    return test_meshify(sigmas, mu, rots, color, n_theta, n_sigma, color)


def test_disk_meshify_from_gsplat(sigmas, mu, rots, n_theta=10, color=[1, 0, 0]):
    # Need to change rot from [w,x,y,z] to [x,y,z,w]
    rots = torch.roll(rots, -1, dims=-1)
    sigmas = sigmas[:, :2]
    return test_disk_meshify(sigmas, mu, rots, n_theta)


def test_meshify(sigmas, mu, rots, rgb, n_theta=8, n_sigma=8, color=[1, 0, 0]):
    verts = np.zeros(((sigmas.shape[0]) * n_theta * n_sigma, 3))
    faces = np.zeros((sigmas.shape[0] * ((n_theta - 1) * (n_sigma - 1)) * 2, 3))
    colors = np.zeros((sigmas.shape[0] * ((n_theta - 1) * (n_sigma - 1)) * 2, 3))
    thetas = np.linspace(0, 2 * np.pi, n_theta)
    phis = np.linspace(0, np.pi, n_sigma)
    xs = np.outer(np.cos(thetas), np.sin(phis))
    ys = np.outer(np.sin(thetas), np.sin(phis))
    zs = np.outer(np.ones(n_theta), np.cos(phis))

    # inputs
    sigmas = wp.from_torch(sigmas, dtype=wp.vec3f)
    mu = wp.from_torch(mu, dtype=wp.vec3f)
    rots = wp.from_torch(rots, dtype=wp.quat)
    __import__("pdb").set_trace()
    if rgb is None:
        e_color = wp.ones((sigmas.shape[0], 3), dtype=wp.vec3f)  # type:ignore
        e_color.fill_(color)
    else:
        e_color = wp.from_torch(rgb, dtype=wp.vec3f)

    # outputs
    verts = wp.array(verts, dtype=wp.vec3f)
    faces = wp.array(faces, dtype=wp.vec3i)
    colors = wp.array(colors, dtype=wp.vec3f)
    xs = wp.array(xs, dtype=wp.float32, ndim=2)
    ys = wp.array(ys, dtype=wp.float32, ndim=2)
    zs = wp.array(zs, dtype=wp.float32, ndim=2)

    n_theta = wp.int32(n_theta)
    n_sigma = wp.int32(n_sigma)

    with wp.ScopedTimer(
        f"Meshify {sigmas.shape[0]} ellispoids using theta x{n_theta.value} n_sigma {n_sigma.value}",
        synchronize=True,
    ):
        wp.launch(
            meshify_3d,
            dim=(int(sigmas.shape[0]), n_theta, n_sigma),  # type:ignore
            inputs=[
                sigmas,
                mu,
                rots,
                e_color,
                verts,
                faces,
                colors,
                xs,
                ys,
                zs,
                n_theta,
                n_sigma,
            ],
        )
        # wp.launch(
        #     meshify,
        #     dim=( n_theta, n_sigma), #type:ignore
        #     inputs=[
        #         sigmas,
        #         mu,
        #         rots,
        #         e_color,
        #         verts,
        #         faces,
        #         colors,
        #         xs,
        #         ys,
        #         zs,
        #         n_theta,
        #         n_sigma,
        #     ],
        # )
        # Update face indices to use the deduplicated vertices
        mesh = trimesh.Trimesh(verts.numpy(), faces.numpy(), face_colors=colors.numpy())
        ti = time.time()
        mesh.remove_degenerate_faces()
        print(f"Time to remove degenerate faces {time.time() - ti}")
    mesh.update_faces(mesh.nondegenerate_faces())
    return mesh


if __name__ == "__main__":
    import open3d as o3d

    disk = disk_mesh([0.5, 0.1], n_theta=20)
    o3d.visualization.draw_geometries([to_o3d(disk)])

    mesh = ellipsoid_mesh(np.array([0.5, 0.1, 0.1]), theta_points=20, sigma_points=20)
    print(mesh.vertices.shape)
    print(mesh.faces.shape)
    print(mesh.faces)
    mesh.export("data/meshes/ellipsoid.stl")

    o3d.visualization.draw_geometries([to_o3d(mesh)])
