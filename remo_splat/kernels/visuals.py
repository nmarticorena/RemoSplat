import numpy as np
import open3d as o3d
import torch
import warp as wp


@wp.kernel
def get_lines(
    origin: wp.array(dtype=wp.vec3f),  # type: ignore
    direction: wp.array(dtype=wp.vec3f),  # type: ignore
    distance: wp.array(ndim=2, dtype=wp.float32),  # type: ignore
    points: wp.array(ndim=2, dtype=wp.vec3f),  # type: ignore
    index: wp.array(ndim=2, dtype=wp.vec2i),  # type: ignore
):
    """
    Get the lines from the origin and direction vectors
    ---------
    origin: wp.array(dtype=wp.vec3f)
        points origins [n,3]
    direction: wp.array(dtype=wp.vec3f)
        gradient info [n,3]
    distance: wp.array(dtype=wp.float32)
    points: wp.array(dtype=wp.vec3f)
        points [m,3]
    index: wp.array(dtype=wp.int32)
        index [m,3]
    """

    o, i = wp.tid()  # o origin, i dir
    # origin[0] = wp.vec3f(0.,0.,0.)  # to be local
    if distance[o, i] < 1e5:
        points[o, i * 2] = origin[o]
        points[o, i * 2 + 1] = origin[o] + direction[i] * distance[o, i]
        index[o, i] = wp.vec2i(i * 2, i * 2 + 1)
    else:
        index[o, i] = wp.vec2i(0, 0)
        points[o, i * 2] = wp.vec3f(0.0, 0.0, 0.0)
        points[o, i * 2 + 1] = wp.vec3f(0.0, 0.0, 0.0)


def lines(origins: torch.Tensor, directions: torch.Tensor, distance: torch.Tensor):
    """
    Compute the line
    Arguments
    ---------
    origin: torch.tensor
        points origins [n,3]
    direction: torch.tensor
        gradient info [n,3]
    distance: torch.tensor
        distance [n,1]
    """
    n = directions.shape[0]
    n_s = origins.shape[0]
    points = wp.zeros((n_s, n * 2), dtype=wp.vec3f)  # type: ignore
    print(origins.shape)
    distance = wp.from_torch(distance.cuda(), dtype=wp.float32)  # type: ignore

    origins = wp.from_torch(origins.cuda(), dtype=wp.vec3f)  # type: ignore
    directions = wp.from_torch(directions.cuda().float(), dtype=wp.vec3f)  # type: ignore
    index = wp.zeros((n_s, n), dtype=wp.vec2i)  # type: ignore
    wp.launch(
        get_lines, dim=(n_s, n), inputs=[origins, directions, distance, points, index]
    )  # type: ignore
    return points.numpy(), index.numpy()


def line_to_o3d(points, index, color=[0, 1, 0]):
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
    lines = []
    line = o3d.geometry.LineSet()
    p = []
    lines = []
    index_ = 0
    for i in range(points.shape[0]):
        p.extend(points[i])
        lines.extend(index[i] + index_)
        index_ += points[i].shape[0]

    line.points = o3d.utility.Vector3dVector(np.array(p))
    line.lines = o3d.utility.Vector2iVector(np.array(lines))
    # line.colors = o3d.utility.Vector3dVector([color])

    return line


def line_from_dist_grad(origin, grad, distance):
    """
    Draw a line in open3d
    Arguments
    ---------
    origin: torch.tensor
        points origins [n,3]
    direction: torch.tensor
        gradient info [n,3]
    distance: torch.tensor
        distance [n,1]
    """
    device = origin.device
    grad, distance = grad.to(device), distance.to(device)
    origin = origin.unsqueeze(1)
    origin = torch.repeat_interleave(origin, grad.shape[1], dim=1)

    with torch.no_grad():
        # normalize gradient
        grad = grad / torch.norm(grad, dim=-1, keepdim=True)
        if len(distance.shape) == 1:
            distance = distance.unsqueeze(1)
        line = o3d.geometry.LineSet()
        destination = origin + grad * distance
        origin = origin.reshape(-1, 3)
        destination = destination.reshape(-1, 3)
        index_o = torch.arange(origin.shape[0])
        index_d = torch.arange(destination.shape[0]) + origin.shape[0]

        points = torch.cat([origin, destination], dim=0).cpu().numpy()
        lines = torch.stack([index_o, index_d], dim=1).cpu().numpy()


        line.points = o3d.utility.Vector3dVector(points)
        line.lines = o3d.utility.Vector2iVector(lines)
    return line


def line_from_origin_destination(origin, destination):
    """
    Draw a line in open3d
    Arguments
    ---------
    origin: torch.tensor
        points origins [n,3]
    destination: torch.tensor
        destination [n,3]
    """
    line = o3d.geometry.LineSet()
    index_o = torch.arange(origin.shape[0])
    index_d = torch.arange(destination.shape[0]) + origin.shape[0]

    points = torch.cat([origin, destination], dim=0).cpu().numpy()
    lines = torch.stack([index_o, index_d], dim=1).cpu().numpy()
    points.reshape(-1, 3)


    line.points = o3d.utility.Vector3dVector(points)
    line.lines = o3d.utility.Vector2iVector(lines)

    return line
