import torch
import warp as wp

from remo_splat.gaussians_3d import Gaussians, Gaussians3D


@wp.func
def gaussian2conic(sigma: wp.mat33f, mu: wp.vec3f, n_sigma: float):  # type: ignore
    # we can construct C as a block matrix from three parts: A, b, and c
    A = wp.inverse(sigma)  # 3x3
    # A = sigma
    b = -1.0 * (wp.mul(A, mu))  # 3x1
    c = (
        wp.dot(mu, A @ mu) - n_sigma * n_sigma  # 1
    )  # notice how we have to square n_sigma here!
    # arrange the conic matrix C
    # C = wp.zeros(dtype = wp.float32)

    C = wp.mat44f(
        A[0, 0],
        A[0, 1],
        A[0, 2],
        b[0],
        A[1, 0],
        A[1, 1],
        A[1, 2],
        b[1],
        A[2, 0],
        A[2, 1],
        A[2, 2],
        b[2],
        b[0],
        b[1],
        b[2],
        c,
        shape=(4, 4),
    )
    return C


@wp.func
def check_intersection(x: wp.vec3f, dir: wp.vec3f, C: wp.mat44f):  # type: ignore
    # compute the quadratic coefficients
    x_homo = wp.vec4f(x[0], x[1], x[2], 1.0)
    dir_homo = wp.vec4f(dir[0], dir[1], dir[2], 0.0)
    a = wp.dot(dir_homo, C @ dir_homo)
    b = 2.0 * wp.dot(x_homo, C @ dir_homo)
    c = wp.dot(x_homo, C @ x_homo)

    # compute the discriminant
    delta = b * b - 4.0 * a * c

    # check if the discriminant is positive
    # wp.print(delta)
    return delta > 0, a, b, c, delta


@wp.kernel
def intersections(
    gaussians: wp.array(dtype = wp.mat44f),  # type: ignore
    origin: wp.array(dtype = wp.vec3f), # type: ignore
    direction: wp.array(dtype=wp.vec3f),  # type: ignore
    intersected: wp.array(ndim=3, dtype=wp.bool),  # type: ignore
    distance: wp.array(ndim=3, dtype=wp.float32),  # type: ignore
):
    o, i, j = wp.tid()  # iterate over the gaussians, the directions and the origins
    C = gaussians[i]
    #TODO I am computing this one more than required, there is
    # some functions to check if it was already writen or not
    sol, a, b, c, d = check_intersection(origin[o], direction[j], C)

    if sol:
        intersected[o, i,j] = sol
        # Solve
        d1 = (-b + wp.sqrt(d)) / (2. * a)
        d2 = (-b - wp.sqrt(d)) / (2. * a)
        # distance[i, j] = wp.min(wp.abs(d1), wp.abs(d2)) * wp.sign(d1)
        # only save if it is positive
        if d1 > 0 and d2 > 0:
            distance[o,i, j] = wp.min(d1, d2)
        elif d1 > 0:
            distance[o,i, j] = d1
        elif d2 > 0:
            distance[o,i, j] = d2
            # distance[i,j] = 1e6

@wp.kernel
def toConic(
    g:Gaussians,
    conics: wp.array(dtype = wp.mat44f), #type: ignore
):
    i = wp.tid()
    conics[i] = gaussian2conic(g.covs[i], g.means[i], 1.0)

def draw_intersected(gaussians: Gaussians3D, origin: torch.Tensor, direction: torch.Tensor):
    n_s = origin.shape[0]
    n_g = gaussians.gaussians.means.shape[0]
    n_d = direction.shape[0]
    print("Launching with ", n_g)
    for _ in range(1):
        with wp.ScopedTimer(f"get Intersection with {n_s} X {n_g} X {n_d}", synchronize=True):
            interesected = wp.array(dtype=wp.bool, shape=(n_s, n_g, n_d))
            conics = wp.zeros(dtype=wp.mat44f, shape=(n_g)) #type: ignore
            distance = wp.full((n_s,n_g, n_d), 1.e6, dtype=wp.float32) # type: ignore
            origin_wp = wp.from_torch(origin, dtype = wp.vec3f)
            direction_wp = wp.from_torch(direction, dtype=wp.vec3f)
            wp.launch(toConic,
                      dim = (n_g),
                      inputs = [gaussians.gaussians, conics]
                      )
            wp.launch(
                intersections,
                dim=(n_s, n_g, n_d),
                inputs=[conics, origin_wp, direction_wp, interesected, distance],
            )

            # min_distance, index_distance = wp.to_torch(distance).min(axis=1) # Positive
            sorted, sorted_index = torch.sort(wp.to_torch(distance), dim=1)

    return None, distance, sorted_index, sorted

@wp.kernel
def distance_lines_gaussians(
    gaussians: Gaussians,  # type: ignore
    origin: wp.vec3f,
    direction: wp.vec3f,
    distances: wp.array(dtype=wp.float32),  # type: ignore
):
    i = wp.tid()
    g = gaussians
    dir = wp.normalize(direction)
    distances[i] = wp.length(wp.cross(wp.sub(g.means[i], origin), dir))


@wp.kernel
def distance_line(
    origin: wp.vec3f,
    direction: wp.vec3f,
    means: wp.array(dtype=wp.vec3f),  # type: ignore
    distances: wp.array(dtype=wp.float32),  # type: ignore
):
    """
    Compute the distance between a line and a set of points
    Arguments
    ---------
        points origins [n,3]
    direction: wp.vec3f
        gradient info [n,3]
    means: wp.array(dtype=wp.vec3f)
        points [m,3]
    """

    i = wp.tid()

    dir = wp.normalize(direction)
    distances[i] = wp.length(wp.cross(wp.sub(means[i], origin), dir))

def get_distance(origin, direction, means) -> wp.array:
    """
    Compute the distance between a line and a set of points
    Arguments
    ---------
    origin: torch.tensor
        points origins [n,3]
    direction: torch.tensor
        gradient info [n,3]
    means: torch.tensor
        points [m,3]
    """
    with wp.ScopedTimer(f"get_distance of {means.shape[0]} points", synchronize=True):
        o = wp.vec3f(origin)

        distances = wp.zeros(means.shape[0], dtype=wp.float32)
        d = wp.vec3f(direction)
        m = wp.array(means, ndim=2, dtype=wp.vec3f)
        wp.launch(distance_line, dim=(means.shape[0]), inputs=[o, d, m, distances])
    return distances


@wp.kernel
def distance_disk(
    p_r:  wp.array(dtype = wp.vec3f),
    mu: wp.array(dtype =wp.vec3f),
    rots: wp.array(dtype= wp.quat),
    scales: wp.array(dtype = wp.vec3f),
    distances: wp.array(ndim = 2,dtype=wp.float32),  # type: ignore
    gradients: wp.array(ndim=2, dtype=wp.vec3f),  # type: ignore
):
    # We asume is already sorted
    i, j = wp.tid() # i for points, j for gaussians

    p_r_local = wp.quat_rotate_inv(rots[j], p_r[i] - mu[j])# + 1e-8

    # Check if inside the disk
    inside = ((p_r_local[0] * p_r_local[0]) / (scales[j][0] * scales[j][0]) + (p_r_local[1] * p_r_local[1])  / (scales[j][1] * scales[j][1])) <= 1.0 + 1e-12

    if inside:
        distances[i,j] = p_r_local[2] * p_r_local[2]
        gradients[i,j] = wp.quat_rotate(rots[j], wp.normalize(wp.vec3f(0.0, 0.0, 2.0 * p_r_local[2])))
    else:
        distances[i,j] = 0.
        gradients[i,j] = wp.vec3f(0.0, 0.0, 0.0)

@wp.func
def root(r:wp.vec2f,z:wp.vec2f,g:float):
    n = r * z
    s0 = z[1] - 1.0
    s1 = wp.length(n) - 1.0
    s = 0.0

    for i in range(10):
        s = 0.5 * (s0 + s1)              # midpoint
        ratio = n / (s + r)              # elementwise
        g = wp.dot(ratio, ratio) - 1.0   # same as torch.sum(...)

        if g >= 0.0:
            s1 = s                       # shrink upper bound
        else:
            s0 = s                       # shrink lower bound

    return s


@wp.func
def solved_2d(p_local: wp.vec2f, scale: wp.vec2f):
    z = p_local / scale
    g = wp.length(z) - 1.0
    r = scale / scale[1]

    lam = root(r, z, g)

    y = r * p_local / (lam + r)

    return y # Closer point on the 2D ellipse in their local frame
