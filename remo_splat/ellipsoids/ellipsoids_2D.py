import pdb

import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(precision=2, suppress=True)


def generate_data(mu=np.zeros(2), angle=np.pi / 3, sx=1, sy=1):
    # generate 1000 random points with a 2d standard normal distribution
    x = np.random.multivariate_normal(np.zeros(2), np.eye(2), 10000)

    # we can get the standard deviation in one direction (i.e. x or y) be s, by scaling the x values by s
    x[:, 0] *= sx
    x[:, 1] *= sy

    # then rotate the points by angle
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    x = R @ x.T

    # move it to the desired mean
    x = x + mu[:, None]

    return x.T

def solve_for_d(x, m, C):
    """
    Solves for d such that (x + d*m)^T C (x + d*m) = 0.
    That means where does the line that passes through the point x with direction vector m, intersect the conic defined by C?
    There will typically be two solutions for d, as the line will intersect the conic in two points.
    Use this in combination with the gradient of the algrebraic distance in x, i.e. 2Cx, to find the point on the conic that is closest to x.
    Parameters:
    x (numpy array): The initial point vector as a homogeneous vector
    m (numpy array): The direction vector, i.e. the derivative of the algebraic distance in x, (2Cx)
    C (numpy array): The symmetric matrix (conic or quadric matrix)

    Returns:
    list: Solutions for d
    """
    # Calculate the coefficients of the quadratic equation
    a = m.T @ C @ m
    b = 2 * (x.T @ C @ m)
    c = x.T @ C @ x

    # Compute the discriminant
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError(
            "No real solutions for d (the line does not intersect the conic/quadric)."
        )

    # Solve using the quadratic formula
    d1 = (-b + np.sqrt(discriminant)) / (2 * a)
    d2 = (-b - np.sqrt(discriminant)) / (2 * a)

    return [d1, d2]

def euclidean_distance(x, C):
    """
    Calculate the Euclidean distance of a point x to the conic defined by the matrix C.

    Parameters:
    x (numpy array): The point vector as homogeneous vector
    C (numpy array): The symmetric matrix (conic or quadric matrix)

    Returns:
    float: The Euclidean distance of the point to the conic
    numpy array: The point on the conic that is closest to x
    """
    # Calculate the gradient of the algebraic distance in x and normalise it, this is the direction vector
    j = 2 * C @ x
    j = j / j[2]
    j[2] = 0

    # now x + d*j is the closest point on the conic to x
    d = solve_for_d(x, j, C)
    # check which of the two solutions is the closest
    x = x / x[2]
    x = x[0:2]
    j = j[0:2]
    p1 = x + d[0] * j
    p2 = x + d[1] * j
    d1 = np.linalg.norm(p1 - x)
    d2 = np.linalg.norm(p2 - x)
    if d1 < d2:
        return d1, p1
    else:
        return d2, p2

def decomposition(x: np.ndarray):
    # calculate the covariance matrix
    cov = np.cov(x.T)

    # cholesky decomposition gives us L such that Sigma = L L^T
    L = np.linalg.cholesky(cov)
    # calculate the eigenvalues and eigenvectors

    return L

def sample_ellipse(mu, L, s, n_points):
    # draw the ellipse with center mu and Cholesky factor L
    theta = np.linspace(0, 2 * np.pi, n_points)
    x = np.cos(theta)
    y = np.sin(theta)
    xy = np.array([x, y])
    xy = L @ xy * s
    xy = xy + mu[:, None]
    return xy

def gaussian2conic(sigma, mu, n_sigma: int = 1):
    # we can construct C as a block matrix from three parts: A, b, and c
    A = np.linalg.inv(sigma)
    # A = sigma
    b = -A @ mu
    c = mu.T @ A @ mu - n_sigma**2  # notice how we have to square n_sigma here!

    # arrange the conic matrix C
    C = np.zeros((3, 3))
    C[0:2, 0:2] = A
    C[2, 0:2] = b
    C[0:2, 2] = b
    C[2, 2] = c
    return C

def algebraic_distance(x, C):
    if x.shape[1] == 2:
        x = homogenize(x)

    # calculate the algebraic distance
    d = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        d[i] = x[i].T @ C @ x[i]
    return d

def grad_algebraic_distance(x, C):
    if x.shape[1] == 2:
        x = homogenize(x)

    # calculate the gradient of the algebraic distance
    grad = np.zeros(x.shape)
    for i in range(x.shape[0]):
        grad[i] = 2 * C @ x[i]
    return grad

def homogenize(x):
    return np.hstack([x, np.ones((x.shape[0], 1))])

def euclidean_distance(p, x):
    """
    Calculate the Euclidean distance between a point p and a set of points x
    Args:
        p: a point in 2D space
        x: a set of points in 2D space
    """
    return np.linalg.norm(x - p, axis=0)

def euclidean_distance_ellipse(p, x):
    """
    Calculate the Euclidean distance between a point p and a set of points x
    Args:
        p: a point in 2D space
        x: a set of points in 2D space
    """
    if len(p.shape) == 1:
        p = p.reshape(2, -1)
    distance = euclidean_distance(p, x)
    min, index = np.min(distance), np.argmin(distance)
    return min, x[:, index]

def test_generate_data():
    mu = np.array([1, 2])
    angle = np.pi / 3
    sx = 1
    sy = 1
    x = generate_data(mu, angle, sx, sy)
    plt.scatter(x[:, 0], x[:, 1])
    plt.axis("equal")
    plt.grid(True)
    plt.title(
        "Generated data using "
        + r"$\mu = $"
        + str(mu)
        + ", "
        + r"$\theta = $"
        + str(angle)
        + ", "
        + r"$s_x = $"
        + str(sx)
        + ", "
        + r"$s_y = $"
        + str(sy)
    )

    return

def test_grad_algebraic_distance():
    mu = np.array([1, 2])
    angle = np.pi / 3
    sx = 1
    sy = 5
    x = generate_data(mu, angle, sx, sy)

    L = decomposition(x)
    for s in [1, 2, 3]:
        xy = sample_ellipse(mu, L, s)
        plt.plot(xy[0], xy[1], "-o", color="black", alpha=0.5, label="s=" + str(s))
    plt.legend()

    C = gaussian2conic(np.cov(x.T), mu, 1)

    X, Y = np.meshgrid(np.linspace(-50, 50, 1000), np.linspace(-50, 50, 1000))
    grid_points = np.array([X.flatten(), Y.flatten()]).T

    d = algebraic_distance(grid_points, C)
    grad = grad_algebraic_distance(grid_points, C)
    plt.contourf(X, Y, d.reshape(X.shape), levels=100, cmap="viridis", legend=0.5)
    plt.show()
    print(f"Algebraic distance:\n{d}")
    print(f"Gradient of algebraic distance:\n{grad}")
    return

def test_euclidean_distance():
    mu = np.array([1, 2])
    angle = np.pi / 3
    sx = 1
    sy = 5
    x = generate_data(mu, angle, sx, sy)

    L = decomposition(x)
    xy = sample_ellipse(mu, L, s=3, n_points=10_0)
    plt.plot(xy[0], xy[1], "-o", color="black", alpha=0.5, label="s=3")
    plt.legend()

    X, Y = np.meshgrid(np.linspace(-50, 50, 50), np.linspace(-50, 50, 50))
    grid_points = np.array([X.flatten(), Y.flatten()]).T
    d = np.zeros(grid_points.shape[0])
    for i in range(grid_points.shape[0]):
        print(grid_points[i, :])
        d[i], closest_point = euclidean_distance_ellipse(
            grid_points[i, :].reshape(2, -1), xy
        )
    plt.contour(X, Y, d.reshape(X.shape), levels=100, cmap="viridis", legend=0.5)
    plt.title("Euclidean distance to the ellipse")
    plt.show()
    print(f"Algebraic distance:\n{d}")
    return

def test_drawing_ellipsoids():
    mu = np.array([1, 2])
    angle = np.pi / 3
    sx = 1
    sy = 5
    x = generate_data(mu, angle, sx, sy)
    plt.scatter(x[:, 0], x[:, 1], label="data")
    plt.axis("equal")
    plt.grid(True)
    plt.title(
        "Generated data using "
        + r"$\mu = $"
        + str(mu)
        + ", "
        + r"$\theta = $"
        + str(angle)
        + ", "
        + r"$s_x = $"
        + str(sx)
        + ", "
        + r"$s_y = $"
        + str(sy)
    )

    L = decomposition(x)
    for s in [1, 2, 3]:
        xy = sample_ellipse(mu, L, s)
        plt.plot(xy[0], xy[1], "-o", color="black", alpha=0.5, label="s=" + str(s))
    plt.legend()

if __name__ == "__main__":
    # test_drawing_ellipsoids()
    # test_grad_algebraic_distance()
    test_euclidean_distance()
