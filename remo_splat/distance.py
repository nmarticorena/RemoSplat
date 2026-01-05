import pdb

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize

np.set_printoptions(precision=2)


# From iSDF
def gradient(inputs, outputs):
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = torch.autograd.grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=False,
        retain_graph=False,
        only_inputs=True,
    )[0]
    return points_grad


def gaussian_pdf(x, mean, cov):
    x = x.double()
    mean = mean.double()
    cov = cov.double()
    """Calculate the Gaussian PDF."""
    diff = x.unsqueeze(1) - mean.unsqueeze(0)
    inv_cov = torch.linalg.inv(cov)
    normalization = 1 / (2 * np.pi * torch.sqrt(torch.linalg.det(cov)))
    exponent = -0.5 * diff @ inv_cov @ diff.transpose(1, 2)
    return normalization * torch.exp(exponent)


def plot_pdfs(mean, cov, x_range=40, y_range=40, grid_size=50):
    x = np.linspace(-x_range, x_range, grid_size)
    y = np.linspace(-y_range, y_range, grid_size)
    X, Y = np.meshgrid(x, y)
    grid_points = np.stack([X.ravel(), Y.ravel()], axis=1)
    distance = (
        gaussian_pdf(torch.tensor(grid_points).cuda(), mean, cov)
        .reshape(X.shape)
        .cpu()
        .numpy()
    )

    # Plot the gradient using contourf
    cmap = get_cmap("Blues")  # Choose a colormap

    # levels = np.linspace(0, 5, 100)
    plt.contourf(X, Y, distance, levels=20, cmap=cmap, alpha=0.7)
    # return X, Y, distance


def mahalanobis_distance(x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor):
    """
    Compute Mahalanobis distance o a set of points x
    and a set of gaussians
    Parameters
    ----------
    x : torch.Tensor
        Input point locations [N,2]
    mu: torch.Tensor
        mean of Gaussians [M,2]
    sigma: torch.Tensor
        variance of Gaussians [M,2,2]
    Returns
    -------
    distance: torch.Tensor
        Distance to gaussians [N]
    """
    x_in = x.unsqueeze(1)  # [N,1,2]
    mu_in = mu.unsqueeze(0)  # [1, M , 2]
    # sigma_in = sigma.unsqueeze(0)

    diff = x_in - mu_in  # N,M,2
    diff.unsqueeze_(-1)  # [N,M,2,1]

    sigma_inv = torch.linalg.inv(sigma).unsqueeze(0)
    result = diff.transpose(-1, -2) @ sigma_inv @ diff

    result = result.reshape(x.shape[0], mu.shape[0])
    return torch.min(result, 1).values


def forward_gradient(x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor):
    """
    Compute the forward gradient of the distance function,

    Returns
    -------
    distance: torch.Tensor
        Distance of all the points [N]
    gradient: torch.Tensor
        Direction that pushes away from the gaussians [N,3]
    """
    x.requires_grad_(True)
    distance = mahalanobis_distance(x, mu, sigma)
    grad = gradient(x, distance)
    return distance, grad


def plot(mu, sigma, grid_size=50, x_range=20, y_range=20):
    x = np.linspace(-x_range, x_range, grid_size)
    y = np.linspace(-y_range, y_range, grid_size)
    X, Y = np.meshgrid(x, y)
    grid_points = np.stack([X.ravel(), Y.ravel()], axis=1)
    print("grid points")
    print(grid_points.shape)
    distance = (
        mahalanobis_distance(torch.tensor(grid_points).cuda(), mu, sigma)
        .reshape(X.shape)
        .cpu()
        .numpy()
    )

    # Plot the gradient using contourf
    cmap = get_cmap("Blues")  # Choose a colormap

    # levels = np.linspace(0, 5, 100)
    plt.contourf(X, Y, distance, levels=20, cmap=cmap, alpha=0.7)
    plt.savefig("results/contour.png")
    return X, Y, distance


def quick_plot(X, Y, distance):
    cmap = get_cmap("Blues")  # Choose a colormap
    plt.contourf(X, Y, distance, levels=20, cmap=cmap, alpha=0.7)
    return


def generate_gaussians(n_gaussians=10, range=20, covs_scale=1, device="cuda"):
    means = (-range * 2) * torch.rand(n_gaussians, 2) + range
    covs = torch.randn(n_gaussians, 2, 2).double() * covs_scale
    covs = covs @ covs.transpose(1, 2).double()

    return means.to(device), covs.to(device)


if __name__ == "__main__":
    mu, sigma = generate_gaussians(covs_scale=1)
    X, Y, distance = plot(mu, sigma, x_range=40, y_range=40)
    n_points = 10
    p = (-20) * torch.rand(n_points, 2) + 10
    # plot_pdfs(mu, sigma)
    # plt.show()

    plt.scatter(p.numpy()[:, 0], p.numpy()[:, 1], color="r")
    plt.savefig("results/sgd.png")
    # plt.show()

    points = np.array(p.unsqueeze(0))
    point = p.cuda().double().requires_grad_(True)

    optimizer = torch.optim.SGD([point], lr=0.001)
    for j in range(points.shape[1]):
        plt.plot(points[:, j, 0], points[:, j, 1], "o-")

    for i in range(200):
        quick_plot(X, Y, distance)
        di, grad = forward_gradient(point, mu, sigma)
        # dis = mahalanobis_distance(point, mu, sigma)
        # dis = dis.sum()
        # dis.backward()

        # pdb.set_trace()
        point = point + grad * 0.01
        # optimizer.step()
        points = np.vstack((points, point.unsqueeze(0).detach().cpu().numpy()))

        for j in range(points.shape[1]):
            plt.plot(points[:, j, 0], points[:, j, 1], "o--")
        plt.savefig(f"results/sgd_{i}.png")
        plt.cla()
        p.grad = None
