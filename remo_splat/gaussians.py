import matplotlib.pyplot as plt
import torch
from matplotlib.cm import get_cmap
from utils import uniform


class Gaussians2D:
    def __init__(self):
        return

    def rot_matrix(self, angles):
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        R = torch.stack(
            [torch.stack([cos, -sin], dim=-1), torch.stack([sin, cos], dim=-1)], dim=-2
        )

        return R.squeeze(1)  # output [N,2,2]

    def pdf(self, x):
        """Calculate the Gaussian PDF."""
        diff = x.unsqueeze(1) - self.mu.unsqueeze(0)  # [n_points, n_gaussians, 2]
        normalization = 1 / (
            2 * torch.pi * torch.sqrt(torch.linalg.det(self.covs))
        )  # ["n_gaussians"]
        exponent = -0.5 * torch.einsum("pgi,gij,pgj->pg", diff, self.inv_covs, diff)
        result = normalization * torch.exp(exponent)
        return result.max(axis=1)[0]

    def generate_random(
        self, n_gaussians, xy_range=20, x_range=[10, 10], y_range=[1, 1]
    ):
        x = uniform((n_gaussians, 1), x_range, device="cuda")
        y = uniform((n_gaussians, 1), y_range, device="cuda")
        self.scales = torch.hstack([x, y])
        print(self.scales.shape)

        # scales = torch.tensor([5.0, 1]).cuda()
        self.rot_angles = uniform(
            (n_gaussians, 1), [-torch.pi, torch.pi], device="cuda"
        )
        self.mu = uniform((n_gaussians, 2), [-xy_range, xy_range], device="cuda")
        self.xy_range = xy_range
        self.initialize_cov()

    def initialize_cov(self):
        R = self.rot_matrix(self.rot_angles).to("cuda")
        self.S = torch.diag_embed(self.scales)
        self.covs = torch.einsum("gik,gkl,gjl->gij", R, self.S, R)
        self.inv_covs = torch.linalg.inv(self.covs)

    def plot_pdfs(self, grid_size):
        x = torch.linspace(-self.xy_range * 2, self.xy_range * 2, grid_size)
        y = torch.linspace(-self.xy_range * 2, self.xy_range * 2, grid_size)
        X, Y = torch.meshgrid(x, y)
        grid_points = torch.stack([X.ravel(), Y.ravel()], axis=1)
        distance = (
            self.pdf(torch.tensor(grid_points).cuda()).reshape(X.shape).cpu().numpy()
        )
        print(distance.min())
        print(distance.max())
        # Plot the gradient using contourf
        cmap = get_cmap("Blues")  # Choose a colormap

        # levels = np.linspace(0, 5, 100)
        plt.contourf(X, Y, distance, levels=4, cmap=cmap, vmin=1e-5, alpha=0.7)

    def rotate_all(self, step=0.1):
        self.rot_angles += step
        self.initialize_cov()
        return


if __name__ == "__main__":
    gaussians = Gaussians2D()
    gaussians.generate_random(20)
    from distance import plot_pdfs

    for i in range(1000):
        gaussians.rotate_all()
        gaussians.plot_pdfs(500)
        plt.show()
    # plot_pdfs(gaussians.mu, gaussians.covs, grid_size=100)

    plt.show()
