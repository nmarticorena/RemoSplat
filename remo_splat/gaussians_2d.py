import matplotlib.pyplot as plt
import numpy as np
import torch
import warp as wp
from matplotlib.animation import FuncAnimation
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
from matplotlib.widgets import Slider
from scipy.stats import chi2

tape = wp.Tape()
step = 1e-7

percentile = 0.99
chi = wp.float32(np.sqrt(chi2.ppf(percentile, 2)))
print(chi)


def uniform(shape, r, device="cuda"):
    sample = (r[0] - r[1]) * torch.rand(*shape) + r[1]
    return sample.to(device)


@wp.func
def rot_matrix(angle: wp.float32, rot: wp.mat22f):
    cos = wp.cos(angle)
    sin = wp.sin(angle)
    rot[0, 0] = cos
    rot[0, 1] = -sin
    rot[1, 0] = sin
    rot[1, 1] = cos
    return rot


@wp.struct
class Gaussian:
    mean: wp.vec2f
    scale: wp.mat22f
    angle: wp.float32
    rot: wp.mat22f
    cov: wp.mat22f


@wp.func
def _mahalanobis_distance(gaussian: Gaussian, point: wp.vec2, chi: wp.float32):
    diff = point - gaussian.mean
    sigma_inv = wp.inverse(gaussian.cov)
    distance = diff @ sigma_inv
    return wp.sqrt(wp.dot(distance, diff)) / chi - 1.0


@wp.kernel
def set_values(
    gaussians: wp.array(dtype=Gaussian),
    means: wp.array(dtype=wp.vec2f),
    angles: wp.array(dtype=float),
    scales: wp.array(ndim=2, dtype=float),
):
    tid = wp.tid()
    gaussians[tid].mean = means[tid]

    gaussians[tid].scale = wp.mat22f(scales[tid, 0], 0.0, 0.0, scales[tid, 1])

    gaussians[tid].angle = angles[tid]
    gaussians[tid].rot = rot_matrix(gaussians[tid].angle, gaussians[tid].rot)

    gaussians[tid].cov = (
        gaussians[tid].rot
        * gaussians[tid].scale
        * gaussians[tid].scale
        * wp.transpose(gaussians[tid].rot)
    )


@wp.kernel
def update_rot(gaussians: wp.array(dtype=Gaussian), rot_delta: float):
    tid = wp.tid()
    gaussians[tid].angle = gaussians[tid].angle + rot_delta
    gaussians[tid].rot = rot_matrix(gaussians[tid].angle, gaussians[tid].rot)

    gaussians[tid].cov = (
        gaussians[tid].rot * gaussians[tid].scale *
        wp.transpose(gaussians[tid].rot)
    )


@wp.kernel
def mahalanobis_distance(
    gaussians: wp.array(dtype=Gaussian),
    points: wp.array(dtype=wp.vec2),
    distances: wp.array(dtype=float),
    chi: wp.float32,
):
    i, j = wp.tid()
    g = gaussians[i]

    wp.atomic_min(
        distances,
        j,
        _mahalanobis_distance(g, points[j], chi),
    )


@wp.kernel
def euclidean_distance(
    gaussians: wp.array(dtype=Gaussian),
    points: wp.array(dtype=wp.vec2),
    distances: wp.array(),
):
    i, j = wp.tid()
    g = gaussians[i]

    diff = points[j] - g.mean

    wp.atomic_min(
        distances,
        j,
        wp.sqrt(wp.dot(diff, diff)),
    )


@wp.kernel
def pdf(
    gaussians: wp.array(dtype=Gaussian),
    points: wp.array(dtype=wp.vec2),
    prob: wp.array(dtype=float),
):
    i, j = wp.tid()  # i for gaussians, j for points
    diff = points[j] - gaussians[i].mean

    normalization = 1.0 / (
        2.0 * wp.pi * wp.sqrt(wp.determinant(gaussians[i].cov))
    )  # ["n_gaussians"]

    distance = wp.inverse(gaussians[i].cov) * diff
    exponent = -0.5 * wp.dot(diff, distance)

    wp.atomic_max(prob, j, normalization * wp.exp(exponent))


@wp.kernel
def update_points(
    points: wp.array(dtype=wp.vec2),
    grad: wp.array(dtype=wp.vec2),
    distance: wp.array(dtype=float),
):
    i = wp.tid()
    points[i] = points[i] - grad[i] * distance[i] * step


@wp.kernel
def reduce_sum(lengths: wp.array(dtype=float), total_length: wp.array(dtype=float)):
    # Accumulate the sum of lengths in total_length[0]
    tid = wp.tid()
    wp.atomic_add(total_length, 0, lengths[tid] * lengths[tid])


class Gaussians2D:
    def __init__(
        self,
        initial_gaussians: int = 2,
        scale_x=[1.0, 1.0],
        scale_y=[1.0, 1.0],
        angles=[-torch.pi, torch.pi],
        xy_range=20,
        x_range=[-10, 10],
        y_range=[-10, 10],
        percentile=0.6827,
    ):
        self.generate_random(
            initial_gaussians, xy_range, x_range, y_range, angles, scale_x, scale_y
        )
        self.percentile = 0.6827
        self.chi = wp.float32(np.sqrt(chi2.ppf(percentile, 2)))

        return

    def update_percentile(self, percentile):
        self.percentile = percentile
        self.chi = wp.float32(np.sqrt(chi2.ppf(percentile, 2)))

    def add_interactive_percentile(self):
        fig = plt.gcf()
        slider = fig.add_axes([0.1, 0.01, 0.8, 0.02])
        self.percentile_slider = Slider(
            slider, "percentile", 0.01, 0.99, valinit=self.percentile
        )

    def generate_random(
        self,
        n_gaussians,
        xy_range=200,
        x_range=[10, 10],
        y_range=[1, 1],
        angles=[-torch.pi, torch.pi],
        scale_x=[1.0, 1.0],
        scale_y=[1.0, 1.0],
    ):
        scales_x = uniform((n_gaussians * 1,), scale_x, device="cuda")
        scales_y = uniform((n_gaussians * 1,), scale_y, device="cuda")
        scales = torch.stack([scales_x, scales_y], axis=1)

        rot_angles = uniform((n_gaussians,), angles, device="cuda")
        mu_x = uniform((n_gaussians * 1,), x_range, device="cuda")
        mu_y = uniform((n_gaussians * 1,), y_range, device="cuda")
        mu = torch.stack([mu_x, mu_y], axis=1)
        self.xy_range = xy_range
        self.gaussians = wp.array(
            [Gaussian() for i in range(n_gaussians)], dtype=Gaussian
        )
        print(mu.shape)
        means = wp.from_torch(mu, dtype=wp.vec2f)
        angles = wp.from_torch(rot_angles)
        scales = wp.from_torch(scales)

        wp.launch(
            kernel=set_values,
            dim=n_gaussians,
            inputs=[self.gaussians, means, angles, scales],
        )

        print(self.gaussians)

    def grad_mahalanobis(self, p):
        num_points = p.shape[0]
        distance = wp.zeros(shape=num_points, dtype=float, requires_grad=True)
        distance.fill_(10000)
        points = wp.array(p, dtype=wp.vec2, requires_grad=True)
        with wp.ScopedTimer(f"Distance and grad", synchronize=True):
            loss = wp.ones(1, dtype=float, requires_grad=True)
            with tape:
                with wp.ScopedTimer(f"Distance", synchronize=True):
                    wp.launch(
                        kernel=mahalanobis_distance,
                        dim=[len(self.gaussians), num_points],
                        inputs=[self.gaussians, points, distance, self.chi],
                    )
                    wp.launch(
                        kernel=reduce_sum, dim=num_points, inputs=[
                            distance, loss]
                    )
            tape.backward(loss)
        print(tape.gradients)
        d = wp.to_torch(distance)
        g = wp.to_torch(tape.gradients[points])
        tape.zero()
        tape.reset()

        return d, g

    def optimize(self, num_points, n_steps):
        fig = plt.figure()
        ax = plt.gca()
        ax.set_aspect("equal", adjustable="datalim")

        points = uniform((num_points, 2), [-20, 20])
        opt = torch.optim.SGD([points], lr=0.1)
        points = wp.from_torch(points, requires_grad=True, dtype=wp.vec2)
        distance = wp.zeros(shape=num_points, dtype=float, requires_grad=True)
        distance.fill_(10000)
        self.plot_pdfs(500)
        scatter = ax.scatter([], [], c="k", label="points", s=0.1)
        plt.legend()
        self.add_interactive_percentile()

        def step(frame):
            self.update_percentile(self.percentile_slider.val)
            print(self.chi.value)
            distance = wp.zeros(
                shape=num_points, dtype=float, requires_grad=True)
            distance.fill_(10000)

            loss = wp.ones(1, dtype=float, requires_grad=True)
            with wp.ScopedTimer(f"Distance and grad", synchronize=True):
                with tape:
                    with wp.ScopedTimer(f"Distance", synchronize=True):
                        wp.launch(
                            kernel=mahalanobis_distance,
                            dim=[len(self.gaussians), num_points],
                            inputs=[self.gaussians, points,
                                    distance, self.chi],
                        )
                        wp.launch(
                            kernel=reduce_sum, dim=num_points, inputs=[
                                distance, loss]
                        )
                tape.backward(loss)
                opt.step()
                tape.zero()
                tape.reset()
            p = wp.array.numpy(points)
            d = wp.array.numpy(distance)
            print("Distance of first point", d[0])
            scatter.set_offsets(np.c_[p[:, 0], p[:, 1]])

        ani = FuncAnimation(fig, step, frames=n_steps, interval=1)
        plt.show()

    def update_rot(self, delta):
        wp.launch(
            kernel=update_rot, dim=len(self.gaussians), inputs=[self.gaussians, delta]
        )
        wp.synchronize()

    def get_pdf(self, points):
        distance = wp.array(shape=points.shape[0], dtype=float)
        distance.fill_(-10000)
        points = wp.from_torch(points, dtype=wp.vec2)
        with wp.ScopedTimer(
            f"get PDF {len(self.gaussians)} x {points.shape[0]}",
            synchronize=True,
        ):
            wp.launch(
                kernel=pdf,
                dim=[len(self.gaussians), points.shape[0]],
                inputs=[self.gaussians, points, distance],
            )
        return distance

    def get_distance(self, points, method="mahalanobis"):
        distance_method = (
            mahalanobis_distance if method == "mahalanobis" else euclidean_distance
        )

        distance = wp.array(shape=points.shape[0], dtype=float)
        distance.fill_(10000)
        points = wp.from_torch(points, dtype=wp.vec2)
        with wp.ScopedTimer(
            f"get distance {method} {len(self.gaussians)} x {points.shape[0]}",
            synchronize=True,
        ):
            wp.launch(
                kernel=distance_method,
                dim=[len(self.gaussians), points.shape[0]],
                inputs=[self.gaussians, points, distance],
            )
        return distance

    def plot_pdfs(self, grid_size):
        X, Y, grid_points = self.get_grid(grid_size)
        distance = self.get_pdf(grid_points)
        norm = Normalize(vmin=0, vmax=self.percentile)
        levels = np.linspace(0, self.percentile, 15)

        self.plot_grid(
            X,
            Y,
            distance,
            "probability",
            cmap="gist_rainbow",
            add_lines=True,
            vmax=percentile,
            levels=levels,
        )
        plt.title("PDF of gaussians")

    def plot_euclidean_distance(self, grid_size):
        X, Y, grid_points = self.get_grid(grid_size)

        distance = self.get_distance(grid_points, "euclidean")

        self.plot_grid(X, Y, distance, "euclidean distance")
        plt.title("euclidean distance")
        plt.xlabel("x")
        plt.ylabel("y")

    def plot_mahalanobis_distance(self, grid_size, title="distance"):
        X, Y, grid_points = self.get_grid(grid_size)

        distance = self.get_distance(grid_points, "mahalanobis")
        self.plot_grid(X, Y, distance, "mahalanobis distance",
                       vmin=None, vmax=5)
        plt.title(title)
        plt.xlabel("x")
        plt.ylabel("y")

    def get_grid(self, grid_size):
        x = torch.linspace(-self.xy_range * 2, self.xy_range * 2, grid_size)
        y = torch.linspace(-self.xy_range * 2, self.xy_range * 2, grid_size)
        X, Y = torch.meshgrid(x, y)
        grid_points = torch.stack([X.ravel(), Y.ravel()], axis=1).cuda()
        return X, Y, grid_points

    def plot_grid(
        self,
        x,
        y,
        distance,
        metric="",
        cmap="jet",
        add_lines=False,
        vmin=1e-5,
        vmax=10,
        levels=10,
    ):
        d = wp.array.numpy(distance)
        d = d.reshape(x.shape)
        d = np.clip(d, vmin, vmax)

        cmap = get_cmap(cmap)  # Choose a colormap
        ax = plt.gca()
        if add_lines:
            contour = ax.contour(
                x,
                y,
                d,
                levels,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                label="gaussians",
            )
        else:
            contour = ax.contourf(
                x,
                y,
                d,
                levels,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                alpha=0.7,
                label="gaussians",
            )
        cbar = plt.colorbar(contour, cmap=cmap)
        cbar.ax.set_ylabel(metric)
        plt.xlabel("x")
        plt.ylabel("y")
        if add_lines:
            cbar.add_lines(contour)
        else:
            ax.clabel(contour, inline=True, fontsize=8)
