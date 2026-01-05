"""
Codebase to compare the predicted distance using differnet sensor approaches
"""

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch

from remo_splat import logger
from remo_splat.lidar import reshape_depth_sensor, reshape_euclidean_sensor

sensor_variants = ["DepthSensor", "EuclideanDistanceGaussian"]


class Comparisson:
    def __init__(
        self,
        gt_seq_name,
        target_name="",
        is_3D: bool = True,
        sensor_variant: str = "DepthSensor",
        load_mesh: bool = False,
    ):
        self.gt_seq_name = gt_seq_name
        dimensions = "3d" if is_3D else "2d"
        if target_name == "":
            target_name = gt_seq_name
        self.target_name = f"distance/{target_name}/{dimensions}/{sensor_variant}"
        self.dimensions = dimensions
        self.euclidean = sensor_variant == "EuclideanDistanceGaussian"
        self.sensor_variant = sensor_variant
        self.similarity = torch.nn.CosineSimilarity(dim=-1)
        self.load_mesh = load_mesh
        if self.euclidean and load_mesh:
            self.mesh = o3d.io.read_triangle_mesh(
                f"logs/experiments/{self.target_name}/0000/mesh.stl"
            )
        else:
            self.mesh = o3d.geometry.TriangleMesh()

    def load_sequence(self, id):
        self.target_data = logger.LoggerLoader(self.target_name, f"{id:04d}", "")
        if self.euclidean and self.load_mesh:
            self.mesh = o3d.io.read_triangle_mesh(
                f"logs/experiments/{self.target_name}/{id:04d}/mesh.stl"
            )
        grad = self.target_data.get_data("g_w")
        distance = self.target_data.get_data("distance")
        if self.euclidean:
            d, g = reshape_euclidean_sensor(distance, grad)
        else:
            d, g = reshape_depth_sensor(distance, grad)
        self.d, self.g = logger.get_mins(d, g)

    def get_data(self, step):
        return self.d[step], self.g[step]

    def plot_error(self, id):
        d, gt_distance = self.get_distances(id)

        error = d - gt_distance
        # get_quintils and fill
        error_25 = np.percentile(error, 25, axis=-1)
        error_75 = np.percentile(error, 75, axis=-1)
        plt.fill_between(range(len(error)), error_25, error_75, alpha=0.5)
        plt.plot(
            np.median(error, axis=-1),
            label=f"mean {self.dimensions} {self.sensor_variant}",
        )

    def get_error(self, id):
        d, gt_distance = self.get_distances(id)
        return d - gt_distance

    def get_all_errors(self):
        # Append all of the error and return a matrix of size [N, n_sensors]
        error = self.get_error(0)
        for i in range(1, 100):
            if i == 48:
                continue
            error = np.concatenate((error, self.get_error(i)), axis=0)
        return error

    def get_all_cosine(self):
        # Append all of the error and return a matrix of size [N, n_sensors]
        cosine = self.step_cosine(0)
        for i in range(1, 100):
            if i == 48:
                continue
            cosine = np.concatenate((cosine, self.step_cosine(i)), axis=0)
        return cosine

    def get_all_distances(self):
        d, gt = self.get_distances(0)
        for i in range(1, 100):
            if i == 48:
                continue
            _d, _gt = self.get_distances(i)
            d = np.concatenate((d, _d), axis=0)
            gt = np.concatenate((gt, _gt), axis=0)
        return d, gt

    def conservative_pred(self, taus=[0.0], epsilon=0.1):
        # we return percentage that is over conservative, percentage that is under
        # and the ones that are within 1%
        pred_distance, gt_distance = self.get_all_distances()
        pred_distance = pred_distance.flatten()
        gt_distance = gt_distance.flatten()
        if isinstance(taus, float):
            taus = [taus]
        safe_acc = np.zeros((len(taus)))
        over_acc = np.zeros((len(taus)))
        acc = np.zeros((len(taus)))

        for i, tau in enumerate(taus):
            safe = np.sum((gt_distance - (pred_distance + tau)) > epsilon)
            unsafe = np.sum((gt_distance - (pred_distance + tau)) <= -epsilon)
            perfect = np.sum(np.abs(gt_distance - (pred_distance + tau)) < epsilon)
            safe_acc[i] = safe / len(pred_distance)
            over_acc[i] = unsafe / len(pred_distance)
            acc[i] = perfect / len(pred_distance)
        return safe_acc

    def col_pred(self, taus=[0.0], epsilon=0.1):
        # we return percentage that is over conservative, percentage that is under
        # and the ones that are within 1%
        pred_distance, gt_distance = self.get_all_distances()
        pred_distance = pred_distance.flatten()
        gt_distance = gt_distance.flatten()
        if isinstance(taus, float):
            taus = [taus]
        safe_acc = np.zeros((len(taus)))
        over_acc = np.zeros((len(taus)))
        acc = np.zeros((len(taus)))

        for i, tau in enumerate(taus):
            safe = np.sum((gt_distance - (pred_distance + tau)) > epsilon)
            over = np.sum((gt_distance - (pred_distance + tau)) < -epsilon)
            perfect = np.sum(np.abs(gt_distance - (pred_distance + tau)) < epsilon)
            safe_acc[i] = safe / len(pred_distance)
            over_acc[i] = over / len(pred_distance)
            acc[i] = perfect / len(pred_distance)
        return safe_acc, over_acc, acc

    def find_max_tau(self, target_accuracy=0.99, tau_min=0.0, tau_max=1.0, tol=1e-3):
        """Binary search for the bigger tau that gives at least  safe predictions."""
        while tau_max - tau_min > tol:
            tau_mid = (tau_min + tau_max) / 2
            safe_acc, _ = self.col_pred(tau_mid)

            if safe_acc >= target_accuracy:
                tau_max = tau_mid  # Try a smaller tau
            else:
                tau_min = tau_mid  # Increase tau

        return tau_min  # Return the smallest tau found

    def find_min_tau(
        self, target_accuracy=0.99, tau_min=0.0, tau_max=1.0, tol=1e-3, epsilon=0.1
    ):
        """Binary search for the smallest tau that gives at least target_accuracy% safe predictions."""
        while tau_max - tau_min > tol:
            tau_mid = (tau_min + tau_max) / 2
            safe_acc, _, _ = self.col_pred(tau_mid, epsilon)

            if safe_acc <= target_accuracy:
                tau_max = tau_mid  # Try a smaller tau
            else:
                tau_min = tau_mid  # Increase tau

        return tau_min  # Return the smallest tau found

    def get_distances(self, id):
        gt_data = logger.LoggerLoader(self.gt_seq_name, f"{id:04d}", "")
        target_data = logger.LoggerLoader(self.target_name, f"{id:04d}", "")

        gt_distances = gt_data.get_data("gt_distance")[:]

        target_distances = target_data.get_data("distance")
        target_gw = target_data.get_data("g_w")

        if self.euclidean:
            d, g = reshape_euclidean_sensor(target_distances, target_gw)
        else:
            d, g = reshape_depth_sensor(target_distances, target_gw)

        d, g = logger.get_mins(d, g)

        # plt.plot(d[:,-7], label = "predicted")
        # plt.plot(gt_distances[:,-7], label = "gt")

        return d, gt_distances

    def step_cosine(self, id):
        gt_data = logger.LoggerLoader(self.gt_seq_name, f"{id:04d}", "")
        target_data = logger.LoggerLoader(self.target_name, f"{id:04d}", "")

        gt_distances = gt_data.get_data("gt_distance")[:]
        gt_grad = gt_data.get_data("gt_distance_grad")[:]

        target_distances = target_data.get_data("distance")
        target_gw = target_data.get_data("g_w")

        if self.euclidean:
            d, g = logger.reshape_euclidean_sensor(target_distances, target_gw)
        else:
            d, g = logger.reshape_depth_sensor(target_distances, target_gw)

        d, g = logger.get_mins(d, g)

        cosine_distance = (
            1
            - self.similarity(
                torch.from_numpy(gt_grad), -1 * torch.from_numpy(g)
            ).numpy()
        )
        return cosine_distance
