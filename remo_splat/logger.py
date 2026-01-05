import copy
import json
import os
import time
from collections import defaultdict
from typing import Optional, Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import spatialmath as sm
import torch
from matplotlib.axes import Axes
from neural_robot.robot import NeuralRobot
from neural_robot.unity_frankie import NeuralFrankie
from scipy.integrate import cumulative_trapezoid

from remo_splat.configs.controllers import NeoConfig, filter_data
from remo_splat.configs.experiments import load_mesh, obstacles_to_mesh
from remo_splat.lidar import reshape_depth_sensor, reshape_euclidean_sensor


class Logger:
    def __init__(self, exp_name: str, config: NeoConfig = NeoConfig()):
        self.config = config
        self.initialize(exp_name, config)

    def __print__(self):
        print("[Logger] Logging to", f"logs/experiments/{self.exp_name}")

    def log(self, data, variable_data={}):
        for k, v in data.items():
            self.add_data(k, v)
        for k, v in variable_data.items():
            self.add_variable_data(k, v)

    def initialize(self, exp_name, config):
        if config is None:
            config = self.config
        self.data = defaultdict(list)
        self.exp_name = exp_name
        self.__print__()

        self.folder_name = f"logs/experiments/{exp_name}"
        os.makedirs(self.folder_name, exist_ok=True)
        self.info_file = f"{self.folder_name}/info.txt"
        if isinstance(config.lamda_q_vec, np.ndarray):
            config.lamda_q_vec = config.lamda_q_vec.tolist()
        if isinstance(config.lamda_q, np.ndarray):
            config.lamda_q = config.lamda_q.tolist()
        if isinstance(config.only_base, np.ndarray):
            config.only_base = config.only_base.tolist()
        if isinstance(config.max_acceleration_step, np.ndarray):
            config.max_acceleration_step = config.max_acceleration_step.tolist()

        with open(f"{self.folder_name}/config.json", "w") as f:
            f.write(json.dumps(config.__dict__, indent=4))
        self.variable_data = defaultdict(list)

    def add_variable_data(self, name, values):
        """Add data with variable size

        Parameters
        ----------
        name : str
            key
        values : np.ndarray
            values to store
        """
        self.variable_data[name].append(values.tolist())

    def add_data(self, name, values):
        self.data[name].append(values)

    def save(self, robot="frankie"):
        with open(self.info_file, "w") as f:
            # Write the date
            f.write(f"Date: {time.ctime()}\n")

            for key, value in self.data.items():
                f.write(f"{key}\n")

        with open(f"{self.folder_name}/matrix.json", "w") as f:
            f.write(json.dumps(self.variable_data, indent=4))
        self.data["robot"] = [robot]
        np.savez_compressed(f"{self.folder_name}/data.npz", **self.data)


class LoggerLoader:
    def __init__(
        self, exp_name, traj_id="0000", run_id="", robot: Optional[NeuralRobot] = None
    ):
        # print(f"Loading {exp_name}")
        project_root = os.path.dirname(os.path.abspath(__file__))
        self.folder_name = os.path.join(
            project_root, "..", "logs", "experiments", exp_name
        )
        try:
            with open(f"{self.folder_name}/{traj_id}/config.json", "r") as f:
                config_data = json.load(f)
                self.config = NeoConfig(**filter_data(config_data))
        except FileNotFoundError:
            print(f"Could not find config file for {self.folder_name}/config.json")
            self.config = NeoConfig()
        self.data_folder = f"{self.folder_name}/{traj_id}/{run_id}"
        self.data = dict(np.load(f"{self.folder_name}/{traj_id}/{run_id}/data.npz"))
        for k, v in self.data.items():
            self.data[k] = np.ascontiguousarray(v)
        self.variable_data = json.load(
            open(f"{self.folder_name}/{traj_id}/{run_id}/matrix.json", "r")
        )
        self.time = self.get_time()
        self.gt_mesh, self.target = self.load_mesh()
        if robot:
            self.robot = robot  # TODO: Make it to read the recent added robot_name saved on the logscontroller
        else:
            self.robot = None

    def __str__(self):
        return f"[Logger] Loading from {self.data_folder}"

    def load_mesh(self):
        import open3d as o3d

        scene_json = f"{self.folder_name}/scene.json"
        meshes = o3d.geometry.TriangleMesh()
        target = o3d.geometry.TriangleMesh()
        if os.path.exists(scene_json):
            meshes, target = load_mesh(scene_json)
            # target.transform(objective.A)
        return meshes, target

    def get_traj_o3d(self, robot):
        import open3d as o3d

        base = self.get_data("T_WB")
        q = self.get_data("q")

        eef = []
        base_pos = []
        for T_WB, qi in zip(base,q):
            robot.q = qi
            b = sm.SE3(T_WB, check=False).norm()
            robot.base = b
            base_pos.append(b.t)  # Get the base position
            # Get the end-effector position
            eef.append(robot.fkine(qi).t)

        eef = np.array(eef)
        base_pos = np.array(base_pos)
        base_traj = o3d.geometry.LineSet()
        base_traj.points = o3d.utility.Vector3dVector(base_pos)
        base_traj.lines = o3d.utility.Vector2iVector(
            [[i, i + 1] for i in range(len(base_pos) - 1)]
        )
        base_traj.paint_uniform_color([0, 0, 1])  # Blue for base trajectory
        eef_traj = o3d.geometry.LineSet()
        eef_traj.points = o3d.utility.Vector3dVector(eef)
        eef_traj.lines = o3d.utility.Vector2iVector(
            [[i, i + 1] for i in range(len(eef) - 1)]
        )
        eef_traj.paint_uniform_color([1, 0, 0])  # Red for end-effector trajectory

        return base_traj, eef_traj






    def get_time(self):
        q = self.data["q"]  # We always save the joints
        steps = q.shape[0]
        time = np.arange(0, steps * self.config.step_time, self.config.step_time)
        return time[:steps]

    def get_benchmark_metrics(self) -> Dict:
        stored_keys = self.data.keys()
        benchmark = {}
        total_captured = np.zeros_like(self.time)
        for key in stored_keys:
            if key.startswith("t_"):
                benchmark[key] = self.data.get(key)
                if "total" not in key:
                    total_captured += self.data.get(key)
        benchmark["total_captured"] = total_captured
        return benchmark

    def get_data_time(self, key) -> tuple[np.ndarray, np.ndarray]:
        """
        Return data and the time vector
        """
        return self.get_time(), self.get_data(key)

    def get_distance_target(self) -> float:
        """
        Return the distance to the target pose
        """
        return self.get_data("et")[-1]

    def get_min_distance(self) -> float:
        """
        Return the minimum gt distance to obstacles
        """
        return self.get_data("gt_distance").min()

    def get_min_pred_distance(self) -> float:
        """
        Return the minimum predicted distance to obstacles
        """
        return self.get_data("pred_distance").min()

    def get_data(self, key):
        try:
            return self.data[key]
        except KeyError:
            print(f"Could not find key {key} at {self.data_folder}")
            print("Available keys:")
            with open(self.data_folder + "/info.txt", "r") as f:
                print(f.read())
            raise KeyError

    def get_variable_data(self, key):
        return self.variable_data[key]

    def plot_data(self, key, label="", min=False, **kwargs):
        data = self.data[key]
        if min:
            data = data.min(axis=1)
        plt.plot(self.time, data, label=label, **kwargs)

    def plot_range(self, key, key_max, key_min, label="", **kwargs):
        data = self.data[key]
        data_max = self.data[key_max]
        data_min = self.data[key_min]
        plt.plot(self.time, data_max, label="75")
        plt.plot(self.time, data_min, label="25")
        plt.fill_between(self.time, data_min, data_max, alpha=0.5, **kwargs)
        plt.plot(self.time, data, label=label, **kwargs)

    def collided(self):
        try:
            gt_distance = self.get_data("gt_distance")
        except KeyError:
            print("No gt_distance found, assuming no collision")
            return False
        return np.any(gt_distance < 0.0)

    def pred_collided(self):
        pred_distance = self.get_data("pred_distance")
        return np.any(pred_distance < 0.0)

    def get_all_times(self):
        """
        Get all the measured times from the data
        """
        times = {}
        for key in self.data.keys():
            if key.startswith("t_"):
                times[key] = self.data[key]

        # add a measured time using all the keys but not t_total
        t_measured = np.zeros_like(self.time)
        for key in times.keys():
            if key != "t_total":
                t_measured += times[key]
        times["t_measured"] = t_measured
        return times


    def reached(self):
        target_distance = self.get_data("et")
        return np.any(target_distance < 0.02)

    def closest_target_distance(self):
        target_distance = self.get_data("et")
        return target_distance.min()

    def average_distance(self):
        try:
            gt_distance = self.get_data("gt_distance")
            closest = np.min(gt_distance, axis=1)
        except KeyError:
            print("No gt_distance found, assuming no distance")
            return 0.0
        return np.mean(closest)

    def average_manipulability(self, robot):
        q_seq = self.get_data("q")
        mani = []
        for q in q_seq:
            mani.append(robot.manipulability(q))
        return np.mean(mani)

    def get_length_trajectory(self, robot:NeuralRobot) -> float:
        """
        Get length of the trajectory of the end-effector
        """
        q_seq = self.get_data("q")
        T_WB = self.get_data("T_WB")
        eef_traj = []
        for q, T in zip(q_seq, T_WB):
            robot.q = q
            robot.base = sm.SE3(T, check=False).norm()
            eef_traj.append(robot.fkine(q).t)
        eef_traj = np.array(eef_traj)
        return np.linalg.norm(np.diff(eef_traj, axis=0), axis=-1).sum()


    def get_gt_distance_link(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the minimum gt distance of each link
        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing:
            - Distance of each spehre to the closet obstacje [steps, N]
            - Gradient of the distance [steps, N , 3]
        """
        distance = self.get_data("gt_distance")
        grad = self.get_data("gt_distance_grad")
        return distance, grad

    def pred_error_closest(
        self, pred_distance: np.ndarray, gt_distance: np.ndarray, euclidean=False
    ) -> np.ndarray:
        """
        Compute gt - pred = error
        error < 0 unsafe
        error > 0 overconstraint
        Args:
            pred_distance (np.ndarray): Predicted distance per link [step, N, 1],
                if not euclidean we need to get min accross last dim
            gt_distance (np.ndarray): GT distance per link [step, N]
        Returns:
            gt - pred (np.ndarray) [steps, N]
        """
        if not euclidean:
            pred_distance = np.min(pred_distance, axis=-1)

        mask = gt_distance < self.config.di # Only consider error when the distance is considered
        pred_distance = pred_distance.squeeze()


        error = (gt_distance - pred_distance)
        error[~mask] = 0.0
        return error

    def pred_error(self,
        pred_distance: np.ndarray,
        pred_grad:np.ndarray,
        gt_distance: np.ndarray,
        gt_grad:np.ndarray,
        euclidean=False
    ) -> np.ndarray:
        """
        Compute gt - pred = error
        error < 0 unsafe
        error > 0 overconstraint
        Args:
            pred_point (np.ndarray): Predicted distance per link [step, N, 3],
                if not euclidean we need to get min accross last dim
            gt_distance (np.ndarray): GT distance per link [step, N]
        Returns:
            gt - pred (np.ndarray) [steps, N]
        """

        mask = gt_distance < self.config.di # Shape (Steps, N)
        pred_d = np.copy(pred_distance)
        gt_d = np.copy(gt_distance)

        gt_point = gt_grad * gt_d[..., np.newaxis]
        pred_point = pred_grad * pred_d[..., np.newaxis]

        gt_point = gt_point[:,:,np.newaxis,:] # Shape (steps, N, 1, 3)

        distance = np.linalg.norm(gt_point - pred_point, axis = -1) #/gt_distance[...,np.newaxis]
        closest = distance.min(axis = -1) #/ gt_distance

        # dot_product = np.sum(gt_point * pred_point, axis = -1) # shape (steps, N, 6)
        # dot_product = np.nan_to_num(dot_product, nan = -np.inf)
        # # dot_product[~mask] = 1.0
        # closest = dot_product.max(axis = -1)
        return closest




    def pred_error_min(
        self, pred_distance: np.ndarray, gt_distance: np.ndarray, euclidean=False
    ) -> np.ndarray:
        """
        Compute error of the closest gt distance - pred min distance
        error < 0 unsafe
        error > 0 overconstraint
        Args:
            pred_distance (np.ndarray): Predicted distance per link [step, N, 1],
                if not euclidean we need to get min accross last dim
            gt_distance (np.ndarray): GT distance per link [step, N]
        Returns:
            gt - pred (np.ndarray) [steps, 1]
        """
        if not euclidean:
            pred_distance = np.min(pred_distance, axis=-1)

        pred_distance = pred_distance.squeeze()
        pred_distance = np.min(pred_distance, axis=-1)
        gt_distance = np.min(gt_distance, axis=-1)
        error = (gt_distance - pred_distance)

        return error



    def plot_error(self, ax:Axes, euclidean = False, label = ""):
        gt_distance, gt_grad = self.get_gt_distance_link()
        pred_distance, pred_grad = self.get_distance_link(euclidean=euclidean)


        error = self.pred_error(pred_distance, pred_grad, gt_distance, gt_grad, euclidean=euclidean)

        # Plot the error
        ax.plot(self.time, error.mean(axis=-1), label="Error")
        # t = np.repeat(self.time, error.shape[-1])
        # ax.scatter(t, error )
        ax.fill_between(
            self.time,
            error.min(axis=-1),
            error.max(axis=-1),
            alpha=0.5,
            label=label,
        )
        return

    def plot_error_individual(self, ax:Axes, euclidean = False, label = ""):
        gt_distance, gt_grad = self.get_gt_distance_link()
        pred_distance, pred_grad = self.get_distance_link(euclidean=euclidean)
        error = self.pred_error_closest(pred_distance, gt_distance, euclidean=euclidean)

        # Plot the error
        ax.plot(self.time, error.mean(axis=-1), label="Error")
        ax.fill_between(
            self.time,
            error.min(axis=-1),
            error.max(axis=-1),
            alpha=0.5,
            label=label,
        )
        ax.set_xlim(self.time.min(), self.time.max())
        return

    def plot_error_min(self, ax:Axes, euclidean = False, label = ""):
        gt_distance, gt_grad = self.get_gt_distance_link()
        pred_distance, pred_grad = self.get_distance_link(euclidean=euclidean)
        error = self.pred_error_min(pred_distance, gt_distance, euclidean=euclidean)

        # Plot the error
        ax.plot(self.time, error)
        ax.fill_between(
            self.time,
            error.min(axis=-1),
            error.max(axis=-1),
            alpha=0.5,
            label=label,
        )
        ax.set_xlim(self.time.min(), self.time.max())
        return



    def plot_comparisson(self, ax: Axes):
        ds = self.config.ds
        di = self.config.di

        # Get nicer red and yellow from seaborn's "bright" palette
        palette = sns.color_palette("bright")
        sb_red = palette[3]  # Red (index 3 in "bright")
        sb_yellow = palette[1]  # Yellow (index 1 in "bright")

        ax.fill_between(
            self.time,
            ds,
            -0.01,
            color=sb_red,
            alpha=0.3,
        )
        ax.fill_between(
            self.time,
            ds,
            di,
            color=sb_yellow,
            alpha=0.3,
        )

        ax.set_ylim(-0.01, di + 0.2)
        ax.set_xlim(self.time.min(), self.time.max())
        ax.axhline(ds, label="ds", color=sb_red)
        ax.axhline(di, label="di", color=sb_yellow)
        ax.plot(
            self.time,
            self.get_data("gt_distance").min(axis=-1),
            label=r"$d_{gt}$",
            color="blue",
        )
        ax.plot(
            self.time,
            self.get_data("d_distance").min(axis=-1),
            label=r"$\hat{d}$",
            color="green",
        )
        ax.legend()
        ax.set_xlabel("Time [s]")
        ax.set_title("Distance to obstacles")
        ax.set_ylabel(r"$d$")
        return

    def plot_distance_target(self, ax:Axes):
        et = self.get_data("et")
        ax.plot(
            self.time,
            et,
            color = "black",
            linestyle=":"
        )
        return



    def get_distance_link(self, euclidean=False) -> tuple[np.ndarray, np.ndarray]:
        """Get the minimum distance per link for the
        Args:
            euclidean (bool): if using euclidean or depth sensor
        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing:
            - Distance of each sphere to the closest obstacle [Steps, N, 1 if
                euclidean or 6 if depth]
            - Gradient of the distane [Steps, N, 1 or 6 , 3]
        """
        distance = self.get_data("d_distance")
        gradient = self.get_data("sensor_grad")

        if euclidean:
            return reshape_euclidean_sensor(distance, gradient)
        else:
            return reshape_depth_sensor(distance, gradient)

    def compute_collision_jacobian(
        self, euclidean=False
    ) -> (
        np.ndarray
    ):  # TODO: add load of the robot_name, at the moment defaulted to curobo
        """
        Compute the spatial jacobian of each of the sampled point on the robot
        Returns:
            J_v (np.ndarray): Manipulation jacobian size [Steps, n_dof, ]
        """
        from remo_splat.controller import (
            compute_collision_constraints, compute_collision_cost)

        distance = torch.from_numpy(self.get_data("d_distance")).cuda()
        gradient = torch.from_numpy(self.get_data("sensor_grad")).cuda()
        gt_grad = torch.from_numpy(self.get_data("gt_grad")).cuda()
        gt_distance = torch.from_numpy(self.get_data("gt_distance")).cuda()
        q = self.get_data("q")
        T_WB = self.get_data("T_WB")
        J_c = []
        J_d = []
        c = []
        config = copy.deepcopy(self.config)
        config.collision_cost = "w_avg"
        if not self.robot:
            self.robot = NeuralFrankie("curobo", spheres=True)
        for ix, q_i in enumerate(q):
            self.robot.q = q_i
            self.robot.transform = torch.from_numpy(T_WB[ix][:3, :3]).cuda().float()
            A_in, c_in = compute_collision_constraints(
                self.robot, gradient[ix], distance[ix], config
            )
            # gt_Ain, gt_cin = compute_collision_constraints(self.robot,gt_grad[ix] )
            mask = torch.isinf(c_in)
            c_in = c_in[~mask]
            A_in = A_in[~mask]
            d = distance[ix]
            # Filter nan
            d = d[~mask]

            J_d.append(A_in)
            J_c.append(compute_collision_cost(self.robot, A_in, c_in, d, config))

            pass

        J_c = torch.stack(J_c).cpu().numpy()
        plt.plot(J_c)

        plt.gca().set_prop_cycle()
        plt.show()
        # J_d = torch.stack(J_d) # [Steps, N_sensors * 6, dof]

        # Test 1, Flat them all togeher


class LoggerFolderLoader:
    def __init__(self, exp_name):
        self.episodes = load_folder(exp_name)
        n_episodes = len(self.episodes)
        self.exp_name = exp_name

    def __str__(self):
        return f"[Logger] Loading folder {self.exp_name} with a total of {len(self.episodes)} episodes"

    def proccess(self):
        process_data = {"avg_distance": [], "reached": []}
        for episode in self.episodes:
            data = LoggerLoader(episode, "", "")
            process_data["avg_distance"].append(data.average_distance())
            process_data["reached"].append(data.reached())
        return process_data


def central_diff(vel: np.ndarray, t: np.ndarray) -> np.ndarray:
    a = np.zeros_like(vel)
    dt = np.diff(t, axis=0)
    a[0] = (vel[1] - vel[0]) / dt[0]
    a[-1] = (vel[-1] - vel[-2]) / dt[-1]
    for i in range(1, len(vel) - 1):
        a[i] = (vel[i + 1] - vel[i - 1]) / (t[i + 1] - t[i - 1])
    return a

def acc_eef(data, robot, dt=0.05):
    qd = data["qd"]
    q = data["q"]
    eef_accs = []

    t = np.arange(len(qd)) * dt
    qdd = central_diff(qd, t)

    for i in range(len(q)):
        robot.q = q[i]
        robot.base = sm.SE3(data["T_WB"][i], check=False).norm()
        J = robot.jacob0(q[i])
        J_dot = robot.jacob0_dot(q[i], qd[i], J)
        eef_acc = J @ qdd[i] + J_dot @ qd[i]
        eef_accs.append(eef_acc)

    eef_accs = np.array(eef_accs)
    accs = eef_accs[:, :3]  # Remove angular acceleration

    acc = np.linalg.norm(accs, axis=1)

    cumAcc = cumulative_trapezoid(acc, t)

    return acc, cumAcc


def load_folder(folder: str):
    """
    Load log folder under ./logs/experiments/{folder}/ containing either the variations of the episodes
    """
    project_root = os.path.dirname(os.path.abspath(__file__))
    folder_name = os.path.join(project_root, "..", "logs", "experiments", folder)

    files = os.listdir(folder_name)
    data = []
    for f in files:
        if os.path.isdir(os.path.join(folder_name, f)):
            data.append(f"{folder}/{f}")

    # Sort them
    data.sort(key=lambda x: int(x.split("/")[-1]))
    return data


def plot_range(data, label="", axis=0, **kwargs):
    data_max = np.max(data, axis=-1)
    data_min = np.min(data, axis=-1)
    quintil_75 = np.quantile(data, 0.75, axis=-1)
    quintil_25 = np.quantile(data, 0.25, axis=-1)
    # plt.plot(quintil_75, label = "75")
    # plt.plot(quintil_25, label = "25")

    # plt.plot(data_max, label = "75")
    # plt.plot(data_min, label = "25")
    plt.fill_between(
        np.arange(data.shape[0]),
        quintil_25,
        quintil_75,
        alpha=0.5,
        label=label,
        **kwargs,
    )
    plt.plot(data.mean(axis=-1), label=label, **kwargs)


def get_z_angle(data):
    poses = [sm.SE3(T) for T in data]
    angles = [p.rpy() for p in poses]
    return np.array(angles)




def separate_links(robot, data):
    """
    Separate the data into the links of the robot
    the data should be in the shape [N_steps, N_sensors, 6, 1 or 3]
    We want to separate the N_sensors into the links of the robot
    """
    __import__("pdb").set_trace()


def get_mins(distance, gradients):
    """
    Get the minimum distance and gradient for each sensor

    Returns
    -------
    sensor_predictions: np.ndarray
        The minimum distance for each sensor [N_steps, n_sensors]
    sensor_grad_predictions: np.ndarray
        The gradient for the minimum distance [N_steps, n_sensors, 3]
    """
    n_sensor = distance.shape[1]
    index_min = np.argmin(distance, axis=-1)
    rows = np.arange(distance.shape[0])[:, None]  # Shape [N, 1] for broadcasting
    sensor_predictions = distance[
        rows, np.arange(n_sensor), index_min
    ]  # Shape: [N, n_sensors]

    # Use the same indices to gather the corresponding gradients
    sensor_grad_predictions = gradients[
        rows, np.arange(n_sensor), index_min, :
    ]  # Shape: [N, n_sensors, 3]
    return sensor_predictions, sensor_grad_predictions
