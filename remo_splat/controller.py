import math
import time
from typing import Optional, Tuple

import numpy as np
import qpsolvers as qp
import roboticstoolbox as rtb
import spatialmath as sm
import torch
from neural_robot.robot import NeuralRobot
from neural_robot.utils import math as helper_math

from remo_splat.configs.controllers import NeoConfig
from remo_splat.sdf import IdealSDF
from remo_splat.kernels import visuals
from remo_splat.lidar import DepthSensor, DistanceSensor, reshape_depth_sensor
from remo_splat.logger import Logger
from remo_splat.o3d_visualizer import Visualizer


def torch_wavg(tensor: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return torch.sum(tensor * weights[:, None], dim=0) / torch.sum(weights, dim=0)

def compute_collision_cost(robot:NeuralRobot, c_Ain:torch.Tensor, c_bin:torch.Tensor, distance:torch.Tensor, controller_config:NeoConfig) -> torch.Tensor:
    mask = c_bin < 1
    c_bin = c_bin[mask].detach()
    c_Ain = c_Ain[mask].detach()

    if c_Ain.shape[0] != 0:
        if controller_config.collision_cost == "min":
            min_distance = torch.argmin(c_bin)
            collision = c_Ain[min_distance, :]
        elif controller_config.collision_cost == "avg":
            collision = torch.mean(c_Ain, dim=0)
        elif controller_config.collision_cost == "w_avg":
            weights = 1 - c_bin
            collision = torch_wavg(c_Ain, weights)
        elif controller_config.collision_cost == "w2_avg":
            weights = 1 - c_bin**2
            collision = torch_wavg(c_Ain, weights)
        elif controller_config.collision_cost == "w3_avg":
            weights = 1 - c_bin**2
            collision = torch_wavg(c_Ain, weights**2)
        else:
            collision = torch.zeros(robot.n + 6).cuda()
    else:
        collision = torch.zeros(robot.n + 6).cuda()
    if controller_config.gt_collisions:
        x = min_distance
    else:
        x = torch.min(distance).detach().cpu().item()

    di = controller_config.di
    ds = controller_config.ds
    collision_lamda = (controller_config.collision_gain / (di - ds) ** 2) * (x - di) ** 2
    if c_Ain.shape[0] != 0:
        # append an extra 6 zeros to match the size of the QP problem
        collision = torch.cat((collision, torch.zeros(6, device="cuda")))
    return (collision_lamda * collision).detach()


def compute_collision_constraints(
    robot:NeuralRobot, g_w:torch.Tensor, distance:torch.Tensor, controller_config:NeoConfig
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the collision constraints for the QP problem.
    Args:
        g_w (torch.Tensor): The spatial gradient of the distance in the world frame [n,3].
        distances (torch.Tensor): Each of the distances [n,1].
    Returns:
        torch.Tensor: The Ain matrix for the QP problem. [n*+6, n*+6]
        torch.Tensor: The bin matrix for the QP problem. [n*+6]
        where n* is the len(distances<config.di)
    """

    n = g_w.shape[0]
    n_points = len(robot.radii)
    ratio = int(n / n_points)

    Jac = robot.get_jacobians_collision().repeat_interleave(ratio, dim = 0)

    mask = distance < controller_config.di

    g_w = g_w[mask].clone()
    d = distance[mask]
    Jac = Jac[mask]
    g_b = helper_math.rotate_vec(g_w, robot.transform)
    norm_h = torch.zeros(g_b.shape[0], 6).cuda()
    norm_h[:, :3] = g_b
    c_Ain = 1 * norm_h.unsqueeze(1) @ Jac
    c_bin = (
        controller_config.xi
        * (d - controller_config.ds)
        / (controller_config.di - controller_config.ds)
    )
    c_Ain = c_Ain.squeeze()
    if len(c_Ain.shape) == 1:
        c_Ain = c_Ain.unsqueeze(0)


    return c_Ain, c_bin

class MMController:
    def __init__(
        self,
        robot: NeuralRobot,
        sensor: Optional[DistanceSensor],
        config: NeoConfig = NeoConfig(),  # Add config
        gui: Optional[Visualizer] = None,
        logger: Optional[Logger] = None,
    ):
        self.robot = robot
        self.sensor = sensor
        self.config = config
        # self.config.ds = 0.01 # Force higher distance
        self.gui = gui
        self.logger = logger

    def get_ve(self, T_WEp) -> Tuple[np.ndarray, float]:
        """
        Calculate the end-effector velocity based on the current and desired poses.
        Args:
            T_We (sm.SE3): Current pose of the end-effector in the world frame.
            T_WEp (sm.SE3): Desired pose of the end-effector in the world frame.
        Returns:
            v_e (np.ndarray): The end-effector target velocity represented as a twist (v,w).
            pose_error (float): The error between the current and desired poses.
        """
        T_We = self.robot.fkine(self.robot.q)
        T_eEp = T_We.inv() * T_WEp

        pose_error = np.sum(np.abs(T_eEp.t)) + 1e-10
        v_e, _ = rtb.p_servo(
            T_We,
            T_WEp,
            self.config.beta,
            method="rpy",
            threshold=self.config.precision,
        )


        norm_ev = np.linalg.norm(v_e[:3])
        if norm_ev > self.config.max_ev:
            # only cap the translational component
            v_e[:] = (self.config.max_ev / norm_ev) * v_e[:]

        return v_e, pose_error

    def create_Q(self, et: float) -> np.ndarray:
        """
        Create the Q matrix for the QP problem.
        Args:
            et (float): The error between the current and desired poses, used for the slack
            gain
        Returns:
            np.ndarray: The Q matrix for the QP problem. [n+6, n+6]
        """
        Q = np.eye(self.robot.n + 6)

        Q[: self.robot.n, : self.robot.n] *= self.config.lamda_q

        Q[: self.robot.base_dofs, : self.robot.base_dofs] *= 1.0 / (et)
        et_for_slack = 1 / (et * 1)

        Q[self.robot.n :, self.robot.n :] = et_for_slack * np.eye(6)
        return Q

    def initialize_constraints(
        self, v_e: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Initialize the constraints for the QP problem.
        Returns:
            np.ndarray: The Aeq matrix for the QP problem. [6, n+6]
            np.ndarray: The beq matrix for the QP problem. [6]
            np.ndarray: The Ain matrix for the QP problem. [n+6, n+6]
            np.ndarray: The bin matrix for the QP problem. [n+6]
        """
        Aeq = np.c_[self.robot.jacobe(self.robot.q), np.eye(6)]
        Ain = np.zeros((self.robot.n + 6, self.robot.n + 6))
        bin = np.zeros(self.robot.n + 6)
        beq = v_e.reshape((6,))
        return Aeq, beq, Ain, bin




    def step(
        self, T_WEp: sm.SE3, T_GW: sm.SE3 = sm.SE3(), gt_sdf: Optional[IdealSDF] = None
    ) -> Tuple[bool, np.ndarray, bool]:
        """
        Step the controller to update the robot's state based on a desired end-effector pose.

        Args:
            T_WEp (sm.SE3): Desired pose of the end-effector in the world frame.
            T_GW (sm.SE3): Transform from the world to the gaussian world.

        Returns:
            tuple:
                done (bool): Whether the controller has completed its task.
                qd (list[float]): Current joint velocities of the robot.
                failed (bool): Whether the controller encountered a failure.
        """
        total_ti = time.perf_counter()
        ti = time.perf_counter()
        executed_times = {}
        v_e, pose_error = self.get_ve(T_WEp)

        self.robot.transform[:3, :3] = torch.tensor(
            self.robot.base.R.T, device="cuda", dtype=torch.float32
        )  # R_BW
        Q = self.create_Q(pose_error)

        Aeq, beq, Ain, bin = self.initialize_constraints(v_e)

        # Get robot poses
        Ain[: self.robot.n, : self.robot.n], bin[: self.robot.n] = (
            self.robot.joint_velocity_damper(
                self.config.ps, self.config.pi, self.robot.n
            )
        )
        # Manipulability
        c = np.concatenate(
            (
                np.zeros(self.robot.base_dofs),
                -self.robot.jacobm(
                    start=self.robot.links[self.robot.base_dofs + 2]
                ).reshape((self.robot.n - self.robot.base_dofs,)),
                np.zeros(6),
            )
        )
        executed_times["initialization"] = time.perf_counter() - ti
        if self.config.collisions:
            ti = time.perf_counter()

            T_WG = torch.inverse(T_GW)
            if not self.config.ideal:
                if isinstance(self.sensor, DepthSensor):
                    X_WSp = self.robot.get_point_poses().detach()
                    X_GSp = T_GW @ X_WSp
                else:
                    X_WSp = self.robot.transform_points().detach()
                    X_GSp = helper_math.transform_points(X_WSp, T_GW)

                gradient, distance = self.sensor.get_distance(
                    X_GSp, self.robot.SpheresRadius
                )
                if self.config.only_min and isinstance(self.sensor, DepthSensor):
                    distance , gradient = reshape_depth_sensor(distance, gradient) # Shape [n,6,1] and [n,6,3]
                    index = torch.argmin(distance, dim=1)
                    distance = distance[torch.arange(distance.shape[0]), index]
                    gradient = gradient[torch.arange(gradient.shape[0]), index]

            else:
                if self.config.ideal_all:
                    gradient, distance, _ = self.robot.get_distance_all(gt_sdf) # [n_points, n_obs, 3], [n_points, n_obs]
                    n_obs = gradient.shape[1]
                    gradient = gradient.reshape(-1, 3)
                    distance = distance.reshape(-1)
                else:
                    gradient, distance, _ = self.robot.get_distance(gt_sdf) #[n_points, 3], [n_points]
                gradient = gradient * -1
            executed_times["distance"] = time.perf_counter() - ti
            if (
                gt_sdf is not None
                and self.gui is not None
                and isinstance(self.sensor, DepthSensor)
            ):
                gt_grad, gt_distance, _ = self.robot.get_distance(gt_sdf)

                gt_lines = visuals.line_from_dist_grad(
                    X_WSp[:, :3, -1], -gt_grad, gt_distance + self.robot.SpheresRadius
                )
                self.gui.update_geometry("gt_lines", gt_lines)

            # if isinstance(self.sensor, DepthSensor) and self.gui is not None:
            if self.gui is not None:
                self.sensor.gui_debug(gradient, distance, self.gui, gt_sdf, None)
                # lines = self.sensor.draw_lines(None, gradient, distance)
                # camera_lines = self.sensor.panorama_sensor.draw_camera_lines(
                #     self.sensor.T_CWs, 0.01
                # )
                # self.gui.update_geometry("lines", lines)
                # self.gui.update_geometry("camera_lines", camera_lines)
                # if gt_sdf is not None:
                #     pcd = self.sensor.debug_pcd(error)
                # else:
                #     pcd = self.sensor.debug_pcd()
                #
                # self.gui.update_geometry("pcd", pcd)
            ti = time.perf_counter()
            # This gradient is wrt to the gaussian frame
            d_distance = distance.min().item()
            # Question is this gradient in the world frame or the gaussian world frame, at the moment is in the gaussian frame
            gradient = helper_math.rotate_vec(
                gradient, T_WG[:3, :3]
            )  # This is the gradient in the base robot frame
            c_Ain, c_bin = compute_collision_constraints(self.robot, gradient, distance, self.config)
            # Clean all the nans
            mask = torch.isinf(c_bin)

            c_bin = c_bin[~mask]
            c_Ain = c_Ain[~mask]

            executed_times["collision_constraints"] = time.perf_counter() - ti


            if self.config.collision_cost != "" and c_Ain.shape[0] > 0:
                ti = time.perf_counter()
                c += compute_collision_cost(self.robot, c_Ain, c_bin, distance, self.config).cpu().numpy()
                executed_times["collision_cost"] = time.perf_counter() - ti

            c_Ain = c_Ain.detach().cpu().numpy()
            c_bin = c_bin.detach().cpu().numpy()

            c_Ain = np.c_[c_Ain, np.zeros((c_Ain.shape[0], 6))]

            if c_Ain is not None and c_bin is not None:
                Ain = np.r_[Ain, c_Ain]
                bin = np.r_[bin, c_bin]

        ti = time.perf_counter()
        # orientation cost
        kε = 0.5
        bTe = self.robot.fkine(self.robot.q, include_base=False).A
        θε = math.atan2(bTe[1, -1], bTe[0, -1])
        ε = kε * θε
        c[self.robot.base_dofs - 1] = c[self.robot.base_dofs - 1] - ε


        lb = -1 * np.r_[self.robot.qdlim[: self.robot.n], 10 * np.ones(6)]
        ub = np.r_[self.robot.qdlim[: self.robot.n], 10 * np.ones(6)]

        mask = np.all(Ain == 0, axis=1)
        Ain = Ain[~mask]
        bin = bin[~mask]
        executed_times["extra_costs"] = time.perf_counter() - ti

        torch.cuda.synchronize()
        ti = time.perf_counter()
        qd = qp.solve_qp(Q, c, Ain, bin, Aeq, beq, lb=lb, ub=ub, solver="quadprog")

        executed_times["qp"] = time.perf_counter() - ti

        failed = qd is None

        if qd is not None:
            oqd = qd.copy()
            qd = qd[: self.robot.n]

            finished = pose_error < self.config.precision
        else:
            qd = np.zeros(self.robot.n)
            oqd = np.zeros(self.robot.n + 6)
            print(failed)
            finished = True
        self.debug_qd = qd
        executed_times["total"] = time.perf_counter() - total_ti
        if self.logger is not None:
            data = {
                "q": np.copy(self.robot.q[: self.robot.n]),
                "qd": qd,
                "oqd": oqd,
                "et": pose_error,
                "v_e": v_e,
                "T_WB": np.copy(self.robot._T),
            }
            for key, value in executed_times.items():
                data["t_"+key] = value
            if self.config.collisions:
                data["n_collisions"] = c_Ain.shape[0]
                data["d_distance"] = distance.detach().cpu().numpy()
                data["sensor_grad"] = gradient.detach().cpu().numpy()
                data["pred_distance"] = d_distance

            if gt_sdf is not None:
                gt_grad, gt_distance, _ = self.robot.get_distance(gt_sdf)
                gt_distance = gt_distance.cpu().numpy()
                data["gt_distance"] = gt_distance
                data["gt_distance_grad"] = gt_grad.detach().cpu().numpy()

            self.logger.log(data)

        return finished, qd, failed


if __name__ == "__main__":
    example = MMController(None, None, None)
    a, b, c = example.step(sm.SE3())
