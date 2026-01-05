import enum
import json
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np
import open3d as o3d
import spatialgeometry as sg
import spatialmath as sm
from remo_splat.sdf import load_from_json_full

from remo_splat.arkit_utils import load_arkit
import remo_splat

from . import controllers, gs



@dataclass
class SensorParams:
    width: int = 80
    height: int = 80
    fov: int = 90


class Sensor(enum.IntEnum):
    depth = 0
    "Using backprojection"
    euclidean = 1
    "Using euclidean distance"
    euclidean_less = 2
    "Using euclidean distance with less maximum steps"
    euclidean_all = 3
    "Using euclidean but with all the points"
    depth_min = 4
    "Using depth projection but only the closer one in the 6 directions"
    gt = -1
    "Using the gt information of the scene, only valid on sim"
    gt_active = -2
    "GT info and active collision cost"
    gt_active_faster = -3
    "Gt information with active collision cost but a faster control loop"
    gt_all = -4
    "SDF using the distance to each primitive"

    def __str__(self) -> str:
        return self.name


@dataclass
class ExperimentConfig:
    gsplat: gs.GSplatLoader
    exp_name: str = "bookshelf_test"
    T_WB: sm.SE3 = field(default_factory=lambda: sm.SE3(0, 0, 0))
    robot_name: str = "gsplat_simple"
    trajectory_file: str = "reaching"
    log: bool = True
    gui: bool = True
    step_time: float = 0.01
    cameras: SensorParams = field(default_factory=lambda: SensorParams())
    controller: controllers.NeoConfig = field(
        default_factory=lambda: controllers.NeoConfig()
    )
    n_repeat: int = 1
    sensor: Sensor = Sensor.depth
    mesh_file: str = ""
    "Mesh file to load"
    mesh: sg.Mesh = field(init=False)

    def __post_init__(self):
        from remo_splat.utils import CameraParams

        self.mesh_file = os.path.abspath(
            os.path.join("results", "meshes", f"{self.gsplat.scene}.ply")
        )
        self.mesh = sg.Mesh(self.mesh_file)
        self.mesh.T = self.gsplat.T_WG
        self.traj_folder = os.path.abspath(
            os.path.join("data", "trajs", self.gsplat.scene)
        )
        if os.path.exists(self.traj_folder):
            self.traj_files = os.listdir(self.traj_folder)
            self.traj_files = [
                os.path.join(self.traj_folder, f) for f in self.traj_files
            ]
        else:
            self.traj_files = []

        self.camera = CameraParams(
            self.cameras.width, self.cameras.height, fov=self.cameras.fov
        )


def obstacles_to_mesh(obstacles: List[sg.CollisionShape]):
    mesh = o3d.geometry.TriangleMesh()
    for obs in obstacles:
        if isinstance(obs, sg.Cuboid):
            box = o3d.geometry.TriangleMesh.create_box(
                obs.scale[0], obs.scale[1], obs.scale[2]
            )
            box.translate(obs.T[:3,-1], relative = False)
            mesh += box
        elif isinstance(obs, sg.Cylinder):
            cylinder = o3d.geometry.TriangleMesh.create_cylinder(obs.radius, obs.length)
            cylinder.translate(obs.T[:3,-1], relative = False)
            mesh += cylinder
    return mesh


def load_mesh(
    scene_info_path: str,
) -> Tuple[o3d.geometry.TriangleMesh, o3d.geometry.TriangleMesh]:
    def get_cyl(cylinder):
        position = np.array(cylinder["position"])
        radius = cylinder["radius"]
        height = cylinder["heigth"] * 2  # typo in the json file
        return o3d.geometry.TriangleMesh.create_cylinder(radius, height).translate(
            position, relative=False
        )

    def get_box(box):
        position = np.array(box["position"])
        scale = np.array(box["scale"]) * 2
        return o3d.geometry.TriangleMesh.create_box(*scale).translate(
            position, relative=False
        )

    with open(scene_info_path) as f:
        data = json.load(f)
        mesh = o3d.geometry.TriangleMesh()
        for cylinder in data.get("cylinders",[]):
            mesh += get_cyl(cylinder)
        for box in data.get("cubes",[]):
            mesh += get_box(box)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.5, 0.5, 0.5])
        target_pose = data.get("target", None)
        if target_pose is not None:
            target = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
            target.transform(np.array(target_pose))
        poses = data.get("poses", None)
        if poses is not None:
            target  = o3d.geometry.TriangleMesh()
            for pose in poses:
                _target = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
                _target.transform(np.array(pose))
                target += _target

    return mesh, target


@dataclass
class ReachingExperimentConfig:
    gsplat: gs.GSplatLoader

    exp_name: str = "reaching_test"
    "Exp name"
    T_WB: sm.SE3 = field(default_factory=lambda: sm.SE3(0, 0, 0))
    "Base pose w/r to the world"
    robot_name: str = "gsplat_simple"
    "Robot geometry to load"
    spheres: bool = True
    "Use spheres instead of points"
    trajectory_file: str = "reaching"
    "Exp type"
    log: bool = True
    "Log the steps"
    gui: bool = True
    "Swift gui"
    o3d_vis: bool = False
    "open3d custom gui"
    cameras: SensorParams = field(default_factory=lambda: SensorParams())
    "Default sensor to use if depth sensor is used"
    controller: controllers.NeoConfig = field(
        default_factory=lambda: controllers.NeoConfig()
    )
    "Controller config"
    env_type: Optional[str] = None
    ""
    sensor: Sensor = Sensor.depth
    "Sensor to use"
    length: float = 15
    "Time duration of experiment"
    mesh_file: str = ""
    "Mesh file to load"
    mesh: sg.Mesh = field(init=False)
    "SG mesh to load on swift"



    def __post_init__(self):
        if self.env_type is None:
            self.env_type = self.gsplat.scene
        from remo_splat.utils import CameraParams

        self.mesh_file = os.path.abspath(
            os.path.join("results", "meshes", f"{self.gsplat.scene}.ply")
        )
        if os.path.exists(self.mesh_file):
            self.mesh = sg.Mesh(self.mesh_file)
            self.mesh.T = self.gsplat.T_WG
        else:
            self.mesh = sg.Mesh()
            self.mesh.T = self.gsplat.T_WG
            print(f"Mesh file {self.mesh_file} not found, using empty mesh")

        self.steps = int(self.length / self.controller.step_time )
        self.load_scene()
        self.camera = CameraParams(
            self.cameras.width, self.cameras.height, fov=self.cameras.fov
        )
        self.controller.robot_config = self.robot_name

    def load_scene(self):
        self.json_path = f"{remo_splat.__path__[0]}/../data/{self.gsplat.scene}.json"
        self.gt_sdf, self.obstacles, self.objective, self.T_WB = load_from_json_full(
            self.json_path
        )
        self.o3d_mesh = obstacles_to_mesh(self.obstacles)

    def load(self, index:int):
        """
        Load the next gsplat
        """
        macro = self.gsplat.scene.split("_")
        macro[-1] = f"{index:04d}"
        name = "_".join(macro)
        self.gsplat.load(name)

        self.json_path = f"{remo_splat.__path__[0]}/../data/{self.gsplat.scene}.json"
        self.load_scene()

@dataclass
class Bookshelf(ExperimentConfig):
    gsplat: gs.GSplatLoader = field(default_factory=lambda: gs.ExampleBookshelf())
    T_WB: sm.SE3 = field(default_factory=lambda: sm.SE3(-0.5, 0, 0))
    robot_name: str = "gsplat_simple"
    exp_name: str = "bookshelf"


@dataclass
class Bookshelf2D(ExperimentConfig):
    gsplat: gs.GSplatLoader = field(
        default_factory=lambda: gs.ExampleBookshelf(is_3D=False)
    )
    T_WB: sm.SE3 = field(default_factory=lambda: sm.SE3(-0.5, 0, 0))
    robot_name: str = "gsplat_simple"
    exp_name: str = "bookshelf"


@dataclass
class ReplicaCAD(ExperimentConfig):
    gsplat: gs.GSplatLoader = field(default_factory=lambda: gs.ExampleReplica())
    T_WB: sm.SE3 = field(default_factory=lambda: sm.SE3(0.1, 0, 0))
    robot_name: str = "gsplat_simple"
    exp_name: str = "bookshelf"


@dataclass
class ReplicaCAD2D(ExperimentConfig):
    gsplat: gs.GSplatLoader = field(
        default_factory=lambda: gs.ExampleReplica(is_3D=False)
    )
    T_WB: sm.SE3 = field(default_factory=lambda: sm.SE3(0.1, 0, 0))
    robot_name: str = "gsplat_simple"
    exp_name: str = "bookshelf"


@dataclass
class ARKit(ExperimentConfig):
    gsplat: gs.GSplatLoader = field(default_factory=lambda: gs.ExampleReal(is_3D=True))
    T_WB: sm.SE3 = field(default_factory=lambda: sm.SE3(0.1, 0, 0))
    robot_name: str = "gsplat_simple"
    exp_name: str = "bookshelf"


@dataclass
class ReplicaCADMCMC(ExperimentConfig):
    gsplat: gs.GSplatLoader = field(
        default_factory=lambda: gs.ExampleReplica(exp_name="apt_2_nav_mcmc")
    )
    T_WB: sm.SE3 = field(default_factory=lambda: sm.SE3(0.1, 0, 0))
    robot_name: str = "gsplat_simple"
    exp_name: str = "bookshelf"


@dataclass
class ReachingBookshelf(ReachingExperimentConfig):
    gsplat: gs.GSplatLoader = field(default_factory=lambda: gs.ReachingBookshelf())
    T_WB: sm.SE3 = field(default_factory=lambda: sm.SE3(-0.5, 0, 0))
    robot_name: str = "gsplat_simple"
    exp_name: str = "bookshelf"
    env_type: str = "bookshelf_cage"


@dataclass
class ReachingTable(ReachingExperimentConfig):
    gsplat: gs.GSplatLoader = field(default_factory=lambda: gs.ReachingTable())
    T_WB: sm.SE3 = field(default_factory=lambda: sm.SE3(-0.5, 0, 0))
    robot_name: str = "gsplat_simple"
    exp_name: str = "table_new"
    env_type: str = "table_new"


@dataclass
class ReachingRealWorldConfig(ReachingExperimentConfig):
    gsplat: gs.ExampleReal#  = field(default_factory=lambda: gs.ExampleReal())# type: ignore
    T_WB: sm.SE3 = field(default_factory=lambda: sm.SE3(0.1, 0, 0))
    robot_name: str = "gsplat_rbf"
    exp_name: Optional[str] = None

    def load_scene(self):
        if self.exp_name is None:
            self.exp_name = self.gsplat.exp_name
        self.json_path = f"{remo_splat.__path__[0]}/../configs/real_world/{self.gsplat.scene}.json"
        self.gt_sdf = None
        self.obstacles = []
        self.T_WB, self.T_WEp = load_arkit(self.exp_name)
        self.objective = self.T_WEp[0]
        self.T_GC = sm.SE3(self.gsplat.dataset_helper.get_transforms_cv2(),check=False).norm()
        self.T_WC = self.gsplat.T_WG * self.T_GC

    def load(self, index:int):
        """
        Load the next pose
        """
        self.objective = self.T_WEp[index]

