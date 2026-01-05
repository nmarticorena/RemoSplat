import json
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import spatialmath as sm
from nerf_tools.dataset.nerf_dataset import NeRFDataset, load_from_json

import remo_splat

path = remo_splat.__path__[0]


def get_arkit_pose(exp_name: str) -> sm.SE3:
    with open(f"{path}/../configs/real_world/{exp_name}.json") as f:
        data = json.load(f)
    try:
        pose = data["scene"]["T_WG"]
    except KeyError:
        print("Key 'T_WG' not found in the JSON data. Using default identity pose.")
        pose = np.eye(4)
    T_WG = sm.SE3(pose, check=False).norm()
    return T_WG


@dataclass
class GSLoader:
    # 7000 for low, 30000 for high
    high: bool = True
    # Name of the scene
    scene: str = "printer_2"
    dataset_folder: str = os.environ["NERF_CAPTURE"]
    is_3D: bool = True
    hist: bool = False
    ply: bool = True
    exp_name: str = ""
    # Transformation from world to gaussian splat frame
    T_WG: sm.SE3 = field(default_factory=lambda: sm.SE3())
    step: Optional[int] = None
    inria: bool = False  # Flag to use gsplat inria wrappers
    last: bool = False  # Flag to use last checkpoint

    def __post_init__(self):
        gs_folder = "3DGS" if self.is_3D else "2DGS"
        gsplat_folder = "gsplat" if self.is_3D else "gsplat_2D"
        if self.step is None:
            self.step = 30000 if self.high else 7000
        folder = "point_cloud/iteration"

        if self.exp_name == "":
            self.exp_name = f"{self.scene}"
            print("Exp name not provided, using scene name " + self.exp_name)

        if self.ply:
            self.folder = f"data/{gs_folder}/{self.exp_name}/{folder}_{self.step}"
            self.file = f"{self.folder}/point_cloud.ply"
        else:
            self.folder = f"data/{gsplat_folder}/{self.exp_name}/ckpts"
            if self.last:
                self.file = f"{self.folder}/last.pt"
            else:
                if self.is_3D:
                    self.file = f"{self.folder}/ckpt_{self.step-1}_rank0.pt"
                else:
                    self.file = f"{self.folder}/ckpt_{self.step-1}.pt"
        print(self.file)

        self.info_file = f"{self.dataset_folder}/{self.scene}/transforms.json"
        self.dataset_helper: NeRFDataset = load_from_json(self.info_file)

    def load(self, scene: str):
        self.scene = scene
        self.exp_name = scene
        self.__post_init__()


@dataclass
class GSplatLoader(GSLoader):
    hist: bool = False
    ply: bool = False

    def load_mesh(self):
        mesh_folder = f"{self.dataset_folder}/{self.scene}/mesh/"
        if not os.path.exists(os.path.join(mesh_folder,"all.stl")):
            os.makedirs(mesh_folder, exist_ok=True)
            print("mesh does not exist")
            return ""

        return f"{mesh_folder}/all.stl"

    def mesh_path(self):
        return f"{self.dataset_folder}/{self.scene}/mesh/all.stl"

@dataclass
class ExampleBookshelf(GSplatLoader):
    scene: str = "bookshelf"
    dataset_folder: str = "/media/nmarticorena/DATA/datasets/nerf_standard"
    high: bool = False


@dataclass
class ReachingLoader(GSplatLoader):
    dataset_folder: str = "data/scenes/ReMoSplat-synthetic"
    high: bool = False
    T_WG: sm.SE3 = field(default_factory=lambda: sm.SE3(0, 0, 0))
    is_3D: bool = False
    step: Optional[int] = 1000


@dataclass
class ReachingBookshelf(ReachingLoader):
    scene: str = "bookshelf_cage/bookshelf_cage_0000"


@dataclass
class ReachingTable(ReachingLoader):
    scene: str = "table_new/table_new_0000"


@dataclass
class ExampleReal(GSplatLoader):
    is_3D: bool = False
    last: bool = True
    scene: str = "bookshelf_room"

    def __post_init__(self):
        self.T_WG: sm.SE3 = get_arkit_pose(self.scene)
        super().__post_init__()


@dataclass
class ExampleDesk(GSLoader):
    scene: str = "desk60gs"
    high: bool = False


@dataclass
class ExampleReplica(GSplatLoader):
    scene: str = "apt_2_nav"
    high: bool = False
    dataset_folder: str = "/media/nmarticorena/DATA/iSDF_data/seqs/"
    T_WG: sm.SE3 = field(
        default_factory=lambda: sm.SE3.Rx(np.pi / 2) * sm.SE3.Ry(np.pi / 2)
    )


@dataclass
class ExampleRealGSplat(GSLoader):
    scene: str = "printer_2"
    high: bool = False
    hist: bool = True
    ply: bool = False
    T_WG: sm.SE3 = field(default_factory=lambda: get_arkit_pose("printer_2"))


@dataclass
class S12Example(GSLoader):
    scene: str = "bookshelf_final"
    high: bool = False
    hist: bool = True
    ply: bool = False
    dataset_folder: str = os.environ["DATASETS"] + "/S12Dataset/real_scans"


if __name__ == "__main__":
    import tyro

    tyro.cli(S12Example)
