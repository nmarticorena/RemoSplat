from typing import Tuple
import json

import numpy as np
from spatialmath import SE3

import remo_splat
path = remo_splat.__path__[0]

def load_arkit(exp_name: str) -> Tuple[SE3,SE3]:
    """
    Load arkit scene information
    Parameters
    ----------
    exp_name
        name of the scene

    Returns
    -------
    T_WB: SE3
        Pose of the base of the robot
    T_Wep: SE3
        List of the target end effector poses
    """
    with open(f"{path}/../configs/real_world/{exp_name}.json") as f:
        data = json.load(f)
    T_WB = data["base"]
    T_WEp = [np.array(p) for p in data["poses"]]
    T_WB = SE3(T_WB, check=False).norm()
    T_WEp = SE3(T_WEp, check = False).norm()
    return T_WB, T_WEp

