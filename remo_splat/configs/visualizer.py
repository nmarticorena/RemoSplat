from dataclasses import dataclass, field
from typing import List

@dataclass
class ExperimentVisualizerConfig:
    folder_name: str = "final"
    "Folder name to load the experiment under experiments/"
    env_names: List[str] = field(default_factory= lambda:["bookshelf", "table_new"])
    "List of env names to load"
    dimensions: List[str] = field(default_factory= lambda:["2D", "3D"])
    "List of dimensions to load"
    sensors: List[str] = field(default_factory= lambda:[ "depth", "euclidean_less", "gt"])
    "List of sensors to load"
    active: List[str] = field(default_factory=lambda:["null", "active"])
    "List of active sensors to load"
    robot_name: str = "curobo"


