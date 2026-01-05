from dataclasses import dataclass, field
from typing import List


@dataclass
class PostProcessConfig:
    exp_name: str
    envs: List[str] = field(default_factory=lambda: ["bookshelf", "table_new"])
    dimensions: List[str] = field(default_factory=lambda: ["2D", "3D"])
    sensors: List[str] = field(
        default_factory=lambda: ["depth", "depthactive",
                                 "depth_min", "depth_minactive",
                                 "euclidean", "euclideanactive",
                                 "euclidean_less", "euclidean_lessactive",
                                 "gt", "gtactive", "gt_active_faster",
                                 "gt_all", "gt_allactive"]
    )
    n_episodes: int = 500

    def get_path(self, env, dim , sensor):
        if "gt" in sensor:
            return f"{self.exp_name}/{env}/{sensor}"
        return f"{self.exp_name}/{env}/{dim}/{sensor}"

