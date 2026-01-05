import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tyro
from matplotlib.colors import BoundaryNorm, ListedColormap
from tqdm import tqdm

from remo_splat import logger


@dataclass
class Args:
    exp_name: str
    envs: List[str] = field(default_factory=lambda: ["bookshelf", "table_new"])
    dimensions: List[str] = field(default_factory=lambda: ["2D", "3D"])
    sensors: List[str] = field(
        default_factory=lambda: ["depth", "depthactive",
                                 "euclidean", "euclideanactive",
                                 "euclidean_less", "euclidean_lessactive",
                                 "gt", "gt_active", "gt_active_faster"]
    )
    load: bool = False
    n_episodes: int = 500


# Script that post process the logs of the rmmi.py benchmark
class PostProcess:
    def __init__(self, args: Args):
        self.args = args
        self.n_episodes = args.n_episodes
        self.failed = []
        self.results = defaultdict(
            lambda: [0] * self.n_episodes * len(args.envs)
        )
        self.distance = defaultdict(
            lambda: [100] * self.n_episodes * len(args.envs)
        )

    def get_path(self, env, dim, sensor):
        if "gt" in sensor:
            return f"{self.args.exp_name}/{env}/{sensor}"
        return f"{self.args.exp_name}/{env}/{dim}/{sensor}"

    def process_episode(self, path, episode_id, env, key):
        """
        Get the results of a particular episode of a particular variation
        """
        data = logger.LoggerLoader(path, f"{episode_id:04d}", "")

        # Reached
        reached = data.reached()

        # Collided
        collided = data.collided()

        # Reached wo/collision
        successfull = reached and (not collided)

        if successfull:
            result = 1
        elif collided:
            result = -1
        else:
            result = 0
    
        id = episode_id + self.n_episodes * self.args.envs.index(env)
        closest_distance = data.closest_target_distance()


        self.results[key][id] = result
        self.distance[key][id] = closest_distance

    def loop(self):
        for env in tqdm(self.args.envs):
            for sensor in tqdm(self.args.sensors, leave=False):
                for dim in tqdm(self.args.dimensions, leave=False):
                    key = (dim, sensor)
                    full_path = os.path.join("logs/experiments", self.get_path(env, dim ,sensor))
                    if os.path.isdir(full_path):
                        for e in tqdm(range(self.n_episodes), leave=False):
                            self.process_episode(self.get_path(env, dim, sensor), e, env, key)
                        if "gt" in sensor:
                            break # Go next sensor
                    else:
                        print(f"{key} does not exist on the folder {self.args.exp_name}")

    def save_results(self, env):
        data = pd.DataFrame(self.results).T
        data.to_pickle(f'results_{env}.pkl')
        distance = pd.DataFrame(self.distance).T
        distance.to_pickle(f'distance_{env}.pkl')


def generate_plot(env):
    """Formats and prints the results as a table."""
    data = pd.read_pickle(f"results_{env}.pkl")
    distance = pd.read_pickle(f"distance_{env}.pkl")
    

    cmap = ListedColormap(['red', 'blue', 'green'])
    bounds = [-1.5, -0.5, 0.5, 1.5]  # Bounds between values
    norm = BoundaryNorm(bounds, cmap.N)
    # Plot
    fig, ax = plt.subplots(2,1, figsize = (20,6))
    sns.heatmap(data, cmap=cmap, norm=norm, cbar=True, linewidths=0.0 , ax = ax[0])
    # Customize colorbar ticks
    plt.title('State Matrix Plot')
    plt.xlabel('Episode')
    plt.ylabel('Method')
    plt.tight_layout()

    sns.heatmap(distance, cbar = True, ax = ax[1])
    plt.title("Distance to target heatmap")
    plt.xlabel('Episode')
    plt.ylabel('Method')
    plt.tight_layout()
    plt.savefig(f"distance{env}.png")
    plt.show()
    plt.cla()



if __name__ == "__main__":
    import tyro

    args = tyro.cli(Args)
    args.envs = ["bookshelf", "table_new"]
    args.sensors = ["depthactive", "euclidean_lessactive","gt_active"]
    
    if not args.load:
        processor = PostProcess(args)
        processor.loop()
        processor.save_results("all")
        exit()
        args.envs = ["table_new"]
        processor = PostProcess(args)
        processor.loop()
        processor.save_results("table_new")
        generate_plot("bookshelf")
        generate_plot("table_new")

    else:
        generate_plot("bookshelf")
        generate_plot("table_new")

