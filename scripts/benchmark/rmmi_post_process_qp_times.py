import os
from collections import defaultdict
from itertools import product
from pathlib import Path
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import seaborn as sns
import tyro
from neural_robot.unity_frankie import NeuralFrankie as Robot
from tqdm import tqdm

from remo_splat import logger
from remo_splat.configs.postprocess import PostProcessConfig

ROBOT = Robot("curobo", spheres = True)

# Script that post process the logs of the rmmi.py benchmark
class PostProcess:
    def __init__(self, args: PostProcessConfig):
        self.args = args
        self.failed = []
        self.results = defaultdict(list)


    def process_episode(self, path, episode_id, key):
        """
        Get the results of a particular episode of a particular variation
        """
        data = logger.LoggerLoader(path, f"{episode_id:04d}", "", None)

        self.results[key].extend(data.get_data("t_qp") * 1000)

    def loop(self):
        envs = self.args.envs
        sensors = self.args.sensors
        dims = self.args.dimensions

        for env, sensor, dim in tqdm(product(envs, sensors, dims), leave = False):
            key = (dim, sensor)
            print(f"Processing {key}")
            full_path = os.path.join("logs/experiments", self.args.get_path(env, dim ,sensor))
            if os.path.isdir(full_path):
                for e in tqdm(range(self.args.n_episodes), leave=False):
                    self.process_episode(self.args.get_path(env, dim, sensor), e, key)
            else:
                print(f"{key} does not exist on the folder {self.args.exp_name}")
        self.generate_report()

    def save_path(self, key):
        return "_".join(key)

    def generate_report(self):
        """Plot box plot of the qp times all togeher"""
        results = {}
        for k, v in self.results.items():
            print(f"{k}: {np.mean(v)} +- {np.std(v)}")
            results[self.save_path(k)] = f"{np.mean(v):.2f} +- {np.std(v):.2f}"
        df = pd.DataFrame({
            "dim": [k[0] for k, v in self.results.items() for _ in v],
            "sensor": [k[1] for k, v in self.results.items() for _ in v],
            "t_qp": [item for k, v in self.results.items() for item in v],
        })
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x="sensor", y="t_qp", hue="sensor")
        plt.savefig("qp_times.pdf")
        # plt.yscale("log")
        plt.show()
        pd.DataFrame.from_dict(results, orient='index', columns=["t_qp (ms)"]).to_csv(f"qp_times.csv")



if __name__ == "__main__":
    import tyro

    args = tyro.cli(PostProcessConfig)

    processor = PostProcess(args)
    processor.loop()
    os.makedirs("results_tables/details/results_" + args.exp_name, exist_ok=True)
    with open(f"results_tables/details/results_{args.exp_name}/failed.txt", 'w') as f:
        for failed in processor.failed:
            f.write(f"{failed}\n")
    print(len(processor.failed))
