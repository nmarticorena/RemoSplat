import os
from collections import defaultdict
from itertools import product
from pathlib import Path

import pandas as pd
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
        self.n_episodes = args.n_episodes
        self.failed = []
        self.results = defaultdict(
            lambda: {
                "reached": False,
                "collided": False,
                "successfull": False,
                "eef_acc": 0,
                "average_distance": 0.0,
                "average_manipulability": 0.0,
                "pred_collided": False,
                "total": 0,
            }
        )

    def process_episode(self, path, episode_id, key):
        """
        Get the results of a particular episode of a particular variation
        """
        data = logger.LoggerLoader(path, f"{episode_id:04d}", "", ROBOT)

        # Reached
        reached = data.reached()

        # Collided
        collided = data.collided()
        if collided:
            self.failed.append(data.data_folder)

        pred_collided = data.pred_collided()

        average_distance = data.average_distance()
        # average_manipulability = data.average_manipulability()

        # Reached wo/collision
        successfull = reached and (not collided)
        self.results[key]["reached"] += int(reached)
        self.results[key]["collided"] += int(collided)
        self.results[key]["successfull"] += int(successfull)
        self.results[key]["average_distance"] += float(average_distance)
        self.results[key]["pred_collided"] += int(pred_collided)
        # self.results[key]["average_manipulability"] += float(average_manipulability)
        self.results[key]["total"] += 1

    def loop(self):
        envs = self.args.envs
        sensors = self.args.sensors
        dims = self.args.dimensions

        for env,sensor, dim in tqdm(product(envs, sensors, dims), leave = False):
            if ("gt" in sensor) and (dim == "3D"):
                print("continue")
                continue
            key = (env, dim, sensor)
            print(f"Processing {key}")
            full_path = os.path.join("logs/experiments", self.args.get_path(env, dim ,sensor))
            if os.path.isdir(full_path):
                for e in tqdm(range(self.n_episodes), leave=False):
                    self.process_episode(self.args.get_path(env, dim, sensor), e, key)
            else:
                print(f"{key} does not exist on the folder {self.args.exp_name}")


        self.generate_report()

    def generate_report(self):
        """Formats and prints the results as a table."""
        data = []
        for (env, dim, sensor), stats in self.results.items():
            total = max(stats["total"], 1)  # Avoid division by zero
            data.append(
                [
                    env,
                    dim,
                    sensor,
                    stats["reached"] / total,
                    stats["collided"] / total,
                    stats["average_distance"] / total,
                    stats["pred_collided"]/total,
                    # stats["average_manipulability"] / total,
                    stats["successfull"] / total,
                ]
            )

        # Create DataFrame
        df = pd.DataFrame(
            data,
            columns=[
                "Env",
                "Dim",
                "Sensor",
                "Avg Reached",
                "Avg Collided",
                "Avg Distance",
                "Avg pred collided",
                # "Avg manipulability",
                "Avg Successful",
            ],
        )
        print(df.to_string(index=False))  # Print a clean table
        path = f"result_tables/results_{self.args.exp_name}.csv"
        parent_dir = Path(path).parent
        if not os.path.exists(path):
            os.makedirs(parent_dir, exist_ok= True)
            print("Creating folder parent_dir")


        df.to_csv(path)


if __name__ == "__main__":
    import tyro

    args = tyro.cli(PostProcessConfig)

    processor = PostProcess(args)
    processor.loop()
    print(processor.failed)
    with open(f"results_tables/{args.exp_name}_failed.txt", 'w') as f:
        for failed in processor.failed:
            f.write(f"{failed}\n")
    print(len(processor.failed))
