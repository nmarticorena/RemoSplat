import os
from dataclasses import dataclass, field
from typing import List

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import tqdm
import tyro
from matplotlib import pyplot as plt
from neural_robot.neural_frankie import NeuralFrankie

from remo_splat import logger


def configure_matplotlib():
    sns.set()
    sns.set_style(style="whitegrid")
    sns.set_palette("colorblind", 6)
    sns.set_context("paper")
    matplotlib.rcParams["ps.useafm"] = True
    matplotlib.rcParams["pdf.use14corefonts"] = True
    matplotlib.rcParams["legend.columnspacing"] = 1.0
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["xtick.major.pad"] = "0"
    plt.rcParams["ytick.major.pad"] = "0"
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    plt.rcParams.update({"font.size": 8})
    plt.rcParams.update({"axes.labelsize": 10})
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams["svg.fonttype"] = "none"
    # default xlim[]
    plt.rcParams["axes.xmargin"] = 0
    plt.rcParams.update({"figure.figsize": (3.5, 3.5)})


configure_matplotlib()


@dataclass
class Config:
    envs: List = field(default_factory=lambda: ["bookshelf_cage", "table_new"])
    exp_name: str = "test_random/"


args = tyro.cli(Config)
experiment_name = args.exp_name

master_folder = args.envs
master_folder = [experiment_name + "/" + i for i in master_folder]

experiment_path = f"logs/experiments/{experiment_name}"
folders = os.listdir("logs/experiments/" + master_folder[0])
# print(folders)
# exit(0)
os.makedirs(f"logs/experiments/{experiment_name}/results", exist_ok=True)
os.makedirs(f"logs/experiments/{experiment_name}/latex_tables", exist_ok=True)


robot = NeuralFrankie("points_9", spheres=False)


def plot(data: logger.LoggerLoader, name):
    try:
        data.plot_data("gt_distance", "gt_distance", min=True)
        data.plot_data("d_distance", "gsplat_distance", linestyle="--")
        data.plot_data("et", "distance_target")
        plt.legend()
        plt.title(name)
        name = name.replace("/", "_")
        plt.savefig(f"{experiment_path}/results/{name}.png")
        plt.close()
        plt.cla()
    except Exception as e:
        print(f"Error plotting {name}: {e}")


for m in master_folder:
    env_name = m.split("/")[-1]
    env_df = pd.DataFrame()
    env_successfull_df = pd.DataFrame()
    data_path = logger.load_folder(m)
    df = pd.DataFrame()
    df_2 = pd.DataFrame()
    for i in tqdm.tqdm(data_path):
        j = ""
        data = logger.LoggerLoader(i, "", "")
        collided = data.collided()
        q = data.get_data("q")
        if len(q) == 1:
            print(f"Skipping {i} because it has only one pose")
            continue
        # eef_acc, eef_cum_acc = logger.acc_eef(data, robot)
        # eef_jerk, eef_cum_jerk = logger.jerk_eef(data, robot)
        reached = data.reached()

        # real_collided = logger.real_collided(data)
        values = {
            "name": i,
            "real collided": collided,
            "reached": reached,
            # "eef acc": eef_cum_acc[-1],
            # "max eef acc": np.max(eef_acc),
            # "mean eef acc": np.mean(eef_acc),
            # "min eef acc": np.min(eef_acc),
            # "eef jerk": eef_cum_jerk[-1],
            # "max eef jerk": np.max(eef_jerk),
            # "mean eef jerk": np.mean(eef_jerk),
        }
        # Create a DataFrame object
        df_temp = pd.DataFrame([values])
        if values["reached"]:
            df_2 = pd.concat([df_2, df_temp], ignore_index=True)
        plot(data, i)

        df = pd.concat([df, df_temp], ignore_index=True)
    df.to_excel(f"{experiment_path}/results/{env_name}_full.xlsx")
    # concatenate the dataframes
    results = df.drop("name", axis=1).mean().T

    filter_results = df_2.drop("name", axis=1).mean().T
    env_successfull_df = pd.concat(
        [env_successfull_df, filter_results.rename(f"{j.replace('_', ' ')}")],
        axis=1,
    )

    with open(f"{experiment_path}/results/env_name_result.txt", "w") as f:
        f.write(results.to_string())
    env_df = pd.concat([env_df, results.rename(f"{j.replace('_', ' ')}")], axis=1)
    df.to_excel(f"{experiment_path}/results/{env_name}.xlsx")

    env_successfull_df.to_excel(
        f"{experiment_path}/results/{env_name}_successfull.xlsx"
    )
    env_successfull_df.to_csv(f"{experiment_path}/results/{env_name}_successfull.csv")
    env_df.to_excel(f"{experiment_path}/results/{env_name}.xlsx")
    env_df.to_csv(f"{experiment_path}/results/{env_name}.csv")
    env_df.to_latex(
        f"{experiment_path}/latex_tables/{env_name}.tex",
        float_format="%.3f",
        formatters={"name": str.upper},
    )
    env_successfull_df.to_latex(
        f"{experiment_path}/latex_tables/{env_name}_successfull.tex",
        float_format="%.3f",
        formatters={"name": str.upper},
    )
