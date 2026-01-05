import pandas as pd
import os
import glob
from collections import defaultdict
from functools import reduce

import tyro
from dataclasses import dataclass

@dataclass
class ProcessResultsConfig:
    "Configuration for processing results"
    input_folder: str = "."
    output_name: str = "example"
    output_all: str = "average_metrics_all.csv"
    output_common: str = "average_metrics_successful_common.csv"

args = tyro.cli(ProcessResultsConfig)

base_path = os.path.dirname(args.input_folder)
breakpoint()
environments = ["bookshelf_cage", "table_new"]

# Group CSVs by environment and method
data_by_env_and_method = defaultdict(dict)

for csv_file in glob.glob(f"{base_path}/*.csv"):
    filename = os.path.basename(csv_file)
    for env in environments:
        if env in filename:
            method = filename.replace(env + "_2D_", "").replace(".csv", "")
            data_by_env_and_method[env][method] = csv_file

# Metrics to average
metrics = [
    "reached", "collided", "successfull", "eef_acc","eef_cum_acc", "len_target",
    "average_manipulability", "average_distance",
    "pred_collided", "min_distance", "min_pred_distance", "distance_target",
    "qp_solving_time"
]

results_all = []
results_common_success = []

# For global averaging
global_all_rows = defaultdict(list)
global_common_rows = defaultdict(list)

for env, methods in data_by_env_and_method.items():
    dfs = {}
    success_ids = {}

    # Load all dataframes and collect success IDs
    for method, path in methods.items():
        df = pd.read_csv(path)
        df["env"] = env
        df["method"] = method
        dfs[method] = df
        success_ids[method] = set(df[df["successfull"] == True]["episode_id"])

        # Store for global average
        global_all_rows[method].append(df)

    # Find common successful episode_ids across all methods
    common_success_ids = sorted(reduce(lambda a, b: a & b, success_ids.values()))

    # Save these common episode ids
    txt_filename = f"{env}_common_successful_ids.txt"
    with open(txt_filename, "w") as f:
        for ep_id in common_success_ids:
            f.write(f"{ep_id}\n")

    for method, df in dfs.items():
        # Average over all episodes
        avg_all = df[metrics].mean()
        avg_all["env"] = env
        avg_all["method"] = method
        results_all.append(avg_all)

        # Filter only common successful episodes
        df_common = df[df["episode_id"].isin(common_success_ids)]
        avg_common = df_common[metrics].mean()
        avg_common["env"] = env
        avg_common["method"] = method
        results_common_success.append(avg_common)

        # Store for global common
        global_common_rows[method].append(df_common)

# Compute global averages across environments
for method in global_all_rows:
    combined_df = pd.concat(global_all_rows[method], ignore_index=True)
    avg_all = combined_df[metrics].mean()
    avg_all["env"] = "all"
    avg_all["method"] = method
    results_all.append(avg_all)

for method in global_common_rows:
    if global_common_rows[method]:  # Make sure there's data
        combined_df = pd.concat(global_common_rows[method], ignore_index=True)
        avg_common = combined_df[metrics].mean()
        avg_common["env"] = "all"
        avg_common["method"] = method
        results_common_success.append(avg_common)

# Save tables
df_all = pd.DataFrame(results_all)
df_common_successful = pd.DataFrame(results_common_success)

folder = f"results/final_tables/{args.output_name}"
os.makedirs(folder, exist_ok=True)

df_all.to_excel(f"{folder}/average_metrics_all.xlsx", index=False)
df_common_successful.to_excel(f"{folder}/average_metrics_successful_common.xlsx", index=False)

print("Done. Per-environment and global averages written.")
