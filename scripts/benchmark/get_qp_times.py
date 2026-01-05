import pandas as pd
import os
import glob
from collections import defaultdict
from functools import reduce
import matplotlib.pyplot as plt
import seaborn as sns

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
environments = ["bookshelf_cage", "table_new"]

# Group CSVs by environment and method
data_by_env_and_method = defaultdict(dict)

for csv_file in glob.glob(f"{base_path}/*.csv"):
    filename = os.path.basename(csv_file)
    for env in environments:
        if env in filename:
            method = filename.replace(env + "_2D_", "").replace(".csv", "")
            data_by_env_and_method[env][method] = csv_file

# Metrics to save
metrics = [
    "qp_solving_time"
]

results_all = []

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
        # Store for global average
        global_all_rows[method].append(df)

times = {}

# Compute global averages across environments
for method in global_all_rows:
    combined_df = pd.concat(global_all_rows[method], ignore_index=True)
    times[method] = combined_df["qp_solving_time"] * 1000


# Plotting
sns.violinplot(data=pd.DataFrame(times))
plt.ylabel("Time (ms)")
plt.title("QP Solving Time")
plt.show()


