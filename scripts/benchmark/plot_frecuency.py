

import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd

# Load all CSV logs
folder = "results_check_frecuency"
logs = [f for f in os.listdir(folder) if f.endswith(".csv")]
paths = [os.path.join(folder, log) for log in logs]

# folder = "results_check_frecuency_mars"
# logs = [f for f in os.listdir(folder) if f.endswith(".csv")]
# path_mars = [os.path.join(folder, log) for log in logs]
# paths = [*paths, *path_mars]

# Regex to extract Hz from filename
hz_pattern = re.compile(r"(\d+)")

# Store: {(Dim, Sensor, Env): list of (Hz, Avg Successful)}

targets = ["Avg Successful", "Avg Collided"]

for target_data in targets:


    data_grouped = defaultdict(list)
    for path in paths:
        # Extract Hz from filename
        match = hz_pattern.search(os.path.basename(path))
        if match:
            hz = int(match.group(1))
        else:
            print(f"No Hz found in {path}")
            continue

        df = pd.read_csv(path)
        df["Hz"] = hz

        # Collect data
        for _, row in df.iterrows():
            key = (row["Dim"], row["Sensor"], row["Env"])
            data_grouped[key].append((hz, row[target_data]))

    # Aggregate by averaging over Env
    # Final dict: {(Dim, Sensor): {Hz: avg_successful}}
    aggregated = defaultdict(lambda: defaultdict(list))
    for (dim, sensor, env), values in data_grouped.items():
        for hz, sr in values:
            aggregated[(dim, sensor)][hz].append(sr)

    # Average over environments
    for key in aggregated:
        for hz in aggregated[key]:
            values = aggregated[key][hz]
            aggregated[key][hz] = sum(values) / len(values)

    # Plot
    plt.figure(figsize=(10, 6))
    for (dim, sensor), hz_dict in aggregated.items():
        sorted_items = sorted(hz_dict.items())  # Sort by Hz
        hz_values, sr_values = zip(*sorted_items)
        label = f"{dim}-{sensor}"
        linestyle = "--" if "depth" in sensor else "-"
        if "gt" in sensor:
            continue
        plt.plot(hz_values, sr_values, marker='o', label=label, linestyle= linestyle)

    plt.xlabel("Hz")
    plt.ylabel(target_data)
    plt.title(f"{target_data} vs Refresh Rate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

