
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Path to your results folder
results_dir = Path("results_tables/details/check_frecuency/20")

# All CSVs
csv_files = list(results_dir.glob("*.csv"))

# Known envs
valid_envs = {"bookshelf", "table_new"}

data = []
for csv_file in csv_files:
    parts = csv_file.stem.split("_")

    # Extract env
    if parts[0] == "table" and parts[1] == "new":
        env = "table_new"
        dim, sensor = parts[2], parts[3:]
        sensor_dim = f"{dim}_{sensor}"
    elif parts[0] in valid_envs:
        env = parts[0]
        dim, sensor = parts[1], parts[2:]
        sensor_dim = f"{dim}_{sensor}"
    else:
        print(f"Skipping {csv_file}, unknown environment format.")
        continue

    df = pd.read_csv(csv_file)

    if "distance_target" in df.columns:
        for val in df["distance_target"]:
            data.append({
                "sensor_dim": sensor_dim,
                "distance_target": val
            })
    else:
        print(f"Skipping {csv_file}, missing 'distance_target' column.")

# Combine and plot
df_all = pd.DataFrame(data)

plt.figure(figsize=(10, 5))
sns.boxplot(x="sensor_dim", y="distance_target", data=df_all)
plt.title("Distance to Target by Sensor-Dim (Bookshelf + Table_new Combined)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
sns.kdeplot(data=df_all, x="distance_target", hue="sensor_dim", common_norm=False, fill=True)
plt.title("Distance to Target KDE by Sensor-Dim (Bookshelf + Table_new Combined)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
sns.histplot(data=df_all, x="distance_target", hue="sensor_dim", multiple="layer", element="step", stat="density")
plt.title("Distance to Target Histogram by Sensor-Dim (Bookshelf + Table_new Combined)")
plt.tight_layout()
plt.show()



