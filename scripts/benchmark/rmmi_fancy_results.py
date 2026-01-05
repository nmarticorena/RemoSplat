import pandas as pd
import tyro


def main(filename:str) -> None:
    print(f"pretty results of {filename}")
    df = pd.read_csv(filename, index_col = 0)

    # Assuming your DataFrame is called df
    df['Dim_Sensor'] = df['Dim'] + '_' + df['Sensor']

    # Group by Dim_Sensor and average across Env
    grouped = df.groupby('Dim_Sensor').mean(numeric_only=True)

    print(grouped)
    output_filename = filename
    output_filename.replace(".csv","fancy.csv")
    grouped.to_csv(f"{output_filename}",float_format="%.3f")

if __name__ == "__main__":
    tyro.cli(main)

