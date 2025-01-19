import argparse

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import numpy as np

def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)



def prepare_data(data: pd.DataFrame, method="IQR") -> pd.DataFrame:
    # Strip whitespace from column names
    data.columns = data.columns.str.strip()

    # remove rows with null values
    data = data[data.notnull().all(axis=1)]

    # remove consecutive duplicates
    not_duplicate = data.diff(-1).any(axis=1)
    not_duplicate[not_duplicate.size-1] = True
    data = data[not_duplicate.values]

    # remove outliers
    if method == "IQR":
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        data = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]
    elif method == "zscore":
        data = data[(np.abs(stats.zscore(data.select_dtypes(include=[np.number]))) < 3).all(axis=1)]
    else:
        raise ValueError("Invalid outlier detection method. Choose 'IQR' or 'zscore'.")

    return data


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=Path, default="data.csv", help="path to the csv file with data"
    )
    return parser.parse_args()


def visualize_data(data: pd.DataFrame, save_plots=True):
    # Ensure directory for saving plots
    output_dir = "plots"
    if save_plots:
        Path(output_dir).mkdir(parents=True, exist_ok=True)  # Creates the directory if it doesn't exist

    # Histograms for numeric columns
    data.hist(bins=20, figsize=(20, 15))
    plt.suptitle("Histograms of Numeric Features", fontsize=20)
    if save_plots:
        plt.savefig(f"{output_dir}/histograms.png")
    plt.show()

    # Correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap", fontsize=18)
    if save_plots:
        plt.savefig(f"{output_dir}/correlation_heatmap.png")
    plt.show()

    # Boxplots for outlier detection (all on one page)
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    num_plots = len(numeric_columns)
    cols_per_row = 3  # Number of boxplots per row
    rows = (num_plots + cols_per_row - 1) // cols_per_row  # Calculate rows needed

    fig, axes = plt.subplots(rows, cols_per_row, figsize=(15, 5 * rows))
    axes = axes.flatten()  # Flatten for easy indexing

    for i, col in enumerate(numeric_columns):
        sns.boxplot(x=data[col], ax=axes[i])
        axes[i].set_title(f"Boxplot for {col}", fontsize=12)
        if save_plots:
            plt.savefig(f"{output_dir}/boxplot_{col.replace('/', '_').replace(' ', '_')}.png")

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


    # Scatter plots for key relationships
    if 'Gas Turbine (GT) shaft torque (GTT) [kN m]' in data.columns and 'Fuel flow (mf) [kg/s]' in data.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='Gas Turbine (GT) shaft torque (GTT) [kN m]', y='Fuel flow (mf) [kg/s]', data=data)
        plt.title("Shaft Torque vs Fuel Flow", fontsize=15)
        if save_plots:
            plt.savefig(f"{output_dir}/scatter_torque_vs_fuel.png")
        plt.show()

    # Time series plots for decay coefficients
    if 'GT Compressor decay state coefficient' in data.columns and 'GT Turbine decay state coefficient' in data.columns:
        plt.figure(figsize=(12, 6))
        plt.plot(data['GT Compressor decay state coefficient'], label='Compressor Decay Coefficient', marker='o')
        plt.plot(data['GT Turbine decay state coefficient'], label='Turbine Decay Coefficient', marker='o')
        plt.title("Decay Coefficients Over Time", fontsize=15)
        plt.xlabel("Index")
        plt.ylabel("Decay Coefficient")
        plt.legend()
        if save_plots:
            plt.savefig(f"{output_dir}/decay_coefficients.png")
        plt.show()

def summarize_data(data: pd.DataFrame):
    print("Summary Statistics:\n")
    print(data.describe())
    print("\nMissing Values:\n")
    print(data.isnull().sum())

    # Ensure the correct column name is used for correlation
    try:
        print("\nCorrelation with Fuel Flow (mf):\n")
        print(data.corr()['Fuel flow (mf) [kg/s]'].sort_values(ascending=False))
    except KeyError:
        print("Error: 'Fuel flow (mf) [kg/s]' column not found. Check column names.")


def main():
    args = parse_arguments()
    data = load_data(args.data)
    print(f"Loaded {len(data)} rows of data.")

    clean_data = prepare_data(data, method="IQR")
    print(f"Cleaned data has {len(clean_data)} rows (removed {len(data) - len(clean_data)} rows).")

    summarize_data(clean_data)
    visualize_data(clean_data)


if __name__ == "__main__":
    main()