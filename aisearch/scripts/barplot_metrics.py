"""
Script to plot results with traditional metrics using local models
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

INPUT_PATH = os.getenv("INPUT_PATH")
if INPUT_PATH is None:
    raise FileNotFoundError("Set INPUT_PATH or path to results directory")

OUTPUT_PATH = os.getenv("OUTPUT_PATH")
if OUTPUT_PATH is None:
    raise FileNotFoundError("Set OUTPUT_PATH or path to save final plot")


def format_results(path: str) -> pd.DataFrame:
    files = os.listdir(path)
    df_list = []
    for f in files:
        model = f.split("_")[0]
        data = pd.read_csv(f"{path}/{f}", index_col=0)
        trans_data = data.transpose()
        trans_data["model"] = model
        trans_data["system"] = [sys[:-2] for sys in list(trans_data.index)]
        df_list.append(trans_data)
    df_concat = pd.concat(df_list)
    return df_concat


def plot_results(data: pd.DataFrame, output_path: str):
    fig, axes = plt.subplots(3, 2, figsize=(10, 10))
    axes = axes.flatten()
    columns = list(data.columns)[:5]
    for ind, column in enumerate(columns):
        sns.barplot(
            data=data, ax=axes[ind], x="system", y=column, hue="model", palette="Set1"
        )
    fig.suptitle("Performance with traditional metrics", fontsize=15)
    fig.delaxes(axes[-1])
    plt.tight_layout()
    plt.savefig(output_path, dpi=1000)
    plt.show()


if __name__ == "__main__":
    df = format_results(INPUT_PATH)
    plot_results(df, OUTPUT_PATH)
