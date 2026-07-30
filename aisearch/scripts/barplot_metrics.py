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
        model = f.strip("_simple_results.csv")
        data = pd.read_csv(f"{path}/{f}", index_col=0)
        trans_data = data.transpose()
        trans_data["model"] = model
        trans_data["system"] = [sys[:-2] for sys in list(trans_data.index)]
        df_list.append(trans_data)
    df_concat = pd.concat(df_list)
    return df_concat


def plot(data: pd.DataFrame, columns: list[str], output_path: str, title: str):
    fig, axes = plt.subplots(len(columns), 1, figsize=(10, 10))
    axes = axes.flatten()
    for ind, column in enumerate(columns):
        sns.barplot(
            data=data,
            ax=axes[ind],
            x="system",
            y=column,
            hue="model",
            palette="Set1",
            estimator="median",
        )
    fig.suptitle(title, fontsize=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=1000)
    plt.show()


def plot_results(data: pd.DataFrame, output_path: str):
    metrics = list(data.columns)[:3]
    plot(
        data,
        metrics,
        f"{output_path}/barplot_traditional_metrics.png",
        "Performance with traditional metrics",
    )
    extras = list(data.columns)[3:5]
    plot(
        data,
        extras,
        f"{output_path}/barplot_extra_metrics.png",
        "Insights from additional metrics",
    )


if __name__ == "__main__":
    df = format_results(INPUT_PATH)
    plot_results(df, OUTPUT_PATH)
