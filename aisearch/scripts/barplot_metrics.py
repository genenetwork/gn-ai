"""
Script to plot results with traditional metrics using local models
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

INPUT1_PATH = os.getenv("INPUT1_PATH")
if INPUT1_PATH is None:
    raise FileNotFoundError("Set INPUT1_PATH or path to simple results directory")

INPUT2_PATH = os.getenv("INPUT2_PATH")
if INPUT2_PATH is None:
    raise FileNotFoundError("Set INPUT2_PATH or path to dspy results directory")

OUTPUT_PATH = os.getenv("OUTPUT_PATH")
if OUTPUT_PATH is None:
    raise FileNotFoundError("Set OUTPUT_PATH or path to save final plot")


def format_results(simple_path: str, dspy_path: str) -> pd.DataFrame:
    df_list = []
    simple_files = os.listdir(simple_path)
    for f in simple_files:
        model = f.strip("_simple_results.csv")
        data = pd.read_csv(f"{simple_path}/{f}", index_col=0)
        trans_data = data.transpose()
        trans_data = pd.concat(
            [trans_data[column] * 100 for column in trans_data.columns], axis=1
        )
        trans_data["model"] = model
        trans_data["system"] = [sys[:-2] for sys in list(trans_data.index)]
        df_list.append(trans_data)
    final_concat = pd.concat(df_list)
    dspy_files = os.listdir(dspy_path)
    cumul = []
    for f in dspy_files:
        data = pd.read_csv(f"{dspy_path}/{f}")
        aligned = pd.concat([data[column] for column in data.columns])
        cumul.extend(aligned)
    final_concat["satisfaction frequency (%)"] = cumul
    return final_concat


def plot(data: pd.DataFrame, columns: list[str], output_path: str, title: str):
    fig, axes = plt.subplots(len(columns), 1, figsize=(10, 10))
    axes = axes.flatten()
    for ind, column in enumerate(columns):
        sns.barplot(
            data=data,
            ax=axes[ind],
            y="system",
            x=column,
            hue="model",
            palette="ocean_r",
        )
    fig.suptitle(title, fontsize=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=1000)
    plt.show()


def plot_results(data: pd.DataFrame, output_path: str):
    metrics = list(data.columns)[:3]
    metrics.append(data.columns[-1])
    plot(
        data,
        metrics,
        f"{output_path}/barplot_standard_metrics.png",
        "Performance comparison of standard and new metrics",
    )
    extras = list(data.columns)[3:5]
    plot(
        data,
        extras,
        f"{output_path}/barplot_extra_metrics.png",
        "Insights from additional metrics",
    )


if __name__ == "__main__":
    df = format_results(INPUT1_PATH, INPUT2_PATH)
    plot_results(df, OUTPUT_PATH)
