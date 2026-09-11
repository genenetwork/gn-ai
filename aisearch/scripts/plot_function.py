"""
Script to plot results with new metric using local models
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

INPUT_PATH = os.getenv("INPUT_PATH")
if INPUT_PATH is None:
    raise FileNotFoundError("Set INPUT_PATH or path to results directory")

OUTPUT1_PATH = os.getenv("OUTPUT1_PATH")
if OUTPUT1_PATH is None:
    raise FileNotFoundError(
        "Set OUTPUT1_PATH or path to save final boxplot for systems"
    )

OUTPUT2_PATH = os.getenv("OUTPUT2_PATH")
if OUTPUT2_PATH is None:
    raise FileNotFoundError("Set OUTPUT2_PATH or path to save final barplot for models")


def format_results(path: str) -> pd.DataFrame:
    files = os.listdir(path)
    df_list = []
    for f in files:
        model = f.strip("_dspy_results.csv")
        data = pd.read_csv(f"{path}/{f}")
        new = pd.DataFrame(
            data.aggregate("median"), columns=["satisfaction frequency (%)"]
        )
        new["model"] = model
        new["system"] = list(new.index)
        df_list.append(new)
    df_concat = pd.concat(df_list)
    df_concat["model_order"] = pd.Categorical(
        df_concat["model"],
        categories=[
            "gptoss_20b",
            "qwen3_30b",
            "haiku_4.5",
            "qwen3_30b_opus_4.8",
            "haiku_4.5_opus_4.8",
            "opus_4.8",
        ],
        ordered=False,
    )
    df_concat["system_order"] = pd.Categorical(
        df_concat["system"],
        categories=["base", "rag", "graph rag", "agent", "hybird"],
        ordered=False,
    )
    return df_concat.sort_values(by=["system_order", "model_order"])


def plot_results(data: pd.DataFrame, output1_path: str, output2_path: str):
    sns.boxplot(data=data, x="system", y="satisfaction frequency (%)", color="#708090")
    sns.swarmplot(
        data=data,
        x="system",
        y="satisfaction frequency (%)",
        hue="model",
        palette="ocean_r",
    )
    plt.savefig(output1_path, dpi=1000)
    plt.show()
    axb = sns.barplot(
        data=data,
        x="system",
        y="satisfaction frequency (%)",
        hue="model",
        palette="ocean_r",
        width=0.5,
        edgecolor="none",
    )
    for ind, container in enumerate(axb.containers):
        if ind == 0 or ind == 5:
            axb.bar_label(container, fmt="%.0f", padding=1, fontsize=7)
    plt.suptitle("Model performance with customized metric", fontsize=10)
    plt.savefig(output2_path, dpi=1000)
    plt.show()


if __name__ == "__main__":
    df = format_results(INPUT_PATH)
    plot_results(df, OUTPUT1_PATH, OUTPUT2_PATH)
