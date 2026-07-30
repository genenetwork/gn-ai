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

OUTPUT_PATH = os.getenv("OUTPUT_PATH")
if OUTPUT_PATH is None:
    raise FileNotFoundError("Set OUTPUT_PATH or path to save final plot")


def format_results(path: str) -> pd.DataFrame:
    files = os.listdir(path)
    df_list = []
    for f in files:
        model = f.strip("_dspy_results.csv")
        data = pd.read_csv(f"{path}/{f}")
        new = pd.DataFrame(data.aggregate("median"), columns=["satisfaction frequency"])
        new["model"] = model
        new["system"] = list(new.index)
        df_list.append(new)
    df_concat = pd.concat(df_list)
    return df_concat


def plot_results(data: pd.DataFrame, output_path: str):
    sns.barplot(data=data, x="system", y="satisfaction frequency", hue="model", palette="Set1")
    plt.suptitle("Performance with customized metric", fontsize=15)
    plt.savefig(output_path, dpi=1000)
    plt.show()


if __name__ == "__main__":
    df = format_results(INPUT_PATH)
    plot_results(df, OUTPUT_PATH)
