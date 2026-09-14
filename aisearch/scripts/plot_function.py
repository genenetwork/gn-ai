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


def format_results(path: str) -> tuple[pd.DataFrame]:
    files = os.listdir(path)
    full_list = []
    aggr_list = []
    for f in files:
        model = f.strip("_dspy_results.csv")
        data = pd.read_csv(f"{path}/{f}")

        aggr = pd.DataFrame(
            data.aggregate("median"), columns=["satisfaction frequency (%)"]
        )
        aggr["model"] = model
        aggr["system"] = list(aggr.index)
        aggr_list.append(aggr)

        full = data.transpose()
        full = pd.concat([full[column] for column in full.columns])
        full_df = pd.DataFrame(full, columns=["satisfaction frequency (%)"])
        full_df["model"] = model
        full_df["system"] = list(full.index)
        full_list.append(full_df)

    full_concat = pd.concat(full_list)
    full_concat["model_order"] = pd.Categorical(
        full_concat["model"],
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
    full_concat["system_order"] = pd.Categorical(
        full_concat["system"],
        categories=["base", "rag", "graph rag", "agent", "hybrid"],
        ordered=False,
    )

    aggr_concat = pd.concat(aggr_list)
    aggr_concat["model_order"] = pd.Categorical(
        aggr_concat["model"],
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
    aggr_concat["system_order"] = pd.Categorical(
        aggr_concat["system"],
        categories=["base", "rag", "graph rag", "agent", "hybrid"],
        ordered=False,
    )
    return full_concat.sort_values(
        by=["system_order", "model_order"]
    ), aggr_concat.sort_values(by=["system_order", "model_order"])


def plot_results(
    full_data: pd.DataFrame,
    aggr_data: pd.DataFrame,
    output1_path: str,
    output2_path: str,
):
    sns.boxplot(
        data=full_data, x="system", y="satisfaction frequency (%)", color="#708090"
    )
    sns.swarmplot(
        data=full_data,
        x="system",
        y="satisfaction frequency (%)",
        hue="model",
        palette="ocean_r",
    )
    plt.savefig(output1_path, dpi=1000)
    plt.show()
    # heatmap_data = aggr_data.pivot(
    #     index="model", columns="system", values="satisfaction frequency (%)"
    # )
    # heatmap_data = heatmap_data.reindex(
    #     index=[
    #         "opus_4.8",
    #         "haiku_4.5_opus_4.8",
    #         "qwen3_30b_opus_4.8",
    #         "haiku_4.5",
    #         "qwen3_30b",
    #         "gptoss_20b",
    #     ],
    #     columns=["base", "rag", "graph rag", "agent", "hybrid"],
    # )
    sns.barplot(
        full_data,
        x="system",
        y="satisfaction frequency (%)",
        hue="model",
        palette="ocean_r",
    )
    plt.suptitle("Model performance with customized metric", fontsize=10)
    plt.tight_layout()
    plt.savefig(output2_path, dpi=1000)
    plt.show()


if __name__ == "__main__":
    full_df, aggr_df = format_results(INPUT_PATH)
    plot_results(full_df, aggr_df, OUTPUT1_PATH, OUTPUT2_PATH)
