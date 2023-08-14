import pdb

import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from matplotlib import pyplot as plt

from aletheia.scripts.linear_classifier import predict, evaluate1
from aletheia.utils import cache_json

sns.set_context("talk")


def entropy_normed(x):
    if x == 0 or x == 1:
        return 1.0
    else:
        max_entropy = -np.log(0.5)
        H = -x * np.log(x) - (1 - x) * np.log(1 - x)
        return 1.0 - H / max_entropy


def max_normed(x):
    return 2 * (np.maximum(x, 1 - x) - 0.5)


def get_reliability_metrics(output, func, subset_func=None):
    df = pd.DataFrame({"pred": output["pred"], "true": output["true"]})
    df["reliability"] = df["pred"].map(func)
    if subset_func:
        # evaluate only on fake samples as Salvi et al.
        df = subset_func(df)
    num_samples = len(df)

    def evaluate_reliability(τ):
        idxs = df["reliability"] >= τ
        df_reliable = df[idxs]
        pred_binary = df_reliable["pred"] > 0.5
        true = df[idxs]["true"]
        accuracy = np.mean(true == pred_binary)
        num_kept = len(df_reliable)
        return {
            "dataset-name": output["dataset_name"],
            "real-dataset-name": output["real-dataset-name"],
            "τ": τ,
            "accuracy": 100 * accuracy,
            "frac-kept": 100 * num_kept / num_samples,
        }

    δ = 0.01
    return [evaluate_reliability(τ) for τ in np.arange(0.0, 1.0 + δ, δ)]


def main():
    TR_DATASET_NAME = "asvspoof19"
    SPLITS = ["train", "dev"]
    SUBSET = "all"
    FEATURE_TYPE = "wav2vec2-xls-r-2b"
    REAL_DATASET_NAMES = [
        "librispeech",
        "common-voice",
        "voxpopuli",
        "tedlium",
        "gigaspeech",
        "spgispeech",
        "earnings22",
        "ami",
    ]

    TR_DATASETS = [
        {
            "dataset_name": TR_DATASET_NAME,
            "split": split,
            "feature_type": FEATURE_TYPE,
            "subset": SUBSET,
        }
        for split in SPLITS
    ]
    
    def get_real_dataset(dataset_name):
        return {
            "dataset_name": dataset_name,
            "split": "dev",
            "feature_type": FEATURE_TYPE,
            "subset": "all",
        }
    
    path = "output/results/predictions-evaluate-reliability-real-dataset-{}.json"
    outputs = [
        {**output, "real-dataset-name": real_dataset_name}
        for real_dataset_name in REAL_DATASET_NAMES
        for output in cache_json(
            path.format(real_dataset_name),
            predict,
            TR_DATASETS + [get_real_dataset(real_dataset_name)],
            C=1,
            verbose=True,
        )
    ]
    outputs = [{**output, **evaluate1(**output, verbose=True)} for output in outputs]
    results = [
        {
            "dataset-name": output["dataset_name"],
            "real-dataset-name": output["real-dataset-name"],
            "eer": output["eer"],
            "ece": output["ece"],
        }
        for output in outputs
    ]
    df_results = pd.DataFrame(results)
    df_results = df_results.pivot(
        index="real-dataset-name", columns="dataset-name", values=["eer", "ece"]
    )
    st.write(df_results)

    select_only_fakes = lambda df: df[df["true"] == 1]
    dfs = [
        pd.DataFrame(
            get_reliability_metrics(
                output, entropy_normed, subset_func=select_only_fakes
            )
        )
        for output in outputs
    ]
    df = pd.concat(dfs)
    df = df.reset_index(drop=True)
    # df.drop(columns=["τ"], inplace=True)

    DATATASET_SHOW_NAMES = {
        "asvspoof19": "ASVspoof'19 (fake only)",
        "in-the-wild": "In the Wild (fake only)",
    }
    df = df.replace({"dataset-name": DATATASET_SHOW_NAMES})

    fig = sns.relplot(
        data=df,
        x="frac-kept",
        y="accuracy",
        col="dataset-name",
        hue="real-dataset-name",
        kind="line",
    )
    SALVI_VALUES = [
        {
            "dataset-name": "asvspoof19",
            "accuracy": 97.0,
            "frac-kept": 77.0,
            "ha": "right",
            "va": "bottom",
        },
        {
            "dataset-name": "in-the-wild",
            "accuracy": 100.0,
            "frac-kept": 100.0 - 35.0,
            "ha": "left",
            "va": "top",
        },
    ]
    for i, ax in enumerate(fig.axes.flat):
        values = SALVI_VALUES[i]
        ax.scatter(
            [values["frac-kept"]], [values["accuracy"]], color="black", marker="x"
        )
        ax.annotate(
            " Salvi, et al. ",
            (values["frac-kept"], values["accuracy"]),
            ha=values["ha"],
            va=values["va"],
        )

    fig.set(xlabel="Fraction of samples kept (%)", ylabel="Accuracy (%)")
    fig.set_titles("{col_name}")
    st.pyplot(fig)
    fig.savefig("output/icassp/reliability-fake.png")

    dfs = [
        pd.DataFrame(get_reliability_metrics(output, entropy_normed))
        for output in outputs
    ]
    df = pd.concat(dfs)
    df = df.reset_index(drop=True)
    # df.drop(columns=["τ"], inplace=True)

    DATATASET_SHOW_NAMES = {
        "asvspoof19": "ASVspoof'19 (all)",
        "in-the-wild": "In the Wild (all)",
    }

    df = df.replace({"dataset-name": DATATASET_SHOW_NAMES})
    fig = sns.relplot(
        data=df,
        x="frac-kept",
        y="accuracy",
        col="dataset-name",
        hue="real-dataset-name",
        kind="line",
    )
    fig.set(xlabel="Fraction of samples kept (%)", ylabel="Accuracy (%)")
    fig.set_titles("{col_name}")
    st.pyplot(fig)
    
    fig.savefig("output/icassp/reliability-with-real-data.png")

    # fig, axs = plt.subplots(ncols=2, figsize=(10, 5), sharey=True, sharex=True)
    # axs = iter(axs)

    # st.write("results", results)
    # for k, df in dfs.items():
    #     st.write(k)
    #     st.write(df)

    #     fig1, ax = plt.subplots()
    #     sns.histplot(data=df, x="pred", ax=ax, bins=15, multiple="stack", hue="true")
    #     st.pyplot(fig1)

    #     metrics = get_reliability_metrics(df, max_normed)

    #     ax = next(axs)
    #     xs = [m["frac-kept"] for m in metrics]
    #     ys = [m["accuracy"] for m in metrics]
    #     ax.plot(xs, ys)
    #     ax.set_xlabel("Fraction of samples kept (%)")
    #     ax.set_ylabel("Accuracy (%)")
    #     ax.set_title(k)

    # fig.tight_layout()
    # st.pyplot(fig)


if __name__ == "__main__":
    main()
