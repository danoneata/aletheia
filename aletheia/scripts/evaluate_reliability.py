import pdb

import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score

from aletheia.scripts.linear_classifier import predict, evaluate1
from aletheia.utils import cache_json

sns.set_context("talk")


δ = 0.01


def entropy_normed(x):
    if x == 0 or x == 1:
        return 1.0
    else:
        max_entropy = -np.log(0.5)
        H = -x * np.log(x) - (1 - x) * np.log(1 - x)
        return 1.0 - H / max_entropy


def max_normed(x):
    return 2 * (np.maximum(x, 1 - x) - 0.5)



select_only_fakes = lambda df: df[df["true"] == 1]


def evaluate_reliability(df, subset_func=None, τ=0.5):
    if subset_func:
        # evaluate only on fake samples as Salvi et al.
        df = subset_func(df)

    idxs = df["reliability"] >= τ

    pred_binary = df[idxs]["pred-binary"]
    pred = df[idxs]["pred"]
    true = df[idxs]["true"]
    accuracy = np.mean(true == pred_binary)

    num_kept = sum(idxs)
    num_samples = len(df)
    frac_kept = num_kept / num_samples

    # print(num_kept)
    # print(num_samples - num_kept)
    # print(num_samples)

    if num_kept == 0 or true.nunique() == 1:
        auc_roc = np.nan
    else:
        auc_roc = roc_auc_score(true, pred)

    return {
        "accuracy": 100 * accuracy,
        "frac-kept": 100 * frac_kept,
        "auc-roc": 100 * auc_roc,
    }


def plot_hists_reliab(df):
    idxs = df["pred"] > 0.5
    step = 1 / 100
    bins = np.arange(0.0, 1.0 + 2 * step, step)
    fig, ax = plt.subplots()
    ax.hist(df[idxs]["reliability"], bins=bins)
    ax.hist(df[~idxs]["reliability"], bins=bins)
    # ax.scatter(df["reliability"], df["pred"])
    ax.set_xlabel("Reliability")
    ax.set_ylabel("Counts")
    st.pyplot(fig)


def get_reliability_metrics(output, func, subset_func):
    df = pd.DataFrame({"pred": output["pred"], "true": output["true"]})
    df["reliability"] = df["pred"].map(func)
    df["pred-binary"] = df["pred"] > 0.5

    # plot_hists_reliab(df)
    # pdb.set_trace()

    return [
        {
            # "dataset-name": output["dataset_name"],
            # "C": output["C"],
            "τ": τ,
            **evaluate_reliability(df, subset_func, τ),
        }
        for τ in np.arange(0.0, 1.0 + δ, δ)
    ]


def main():
    TR_DATASET_NAME = "asvspoof19"
    SPLITS = ["train", "dev"]
    SUBSET = "all"
    FEATURE_TYPE = "wav2vec2-xls-r-2b"

    TR_DATASETS = [
        {
            "dataset_name": TR_DATASET_NAME,
            "split": split,
            "feature_type": FEATURE_TYPE,
            "subset": SUBSET,
        }
        for split in SPLITS
    ]
    path = "output/results/predictions-evaluate-reliability-C-{}.json"
    outputs = [
        {**output, "C": C}
        for C in [1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
        for output in cache_json(
            path.format(C),
            predict,
            TR_DATASETS,
            C=C,
            verbose=True,
        )
    ]
    outputs = [{**output, **evaluate1(**output, verbose=True)} for output in outputs]
    results = [
        {
            "dataset-name": output["dataset_name"],
            "C": output["C"],
            "eer": output["eer"],
            "ece": output["ece"],
        }
        for output in outputs
    ]
    df_results = pd.DataFrame(results)
    df_results = df_results.pivot(
        index="C", columns="dataset-name", values=["eer", "ece"]
    )
    st.write(df_results)

    dfs = [
        pd.DataFrame(
            {
            "dataset-name": output["dataset_name"],
            "C": output["C"],
            **metric,
            }
        )
        for output in outputs
        for metric in get_reliability_metrics(
            output, entropy_normed, subset_func=select_only_fakes
        )
    ]
    df = pd.concat(dfs)
    df = df.reset_index(drop=True)
    # df.drop(columns=["τ"], inplace=True)
    df["C"] = df["C"].map(lambda x: f"{x:.0f}")

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
        hue="C",
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
        pd.DataFrame(
            {
            "dataset-name": output["dataset_name"],
            "C": output["C"],
            **metric,
            }
        )
        for output in outputs
        for metric in get_reliability_metrics(output, entropy_normed, subset_func=None)
    ]
    df = pd.concat(dfs)
    df = df.reset_index(drop=True)
    # df.drop(columns=["τ"], inplace=True)
    df["C"] = df["C"].map(lambda x: f"{x:.0f}")

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
        hue="C",
        kind="line",
    )
    fig.set(xlabel="Fraction of samples kept (%)", ylabel="Accuracy (%)")
    fig.set_titles("{col_name}")
    st.pyplot(fig)

    fig.savefig("output/icassp/reliability-all.png")

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
