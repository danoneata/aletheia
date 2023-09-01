import pdb

import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from matplotlib import pyplot as plt


sns.set_theme(context="talk", font="Arial")

from aletheia.scripts.evaluate_reliability import (
    δ,
    cache_json,
    entropy_normed,
    evaluate1,
    evaluate_reliability,
    get_reliability_metrics,
    select_only_fakes,
    predict,
)


SUBSET_FUNC = select_only_fakes
SUBSET_FUNC = None


def load_ours():
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
    path = "output/results/predictions-evaluate-reliability-2.json"
    outputs = cache_json(path, predict, TR_DATASETS, verbose=True)
    outputs = [{**output, **evaluate1(**output, verbose=True)} for output in outputs]

    return pd.DataFrame(
        [
            {
                "dataset-name": output["dataset_name"],
                "method-name": "ours",
                **metric,
            }
            for output in outputs
            for metric in get_reliability_metrics(
                output, entropy_normed, subset_func=SUBSET_FUNC,
            )
        ]
    )


def load_salvi():
    DATASET_SHORT_NAME = {
        "asvspoof19": "asv",
        "in-the-wild": "intw",
    }

    def compute_score(row):
        score = row["score"]
        reliab = row["reliab"]
        return np.dot(score, reliab) / np.sum(reliab)

    def load_data(dataset_name):
        def parse_array(s):
            xs = s.strip("[]").split(", ")
            return np.array([float(x) for x in xs])

        def parse_label(s):
            return int(s.strip("[]"))

        CONVERTERS = {
            "reliab": parse_array,
            "score": parse_array,
            "label": parse_label,
        }

        dataset_short = DATASET_SHORT_NAME[dataset_name]
        path = f"../aletheia/rawnet2-antispoofing/reliability_df_{dataset_short}.csv"
        df = pd.read_csv(path, index_col=0, converters=CONVERTERS)

        df["reliability"] = df["reliab"].map(lambda xs: xs.max())
        df["pred"] = df.apply(compute_score, axis=1)
        df["label"] = 1 - df["label"]
        df = df.rename(columns={"label": "true"})
        return df

    def get_reliability_metrics(output, subset_func):
        df = output.copy()
        df["pred-binary"] = df["pred"] > 0.5

        return [
            {
                # "dataset-name": output["dataset_name"],
                # "C": output["C"],
                "τ": τ,
                **evaluate_reliability(df, subset_func, τ),
            }
            for τ in np.arange(0.0, 1.0 + δ, δ)
        ]

    return pd.DataFrame(
        [
            {
                "dataset-name": dataset_name,
                "method-name": "salvi",
                **metric,
            }
            for dataset_name in ["asvspoof19", "in-the-wild"]
            for metric in get_reliability_metrics(load_data(dataset_name), subset_func=SUBSET_FUNC)
        ]
    )


def load_salvi_orig():
    def compute_score(row):
        score = row["score"]
        reliab = row["reliab"]
        return np.dot(score, reliab) / np.sum(reliab)

    def load_data(dataset_name):
        def parse_array(s):
            xs = s.strip("[]").split(" ")
            return np.array([float(x) for x in xs if x])

        CONVERTERS = {
            "reliab": parse_array,
            "score": parse_array,
        }

        path = f"output/salvi-{dataset_name}.csv"
        df = pd.read_csv(path, index_col=0, converters=CONVERTERS)

        df["reliab"] = 1 - df["reliab"]
        df["score"] = - df["score"]

        df["reliability"] = df["reliab"].map(lambda xs: xs.max())
        df["pred"] = df.apply(compute_score, axis=1)
        df = df.rename(columns={"label": "true"})
        return df

    def get_reliability_metrics(output, subset_func):
        df = output.copy()
        df["pred-binary"] = df["pred"] > 0.5

        return [
            {
                # "dataset-name": output["dataset_name"],
                # "C": output["C"],
                "τ": τ,
                **evaluate_reliability(df, subset_func, τ),
            }
            for τ in np.arange(0.0, 1.0 + δ, δ)
        ]

    return pd.DataFrame(
        [
            {
                "dataset-name": dataset_name,
                "method-name": "salvi-orig",
                **metric,
            }
            for dataset_name in ["asvspoof19", "in-the-wild"]
            for metric in get_reliability_metrics(load_data(dataset_name), subset_func=SUBSET_FUNC)
        ]
    )


def main():
    # results = [
    #     {
    #         "dataset-name": output["dataset_name"],
    #         "eer": output["eer"],
    #         "ece": output["ece"],
    #     }
    #     for output in outputs
    # ]
    # st.write(results)

    df1 = load_ours()
    df2 = load_salvi_orig()
    # df3 = load_salvi()
    # df = pd.concat([df1, df2, df3])
    df = pd.concat([df1, df2])

    st.write(df)
    df = df.drop("auc-roc", axis=1)
    df = df.dropna(axis=0, how="any")

    DATATASET_SHOW_NAMES = {
        "asvspoof19": "ASVspoof'19",
        "in-the-wild": "In the Wild",
    }

    METHOD_SHOW_NAMES = {
        "ours": "Ours",   
        "salvi-orig": "Salvi et al.", 
        "salvi": "Salvi et al. (ours)", 
    }

    df = df.replace({"dataset-name": DATATASET_SHOW_NAMES, "method-name": METHOD_SHOW_NAMES})
    df = df.rename(columns={"method-name": "Method"})

    fig = sns.relplot(
        data=df,
        x="frac-kept",
        y="accuracy",
        col="dataset-name",
        hue="Method",
        kind="line",
    )
    fig.set(xlabel="Fraction of samples kept (%)", ylabel="Accuracy (%)")
    fig.set_titles("{col_name}")
    st.pyplot(fig)

    fig.savefig("output/icassp/reliability-ours-vs-salvi.png")


if __name__ == "__main__":
    main()
