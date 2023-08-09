import pdb

from itertools import groupby
from toolz import merge

import pandas as pd
import seaborn as sns
import streamlit as st

from aletheia.scripts.linear_classifier import evaluate
from aletheia.utils import cache_json


sns.set_context("talk")


def main():
    DATASET_NAME = "asvspoof19"
    SPLITS = ["train", "dev"]
    FEATURE_TYPES = [
        "wav2vec2-large",
        "wav2vec2-xls-r-300m",
    ]
    NUM_SAMPLES = 4000
    SEEDS = [0, 1, 2]

    def get_tr_dataset(*, split, feature_type, seed):
        return {
            "dataset_name": DATASET_NAME,
            "split": split,
            "feature_type": feature_type,
            "subset": f"{NUM_SAMPLES}-{seed}",
        }

    def get_tr_datasets(*, feature_type, seed):
        return [
            get_tr_dataset(split=split, feature_type=feature_type, seed=seed)
            for split in SPLITS
        ]

    def get_cache_path(feature_type, seed):
        return f"output/results/pretraining-datasets-{feature_type}-{seed}.json"

    results = [
        merge(
            result,
            {
                "feature-type": feature_type,
                "seed": seed,
            },
        )
        for feature_type in FEATURE_TYPES
        for seed in SEEDS
        for result in cache_json(
            get_cache_path(feature_type, seed),
            evaluate,
            get_tr_datasets(feature_type=feature_type, seed=seed),
            verbose=True,
        )
    ]

    df = pd.DataFrame(results)
    df = df.groupby(["feature-type", "te-dataset"])
    df = df.agg({"eer": ["mean", "std"], "ece": ["mean", "std"]})

    to_str_1 = lambda x: f"{x:.1f}"
    to_str_m = lambda m: df[(m, "mean")].map(to_str_1) + "±" + df[(m, "std")].map(to_str_1)

    df["eer-str"] = to_str_m("eer")
    df["ece-str"] = to_str_m("ece")

    print(df)


if __name__ == "__main__":
    main()
