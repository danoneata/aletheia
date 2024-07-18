import pdb

from itertools import chain, combinations
from toolz import merge

import pandas as pd
import seaborn as sns
import streamlit as st

from aletheia.data import DATASETS
from aletheia.scripts.linear_classifier import evaluate
from aletheia.utils import cache_json


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def main():
    DATASET_NAME = "asvspoof19"
    SPLITS = ["train", "dev"]
    FEATURE_TYPE = "wav2vec2-xls-r-2b"
    SUBSET = "all"

    get_tr_datasets = lambda systems: [
        {
            "dataset_name": DATASET_NAME,
            "split": split,
            "feature_type": FEATURE_TYPE,
            "subset": SUBSET,
            "systems": systems,
        }
        for split in SPLITS
    ]

    dataset = DATASETS[DATASET_NAME]("train")
    systems = [
        dataset.get_system(i)
        for i in range(len(dataset))
        if dataset.get_label(i) == "fake"
    ]
    systems = sorted(set(systems))

    system_real = [
        dataset.get_system(i)
        for i in range(len(dataset))
        if dataset.get_label(i) == "real"
    ]
    system_real = sorted(set(system_real))
    assert len(system_real) == 1
    system_real = system_real[0]

    def get_cache_path(systems):
        systems_str = ",".join(systems)
        return f"output/results/per-system-subset-{systems_str}.json"

    results = [
        merge(
            result,
            {
                "systems": ss,
            },
        )
        for ss in powerset(systems)
        if ss
        for result in cache_json(
            get_cache_path(ss),
            evaluate,
            get_tr_datasets(ss + (system_real, )),
            verbose=True,
        )
    ]

    df = pd.DataFrame(results)
    df = df.pivot(index=["systems"], columns=["te-dataset"], values=["eer", "ece"])
    df = df.reset_index()
    # df = df.sort_values(by=["systems"])
    df_systems = df["systems"].str.join("|").str.get_dummies()
    df_values = df.drop(columns=["systems"])
    xs = [df_systems, df_values]
    df = pd.concat(xs, axis=1)
    df["num-systems"] = df_systems.sum(axis=1)
    df = df.sort_values(by=["num-systems"] + systems)
    print(df.to_csv())


if __name__ == "__main__":
    main()
