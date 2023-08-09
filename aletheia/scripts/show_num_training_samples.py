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
    FEATURE_TYPE = "wav2vec2-xls-r-2b"
    NUM_SAMPLES = [1000, 2000, 4000, 8000, 16000]
    SEEDS = [0, 1, 2]

    def get_tr_dataset(*, split, num, seed):
        return {
            "dataset_name": DATASET_NAME,
            "split": split,
            "feature_type": FEATURE_TYPE,
            "subset": f"{num}-{seed}",
        }

    def get_tr_datasets(*, num, seed):
        return [get_tr_dataset(split=split, num=num, seed=seed) for split in SPLITS]

    def get_cache_path(num, seed):
        return f"output/results/num-training-samples-{num}-{seed}.json"

    results = [
        merge(
            result,
            {
                "num": len(SPLITS) * num,
                "seed": seed,
            },
        )
        for num in NUM_SAMPLES
        for seed in SEEDS
        for result in cache_json(
            get_cache_path(num, seed),
            evaluate,
            get_tr_datasets(num=num, seed=seed),
            verbose=True,
        )
    ]

    df = pd.DataFrame(results)
    st.dataframe(df)

    DATATASET_SHOW_NAMES = {
        "asvspoof19": "ASVspoof'19",
        "in-the-wild": "In the Wild",
    }
    df = df.replace({"te-dataset": DATATASET_SHOW_NAMES})
    fig = sns.relplot(
        data=df,
        x="num",
        y="eer",
        col="te-dataset",
        # height=4,
        # aspect=0.7,
        kind="line",
        ci="sd",
    )
    fig.set(xlabel="Number of training samples", ylabel="EER (%)")
    fig.set_titles("{col_name}")

    st.pyplot(fig)
    fig.savefig("output/icassp/num-training-samples.png")


if __name__ == "__main__":
    main()
