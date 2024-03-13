from toolz import merge

from matplotlib import pyplot as plt

import pandas as pd
import seaborn as sns
import streamlit as st

from aletheia.scripts.linear_classifier import evaluate
from aletheia.utils import cache_json


# sns.set_context("talk")
# sns.set_theme(context="talk", font="Arial")
sns.set_theme(style="whitegrid", context="talk", font="Arial", font_scale=1.0)


def main():
    DATASET_NAME = "asvspoof19"
    SPLITS = ["train", "dev"]
    FEATURE_TYPE = "wav2vec2-xls-r-2b"
    C = 1e3

    NUM_SAMPLES = [1000, 2000, 4000, 8000, 16000, "all"]
    SEEDS = [0, 1, 2]

    def get_tr_dataset(split, num, seed):
        if num == "all":
            subset = "all"
        else:
            subset = f"{num}-{seed}"
        return {
            "dataset_name": DATASET_NAME,
            "split": split,
            "feature_type": FEATURE_TYPE,
            "subset": subset,
        }

    def get_tr_datasets(num, seed):
        return [get_tr_dataset(split, num, seed) for split in SPLITS]

    def get_cache_path(num, seed):
        return f"output/results/num-training-samples-{num}-{seed}.json"

    def get_num(num):
        if num == "all":
            return 25112
        else:
            return int(num)

    def get_seed(subset):
        if subset == "all":
            return 0
        else:
            _, seed = subset.split("-")
            return int(seed)

    results = [
        merge(
            result,
            {
                "num": len(SPLITS) * get_num(num),
                "seed": seed,
            },
        )
        for num in NUM_SAMPLES
        for seed in SEEDS
        for result in cache_json(
            get_cache_path(num, seed),
            evaluate,
            get_tr_datasets(num, seed),
            seed=seed,
            C=C,
            verbose=True,
        )
    ]

    df = pd.DataFrame(results)

    DATATASET_SHOW_NAMES = {
        "asvspoof19": "ASVspoof'19",
        "in-the-wild": "In the Wild",
        "timit-tts": "TIMIT-TTS",
        "fake-or-real": "FakeOrReal",
    }
    df = df.replace({"te-dataset": DATATASET_SHOW_NAMES})
    df = df.rename(columns={
        "te-dataset": "Test dataset",
        "num": "Num. train samples",
        "eer": "EER (%)",
        "ece": "ECE (%)",
    })
    st.dataframe(df)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    sns.lineplot(
        data=df,
        x="Num. train samples",
        y="EER (%)",
        style="Test dataset",
        hue="Test dataset",
        # col="te-dataset",
        # height=4,
        # aspect=0.7,
        errorbar="sd",
        ax=axs[0],
        legend=False,
    )
    sns.lineplot(
        data=df,
        x="Num. train samples",
        y="ECE (%)",
        style="Test dataset",
        hue="Test dataset",
        # col="te-dataset",
        # height=4,
        # aspect=0.7,
        errorbar="sd",
        ax=axs[1],
    )
    sns.move_legend(axs[1], "upper left", bbox_to_anchor=(1.0, 1.0), frameon=False)
    # fig.set(xlabel="Number of training samples", ylabel="EER (%)")
    # fig.set_titles("{col_name}")

    fig.tight_layout()
    st.pyplot(fig)
    fig.savefig("output/icassp/num-training-samples.pdf")


if __name__ == "__main__":
    main()
