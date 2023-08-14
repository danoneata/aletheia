import pdb

import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from matplotlib import pyplot as plt
from toolz import merge

from aletheia.scripts.linear_classifier import evaluate
from aletheia.utils import cache_json


sns.set_context("talk")


def main():
    TR_DATASET_NAME = "asvspoof19"
    SPLITS = ["train", "dev"]
    FEATURE_TYPE = "wav2vec2-xls-r-2b"

    # prepare subsets
    NUM_SAMPLES = [1000, 2000, 4000, 8000, 16000]
    SEEDS = [0, 1, 2]
    SUBSETS = [f"{num}-{seed}" for num in NUM_SAMPLES for seed in SEEDS]
    SUBSETS.append("all")

    DATASETS_REAL_EXTRA = [
        [],
        ["librispeech"],
        ["common-voice"],
        ["voxpopuli"],
        ["tedlium"],
        ["gigaspeech"],
        ["spgispeech"],
        ["earnings22"],
        ["ami"],
        [
            "librispeech",
            "common-voice",
            "voxpopuli",
            "tedlium",
            "gigaspeech",
            "spgispeech",
            "earnings22",
            "ami",
        ],
    ]

    def get_tr_datasets(*, subset, data_extra):
        datasets1 = [
            {
                "dataset_name": TR_DATASET_NAME,
                "split": split,
                "feature_type": FEATURE_TYPE,
                "subset": subset,
            }
            for split in SPLITS
        ]
        datasets2 = [
            {
                "dataset_name": dataset_name,
                "split": "dev",
                "feature_type": FEATURE_TYPE,
                "subset": "all",
            }
            for dataset_name in data_extra
        ]
        return datasets1 + datasets2

    def get_cache_path(subset, data_extra):
        data_extra_str = "-".join(data_extra)
        return f"output/results/real-data-{subset}-{data_extra_str}.json"

    def get_seed(subset):
        if subset == "all":
            return 0
        else:
            _, seed = subset.split("-")
            return int(seed)

    def get_num(subset):
        if subset == "all":
            return 25_380
        else:
            num, _ = subset.split("-")
            return int(num)

    results = [
        merge(
            result,
            {
                "data-extra": data_extra,
                "num-asvspoof": len(SPLITS) * get_num(subset),
                "seed": get_seed(subset),
            },
        )
        for subset in SUBSETS
        for data_extra in DATASETS_REAL_EXTRA
        for result in cache_json(
            get_cache_path(subset, data_extra),
            evaluate,
            get_tr_datasets(subset=subset, data_extra=data_extra),
            verbose=True,
        )
        # for result in [
        #     {
        #         "eer": 100 * np.random.rand(),
        #         "ece": 100 * np.random.rand(),
        #         "te-dataset": d,
        #     }
        #     for d in "asvspoof19 in-the-wild".split(" ")
        # ]
    ]

    df = pd.DataFrame(results)
    st.dataframe(df)

    DATATASET_SHOW_NAMES = {
        "asvspoof19": "ASVspoof'19",
        "in-the-wild": "In the Wild",
    }

    DATA_EXTRA_SHOW_NAMES = {
        "": "∅",
        "librispeech": "LibriSpeech",
        "common-voice": "CommonVoice",
        "voxpopuli": "VoxPopuli",
        "tedlium": "TEDLIUM",
        "gigaspeech": "GigaSpeech",
        "spgispeech": "SPGISpeech",
        "earnings22": "Earnings22",
        "ami": "AMI",
        "all": "All",
    }

    # df = df.replace({"te-dataset": DATATASET_SHOW_NAMES})
    def data_to_str(xs):
        if len(xs) == 0:
            return ""
        elif len(xs) == 1:
            return xs[0]
        elif len(xs) == 8:
            return "all"
        else:
            assert False

    df["data-extra-str"] = df["data-extra"].apply(data_to_str)

    fig, axs = plt.subplots(2, 2, figsize=(6 * 2, 8 * 2), sharex=True, sharey=True)
    axs = iter(axs.flatten())
    for metric in ["eer", "ece"]:
        for dataset in ["asvspoof19", "in-the-wild"]:
            ax = next(axs)
            cols = ["data-extra-str", "num-asvspoof", "seed", metric]
            idxs = df["te-dataset"] == dataset
            dfss = df[cols][idxs]
            dfss = dfss.groupby(["data-extra-str", "num-asvspoof"])[metric].mean()
            dfss = dfss.reset_index()
            dfss = dfss.pivot(
                index="data-extra-str", columns="num-asvspoof", values=metric
            )
            dfss = dfss.reindex(
                index=[
                    "",
                    "librispeech",
                    "common-voice",
                    "voxpopuli",
                    "tedlium",
                    "gigaspeech",
                    "spgispeech",
                    "earnings22",
                    "ami",
                    "all",
                ]
            )
            dfss = dfss.rename(index=DATA_EXTRA_SHOW_NAMES)
            sns.heatmap(
                dfss,
                ax=ax,
                annot=True,
                fmt=".1f",
                cbar=False,
                square=True,
                cmap="viridis_r",
            )
            ax.set_title(
                "{} · {} (%)".format(DATATASET_SHOW_NAMES[dataset], metric.upper())
            )
            ax.set_xlabel("Num. ASVspoof'19 train samples")
            ax.set_ylabel("Additional real data")

    # def draw_heatmap(*args, **kwargs):
    #     data = kwargs.pop("data")
    #     pdb.set_trace()
    #     d = data.pivot(index=args[1], columns=args[0], values=args[2])
    #     sns.heatmap(d, **kwargs)

    # fig = sns.FacetGrid(df, col="")
    # fig.map_dataframe(draw_heatmap, "label1", "label2", "value", cbar=False)
    fig.tight_layout()
    st.pyplot(fig)
    fig.savefig("output/icassp/real-data.png")

    # fig = sns.relplot(
    #     data=df,
    #     x="num",
    #     y="eer",
    #     col="te-dataset",
    #     # height=4,
    #     # aspect=0.7,
    #     kind="line",
    #     ci="sd",
    # )
    # fig.set(xlabel="Number of training samples", ylabel="EER (%)")
    # fig.set_titles("{col_name}")

    # st.pyplot(fig)
    # fig.savefig("output/icassp/num-training-samples.png")


if __name__ == "__main__":
    main()
