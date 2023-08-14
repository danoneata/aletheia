import sys
import pdb
import random

from itertools import chain, combinations
from pathlib import Path

import pandas as pd
import numpy as np
import seaborn as sns
import soundfile as sf
import streamlit as st
import torch
import tqdm

import matplotlib
from matplotlib import pyplot as plt
from sklearn import manifold

from aletheia.scripts.linear_classifier import load_data_npz


sns.set_context("talk")

TR_DATASET_NAME = "asvspoof19"
SPLITS = ["train", "dev"]
FEATURE_TYPE = "wav2vec2-xls-r-2b"
SUBSET = "500-0"


def clear_ticks(ax):
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def plot_real_data_1(dataset_real_names):
    pivot_dataset_name = "asvspoof19"
    datasets = [
        {
            "dataset_name": pivot_dataset_name,
            "split": split,
            "feature_type": FEATURE_TYPE,
            "subset": "2000-0",
        }
        for split in ["train", "dev"]
    ] + [
        {
            "dataset_name": dataset_real_name,
            "split": "dev",
            "feature_type": FEATURE_TYPE,
            "subset": "all",
        }
        for dataset_real_name in dataset_real_names
    ]

    data = [load_data_npz(**d) for d in datasets]
    Xs, ys = zip(*data)

    X = np.vstack(Xs)
    y = np.hstack(ys)
    names = [len(x) * [d["dataset_name"]] for x, d in zip(Xs, datasets)]
    names = list(chain(*names))

    t_sne = manifold.TSNE(
        # perplexity=30,
        # init="random",
        # n_iter=250,
        # random_state=0,
    )

    features_2d = t_sne.fit_transform(X)
    x1, x2 = features_2d.T
    df = pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "label": y,
            "dataset": names,
        }
    )

    df = df.replace({"label": {0: "real", 1: "fake"}})
    df = df.sample(frac=1).reset_index(drop=True)

    # fig, ax = plt.subplots()
    fig, axs = plt.subplots(ncols=2, figsize=(8, 4), sharex=True, sharey=True)
    cm = matplotlib.cm.get_cmap("Set1")
    sns.scatterplot(
        df[df["dataset"] == pivot_dataset_name],
        x="x1",
        y="x2",
        hue="label",
        style="label",
        hue_order=["real", "fake"],
        style_order=["real", "fake"],
        palette=[cm(2), cm(0)],
        ax=axs[0],
    )
    sns.scatterplot(
        df,
        x="x1",
        y="x2",
        hue="dataset",
        style="label",
        hue_order=[pivot_dataset_name] + dataset_real_names,
        style_order=["real", "fake"],
        # palette=[cm(6), cm(1)],
        ax=axs[1],
    )
    for ax in axs:
        clear_ticks(ax)
        sns.move_legend(ax, "lower left", bbox_to_anchor=(0, 1))
    st.pyplot(fig)

    # fig, axs = plt.subplots(ncols=2, figsize=(8, 4), sharex=True, sharey=True)
    # sns.scatterplot(
    #     df,
    #     x="x1",
    #     y="x2",
    #     hue="label",
    #     ax=axs[0],
    # )
    # sns.scatterplot(
    #     df,
    #     x="x1",
    #     y="x2",
    #     hue="dataset",
    #     ax=axs[1],
    # )
    # for ax in axs:
    #     sns.move_legend(ax, "lower left", bbox_to_anchor=(0, 1))
    #     clear_ticks(ax)
    # ax.set_title(title)

    # st.pyplot(fig)


def plot_real_data():
    DATASETS_REAL = [
        "librispeech",
        "common-voice",
        "voxpopuli",
        "tedlium",
        "gigaspeech",
        "spgispeech",
        "earnings22",
        "ami",
    ]
    plot_real_data_1([])
    plot_real_data_1(DATASETS_REAL)
    for dataset_real_name in DATASETS_REAL:
        st.markdown(f"## {dataset_real_name}")
        plot_real_data_1([dataset_real_name])


def main():
    plot_real_data()


# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier
# from toolz import concat
#
# from aletheia.data import ASVspoof2019, InTheWild
# from aletheia.metrics import compute_eer, compute_ece
# from aletheia.timnet.train import FEATURE_EXTRACTORS
# from aletheia.utils import cache_np
#
#
# random.seed(0)
#
#
# def load_audio_default(dataset, datum):
#     audio = dataset.load_audio(datum)
#     sr = 16_000
#     return audio, sr
#
#
# def load_audio_hf(_, datum):
#     return datum["audio"]["array"], datum["audio"]["sampling_rate"]
#
#
# DATASETS = {
#     "asvspoof-train": ASVspoof2019("train"),
#     "asvspoof-valid": ASVspoof2019("dev"),
#     "asvspoof-test": ASVspoof2019("eval"),
#     "in-the-wild": InTheWild("eval"),
#     # "gigaspeech-clean": load_dataset("esb/diagnostic-dataset", "gigaspeech")["clean"],
#     "librispeech-other": load_dataset("esb/diagnostic-dataset", "gigaspeech")["other"],
#     "gigaspeech-other": load_dataset("esb/diagnostic-dataset", "librispeech")["other"],
#     "common-voice-other": load_dataset("esb/diagnostic-dataset", "common_voice")[
#         "other"
#     ],
# }
#
# AUDIO_LOADERS = {
#     "asvspoof-train": load_audio_default,
#     "asvspoof-valid": load_audio_default,
#     "asvspoof-test": load_audio_default,
#     "in-the-wild": load_audio_default,
#     "librispeech-other": load_audio_hf,
#     "gigaspeech-other": load_audio_hf,
#     "gigaspeech-clean": load_audio_hf,
#     "common-voice-other": load_audio_hf,
# }
#
#
# def random_sample(dataset, n):
#     num_samples = len(dataset)
#     indices = list(range(num_samples))
#     if n >= num_samples:
#         return indices
#     else:
#         return random.sample(indices, n)
#
#
# # The order is crucial since affects the random sampling.
# # Always add a new dataset to the end of the list.
# DATASETS_ORDER = [
#     "asvspoof-valid",
#     "asvspoof-test",
#     "in-the-wild",
#     "librispeech-other",
#     "gigaspeech-other",
#     "common-voice-other",
#     "asvspoof-train",
# ]
# INDICES = {k: random_sample(DATASETS[k], 1000) for k in DATASETS_ORDER}
#
# ADRIANA_SPLITS = {
#     "asvspoof-valid": "dev",
#     "asvspoof-test": "eval",
#     "in-the-wild": "ood",
# }
#
#
# def load_titanet_feature(dataset_name, filename):
#     name = ADRIANA_SPLITS[dataset_name]
#     base_path = Path("/home/tts4trust/DATA/ASVSpoof_wavs")
#     path = base_path / f"asv_{name}_titanet" / f"{filename}.npy"
#     return np.load(path)
#
#
# def wrap_extractor(extractor):
#     # wraps an extractor by:
#     # 1. loading the audio file
#     # 2. extracting the mean representation of the features"""
#     extractor_instance = extractor()
#
#     def wrapped(dataset_name, datum):
#         dataset = DATASETS[dataset_name]
#         audio, sr = AUDIO_LOADERS[dataset_name](dataset, datum)
#         output = extractor_instance(audio, sr)
#         return output.mean(dim=1).squeeze(dim=0).cpu().numpy()
#
#     return wrapped
#
#
# def load_features(dataset_name, feature_name):
#     if feature_name == "titanet":
#
#         def get_feature_1(dataset_name, datum):
#             return load_titanet_feature(dataset_name, datum.filename)
#
#     else:
#         get_feature_1 = wrap_extractor(FEATURE_EXTRACTORS[feature_name])
#
#     def get_features():
#         features = [
#             get_feature_1(dataset_name, dataset[i]) for i in tqdm.tqdm(indices_ss)
#         ]
#         return np.vstack(features)
#
#     indices_ss = INDICES[dataset_name]
#     dataset = DATASETS[dataset_name]
#
#     path = f"output/cache/{feature_name}-{dataset_name}.npy"
#     features = cache_np(path, get_features)
#     try:
#         labels = [dataset[i].label for i in indices_ss]
#         speakers = [dataset[i].speaker for i in indices_ss]
#     except AttributeError:
#         labels = ["real"] * len(features)
#         speakers = [None] * len(features)
#     dataset_names = [dataset_name] * len(features)
#     return features, labels, speakers, dataset_names
#
#
# def do1(model_name, datasets_extra=(), order_datasets=None):
#
#     datasets_base = ("asvspoof-train", "asvspoof-test", "in-the-wild")
#     datasets = datasets_base + datasets_extra
#     data = [load_features(dataset_name, model_name) for dataset_name in datasets]
#
#     features, labels, speakers, datasets = zip(*data)
#     features = np.vstack(features)
#     labels = list(concat(labels))
#     speakers = list(concat(speakers))
#     datasets = list(concat(datasets))
#
#     te_datasets = ["asvspoof-test", "in-the-wild"]
#     tr_datasets = list(set(datasets) - set(te_datasets))
#     idxs_tr = np.array([d in tr_datasets for d in datasets])
#
#     y = np.array(labels) == "fake"
#     y = y.astype(float)
#
#     X_tr = features[idxs_tr]
#     y_tr = y[idxs_tr]
#
#     model = LogisticRegression(C=1e6)
#     # model = KNeighborsClassifier()
#     model.fit(X_tr, y_tr)
#
#     results = []
#
#     for te_dataset in te_datasets:
#         idxs_te = np.array([d == te_dataset for d in datasets])
#         X_te = features[idxs_te]
#         y_te = y[idxs_te]
#         preds = model.predict_proba(X_te)[:, 1]
#         eer = 100 * compute_eer(y_te, preds)
#         ece = 100 * compute_ece(y_te, preds)
#         results1 = {
#             "features": model_name,
#             "train-datasets": tuple(sorted(tr_datasets)),
#             "test-dataset": te_dataset,
#             "eer": eer,
#             "ece": ece,
#         }
#         results.append(results1)
#         # st.write(results1)
#
#     results_str = " · ".join(
#         "{} : eer {:.2f} · ece {:.2f}".format(r["test-dataset"], r["eer"], r["ece"])
#         for r in results
#     )
#     st.markdown("{} ◇ {}".format(model_name, results_str))
#     st.markdown("train datasets: {}".format(", ".join(sorted(tr_datasets))))
#
#     t_sne = manifold.TSNE(
#         # perplexity=30,
#         # init="random",
#         # n_iter=250,
#         # random_state=0,
#     )
#
#     features_2d = t_sne.fit_transform(features)
#     x, y = features_2d.T
#     df = pd.DataFrame(
#         {
#             "x": x,
#             "y": y,
#             "label": labels,
#             "dataset": datasets,
#             "speaker": speakers,
#         }
#     )
#
#     nrows = 1
#     ncols = 3
#     sz = 4
#     fig, axs = plt.subplots(
#         nrows, ncols, figsize=(ncols * sz, sz), sharex=True, sharey=True
#     )
#     titles = ["label", "dataset", "speaker"]
#
#     # shuffle?!
#     df = df.sample(frac=1).reset_index(drop=True)
#     order_labels = ["real", "fake"]
#     if order_datasets is None:
#         order_datasets = [d for d in DATASETS.keys() if d in datasets]
#
#     sns.scatterplot(
#         df,
#         x="x",
#         y="y",
#         hue="label",
#         hue_order=order_labels,
#         ax=axs[0],
#     )
#     sns.scatterplot(
#         df,
#         x="x",
#         y="y",
#         hue="dataset",
#         hue_order=order_datasets,
#         ax=axs[1],
#     )
#     sns.scatterplot(
#         df,
#         x="x",
#         y="y",
#         hue="speaker",
#         style="label",
#         legend=False,
#         ax=axs[2],
#     )
#
#     # sns.move_legend(axs[0], "upper left", bbox_to_anchor=(1, 1))
#
#     for ax, title in zip(axs, titles):
#         clear_ticks(ax)
#         ax.set_title(title)
#
#     fig.tight_layout()
#     st.pyplot(fig)
#
#     path = "output/icassp/tsne-{}-{}.npy".format(model_name, "-".join(datasets_extra))
#     fig.savefig(path)
#
#     return fig, results
#
#
# def show_representations():
#     st.markdown("## Evaluating representations")
#     model_names = ["titanet"] + list(FEATURE_EXTRACTORS.keys())
#     output = [do1(model_name) for model_name in tqdm.tqdm(model_names)]
#     # aggregate results
#     _, results = zip(*output)
#     results = list(concat(results))
#     df = pd.DataFrame(results)
#     df = df.pivot(
#         index="features",
#         columns="test-dataset",
#         values=["eer", "ece"],
#     )
#     st.write(df)
#
#
# def show_datasets_contributions():
#     def powerset(iterable):
#         s = list(iterable)
#         return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))
#
#     st.markdown("## Evaluating datasets contributions")
#     model_name = "wav2vec2-xls-r-2b"
#     datasets_extra = [
#         "librispeech-other",
#         "common-voice-other",
#         "gigaspeech-other",
#     ]
#     output = [
#         do1(model_name, datasets_extra=ds) for ds in tqdm.tqdm(powerset(datasets_extra))
#     ]
#     # aggregate results
#     _, results = zip(*output)
#     results = list(concat(results))
#     df = pd.DataFrame(results)
#     df = df.pivot(
#         index="train-datasets",
#         columns="test-dataset",
#         values=["eer", "ece"],
#     )
#     st.write(df)
#
#
# def main():
#     st.set_page_config(layout="wide")
#     # show_representations()
#     show_datasets_contributions()


if __name__ == "__main__":
    main()
