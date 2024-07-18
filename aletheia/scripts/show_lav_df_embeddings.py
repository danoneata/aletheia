import pdb
import json
import random

from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import torch
import tqdm

from scipy.special import expit

from numpy.lib.stride_tricks import sliding_window_view

# import torch
# from torch.nn.functional import conv1d

# from pydub import AudioSegment
# from toolz import first
from tqdm import tqdm

from matplotlib import pyplot as plt

from sklearn import manifold
from sklearn.linear_model import LogisticRegression

from aletheia.scripts.extract_features import FEATURE_EXTRACTORS, SAMPLING_RATE
from aletheia.scripts.linear_classifier import load_data_multi
from aletheia.scripts.plot_tsne import clear_ticks
from aletheia.utils import cache_json, cache_np, cache_pickle


st.set_page_config(layout="wide")

BASE_PATH = Path("/mnt/student-share/data/lav-df")


def load_metadata():
    json_path = BASE_PATH / "metadata.json"
    with open(json_path, "r") as f:
        return json.load(f)


def load_metadata_subset():
    metadata = load_metadata()
    split = "test"

    def is_in_split(datum, split):
        return datum["file"].startswith(f"{split}/")

    def is_fake(datum):
        return datum["modify_video"] and datum["modify_audio"]

    metadata = [
        datum for datum in metadata if is_in_split(datum, split) and is_fake(datum)
    ]

    return random.sample(metadata, 256)


def extract_features(metadata):
    feature_type = "wav2vec2-xls-r-2b"
    feature_extractor = FEATURE_EXTRACTORS[feature_type]()

    def extract1(datum):
        audio_path = BASE_PATH / datum["file"]
        audio, _ = librosa.load(audio_path, sr=SAMPLING_RATE)
        feature = feature_extractor(audio, sr=SAMPLING_RATE)
        feature = feature.cpu().numpy()
        return feature.squeeze(0)

    def get_path(datum):
        filename = Path(datum["file"])
        return "/tmp/features/" + filename.stem + ".npy"

    return [cache_np(get_path(datum), extract1, datum) for datum in tqdm(metadata[:5])]


def build_labels(feature, datum):
    RESOLUTION = 0.02

    labels = np.zeros(feature.shape[0])
    for start, end in datum["fake_periods"]:
        start = int(start / RESOLUTION)
        end = int(end / RESOLUTION)
        labels[start:end] = 1

    return labels


def average_on_window(features, labels, w):
    size = 2 * w + 1
    features1 = sliding_window_view(features, size, axis=0)
    features1 = features1.mean(axis=2)
    labels1 = sliding_window_view(labels, size, axis=0)
    labels1 = labels1.max(axis=1)
    return features1, labels1


def show_tsne(features, labels, scores):
    def make_tsne_plot(axs, features, labels):
        tsne = manifold.TSNE()
        feat2d = tsne.fit_transform(features)
        x1, x2 = feat2d.T
        df = pd.DataFrame(
            {
                "x1": x1,
                "x2": x2,
                "label": labels,
                "index": np.arange(len(labels)),
            }
        )
        sns.scatterplot(
            df,
            x="x1",
            y="x2",
            hue="label",
            style="label",
            ax=axs[0],
            legend=False,
        )

        sns.scatterplot(
            df,
            x="x1",
            y="x2",
            hue="index",
            style="label",
            ax=axs[1],
            legend=False,
        )

        clear_ticks(axs[0])
        clear_ticks(axs[1])

    n = 8
    nrows = 3
    fig, axs = plt.subplots(ncols=n, nrows=nrows, figsize=(4 * n, 4 * nrows))

    for i in range(n):
        j = i + 5
        features1, labels1 = average_on_window(features, labels, j)
        make_tsne_plot(axs[:2, i], features1, labels1)

        size = 2 * j + 1
        scores1 = sliding_window_view(scores, size, axis=0)
        scores1 = scores1.mean(axis=1)
        scores1 = expit(scores1)

        idxs = np.arange(len(scores1))
        axs[2, i].step(idxs, scores1)
        axs[2, i].step(idxs, labels1)

        axs[0, i].set_title(f"window size: {size}")

    st.pyplot(fig)


def train_model():
    tr_datasets = [
        {
            "dataset_name": "asvspoof19",
            "split": split,
            "feature_type": "wav2vec2-xls-r-2b",
            "subset": "all",
        }
        for split in ["train", "dev"]
    ] + [
        {
            "dataset_name": dataset,
            "split": "dev",
            "feature_type": "wav2vec2-xls-r-2b",
            "subset": "all",
        }
        for dataset in ["librispeech", "tedlium", "ami", "voxpopuli"]
    ]
    verbose = True
    C = 1e2
    X_tr, y_tr = load_data_multi(tr_datasets)
    model = LogisticRegression(C=C, max_iter=5_000, random_state=42, verbose=verbose)
    model.fit(X_tr, y_tr)
    return model


def main():
    metadata = cache_json("/tmp/metadata-ss.json", load_metadata_subset)
    features = extract_features(metadata)
    labels = [
        build_labels(feature, datum) for feature, datum in zip(features, metadata)
    ]
    model = cache_pickle("/tmp/model.pkl", train_model)

    for i in range(5):
        datum = metadata[i]
        st.markdown("## {} · `{}`".format(i, datum["file"]))
        st.markdown(datum["transcript"])
        scores_frame = model.decision_function(features[i])
        # score_utt = model.decision_function(features[i].mean(axis=0).reshape(1, -1))
        # st.write("scores frame:", scores_frame)
        # st.write("scores frame (average):", np.mean(scores_frame))
        # st.write("score utt:", score_utt)
        show_tsne(features[i], labels[i], scores_frame)
        st.markdown("---")

    data = [average_on_window(features[i], labels[i], 4) for i in range(5)]
    features = [d[0] for d in data]
    labels = [d[1] for d in data]

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)

    fig, ax = plt.subplots()
    tsne = manifold.TSNE()
    feat2d = tsne.fit_transform(features)
    x1, x2 = feat2d.T

    df = pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "label": labels,
        }
    )
    sns.scatterplot(
        df,
        x="x1",
        y="x2",
        hue="label",
        style="label",
        ax=ax,
        legend=False,
    )
    st.pyplot(fig)

    # idxs_real = np.where(labels == 0)[0]
    # idxs_fake = np.where(labels == 1)[0]
    # idxs_fake = random.sample(list(idxs_fake), len(idxs_real))


if __name__ == "__main__":
    main()
