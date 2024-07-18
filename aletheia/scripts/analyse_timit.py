import pdb
import random

from typing import List

from matplotlib import pyplot as plt
from scipy.stats import describe

import numpy as np
import seaborn as sns
import streamlit as st
import pandas as pd

from pycalib.metrics import binary_ECE

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve

from aletheia.data import Datum, read_file
from aletheia.scripts.linear_classifier import load_data_multi
from aletheia.metrics import compute_eer, compute_ece
from aletheia.utils import cache_json


sns.set_theme(context="talk", font="Arial", style="whitegrid")


def load_metadata(subset) -> List[Datum]:
    LABEL_MAP = {
        "bonafide": "real",
        "spoof": "fake",
    }

    def parse_line(line: str) -> Datum:
        speaker1, filename1, _, system1, label = line.split()
        *system, _ = filename1.split("/")
        speaker = speaker1.split("_")[0]
        filename = speaker1
        if len(system) == 1:
            system = system[0] + "-single"
        elif len(system) == 2:
            system = system[0] + "-multi"
        else:
            assert system[0] == "real"
            system = None
        return Datum(filename, LABEL_MAP[label], system, speaker)  # type: ignore[arg-type]

    subset1 = subset.replace("-", "_")
    path = f"/home/opascu/AI4TRUST/timit/timit-aggregated-{subset1}/single_and_multi_speaker/ASVspoof2019.LA.cm.eval.trl.txt"
    return read_file(path, parse_line)


def predict(subset, verbose=True, C=1e6):

    tr_datasets = [
        {
            "dataset_name": "asvspoof19",
            "split": split,
            "feature_type": "wav2vec2-xls-r-2b",
            "subset": "all",
        }
        for split in ["train", "dev"]
    ]

    te_datasets = [
        {
            "dataset_name": f"timit-tts-{subset}",
            "split": "eval",
            "feature_type": "wav2vec2-xls-r-2b",
            "subset": "all",
        }
    ]

    X_tr, y_tr = load_data_multi(tr_datasets)
    X_te, y_te = load_data_multi(te_datasets)

    model = LogisticRegression(C=C, max_iter=5_000, random_state=42, verbose=verbose)
    model.fit(X_tr, y_tr)
    pred = model.predict_proba(X_te)[:, 1]

    return {
        "true": y_te.tolist(),
        "pred": pred.tolist(),
    }


def evaluate(subset, C):
    output = cache_json(f"/tmp/{subset}-{C}.json", predict, subset=subset, C=C)
    return {
        "subset": subset,
        "C": C,
        "eer": 100 * compute_eer(output["true"], output["pred"]),
        "ece": 100 * compute_ece(output["true"], output["pred"]),
        # "ece1": 100 * binary_ECE(np.array(output["true"]), np.array(output["pred"]), bins=15),
    }


def validate_C():
    results = [
        evaluate(subset, 10**logC)
        for subset in ["clean", "dtw-aug"]
        for logC in range(7)
    ]
    df = pd.DataFrame(results)
    print(df)


def evaluate_per_system(subset):

    C = 1e2
    metadata = load_metadata(subset)
    systems = sorted(
        set(datum.system for datum in metadata if datum.system is not None)
    )
    output = cache_json(f"/tmp/{subset}-{C}.json", predict, subset=subset, C=C)

    def evaluate1(output, system):
        is_valid = lambda datum: datum.system == system or datum.label == "real"
        true = np.array(
            [output["true"][i] for i, datum in enumerate(metadata) if is_valid(datum)]
        )
        pred = np.array(
            [output["pred"][i] for i, datum in enumerate(metadata) if is_valid(datum)]
        )

        print(len(true), sum(true))
        idxs = list(range(len(true)))
        idxs = random.choices(idxs, k=len(idxs))
        true = true[idxs]
        pred = pred[idxs]

        eer = 100 * compute_eer(true, pred)
        ece = 100 * compute_ece(true, pred)
        ece1 = 100 * binary_ECE(true, pred, bins=15)

        fpr, tpr, thresholds = roc_curve(true, pred, pos_label=1)

        fig, axs = plt.subplots(ncols=2, figsize=(12, 5))
        axs[0].plot(fpr, 1 - tpr)
        axs[0].plot([0, 1], [0, 1], "--")
        axs[0].set_xlabel("False acceptance rate")
        axs[0].set_ylabel("False rejection rate")

        step = 1 / 20
        bins = np.arange(0, 1 + step, step)

        axs[1].hist(pred[true == 0], bins=bins, alpha=0.5, label="real")
        axs[1].hist(pred[true == 1], bins=bins, alpha=0.5, label="fake")
        axs[1].set_xlabel("Fakness score")
        axs[1].set_ylabel("Count")
        axs[1].legend()

        fig.tight_layout()

        st.markdown(
            "### TTS: `{}` · EER: {:.1f}% · ECE: {:.1f}%".format(
                system, eer, ece,
            )
        )
        # st.write(eer)
        # st.write(describe(pred[true == 1]))
        st.pyplot(fig)
        st.markdown("---")

        return {
            "subset": subset,
            "system": system,
            "eer": eer,
            "ece": ece,
            # "ece1": ece1,
        }

    results = [evaluate1(output, system) for system in systems]

    df = pd.DataFrame(results)
    print(df)
    print(evaluate(subset, C))


def show_per_sample(subset, filenames):
    C = 1e2
    metadata = load_metadata(subset)
    output = cache_json(f"/tmp/{subset}-{C}.json", predict, subset=subset, C=C)
    results = [
        {
            "true": true,
            "pred": pred,
            "filename": datum.filename,
        }
        for datum, true, pred in zip(metadata, output["true"], output["pred"])
    ]
    df = pd.DataFrame(results)
    df = df[df["filename"].isin(filenames)]
    fig, ax = plt.subplots()
    sns.stripplot(
        data=df,
        x="pred",
        y="filename",
        hue="true",
        jitter=True,
        alpha=0.5,
        ax=ax,
    )
    ax.set_title(subset)
    st.pyplot(fig)


def main():
    evaluate_per_system("clean")
    # evaluate_per_system("dtw-aug")

    metadata = load_metadata("clean")
    filenames = random.sample(list(set(datum.filename for datum in metadata)), 10)
    show_per_sample("clean", filenames)
    show_per_sample("dtw-aug", filenames)


if __name__ == "__main__":
    main()
