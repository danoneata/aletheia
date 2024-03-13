import os
import pdb

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from matplotlib import pyplot as plt
from toolz import join
from sklearn.metrics import accuracy_score, roc_auc_score

from aletheia.data import ASVspoof2019, InTheWild, Datum
from aletheia.metrics import compute_eer, compute_ece
from aletheia.utils import implies, read_file
from aletheia.timnet.test import PREDICTIONS_DIR


sns.set_context("notebook")


DATASETS = {
    "ASVspoof2019": ASVspoof2019,
    "InTheWild": InTheWild,
}


# def shorten_model_name(f):
#     filename, _ = f.split(".")
#     parts = filename.split("_")
#     return "_".join(parts[3:])


# model_folder = Path("rawnet2-antispoofing/eval_scores")
# model_names = [shorten_model_name(f) for f in os.listdir(model_folder) if f.endswith(".txt")]
# model_names = [n for n in model_names if not n.endswith("intw")]
# model_names = sorted(model_names)


RESULTS = [
    {
        "model-name": "rawnet2-seed1",
        "path": "rawnet2-antispoofing/eval_scores/eval_CM_scores_rawnet2_seed1.txt",
        "scores-type": "real",
        "dataset": "ASVspoof2019",
    },
    {
        "model-name": "rawnet2-seed2",
        "path": "rawnet2-antispoofing/eval_scores/eval_CM_scores_rawnet2_seed2.txt",
        "scores-type": "real",
        "dataset": "ASVspoof2019",
    },
    {
        "model-name": "rawnet2-seed3",
        "path": "rawnet2-antispoofing/eval_scores/eval_CM_scores_rawnet2_seed3.txt",
        "scores-type": "real",
        "dataset": "ASVspoof2019",
    },
    {
        "model-name": "rawnet3-seed1",
        "path": "rawnet2-antispoofing/eval_scores/eval_CM_scores_rawnet3_seed1.txt",
        "scores-type": "real",
        "dataset": "ASVspoof2019",
    },
    {
        "model-name": "timnet",
        "path": "/mnt/student-share/data/ai4trust/TIMNET_scores/timnet_4.12eer_onlytrain_TEST.txt",
        "scores-type": "fake",
        "dataset": "ASVspoof2019",
    },
    {
        "model-name": "rawnet2-seed1",
        "path": "rawnet2-antispoofing/eval_scores/eval_CM_scores_rawnet2_seed1_intw.txt",
        "scores-type": "real",
        "dataset": "InTheWild",
    },
    {
        "model-name": "rawnet2-seed2",
        "path": "rawnet2-antispoofing/eval_scores/eval_CM_scores_rawnet2_seed2_intw.txt",
        "scores-type": "real",
        "dataset": "InTheWild",
    },
    {
        "model-name": "rawnet2-seed3",
        "path": "rawnet2-antispoofing/eval_scores/eval_CM_scores_rawnet2_seed3_intw.txt",
        "scores-type": "real",
        "dataset": "InTheWild",
    },
    {
        "model-name": "rawnet3-seed1",
        "path": "rawnet2-antispoofing/eval_scores/eval_CM_scores_rawnet3_seed1_intw.txt",
        "scores-type": "real",
        "dataset": "InTheWild",
    },
    {
        "model-name": "timnet",
        "path": "/mnt/student-share/data/ai4trust/TIMNET_scores/timnet_4.12eer_onlytrain_OOD.txt",
        "scores-type": "fake",
        "dataset": "InTheWild",
    },
    {
        "model-name": "timnet-k-fold-cv",
        "path": "/mnt/student-share/data/ai4trust/TIMNET_scores/ood-k-fold-cv.txt",
        "scores-type": "fake",
        "dataset": "InTheWild",
    },
]


CONFIGS_SELECTED = [
    "titanet-linear",
    "titanet-mlp-1",
    "titanet-mlp-2",
    "titanet-one-out",
]
DATASET_NAMES = {
    "asvspoof-test": "ASVspoof2019",
    "in-the-wild-test": "InTheWild",
}

for c in CONFIGS_SELECTED:
    for d in DATASET_NAMES:
        path = PREDICTIONS_DIR / ("asvspoof-" + c) / (d + ".txt")
        path = str(path)
        RESULTS.append(
            {
                "model-name": c,
                "path": path,
                "scores-type": "fake",
                "dataset": DATASET_NAMES[d],
            }
        )

for d in DATASET_NAMES:
    c = "titanet-one-out-laplace-2"
    path = PREDICTIONS_DIR / "asvspoof-titanet-one-out" / (d + "-laplace-2.txt")
    path = str(path)
    RESULTS.append(
        {
            "model-name": c,
            "path": path,
            "scores-type": "fake",
            "dataset": DATASET_NAMES[d],
        }
    )


# @st.cache_data
def load_results(dataset, kwargs):
    import random

    path = kwargs["path"]
    scores_type = kwargs["scores-type"]

    def parse_line(line: str):
        filename, _, _, score = line.split()
        score = float(score)
        if scores_type == "real":
            score = 1 - score
        return filename, score

    scores = read_file(path, parse_line)
    scores_data = join(0, scores, lambda datum: datum.filename, dataset)

    return [(score, datum) for (_, score), datum in scores_data]


def load_snrs(dataset_name):
    BASE_PATH = Path("/home/tts4trust/WORK/TIM-Net_SER/pytorch/1.SNRS")
    FILENAMES = {
        "ASVspoof2019": "ASV_LA_test_SNRS",
        "InTheWild": "ASV_LA_eval_SNRS",
    }

    def parse(line):
        filename, snr = line.split(",")
        filename = filename.split(".")[0]
        snr = float(snr)
        return filename, snr

    path = BASE_PATH / (FILENAMES[dataset_name] + ".txt")
    data = read_file(str(path), parse)
    return data


def aggregate_results(results_multi):
    scores, data = zip(*results_multi)
    assert len(set(datum.filename for datum in data)) == 1
    return np.mean(scores), data[0]


@dataclass
class Result:
    score: float
    snr: float
    datum: Datum


def plot_score_histograms(results):
    df = [
        {
            "score": result.score,
            "label": result.datum.label,
            "system": result.datum.system,
        }
        for result in results
    ]
    df = pd.DataFrame(df)
    # df = df.sample(5_000)
    fig, axs = plt.subplots(ncols=2, figsize=(8, 4), sharey=True)
    axs[0] = sns.histplot(
        data=df,
        x="score",
        bins=15,
        multiple="stack",
        ax=axs[0],
        hue="label",
    )
    axs[1] = sns.histplot(
        data=df,
        x="score",
        bins=15,
        multiple="stack",
        ax=axs[1],
        hue="system",
    )
    sns.move_legend(axs[1], loc="upper left", bbox_to_anchor=(1.0, 1.0))
    axs[0].set_title("all")
    axs[1].set_title("per system")
    axs[0].set_xlim(0, 1)
    axs[1].set_xlim(0, 1)
    fig.tight_layout()
    return fig


def compute_metrics_on_results(results):
    y_true = [result.datum.label == "fake" for result in results]
    y_pred = [result.score for result in results]
    y_pred_binary = [score > 0.5 for score in y_pred]
    count_smaller_than_0_5 = sum(score < 0.5 for score in y_pred)
    try:
        auc = 100 * roc_auc_score(y_true, y_pred)
    except ValueError:
        auc = None
    return {
        "eer": 100 * compute_eer(y_true, y_pred),
        "ece": 100 * compute_ece(y_true, y_pred),
        "accuracy": 100 * accuracy_score(y_true, y_pred_binary),
        "area-under-curve": auc,
        "prop-smaller-than-0.5": 100 * count_smaller_than_0_5 / len(y_pred),
    }


def compute_eer_system(results, s):
    results_sys = [
        result
        for result in results
        if result.datum.label == "real" or result.datum.system == s
    ]
    y_true = [result.datum.label == "fake" for result in results_sys]
    y_pred = [result.score for result in results_sys]
    return 100 * compute_eer(y_true, y_pred)


def format_speaker(speaker):
    return speaker.replace("_", " ")


def main():
    with st.sidebar:
        dataset_name = st.selectbox("dataset", ["InTheWild", "ASVspoof2019"])

        dataset = DATASETS[dataset_name]("eval")
        systems = sorted(set(datum.system for datum in dataset))
        speakers = sorted(set(datum.speaker for datum in dataset))

        model_names = sorted(
            [
                result["model-name"]
                for result in RESULTS
                if result["dataset"] == dataset_name
            ]
        )
        selected_models = st.multiselect(
            "models",
            model_names,
            default=["rawnet2-seed1", "rawnet2-seed2", "rawnet2-seed3"],
            help="if multiple models are selected, their scores are averaged",
        )
        to_sort_decreasing = st.selectbox(
            "sort decreasing (by fakeness score)", [True, False]
        )
        top_k = st.number_input(
            "num. of examples", value=50, min_value=1, max_value=1000, step=5
        )
        st.markdown("---")
        st.markdown("## filters")
        selected_system = st.selectbox("system", ["all"] + systems)
        selected_label = st.selectbox("label", ["all", "fake", "real"])
        selected_speaker = st.selectbox("speaker", ["all"] + speakers)

    if not selected_models:
        st.error("No models selected. Please select at least a model.")
        st.stop()

    results1 = [
        result
        for result in RESULTS
        if result["model-name"] in selected_models and result["dataset"] == dataset_name
    ]
    results_all = [load_results(dataset, result) for result in results1]
    results_all = zip(*results_all)

    results = [aggregate_results(rs) for rs in results_all]

    snrs = load_snrs(dataset_name)
    results = join(lambda r: r[1].filename, results, 0, snrs)
    results = [Result(score, snr, datum) for (score, datum), (_, snr) in results]
    # results = [Result(score, 0, datum) for score, datum in results]

    metrics_all = compute_metrics_on_results(results)
    fig_all = plot_score_histograms(results)

    eers = [
        {
            "system": s,
            "eer": compute_eer_system(results, s),
        }
        for s in systems
        if s != "-"
    ]
    df_eers = pd.DataFrame(eers)

    results = sorted(results, reverse=to_sort_decreasing, key=lambda t: t.score)
    results = [
        result
        for result in results
        if implies(selected_system != "all", result.datum.system == selected_system)
        and implies(selected_label != "all", result.datum.label == selected_label)
        and implies(selected_speaker != "all", result.datum.speaker == selected_speaker)
    ]

    metrics_selected = compute_metrics_on_results(results)
    fig_selected = plot_score_histograms(results)

    st.markdown("### quantitative results")
    st.markdown("on all:")
    st.write(metrics_all)
    st.markdown("on filtered:")
    st.write(metrics_selected)
    st.markdown("---")
    st.markdown("### score histograms")
    st.markdown("on all:")
    st.pyplot(fig_all)
    st.markdown("on filtered:")
    st.pyplot(fig_selected)
    st.markdown("---")

    st.markdown("### qualitative results")

    for result in results[:top_k]:
        st.markdown(
            "fakness score: {:.5f} · key: `{}` · label: `{}` · system: `{}` · speaker: {} · SNR: {:.2f}".format(
                result.score,
                result.datum.filename,
                result.datum.label,
                result.datum.system,
                format_speaker(result.datum.speaker),
                result.snr,
            )
        )
        st.audio(str(dataset.get_path_audio(result.datum)))
        st.markdown("---")


if __name__ == "__main__":
    main()
