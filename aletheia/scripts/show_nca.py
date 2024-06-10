import pdb
import random

import streamlit as st

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NeighborhoodComponentsAnalysis

from aletheia.data import DATASETS
from aletheia.scripts.linear_classifier import load_data_multi
from aletheia.metrics import compute_eer
from aletheia.utils import cache_pickle


st.set_page_config(layout="wide")
FEATURE_TYPE = "wav2vec2-xls-r-2b"


def load_data(dataset_name, splits, subset):
    datasets = [
        {
            "dataset_name": dataset_name,
            "split": split,
            "feature_type": FEATURE_TYPE,
            "subset": subset,
        }
        for split in splits
    ]
    return load_data_multi(datasets)


def train_nca(n_components, subset):
    X_tr, y_tr = load_data("asvspoof19", splits=["train", "dev"], subset=subset)
    model = NeighborhoodComponentsAnalysis(n_components=n_components, verbose=True)
    model.fit(X_tr, y_tr)
    return model


def compare_classifiers():
    X_tr, y_tr = load_data("asvspoof19", splits=["train", "dev"], subset="all")
    # X_te, y_te = load_data("asvspoof19", splits=["eval"], subset="all")
    X_te, y_te = load_data("in-the-wild", splits=["eval"], subset="all")

    random.seed(10)
    idxs = random.sample(range(len(X_te)), 1000)
    X_te = X_te[idxs]
    y_te = y_te[idxs]

    # LogReg
    model = LogisticRegression(C=1e6, max_iter=5_000, random_state=42)
    model.fit(X_tr, y_tr)
    y_pred = model.predict_proba(X_te)[:, 1]
    print("log reg: {:.1f}".format(100 * compute_eer(y_te, y_pred)))

    # NCA + kNN
    nca_dim = 256
    nca_subset = "4000-0"
    nca = cache_pickle(
        f"output/nca-{nca_dim}-{nca_subset}",
        train_nca,
        n_components=nca_dim,
        subset=nca_subset,
    )

    for k in range(57, 77, 2):
        # kNN
        model = KNeighborsClassifier(n_neighbors=k, weights="distance")
        model.fit(X_tr, y_tr)
        y_pred = model.predict_proba(X_te)[:, 1]
        print("knn {}:       {:.1f}".format(k, 100 * compute_eer(y_te, y_pred)))

        model = KNeighborsClassifier(n_neighbors=k, weights="distance")
        model.fit(nca.transform(X_tr), y_tr)
        y_pred = model.predict_proba(nca.transform(X_te))[:, 1]
        print("nca + knn {}: {:.1f}".format(k, 100 * compute_eer(y_te, y_pred)))


def show_results():
    with st.sidebar:
        K = st.number_input("number of neighbours", value=5)
        te_dataset_name = st.selectbox("test dataset", ["asvspoof19", "in-the-wild"])
        st.markdown("---")
        num_te_samples = st.number_input("show test samples", value=16)
        filter_label = st.selectbox("filter by label", ["all", "real", "fake"])
        sort_by_uncertainty = st.checkbox("sort by uncertainty", value=False)
        # seed = st.number_input("seed", value=42)

    X_tr, y_tr = load_data("asvspoof19", splits=["train"], subset="all")
    X_te, y_te = load_data(te_dataset_name, splits=["eval"], subset="all")

    random.seed(42)
    te_idxs = random.sample(range(len(X_te)), 1_000)
    X_te = X_te[te_idxs]
    y_te = y_te[te_idxs]

    nca_dim = 256
    nca_subset = "4000-0"
    nca = cache_pickle(
        f"output/nca-{nca_dim}-{nca_subset}",
        train_nca,
        n_components=nca_dim,
        subset=nca_subset,
    )

    X_te_1 = nca.transform(X_te)

    model = KNeighborsClassifier(n_neighbors=K, weights="distance")
    model.fit(nca.transform(X_tr), y_tr)
    y_pred = model.predict_proba(X_te_1)[:, 1]
    tr_dists, tr_idxs = model.kneighbors(X_te_1)

    results = [
        {
            "te-idx": te_idxs[i],
            "label-true": y_te[i],
            "fake-proba": y_pred[i],
            "tr-idxs": tr_idxs[i],
            "tr-dists": tr_dists[i],
        }
        for i in range(len(X_te))
    ]

    st.markdown("EER: {:.1f}%".format(100 * compute_eer(y_te, y_pred)))

    tr_dataset = DATASETS["asvspoof19"]("train")
    te_dataset = DATASETS[te_dataset_name]("eval")

    def show_audio(elem, dataset, i, extra_str=""):
        elem.markdown(
            "name: `{}` · label: {} {}".format(
                dataset.get_file_name(i),
                dataset.get_label(i),
                extra_str,
            )
        )
        elem.audio(str(dataset.get_path_audio(i)))

    if filter_label != "all":
        filter_label_idx = 0 if filter_label == "real" else 1
        results = [r for r in results if r["label-true"] == filter_label_idx]

    if sort_by_uncertainty:
        results = sorted(results, key=lambda x: abs(x["fake-proba"] - 0.5))

    for result in results[:num_te_samples]:
        col1, col2 = st.columns(2)

        col1.markdown("### Test sample")
        show_audio(
            col1,
            te_dataset,
            result["te-idx"],
            extra_str=" · fake proba: {:.2f}".format(result["fake-proba"]),
        )

        col2.markdown(f"### Closest {K} training samples")

        for j, tr_idx in enumerate(result["tr-idxs"]):
            show_audio(
                col2,
                tr_dataset,
                tr_idx,
                extra_str=" · dist: {:.2f}".format(result["tr-dists"][j]),
            )

        st.markdown("---")


if __name__ == "__main__":
    # compare_classifiers()
    show_results()
