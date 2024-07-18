import pdb

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from sklearn.metrics import roc_auc_score


def load_data():
    def parse_array(s):
        xs = s.strip("[]").split()
        return np.array([float(x) for x in xs])

    DATASET_NAME = "in-the-wild"
    CONVERTERS = {
        "reliab": parse_array,
        "score": parse_array,
    }

    path = f"output/salvi-{DATASET_NAME}.csv"
    df = pd.read_csv(path, index_col=0, converters=CONVERTERS)

    df["reliab"] = 1 - df["reliab"]
    df["score"] = - df["score"]

    df["reliability"] = df["reliab"].map(lambda xs: xs.max())
    df = df.rename(columns={"label": "true"})
    return df


def evaluate_reliability(df, subset_func=None, τ=0.5):
    if subset_func:
        df = subset_func(df)

    idxs = df["reliability"] >= τ

    pred_binary = df[idxs]["pred-binary"]
    pred = df[idxs]["pred"]
    true = df[idxs]["true"]
    accuracy = np.mean(true == pred_binary)

    num_kept = sum(idxs)
    num_samples = len(df)
    frac_kept = num_kept / num_samples

    if num_kept == 0 or true.nunique() == 1:
        auc_roc = np.nan
    else:
        auc_roc = roc_auc_score(true, pred)
        # print(roc_auc_score(true, np.random.rand(len(true))))
        # pdb.set_trace()

    return {
        "accuracy": 100 * accuracy,
        "frac-kept": 100 * frac_kept,
        "auc-roc": 100 * auc_roc,
        "num-kept": num_kept,
        "num-samples": num_samples,
        "num-discarded": num_samples - num_kept,
    }


def evaluate1(df, subset_func, τ, verbose=False):
    def compute_score(row, τ):
        score = row["score"]
        reliab = row["reliab"]
        return np.dot(score, reliab) / np.sum(reliab)
        # idxs = reliab >= τ
        # if np.sum(idxs) == 0.0:
        #     return np.nan
        # else:
        #     ss = score[idxs]
        #     rs = reliab[idxs]
        #     return np.dot(ss, rs) / np.sum(rs)

    THRESH_SCORE = 0.0
    df["pred"] = df.apply(lambda row: compute_score(row, τ), axis=1)
    df["pred-binary"] = df["pred"] > THRESH_SCORE

    if verbose:
        st.dataframe(df)

    return evaluate_reliability(df, subset_func, τ)


def get_reliability_metrics(df, subset_func=None):
    δ = 0.01
    return [
        {"τ": τ, **evaluate1(df, subset_func, τ)}
        for τ in np.arange(0.0, 1.0 + δ, δ)
    ]


def main():
    df = load_data()
    select_only_fakes = lambda df: df[df["true"] == 1]

    st.markdown("### Evaluation on all samples")
    st.write(evaluate1(df, None, 0.5))

    st.markdown("### Evaluation only on fakes")
    st.write(evaluate1(df, select_only_fakes, 0.5))

    st.markdown("### Evaluation at multiple thresholds")
    results = pd.DataFrame(get_reliability_metrics(df))
    fig, ax = plt.subplots()
    sns.lineplot(data=results, x="frac-kept", y="accuracy", ax=ax)
    st.pyplot(fig)


if __name__ == "__main__":
    main()
