import pdb
import pickle

from collections import defaultdict
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns

from aletheia.metrics import compute_eer


sns.set_context("paper")
st.set_page_config(layout="wide")


def load_data(split):
    path = f"/home/opascu/AI4TRUST/aletheia/aletheia-icassp24/output/features/asvspoof24-{split}-all-features-opensmile-systembased-ege.h5.npy"
    with open(path, "rb") as f:
        return pickle.load(f)


class multimap(defaultdict):
    def __init__(self, pairs, symmetric=False):
        """Given (key, val) pairs, return {key: [val, ...], ...}.
        If `symmetric` is True, treat (key, val) as (key, val) plus (val, key)."""
        self.default_factory = list
        for key, val in pairs:
            self[key].append(val)
            if symmetric:
                self[val].append(key)


ATTACK_TO_FEATS = {
    # old selected features
    # "A01": "F11 F66 F19".split(),
    # "A02": "F21 F66 F11".split(),
    # "A03": "F11 F66 F19".split(),
    # "A04": "F21 F81 F11".split(),
    # "A05": "F21 F86 F11".split(),
    # "A06": "F21 F86 F11".split(),
    # "A07": "F11 F8 F1".split(),
    # "A08": "F20 F17 F66".split(),
    # "A09": "F86 F85 F44".split(),
    # "A10": "F86 F85 F44".split(),
    # "A11": "F58 F60 F59".split(),
    # "A12": "F83 F82 F84".split(),
    # "A13": "F38 F57 F67".split(),
    # "A14": "F51 F45 F57".split(),
    # "A15": "F78 F77 F76".split(),
    # "A16": "F39 F61 F46".split(),
    "A01": "F19 F66 F17".split(),
    "A02": "F66 F19 F17".split(),
    "A03": "F19 F66 F17".split(),
    "A04": "F21 F81 F19".split(),
    "A05": "F21 F19 F17".split(),
    "A06": "F19 F21 F17".split(),
    "A07": "F66 F1 F17".split(),
    "A08": "F20 F66 F17".split(),
    "A09": "F86 F85 F12".split(),
    "A10": "F85 F86 F12".split(),
    "A11": "F58 F60 F59".split(),
    "A12": "F82 F19 F83".split(),
    "A13": "F67 F39 F49".split(),
    "A14": "F57 F45 F51".split(),
    "A15": "F78 F76 F77".split(),
    "A16": "F66 F18 F20".split(),
}

FEATS_AND_ATTACKS = [(f, a) for a, feats in ATTACK_TO_FEATS.items() for f in feats]
FEAT_TO_ATTACKS = multimap(FEATS_AND_ATTACKS)

ATTACKS = sorted(ATTACK_TO_FEATS.keys())
FEATS = list(FEAT_TO_ATTACKS.keys())


num_attacks = len(ATTACKS)
data_orig = {s: load_data(s) for s in "train dev".split()}
data = {
    a: data2
    for data1 in data_orig.values()
    for a, data2 in data1.items()
    if a != "bonafide"
}
data["bonafide"] = np.vstack(
    [
        data_orig["train"]["bonafide"],
        data_orig["dev"]["bonafide"],
    ]
)


def compute_eers(data, attack, feature):
    ff = int(feature[1:])
    fake = data[attack][:, ff]
    real = data["bonafide"][:, ff]

    pred = np.concatenate([fake, real])
    true = np.concatenate([np.ones_like(fake), np.zeros_like(real)])

    eer1 = 100 * compute_eer(true, pred)
    eer2 = 100 * compute_eer(1 - true, pred)

    return eer1, eer2


FEATS_ALL = ["F{:02d}".format(i) for i in range(0, 88)]
eers = [
    {
        "attack": attack,
        "feature": feature,
        "eer": min(*compute_eers(data, attack, feature)),
    }
    for attack in ATTACKS[:num_attacks]
    for feature in FEATS_ALL
]
df = pd.DataFrame(eers).pivot("attack", "feature", "eer")
# df = df[df.mean(axis=0).sort_values().index[:25]]

S = 12
numf, numa = df.shape
ratio = numa / numf
fig, ax = plt.subplots(1, 1, figsize=(S, ratio * S))
sns.heatmap(
    df,
    cmap="rocket_r",
    # center=50,
    square=True,
    linewidth=0.01,
    annot=True,
    annot_kws={"fontsize": 4},
    fmt=".0f",
    cbar=False,
    ax=ax,
)

num_real = len(data["bonafide"])
num_fake = sum(len(data[a]) for a in ATTACKS[:num_attacks])

st.markdown("## EER heatmap for each attack–feature combination")
st.markdown(
    "EER is computed by using only the feature to score the samples and then taking the minimum of the EERs obtained by setting either (i) fakes to be 1 and reals to be 0 or (ii) vice versa. I've used the entire data to compute this EER ({} bonafide samples and {} spoofed samples).".format(
        num_real, num_fake
    )
)
st.pyplot(fig)
st.markdown("---")

st.markdown("## Feature distribution across attacks")
st.markdown("The selection of best features is that done by Octav.")

for r, f in enumerate(FEATS):

    S = 1.25
    fig, axs = plt.subplots(
        nrows=1,
        ncols=num_attacks,
        figsize=(0.75 * num_attacks * S, S),
        sharex=True,
        sharey=True,
        squeeze=False,
        dpi=1000,
    )

    for c, a in enumerate(ATTACKS[:num_attacks]):
        print(f, a)

        ff = int(f[1:])

        # df = {
        #     "fake": data[a][:, ff],
        #     "real": data["bonafide"][:, ff],
        #     # "real": data_orig["train"]["bonafide"][:, ff],
        #     # "real-tr": data_orig["train"]["bonafide"][:, ff],
        #     # "real-va": data_orig["dev"]["bonafide"][:, ff],
        # }
        fake = [
            {"label": "fake", "value": x}
            for x in data[a][:, ff]
        ]
        real = [
            {"label": "real", "value": x}
            for x in data["bonafide"][:, ff]
        ]
        df = pd.DataFrame(fake + real)

        eer1, eer2 = compute_eers(data, a, f)
        eer_min = min(eer1, eer2)
        eer_max = max(eer1, eer2)

        ax = axs[0, c]
        legend = True if c == num_attacks - 1 else False
        sns.kdeplot(df, x="value", hue="label", ax=ax, legend=legend, bw_adjust=0.1)
        ax.set_title("{}\n{:.1f} · {:.1f}".format(a, eer_min, eer_max))
        ax.set_xlabel("")

        if c == num_attacks - 1:
            sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

        if c == 0:
            label = f
            ax.set_ylabel(label)
            ax.set_yticklabels([])

    fig.tight_layout()
    # fig.savefig(f"output/asv5-feature-distribution-across-attacks-{f}.pdf")
    st.markdown("{} → best for attacks: {}".format(f, ", ".join(FEAT_TO_ATTACKS[f])))
    st.pyplot(fig)
