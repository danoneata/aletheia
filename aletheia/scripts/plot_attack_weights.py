import pdb
import pickle
import numpy as np

from matplotlib import pyplot as plt
from scipy.stats import describe

import seaborn as sns
import streamlit as st

sns.set_context("talk")

NUM_ATTACKS = 8
# NUM_ATTACKS = 16


def load_weights(i):
    PATH = "/home/opascu/AI4TRUST/attack_models/model_A{:02d}.pkl"
    with open(PATH.format(i), "rb") as f:
        model = pickle.load(f)
        return model.coef_


weights = [load_weights(i) for i in range(1, NUM_ATTACKS + 1)]
weights = np.vstack(weights)

n_rows, n_cols = weights.shape
ratio = n_cols // n_rows
S = 3
fig, ax = plt.subplots(1, 1, figsize=(ratio * S, S))
sns.heatmap(
    weights,
    cmap="coolwarm",
    center=0,
    square=True,
    linewidth=.5,
    # annot=True,
    # fmt=".0f",
    ax=ax,
)
# attacks = list(range(1, NUM_ATTACKS + 1))
attacks_str = [f"{i + 0.5:.0f}" for i in ax.get_yticks()]
ax.set(xlabel="features", ylabel="attacks", yticklabels=attacks_str)

st.set_page_config(layout="wide")
st.pyplot(fig)


for i in range(NUM_ATTACKS):
    st.markdown(f"### A{i + 1}")
    st.write(describe(weights[i]))
    idxs = np.argsort(np.abs(weights[i]))[-5:][::-1]
    top_5_largetst_mag = weights[i, idxs]
    st.write(idxs)
    st.write(top_5_largetst_mag)