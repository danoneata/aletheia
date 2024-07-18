import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from matplotlib import pyplot as plt


sns.set_theme(style="whitegrid", context="talk", font="Arial", font_scale=1.0)

results = [
    {
        "kernel-sizes": [23],
        "eer": 11.6,
    },
    {
        "kernel-sizes": [21],
        "eer": 11.5,
    },
    {
        "kernel-sizes": [19],
        "eer": 11.5,
    },
    {
        "kernel-sizes": [17],
        "eer": 11.8,
    },
    {
        "kernel-sizes": [15],
        "eer": 11.6,
    },
    {
        "kernel-sizes": [13],
        "eer": 11.9,
    },
    {
        "kernel-sizes": [11],
        "eer": 12.0,
    },
    {
        "kernel-sizes": [9],
        "eer": 12.2,
    },
    {
        "kernel-sizes": [7],
        "eer": 12.5,
    },
    {
        "kernel-sizes": [5],
        "eer": 13.1,
    },
    {
        "kernel-sizes": [3],
        "eer": 13.8,
    },
    {
        "kernel-sizes": [1],
        "eer": 15.6,
    },
    {
        "kernel-sizes": [10, 5],
        "eer": 9.4,
    },
    {
        "kernel-sizes": [10, 5, 3],
        "eer": 8.6,
    },
    {
        "kernel-sizes": [20, 10, 5, 3],
        "eer": 8.1,
    },
]

for result in results:
    ks = result["kernel-sizes"]
    result["receptive-field"] = sum([k - 1 for k in ks]) - 1
    result["Num. layers"] = "one" if len(ks) == 1 else "multiple"
    result["name"] = "/".join(map(str, ks))

df = pd.DataFrame(results)

fig, ax = plt.subplots()
sns.lineplot(
    df,
    x="receptive-field",
    y="eer",
    hue="Num. layers",
    style="Num. layers",
    markers=True,
    ax=ax,
)
ax.set(xlabel="Receptive field size", ylabel="EER ↓ (%)")
for result in results:
    if result["Num. layers"] == "multiple":
        ax.text(
            result["receptive-field"],
            result["eer"] + 0.1,
            result["name"],
            ha="center",
            va="bottom",
            size=10,
        )
st.pyplot(fig)
