import pdb

import pandas as pd
import seaborn as sns
import streamlit as st

from matplotlib import pyplot as plt
from adjustText import adjust_text

sns.set_theme(style="whitegrid", context="talk", font="Arial", font_scale=1.0)


def add_texts(df, ax, metric):
    def shorten1(feature):
        if feature == "large":
            return "lg"
        elif feature == "base":
            return "b"
        else:
            return feature

    def shorten(feature):
        parts = feature.split("-")
        return "-".join(shorten1(part) for part in parts)

    x = "Time (s)"
    texts = [
        ax.text(x, y, shorten(feature), ha="left", va="center", size=16)
        for x, y, feature in zip(df[x], df[metric], df["Variant"])
    ]
    # print(texts)
    adjust_text(
        texts,
        force_explode=(0.5, 0.5),
        ax=ax,
        time_lim=10,
        arrowprops=dict(arrowstyle="-", color="b", alpha=0.5),
    )


df = pd.read_csv("output/feature-type-benchmark.csv")
df["Variant"] = df["Variant"].fillna("")
st.write(df)

fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
sns.scatterplot(
    data=df,
    x="Time (s)",
    y="EER (%)",
    size="Mem. (MB)",
    hue="Model",
    style="Model",
    legend=False,
    ax=axs[0],
)

# axs[0].set_xscale("log")
axs[0].set_xlim(0, 0.3)
axs[1].set_xlim(0, 0.3)
axs[0].set_ylim(0, 50)
axs[1].set_ylim(0, 30)

sns.scatterplot(
    data=df,
    x="Time (s)",
    y="ECE (%)",
    size="Mem. (MB)",
    hue="Model",
    style="Model",
    legend=True,
    ax=axs[1],
)
add_texts(df, axs[0], "EER (%)")
add_texts(df, axs[1], "ECE (%)")

sns.move_legend(axs[1], "upper left", bbox_to_anchor=(1.0, 1.0), frameon=False)
fig.tight_layout()

st.pyplot(fig)

fig.savefig("output/icassp/feature-type-benchmark-with-rawnet.pdf")
