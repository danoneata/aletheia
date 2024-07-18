import pdb

import pandas as pd
import seaborn as sns
import streamlit as st

from matplotlib import pyplot as plt
from adjustText import adjust_text

from aletheia.scripts.extract_features import FEATURE_EXTRACTORS
from aletheia.utils import cache_json
from aletheia.scripts.extract_features_benchmark import main as benchmark


sns.set_theme(style="whitegrid", context="talk", font="Arial", font_scale=1.0)


def add_texts(df, ax, metric):
    # selected_features = [
    #     "wav2vec2-xls-r-300m",
    #     "wav2vec2-xls-r-1b",
    #     "wav2vec2-xls-r-2b",
    # ]

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


results = [
    cache_json(
        f"output/features-benchmark-{f}.json",
        benchmark,
        ["-f", f],
        standalone_mode=False,
    )
    for f in FEATURE_EXTRACTORS.keys()
]

# results = sum(results, [])
# df1 = pd.DataFrame(results)
# df1 = df1.groupby("feature-type").mean()
# 
# df2 = pd.read_csv("output/feature-type-performance.csv")
# df2 = df2.set_index("feature-type")
# 
# df = df2.join(df1)
# df = df.reset_index()
# 
# df3 = [
#     {
#         "feature-type": "RawNet2",
#         "time": 0.03,
#         "cuda-memory-reserved": 1.3,
#         "eer": 26.78,
#         "ece": 24.78 ,
#     }
# ]
# 
# df3 = pd.DataFrame(df3)
# df = pd.concat([df, df3])
# 
# df["Model"] = df["feature-type"].str.split("-").str[0]
# df["Variant"] = df["feature-type"].str.split("-").str[1:].str.join("-")
# df = df.rename(
#     columns={
#         "feature-type": "Feature type",
#         "time": "Time (s)",
#         "cuda-memory-reserved": "Mem. (MB)",
#         "eer": "EER (%)",
#         "ece": "ECE (%)",
#     }
# )
# 
# st.write(df)
# df.to_csv("output/feature-type-benchmark.csv", index=False)
df = pd.read_csv("output/feature-type-benchmark.csv")

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
