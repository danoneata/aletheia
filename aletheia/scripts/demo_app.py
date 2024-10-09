import math
import time
import pdb

from pathlib import Path

import librosa
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


threshold = 0.7

fake_score = 0.0
df_scores = pd.DataFrame({"start": [], "stop": [], "fake score": []})
# df_scores = pd.DataFrame({"start": np.arange(0, 10, 1), "stop": np.arange(1, 11, 1), "fake score": np.random.rand(10)})
df_points = pd.DataFrame({"time": []})

ROOT_IN_THE_WILD = "/mnt/student-share/data/in-the-wild/LA/ASVspoof2019_LA_eval/wav"
ROOT_IN_THE_WILD = Path(ROOT_IN_THE_WILD)

path_audio = ROOT_IN_THE_WILD / "{}.wav"
path_audio = str(path_audio)

sr = 8_000
audios = [librosa.load(path_audio.format(i), sr=sr)[0] for i in range(3)]
audio = np.concatenate(audios)
num_time = math.ceil(len(audio) / sr)
step_time = 1.0

st.markdown("## Streaming audio deepfake detection")
st.markdown(
    """
    This is a demo of an audio deepfake detection system that operates on long audio clips.
    In order to offer feedback quickly to the end user, we show partial results as the data is processed.
    This is done by streaming the results of the backend deepfake detector to the frontend.
    """
)

button = st.button("Upload video")

st.markdown(
    """
    ### Fakeness score: Full audio

    We first show the score that the **any** part of the audio is fake.
    This score is computed as the maximum of all scores at each time step.
    """
)
fake_score_container = st.empty()

fake_score_container.markdown("- Score: `{:.3f}`".format(fake_score))

# bars = (
#     alt.Chart(df_scores)
#     .mark_bar()
#     .encode(
#         x=alt.X("time", title="Time (s)", scale=alt.Scale(domain=(0, num_time))),
#         y=alt.Y("fake score", title="Fakeness score", scale=alt.Scale(domain=(-1, 1))),
#     )
#     .properties(width=700, height=250)
# )

# highlight = (
#     bars.mark_bar(color="#e45755")
#     .encode(y2=alt.Y2(datum=threshold))
#     .transform_filter(alt.datum["fake score"] > threshold)
# )

# rule = alt.Chart().mark_rule().encode(y=alt.Y(datum=threshold))

plot_audio = (
    alt.Chart(pd.DataFrame({"time": np.arange(len(audio)) / sr, "audio": audio}))
    .mark_line(color="#DBE2EF")
    .encode(
        x=alt.X("time", title="Time (s)"),
        y=alt.Y("audio", title="", axis=alt.Axis(labels=False)),
    )
    .properties(width=700, height=200)
)

plot_scores = (
    alt.Chart(df_scores)
    .mark_rect()
    .encode(
        x=alt.X("start", title="Time (s)", scale=alt.Scale(domain=(0, num_time))),
        x2="stop",
        color=alt.Color(
            "fake score:Q",
            scale=alt.Scale(scheme="redyellowgreen", domain=[0, 1]),
            legend=None,
        ),
    )
    # .configure_legend(orient="bottom")
    .properties(width=700, height=125)
)

st.markdown(
    """
    ### Fakeness scores per time step (Explanations)

    In order to understand which part of the audio file triggered the fake detection, we also show scores at each time step.
    These values are updated as the audio is analyzed.
    """
)
# We also show a horizontal line that indicates a threshold over which we consider the segment (and the audio) to be fake.
# chart_scores = st.altair_chart(plot_audio & plot_scores)
chart_audio = st.altair_chart(plot_audio)
chart_scores = st.altair_chart(plot_scores)

st.markdown(
    """
    ### Splicing points (Anomalies)

    Additionaly, we also show the splicing points detected in the audio file.
    These may suggest that the audio file has been tampered with.
    But note that there may be also false positives (for example, video cuts).
    """
)
plot_points = (
    alt.Chart(df_points)
    .mark_rule(color="#e45755")
    .encode(
        x=alt.X("time", title="Time (s)", scale=alt.Scale(domain=(0, num_time))),
    )
    .properties(width=700, height=125)
)
chart_points = st.altair_chart(plot_points)

# my_table = st.table(df_scores)

if button:
    for t in range(num_time):
        time.sleep(1.0)

        fake_score_curr = np.random.rand()
        fake_score = max(fake_score, fake_score_curr)
        fake_score_container.markdown("- Score: `{:.3f}`".format(fake_score))

        df_scores_curr = pd.DataFrame(
            {
                "start": [float(t)],
                "stop": [float(t) + step_time],
                "fake score": [fake_score_curr],
            }
        )
        chart_scores.add_rows(df_scores_curr)
        # my_table.add_rows(df_scores_curr)

        if np.random.rand() > 0.8:
            chart_points.add_rows(pd.DataFrame({"time": [float(t)]}))
