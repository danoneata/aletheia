import time
import pdb

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


num_time = 30
threshold = 0.7

fake_score = 0.0
df_scores = pd.DataFrame({"time": [], "fake score": []})
df_points = pd.DataFrame({"time": []})

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
    ### Video fakeness score

    We first show the score that the **any** part of the videoclip is fake.
    This score is computed as the maximum of all scores at each time step.
    """
)
fake_score_container = st.empty()

fake_score_container.markdown("- Score: `{:.3f}`".format(fake_score))

bars = (
    alt.Chart(df_scores)
    .mark_bar()
    .encode(
        x=alt.X("time", title="Time (s)", scale=alt.Scale(domain=(0, num_time))),
        y=alt.Y("fake score", title="Fakeness score", scale=alt.Scale(domain=(0, 1))),
    )
    .properties(width=700, height=250)
)

highlight = (
    bars.mark_bar(color="#e45755")
    .encode(y2=alt.Y2(datum=threshold))
    .transform_filter(alt.datum["fake score"] > threshold)
)

rule = alt.Chart().mark_rule().encode(y=alt.Y(datum=threshold))

st.markdown(
    """
    ### Temporal fakeness scores (Explanations)

    In order to understand which part of the audio file triggered the fake detection, we also show scores at each time step.
    These values are also updated as the video clip is analyzed.
    We also show a horizontal line that indicates a threshold over which we consider the segment (and the video) to be fake.
    """
)
chart_scores = st.altair_chart(bars + highlight + rule)

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
    .properties(width=700, height=100)
)
chart_points = st.altair_chart(plot_points)

if button:
    for t in range(num_time):
        time.sleep(0.3)

        fake_score_curr = np.random.rand()
        fake_score = max(fake_score, fake_score_curr)
        fake_score_container.markdown("- Score: `{:.3f}`".format(fake_score))

        df_scores_curr = pd.DataFrame({"time": [float(t)], "fake score": [fake_score_curr]})
        chart_scores.add_rows(df_scores_curr)

        if np.random.rand() > 0.8:
            chart_points.add_rows(pd.DataFrame({"time": [float(t)]}))
