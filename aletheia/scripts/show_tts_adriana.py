import json

from pathlib import Path

from pydub import AudioSegment
from toolz import first

import streamlit as st

BASE_PATH = Path("/mnt/student-share/data/lav-df")


@st.cache_data
def load_metadata():
    json_path = BASE_PATH / "metadata.json"
    with open(json_path, "r") as f:
        return json.load(f)


metadata = load_metadata()


with st.sidebar:
    split = st.selectbox("Split", ["test"])


def find_entry(name):
    for entry in metadata:
        if entry["file"] == f"{split}/{name}.mp4":
            return entry
    assert False


tts_path = BASE_PATH / "ADRIANA_generated_samples_unitspeech" / split
for path_ours in tts_path.iterdir():
    name = path_ours.stem

    entry_fake = find_entry(name)
    entry_real = find_entry(Path(entry_fake["original"]).stem)

    path_original = BASE_PATH / entry_fake["original"]
    path_theirs = BASE_PATH / split / (name + ".mp4")

    s, _ = entry_fake["fake_periods"][0]
    _, e = entry_fake["fake_periods"][-1]

    audio_ours = AudioSegment.from_file(path_ours)
    fake_segment_ours = audio_ours[s * 1000 : e * 1000]

    audio_theirs = AudioSegment.from_file(path_theirs)
    fake_segment_theirs = audio_theirs[s * 1000 : e * 1000]

    fake_word_idx = [
        i for i, (_, α, _) in enumerate(entry_fake["timestamps"]) if α == s
    ][0]
    words = [w for w, _, _ in entry_fake["timestamps"]]
    fake_words = words[fake_word_idx : fake_word_idx + len(entry_fake["fake_periods"])]
    fake_word = " ".join(fake_words)

    st.markdown("## `{}`".format(name))
    col0, col1, col2 = st.columns(3)
    col0.markdown("real")
    col0.audio(str(path_original))
    col0.markdown("{}".format(entry_real["transcript"]))

    col1.markdown("fake (theirs)")
    col1.audio(str(path_theirs))
    col1.markdown("{}".format(entry_fake["transcript"]))
    col1.markdown("---")
    col1.markdown("fake segment: {}".format(fake_word))
    col1.audio(fake_segment_theirs.export().read())

    col2.markdown("fake (ours)")
    col2.audio(str(path_ours))
    col2.markdown("{}".format(entry_fake["transcript"]))
    col2.markdown("---")
    col2.markdown("fake segment: {}".format(fake_word))
    col2.audio(fake_segment_ours.export().read())

    # st.markdown("{}".format(entry_fake["fake_periods"]))
    st.markdown("---")
