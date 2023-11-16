import pdb
import json

from pathlib import Path

from pydub import AudioSegment
from toolz import first

import streamlit as st


st.set_page_config(layout="wide")

BASE_PATH = Path("/mnt/student-share/data/lav-df")


@st.cache_data
def load_metadata():
    json_path = BASE_PATH / "metadata.json"
    with open(json_path, "r") as f:
        return json.load(f)


metadata = load_metadata()
split = "test"


def find_entry(name):
    for entry in metadata:
        if entry["file"] == f"{split}/{name}.mp4":
            return entry
    assert False


folder_imgs = BASE_PATH / "PLOTS_lav-df"
fmtbool = lambda b: "✓" if b else "✗"

for path_img in sorted(folder_imgs.iterdir()):
    name = path_img.stem

    entry_fake = find_entry(name)

    if entry_fake["n_fakes"] == 0:
        continue

    name_real = Path(entry_fake["original"]).stem
    entry_real = find_entry(name_real)

    path_video_fake = BASE_PATH / entry_fake["file"]
    path_video_real = BASE_PATH / entry_fake["original"]

    s, _ = entry_fake["fake_periods"][0]
    _, e = entry_fake["fake_periods"][-1]

    audio_fake = AudioSegment.from_file(path_video_fake)
    segment_fake = audio_fake[s * 1000 : e * 1000]

    fake_word_idx = [
        i for i, (_, α, _) in enumerate(entry_fake["timestamps"]) if α == s
    ][0]
    words = [w for w, _, _ in entry_fake["timestamps"]]
    fake_words = words[fake_word_idx : fake_word_idx + len(entry_fake["fake_periods"])]
    fake_word = " ".join(fake_words)

    is_fake_video = fmtbool(entry_fake["modify_video"])
    is_fake_audio = fmtbool(entry_fake["modify_audio"])

    st.markdown("## fake id: `{}` → real id: `{}` ◇ fake video: {} · fake audio: {}".format(name, name_real, is_fake_video, is_fake_audio))
    col0, col1, col2 = st.columns(3)
    col0.markdown("real")
    col0.audio(str(path_video_real))
    col0.markdown("{}".format(entry_real["transcript"]))

    col1.markdown("fake")
    col1.audio(str(path_video_fake))
    col1.markdown("{}".format(entry_fake["transcript"]))
    col1.markdown("---")
    col1.markdown("fake segment: {}".format(fake_word))
    col1.audio(segment_fake.export().read())

    col2.markdown("fake · audio plot (by Octav)")
    col2.image(str(path_img))

    # st.markdown("{}".format(entry_fake["fake_periods"]))
    st.markdown("---")
