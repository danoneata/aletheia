import random

from typing import List
from pathlib import Path

import streamlit as st

from aletheia.data import Datum
from aletheia.utils import read_file


BASE_PATH = Path("/mnt/student-share/ASVSpoof2024")


def load_data(split) -> List[Datum]:
    assert split in "train dev".split()

    LABEL_MAP = {
        "bonafide": "real",
        "spoof": "fake",
    }

    def parse_line(line: str) -> Datum:
        speaker, filename, gender, _, system, label = line.split()
        return Datum(filename, LABEL_MAP[label], system, speaker)

    path = BASE_PATH / "metadata" / f"ASVspoof5.{split}.metadata.txt"
    path = str(path)
    return read_file(path, parse_line)


subsets = {
    "train": ["A{:02d}".format(i) for i in range(1, 9)] + ["bonafide"],
    "dev": ["A{:02d}".format(i) for i in range(9, 17)] + ["bonafide"],
}


with st.sidebar:
    split = st.selectbox("Split", ["train", "dev"])
    subset = st.selectbox("Attack", subsets[split])


data = load_data(split)
random.shuffle(data)
data = [datum for datum in data if datum.system == subset]
data = data[:15]

for datum in data:
    s = "T" if split == "train" else "D"
    path = BASE_PATH / f"flac_{s}" / (datum.filename + ".flac")
    st.markdown("{} · {}".format(datum.filename, datum.system))
    st.audio(str(path))
