import pdb
import random

from itertools import groupby
from pathlib import Path
from typing import List, Literal, Optional

from matplotlib import pyplot as plt
from toolz import concat, first, partition_all

import numpy as np
import soundfile as sf
import streamlit as st
import seaborn as sns

from aletheia.data import SAMPLING_RATE, Split, Label, Datum, ASVspoof2019
from aletheia.utils import read_file


def to_relative(segments):
    """Transforms a list of absolute times to the relative times between them."""
    times = [0] + list(concat(segments))
    carrier_deltas = [
        t2 - t1
        for t1, t2 in zip(times[:-1], times[1:])
    ]
    return list(partition_all(2, carrier_deltas))


def to_absolute(segments):
    """Transforms a list of relatives durations to absolute times."""
    t = 0
    times = []
    for Δ in concat(segments):
        t += Δ
        times.append(t)
    return list(partition_all(2, times))


def realign(k, group):
    data = [
        (carrier, int(carrier_s), int(carrier_e), sub, int(sub_s), int(sub_e))
        for _, carrier, carrier_s, carrier_e, sub, sub_s, sub_e in group
    ]
    data = sorted(data, key=lambda x: x[1])

    carrier_segments = [(s, e) for _, s, e, *_ in data]
    sub_deltas = [e - s for *_, s, e in data]

    deltas = [
        (Δ0, Δ)
        for (Δ0, _), Δ in zip(to_relative(carrier_segments), sub_deltas)
    ]

    segments = to_absolute(deltas)

    # print("carrier t:", carrier_segments)
    # print("carrier Δ:", to_relative(carrier_segments))
    # print("sub Δ:    ", sub_deltas)
    # print("realign t:", segments)
    # pdb.set_trace()

    return [
        {
            "start-frame": s,
            "end-frame": e,
            "carrier": carrier,
            "carrier-start-frame": carrier_s,
            "carrier-end-frame": carrier_e,
            "sub": sub,
            "sub-start-frame": sub_s,
            "sub-end-frame": sub_e,
        }
        for (carrier, carrier_s, carrier_e, sub, sub_s, sub_e), (s, e) in zip(data, segments)
    ]


def load_concatenate_log(split):
    path = f"output/partial-spoof-concatenate-log/{split}_concatenate.log"
    lines = read_file(path, lambda line: line.split())
    return {
        k: realign(k, group)
        for k, group in groupby(lines, key=first)
    }


class PartialSpoof:
    def __init__(self, split: Split):
        self.base_path = Path(
            "/mnt/student-share/data/ai4trust/partially_spoofed_dataset"
        )
        self.split = split
        self.ext = "wav"
        self.data = self.load_data()
        path = self.base_path / "segment_labels" / f"{split}_seglab_0.01.npy"
        self.segment_labels = np.load(path, allow_pickle=True).item()

    def get_folder_data(self) -> Path:
        return self.base_path / self.split / "con_wav"

    def load_data(self) -> List[Datum]:
        LABEL_MAP = {
            "bonafide": "real",
            "spoof": "fake",
        }

        def parse_line(line: str) -> Datum:
            speaker, filename, _, system, label = line.split()
            return Datum(filename, LABEL_MAP[label], system, speaker)  # type: ignore[arg-type]

        filename = f"PartialSpoof.LA.cm.{self.split}.trl.txt"
        path = self.base_path / "protocols" / "PartialSpoof_LA_cm_protocols" / filename
        path = str(path)
        return read_file(path, parse_line)

    def __len__(self) -> int:
        return len(self.data)

    def get_file_name(self, i: int) -> str:
        return self.data[i].filename

    def get_label(self, i: int) -> Label:
        return self.data[i].label

    def get_label_segments(self, i: int):
        pass

    def get_path_audio(self, i: int) -> Path:
        return self.get_folder_data() / (self.get_file_name(i) + "." + self.ext)

    def get_system(self, i: int) -> Optional[str]:
        return self.data[i].system

    def load_audio(self, i: int):
        audio_path = self.get_path_audio(i)
        audio, sr = sf.read(audio_path)
        assert sr == SAMPLING_RATE
        return audio


def main():
    random.seed(1337)
    split = "eval"
    dataset_src = ASVspoof2019(split)
    dataset = PartialSpoof(split)
    concatenate_log = load_concatenate_log(split)

    src_filename_to_index = {
        dataset_src.get_file_name(i): i for i in range(len(dataset))
    }

    filename_to_index = {
        dataset.get_file_name(i): i for i in range(len(dataset))
    }

    LABEL_MAP = {
        "fake": 0,
        "real": 1,
    }

    def plot1(i):
        def reverse_label(label):
            return "fake" if label == "real" else "real"

        def get_color(label):
            return "lightgreen" if label == "real" else "lightcoral"

        filename = dataset.get_file_name(i)
        segments = concatenate_log[filename]

        audio = dataset.load_audio(i)
        audio = audio / np.max(np.abs(audio))

        t = audio.shape[0] / SAMPLING_RATE

        # num_segments = len(segments)
        fig, axs = plt.subplots(nrows=3, figsize=(t * 5, 10), sharex=True)

        for ax in axs:
            ax.set_ylim(-1.1, 1.1)

        carrier = segments[0]["carrier"]
        carrier_audio = dataset_src.load_audio(src_filename_to_index[carrier])
        carrier_audio = carrier_audio / np.max(np.abs(audio))
        carrier_label = dataset_src.get_label(src_filename_to_index[carrier])
        sub_label = reverse_label(carrier_label)

        color = get_color(carrier_label)
        axs[0].axvspan(xmin=0, xmax=carrier_audio.shape[0], color=color)
        for segment in segments:
            axs[0].axvspan(
                xmin=segment["carrier-start-frame"],
                xmax=segment["carrier-end-frame"],
                color="white",
            )
        axs[0].plot(carrier_audio)
        axs[0].set_title("carrier: {} · label: {}".format(carrier, carrier_label))

        axs[2].axvspan(xmin=0, xmax=audio.shape[0], color=color)
        for segment in segments:
            j = src_filename_to_index[segment["sub"]]
            label = dataset_src.get_label(j)
            midpoint = (segment["start-frame"] + segment["end-frame"]) / 2
            axs[2].axvspan(
                xmin=segment["start-frame"],
                xmax=segment["end-frame"],
                color=get_color(label),
            )
            axs[2].text(
                midpoint,
                -1.0,
                segment["sub"],
                fontsize=8,
                ha="center",
                va="bottom",
                color="black",
            )

        # st.write(segments)

        axs[2].plot(audio)
        axs[2].set_xlabel("t")
        axs[2].set_title("generated: " + filename)

        y = dataset.segment_labels[filename].astype("float")
        x = np.arange(y.shape[0]) * 0.01 * SAMPLING_RATE
        axs[2].step(x, y, linewidth=2, color="k")

        groups = [
            list(group)
            for k, group in groupby(zip(x, y), key=lambda xy: xy[1])
            if k == LABEL_MAP[sub_label]
        ]

        δ = 0.1
        for i, segment in enumerate(segments):
            audio_seg = dataset_src.load_audio(src_filename_to_index[segment["sub"]])
            s = segment["sub-start-frame"]
            e = segment["sub-end-frame"]

            try:
                t, _ = first(groups[i])
            except:
                continue

            idxs = np.arange(t, t + e - s)
            axs[1].vlines(t, -1.1, 1.1, color="gray", linestyle="dashed")
            axs[2].vlines(t, -1.1, 1.1, color="gray", linestyle="dashed")
            axs[1].plot(idxs, audio_seg[s: e] + δ * i)

        # axs[1].step(x, y, linewidth=2, color="k")
        axs[1].set_title("substituions · label: " + sub_label)

        return fig

    # idxs = random.sample(range(len(dataset)), 32)
    idxs = list(range(len(dataset)))
    random.shuffle(idxs)
    # idxs = [filename_to_index["CON_T_0000001"]]

    count = 0

    for i in idxs:
        path = dataset.get_path_audio(i)
        filename = dataset.get_file_name(i)

        if not filename.startswith("CON_"):
            continue

        # carrier = concatenate_log[filename][0]["carrier"]
        # label_carrier = dataset_src.get_label(src_filename_to_index[carrier])

        # if label_carrier != "real":
        #     continue

        # st.write(dataset.data[i])
        # st.write(dataset.segment_labels[path.stem])
        fig = plot1(i)
        st.pyplot(fig)
        st.audio(str(path))
        st.markdown("---")

        if count > 32:
            break
            
        count += 1


if __name__ == "__main__":
    main()
