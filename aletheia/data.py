from typing import List, Literal
from pathlib import Path

from dataclasses import dataclass

import soundfile as sf

from aletheia.utils import read_file


Label = Literal["real", "fake"]
Split = Literal["train", "dev", "eval"]

SAMPLING_RATE = 16_000

@dataclass
class Datum:
    filename: str
    label: Label
    system: str
    speaker: str


class ASVspoof2019:
    def __init__(self, split: Split):
        self.base_path = Path("/mnt/student-share/ASVspoof2019/LA")
        self.split = split
        self.ext = "flac"
        self.data = self.load_data()

    def get_folder_data(self) -> Path:
        return self.base_path / ("ASVspoof2019_LA_" + self.split) / self.ext

    def load_data(self) -> List[Datum]:
        LABEL_MAP = {
            "bonafide": "real",
            "spoof": "fake",
        }

        def parse_line(line: str) -> Datum:
            speaker, filename, _, system, label = line.split()
            return Datum(filename, LABEL_MAP[label], system, speaker)  # type: ignore[arg-type]

        filename = "ASVspoof2019.LA.cm.{}.{}.txt".format(
            self.split,
            "trn" if self.split == "train" else "trl",
        )
        path = self.base_path / "ASVspoof2019_LA_cm_protocols" / filename
        path = str(path)
        return read_file(path, parse_line)

    def __getitem__(self, i: int) -> Datum:
        if i >= len(self):
            raise IndexError
        return self.data[i]

    def __len__(self) -> int:
        return len(self.data)

    def get_path_audio(self, datum: Datum) -> Path:
        return self.get_folder_data() / (datum.filename + "." + self.ext)

    def get_label(self, datum: Datum) -> Label:
        return datum.label

    def load_audio(self, datum: Datum):
        audio_path = self.get_path_audio(datum)
        audio, sr = sf.read(audio_path)
        assert sr == SAMPLING_RATE
        return audio


class InTheWild(ASVspoof2019):
    def __init__(self, split: Split):
        assert split == "eval"
        self.base_path = Path("/mnt/student-share/data/in-the-wild/LA")
        self.split = split
        self.ext = "wav"
        self.data = self.load_data()

    def get_path_audio(self, datum: Datum) -> Path:
        return self.get_folder_data() / (datum.filename + "." + self.ext)

    def get_label(self, datum: Datum) -> Label:
        return datum.label


DATASETS = {
    "asvspoof19": ASVspoof2019,
    "in-the-wild": InTheWild,
}