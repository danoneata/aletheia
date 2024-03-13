from abc import ABC, abstractmethod
from typing import List, Literal, Optional
from pathlib import Path

from dataclasses import dataclass
from functools import cached_property, partial

import soundfile as sf

from datasets import load_dataset, concatenate_datasets, Dataset as HuggingFaceDataset

from aletheia.utils import read_file


Label = Literal["real", "fake"]
Split = Literal["train", "dev", "eval"]

SAMPLING_RATE = 16_000


class MyDataset(ABC):
    def __init__(self, split):
        self.split = split

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def get_file_name(self, i: int) -> str:
        pass

    @abstractmethod
    def get_label(self, i: int) -> Label:
        pass

    @abstractmethod
    def load_audio(self, i: int):
        pass

    def get_system(self, i: int) -> Optional[str]:
        return None


@dataclass
class Datum:
    filename: str
    label: Label
    system: Optional[str]
    speaker: Optional[str]


class ASVspoof2019(MyDataset):
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

    # def __getitem__(self, i: int) -> Datum:
    #     if i >= len(self):
    #         raise IndexError
    #     return self.data[i]

    def __len__(self) -> int:
        return len(self.data)

    def get_file_name(self, i: int) -> str:
        return self.data[i].filename

    def get_label(self, i: int) -> Label:
        return self.data[i].label

    def get_path_audio(self, i: int) -> Path:
        return self.get_folder_data() / (self.get_file_name(i) + "." + self.ext)

    def get_system(self, i: int) -> Optional[str]:
        return self.data[i].system

    def load_audio(self, i: int):
        audio_path = self.get_path_audio(i)
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


class TimitTTS(MyDataset):
    def __init__(self, split, subset):
        self.split = split
        self.subset = subset
        assert subset in {"clean", "dtw-aug"}
        self.data = self.load_data()

    def load_data(self) -> List[Datum]:
        pass

    def __len__(self) -> int:
        pass

    def get_file_name(self, i: int) -> str:
        pass

    def get_label(self, i: int) -> Label:
        pass

    def load_audio(self, i: int):
        pass

    def get_system(self, i: int) -> Optional[str]:
        return None


class HuggingFaceRealDataset(MyDataset):
    @property
    @abstractmethod
    def dataset(self) -> HuggingFaceDataset:
        pass

    def __len__(self) -> int:
        return len(self.dataset)

    def get_file_name(self, i: int) -> str:
        return self.dataset[i]["id"]

    def get_label(self, i: int) -> Label:
        return "real"

    def load_audio(self, i: int):
        audio_dict = self.dataset[i]["audio"]
        assert audio_dict["sampling_rate"] == SAMPLING_RATE
        return audio_dict["array"]


# class ESBDataset(HuggingFaceRealDataset):
#     def __init__(self, name, split: Split):
#         SPLIT_MAP = {
#             "train": "train",
#             "dev": "validation",
#             "eval": "test",
#         }
#         split_esb = SPLIT_MAP[split]
#         if name == "librispeech" and split in {"dev", "eval"}:
#             split_esb = split_esb + ".other"
#         self.data = load_dataset("esb/datasets", name, split=split_esb, use_auth_token=True)
#         import pdb; pdb.set_trace()


class ESBDiagnosticDataset(HuggingFaceRealDataset):
    def __init__(self, name, split: Split):
        self.name = name
        self.split = split
        assert split == "dev"

    @cached_property
    def dataset(self):
        dataset = load_dataset("esb/diagnostic-dataset", self.name, use_auth_token=True)
        if isinstance(dataset, dict):
            datasets = list(dataset.values())
            dataset = concatenate_datasets(datasets)
        return dataset


class ESBDataset(MyDataset):
    def __init__(self, name, split: Split):
        self.name = name
        self.split = split

    @cached_property
    def dataset(self):
        dataset = load_dataset("esb/datasets", self.name, use_auth_token=True, split=self.split)
        # if isinstance(dataset, dict):
        #     datasets = list(dataset.values())
        #     dataset = concatenate_datasets(datasets)
        return dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def get_file_name(self, i: int) -> str:
        return self.dataset[i]["id"]

    def get_label(self, i: int) -> Label:
        return "real"

    def load_audio(self, i: int):
        audio_dict = self.dataset[i]["audio"]
        assert audio_dict["sampling_rate"] == SAMPLING_RATE
        return audio_dict["array"]


DATASETS = {
    "asvspoof19": ASVspoof2019,
    "in-the-wild": InTheWild,
    "common-voice-full": partial(ESBDataset, name="common_voice"),
}  # type: dict[str, type[MyDataset]]

ESB_DATASETS = [
    "librispeech",
    "common_voice",
    "voxpopuli",
    "tedlium",
    "gigaspeech",
    "spgispeech",
    "earnings22",
    "ami",
]

for name in ESB_DATASETS:
    my_name = name.replace("_", "-")
    DATASETS[my_name] = partial(ESBDiagnosticDataset, name=name)
