import click
import pdb
import random

import h5py
import torch
import numpy as np

from transformers import AutoFeatureExtractor, WavLMModel, Wav2Vec2Model
from tqdm import tqdm

from aletheia.data import DATASETS, SAMPLING_RATE
from aletheia.utils import cache_json


class HuggingFaceFeatureExtractor:
    def __init__(self, model_class, name):
        self.device = "cuda"
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(name)
        self.model = model_class.from_pretrained(name)
        self.model.eval()
        self.model.to(self.device)

    def __call__(self, audio, sr):
        inputs = self.feature_extractor(
            audio,
            sampling_rate=sr,
            return_tensors="pt",
            # padding=True,
            # max_length=16_000,
            # truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                # output_attentions=True,
                # output_hidden_states=False,
            )
        return outputs.last_hidden_state


FEATURE_EXTRACTORS = {
    "wav2vec2-base": lambda: HuggingFaceFeatureExtractor(
        Wav2Vec2Model, "facebook/wav2vec2-base"
    ),
    "wav2vec2-large": lambda: HuggingFaceFeatureExtractor(
        Wav2Vec2Model, "facebook/wav2vec2-large"
    ),
    "wav2vec2-large-lv60": lambda: HuggingFaceFeatureExtractor(
        Wav2Vec2Model, "facebook/wav2vec2-large-lv60"
    ),
    "wav2vec2-large-robust": lambda: HuggingFaceFeatureExtractor(
        Wav2Vec2Model, "facebook/wav2vec2-large-robust"
    ),
    "wav2vec2-large-xlsr-53": lambda: HuggingFaceFeatureExtractor(
        Wav2Vec2Model, "facebook/wav2vec2-large-xlsr-53"
    ),
    "wav2vec2-xls-r-300m": lambda: HuggingFaceFeatureExtractor(
        Wav2Vec2Model, "facebook/wav2vec2-xls-r-300m"
    ),
    "wav2vec2-xls-r-1b": lambda: HuggingFaceFeatureExtractor(
        Wav2Vec2Model, "facebook/wav2vec2-xls-r-1b"
    ),
    "wav2vec2-xls-r-2b": lambda: HuggingFaceFeatureExtractor(
        Wav2Vec2Model, "facebook/wav2vec2-xls-r-2b"
    ),
    "wavlm-base": lambda: HuggingFaceFeatureExtractor(
        WavLMModel, "microsoft/wavlm-base"
    ),
    "wavlm-base-sv": lambda: HuggingFaceFeatureExtractor(
        WavLMModel, "microsoft/wavlm-base-sv"
    ),
    "wavlm-base-plus": lambda: HuggingFaceFeatureExtractor(
        WavLMModel, "microsoft/wavlm-base-plus"
    ),
    "wavlm-large": lambda: HuggingFaceFeatureExtractor(
        WavLMModel, "microsoft/wavlm-large"
    ),
}


def get_indices(dataset, subset):
    def read_subset(subset):
        num_samples, seed = subset.split("-")
        num_samples = int(num_samples)
        seed = int(seed)
        return num_samples, seed

    indices = list(range(len(dataset)))
    if subset == "all":
        return indices
    else:
        num_samples, seed = read_subset(subset)
        random.seed(seed)
        return random.sample(indices, num_samples)


@click.command()
@click.option("-d", "--dataset", "dataset_name", type=str, required=True)
@click.option("-s", "--split", type=str, required=True)
@click.option("-f", "--feature-type", type=str, required=True)
@click.option("--subset", type=str, default="all")
def main(
    dataset_name: str,
    split: str,
    feature_type: str,
    subset: str = "all",
):
    dataset = DATASETS[dataset_name](split=split)
    feature_extractor = FEATURE_EXTRACTORS[feature_type]()

    def extract1(audio):
        feature = feature_extractor(audio, sr=SAMPLING_RATE)
        feature = feature.squeeze(dim=0).mean(dim=0)
        feature = feature.cpu().numpy()
        return feature

    path_idxs = f"output/filelists/{dataset_name}-{split}-{subset}.json"
    path_hdf5 = f"output/features/{dataset_name}-{split}-{feature_type}.h5"

    indices = cache_json(path_idxs, get_indices, dataset, subset)

    with h5py.File(path_hdf5, "a") as f:
        for i in tqdm(indices):
            try:
                group = f.create_group(dataset.get_file_name(i))
            except ValueError:
                continue
            feature = extract1(dataset.load_audio(i))
            label_np = np.array(dataset.get_label(i) == "fake", dtype=np.int32)
            index_np = np.array(i, dtype=np.int32)
            group.create_dataset("feature", data=feature)
            group.create_dataset("label", data=label_np)
            group.create_dataset("index", data=index_np)

    def get_feature_dim(f):
        for key in f.keys():
            return f[key]["feature"].shape[0]

    with h5py.File(path_hdf5, "r") as f:
        num_samples = len(indices)
        X = np.zeros((num_samples, get_feature_dim(f)))
        y = np.zeros(num_samples)

        for i, index in enumerate(tqdm(indices)):
            filename = dataset.get_file_name(index)
            group = f[filename]
            X[i] = np.array(group["feature"])
            y[i] = np.array(group["label"])

    path = f"output/features/{dataset_name}-{split}-{feature_type}-{subset}.h5"
    np.savez(path, X=X, y=y)


if __name__ == "__main__":
    main()
