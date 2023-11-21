import time

import click
import pandas as pd
import torch

from tqdm import tqdm
from scipy.stats import describe

from aletheia.scripts.extract_features import (
    FEATURE_EXTRACTORS,
    DATASETS,
    SAMPLING_RATE,
)


@click.command()
@click.option("-f", "--feature-type", type=str, required=True)
def main(feature_type: str):
    split = "eval"
    dataset_name = "asvspoof19"

    dataset = DATASETS[dataset_name](split=split)
    feature_extractor = FEATURE_EXTRACTORS[feature_type]()

    def extract1(audio):
        feature = feature_extractor(audio, sr=SAMPLING_RATE)
        feature = feature.squeeze(dim=0).mean(dim=0)
        return feature

    def benchmark(i):
        audio = dataset.load_audio(i)
        t0 = time.time()
        extract1(audio)
        t1 = time.time()
        return {
            "time": t1 - t0,
            "audio-length": len(audio) / SAMPLING_RATE,
            "cuda-memory-reserved": torch.cuda.memory_reserved() / 2 ** 20,
            "cuda-max-memory-reserved": torch.cuda.max_memory_reserved() / 2 ** 20,
        }


    results = [benchmark(i) for i in tqdm(range(64))]
    df = pd.DataFrame(results)
    print(df.describe())


if __name__ == "__main__":
    main()
