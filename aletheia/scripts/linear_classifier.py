import pdb
import random
from typing import Dict, List
from pathlib import Path

import numpy as np

from sklearn.linear_model import LogisticRegression

from aletheia.data import DATASETS
from aletheia.metrics import compute_eer, compute_ece


DATA_DIR = Path("/mnt/student-share/projects/2024-interspeech")


def load_data_npz(dataset_name, split, feature_type, subset, systems=None):
    path = (
        DATA_DIR
        / "output"
        / "features"
        / f"{dataset_name}-{split}-{feature_type}-{subset}.h5.npz"
    )
    with np.load(path) as f:
        X = f["X"]
        y = f["y"]
    if systems is not None:
        assert subset == "all"
        assert systems != []
        dataset = DATASETS[dataset_name](split=split)
        indices = [i for i in range(len(dataset)) if dataset.get_system(i) in systems]
        X = X[indices]
        y = y[indices]
    return X, y


def load_data_multi(datasets):
    data = [load_data_npz(**d) for d in datasets]
    Xs, ys = zip(*data)

    X = np.vstack(Xs)
    y = np.hstack(ys)

    return X, y


def get_te_datasets(feature_type) -> List[Dict]:
    dataset_to_subsets = {
        "asvspoof19": ["asvspoof19"],
        "in-the-wild": ["in-the-wild"],
        "timit-tts": ["timit-tts-clean"],
        "fake-or-real": ["FakeOrReal"],
        # "timit-tts": ["timit-tts-clean", "timit-tts-dtw-aug"],
    }
    return [
        {
            "dataset_name": name,
            "subsets": [
                {
                    "dataset_name": s,
                    "split": "eval",
                    "feature_type": feature_type,
                    "subset": "all",
                }
                for s in subset
            ],
        }
        for name, subset in dataset_to_subsets.items()
    ]


def get_feature_type(tr_datasets):
    note = "Training datasets should consist of the same feature type."
    feature_types = [d["feature_type"] for d in tr_datasets]
    num_feature_types = len(set(feature_types))
    assert num_feature_types == 1, note
    return feature_types[0]


def predict(tr_datasets, seed=42, C=1e6, verbose=False):
    random.seed(seed)

    def predict1(model, te_datasets):
        X_te, y_te = load_data_multi(te_datasets)
        pred = model.predict_proba(X_te)[:, 1]
        return {
            "true": y_te.tolist(),
            "pred": pred.tolist(),
        }

    X_tr, y_tr = load_data_multi(tr_datasets)

    # idxs = random.choices(range(len(X_tr)), k=len(X_tr))
    # X_tr = X_tr[idxs]
    # y_tr = y_tr[idxs]

    model = LogisticRegression(C=C, max_iter=5_000, random_state=seed, verbose=verbose)

    from time import time
    time_s = time()
    model.fit(X_tr, y_tr)
    time_e = time()

    print("Shape", X_tr.shape)
    print(f"Time: {time_e - time_s:.2f}s")
    print()

    outputs = get_te_datasets(get_feature_type(tr_datasets))
    for i, output in enumerate(outputs):
        outputs[i] = {**output, **predict1(model, output["subsets"])}

    return outputs


def evaluate1(true, pred, dataset_name, verbose=False, **_):
    eer = 100 * compute_eer(true, pred)
    ece = 100 * compute_ece(true, pred)
    results = {
        "te-dataset": dataset_name,
        "eer": eer,
        "ece": ece,
    }
    if verbose:
        print(results)
    return results


def evaluate(tr_datasets, *, seed, C, verbose):
    return [
        evaluate1(**d, verbose=verbose)
        for d in predict(tr_datasets, seed=seed, C=C, verbose=verbose)
    ]