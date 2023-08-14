import pdb

import numpy as np

from sklearn.linear_model import LogisticRegression

from aletheia.metrics import compute_eer, compute_ece


def load_data_npz(dataset_name, split, feature_type, subset):
    path = f"output/features/{dataset_name}-{split}-{feature_type}-{subset}.h5.npz"
    with np.load(path) as f:
        X = f["X"]
        y = f["y"]
    return X, y


def load_data_multi(datasets):
    data = [load_data_npz(**d) for d in datasets]
    Xs, ys = zip(*data)

    X = np.vstack(Xs)
    y = np.hstack(ys)

    return X, y


def get_te_datasets(feature_type):
    return [
        {
            "dataset_name": dataset,
            "split": "eval",
            "feature_type": feature_type,
            "subset": "all",
        }
        for dataset in ["asvspoof19", "in-the-wild"]
    ]


def get_feature_type(tr_datasets):
    note = "Training datasets should consist of the same feature type."
    feature_types = [d["feature_type"] for d in tr_datasets]
    num_feature_types = len(set(feature_types))
    assert num_feature_types == 1, note
    return feature_types[0]


def predict(tr_datasets, C=1e6, verbose=False):
    def predict1(model, te_dataset):
        X_te, y_te = load_data_npz(**te_dataset)
        pred = model.predict_proba(X_te)[:, 1]
        return {
            "true": y_te.tolist(),
            "pred": pred.tolist(),
        }

    X_tr, y_tr = load_data_multi(tr_datasets)

    model = LogisticRegression(C=C, max_iter=5_000, random_state=42, verbose=verbose)
    model.fit(X_tr, y_tr)

    outputs = get_te_datasets(get_feature_type(tr_datasets))
    for i, output in enumerate(outputs):
        outputs[i] = {**output, **predict1(model, output)}

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


def evaluate(tr_datasets, verbose):
    return [evaluate1(**d, verbose=verbose) for d in predict(tr_datasets)]
