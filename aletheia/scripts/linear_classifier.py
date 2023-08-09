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


def evaluate(tr_datasets, verbose=False):
    def evaluate1(model, te_dataset):
        X_te, y_te = load_data_npz(**te_dataset)
        preds = model.predict_proba(X_te)[:, 1]
        eer = 100 * compute_eer(y_te, preds)
        ece = 100 * compute_ece(y_te, preds)
        results = {
            "te-dataset": te_dataset["dataset_name"],
            "eer": eer,
            "ece": ece,
        }
        if verbose:
            print(results)
        return results

    note = "Training datasets should consist of the same feature type."
    feature_types = [d["feature_type"] for d in tr_datasets]
    num_feature_types = len(set(feature_types))
    assert num_feature_types == 1, note

    feature_type = feature_types[0]
    te_datasets = [
        {
            "dataset_name": dataset,
            "split": "eval",
            "feature_type": feature_type,
            "subset": "all",
        }
        for dataset in ["asvspoof19", "in-the-wild"]
    ][1:]

    data = [load_data_npz(**d) for d in tr_datasets]
    Xs_tr, ys_tr = zip(*data)

    X_tr = np.vstack(Xs_tr)
    y_tr = np.hstack(ys_tr)

    model = LogisticRegression(C=1e6, max_iter=5_000, random_state=42)
    model.fit(X_tr, y_tr)

    return [evaluate1(model, te_dataset) for te_dataset in te_datasets]
