import pdb
import h5py

def load_data(dataset_name, split, feature_type):
    path = f"output/features/{dataset_name}-{split}-{feature_type}.h5"
    with h5py.File(path, "r") as f:
        pdb.set_trace()

load_data("asvspoof19", "train", "wav2vec2-xls-r-2b")