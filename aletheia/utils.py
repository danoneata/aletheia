from typing import Callable, List, TypeVar, Tuple
import os
import numpy as np


A = TypeVar("A")


def read_file(path: str, parse_line: Callable[[str], A]) -> List[A]:
    with open(path, "r") as f:
        return list(map(parse_line, f.readlines()))


def implies(p: bool, q: bool):
    return not p or q


def logit(probas: np.ndarray):
    return np.log(probas / (1 - probas))


def sigmoid(logit: np.ndarray):
    return 1 / (1 + np.exp(-logit))


def load_results_asvspoof(path: str) -> List[Tuple[str, str, float]]:
    def parse_line(line: str):
        filename, _, label, score = line.split()
        score = float(score)
        label = "real" if label == "bonafide" else "fake"
        return filename, label, score

    return read_file(path, parse_line)


def cache_np(path, func, *args, **kwargs):
    if os.path.exists(path):
        return np.load(path)
    else:
        result = func(*args, **kwargs)
        np.save(path, result)
        return result