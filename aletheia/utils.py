from typing import Callable, List, TypeVar, Tuple
import json
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


def cache_np(path, func, *args, **kwargs):
    if os.path.exists(path):
        return np.load(path)
    else:
        result = func(*args, **kwargs)
        np.save(path, result)
        return result


def cache_json(path, func, *args, **kwargs):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        result = func(*args, **kwargs)
        with open(path, "w") as f:
            json.dump(result, f)
        return result