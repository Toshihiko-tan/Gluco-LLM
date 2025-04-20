from __future__ import annotations

import sys, os
from typing import Callable, Dict, Tuple, Sequence, Any
import numpy as np

from RAG.distance import calc_mape, calc_correlation

def _euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))

def _manhattan(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).sum())

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    num = float(np.dot(a, b))
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 1.0 if denom == 0 else 1.0 - num / denom

_BUILTIN_METRICS: Dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "euclidean": _euclidean,
    "l2": _euclidean,
    "manhattan": _manhattan,
    "l1": _manhattan,
    "cosine": _cosine,
}
_CUSTOM_METRICS: Dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "mape": calc_mape,
    "correlation": calc_correlation,
}
_METRICS: Dict[str, Callable[[np.ndarray, np.ndarray], float]] = {**_BUILTIN_METRICS, **_CUSTOM_METRICS}

class DartsRetriever:
    def __init__(
        self,
        inputs: Sequence[np.ndarray],
        outputs: Sequence[Any],
        metric: str | Callable[[np.ndarray, np.ndarray], float] = "euclidean",
    ) -> None:
        if len(inputs) != len(outputs):
            raise ValueError("inputs and outputs must have the same length")
        self.inputs = [np.asarray(x, dtype=float) for x in inputs]
        self.outputs = list(outputs)
        if isinstance(metric, str):
            key = metric.lower()
            if key not in _METRICS:
                raise ValueError(f"Unknown metric '{metric}'. Available: {list(_METRICS)}")
            self.metric_fn = _METRICS[key]
        else:
            self.metric_fn = metric
    def retrieve(self, query: np.ndarray, k: int = 5):
        q = np.asarray(query, dtype=float)
        X = np.vstack(self.inputs)
        n = X.shape[0]
        if n == 0:
            return []
        k = min(max(k, 1), n)
        if self.metric_fn in (_METRICS["euclidean"], _METRICS["l2"]):
            dists = np.linalg.norm(X - q, axis=1)
        elif self.metric_fn in (_METRICS["cosine"],):
            num = X @ q
            denom = np.linalg.norm(X, axis=1) * np.linalg.norm(q)
            dists = 1 - num / np.where(denom == 0, 1, denom)
        else:
            dists = np.fromiter((self.metric_fn(q, x) for x in X), dtype=float, count=n)
        idx = np.argpartition(dists, k-1)[:k]
        idx = idx[np.argsort(dists[idx])]
        return [(self.inputs[i], self.outputs[i], float(dists[i])) for i in idx]
    __call__ = retrieve

def _default_encoder(sample: Tuple[np.ndarray, ...]) -> np.ndarray:
    return np.asarray(sample[0], dtype=float).flatten()

def build_retriever_from_dataset(
    dataset: Any,
    k_encoder: Callable[[Tuple[np.ndarray, ...]], np.ndarray] = _default_encoder,
    metric: str | Callable[[np.ndarray, np.ndarray], float] = "euclidean",
) -> DartsRetriever:
    if not hasattr(dataset, "__len__") or not hasattr(dataset, "__getitem__"):
        raise TypeError("dataset must support len() and __getitem__")
    inputs, outputs = [], []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        inputs.append(k_encoder(sample))
        outputs.append(sample)
    return DartsRetriever(inputs, outputs, metric)

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 144))
    retriever = DartsRetriever(X, X, metric="euclidean")
    q = rng.normal(size=144)
    print(retriever(q, k=3))