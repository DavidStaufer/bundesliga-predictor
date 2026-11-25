"""Ensemble: gewichtete Mischung mehrerer Modelle.

Die Score-Matrizen der Einzelmodelle werden linear gemischt ("linear pool").
Die Gewichte werden so gewählt, dass der Log-Loss auf früheren, nicht zum
Training genutzten Vorhersagen minimal wird. Fehlen für ein Spiel die Quoten,
wird nur über die verfügbaren Modelle gemischt.
"""

import numpy as np
from scipy.optimize import minimize

from ..metrics import one_hot
from ..scores import outcome_probs


def _softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def fit_weights(prob_list: list[np.ndarray], results) -> np.ndarray:
    """Gewichte (Summe 1), die den Log-Loss der Mischung minimieren."""
    stacked = np.stack(prob_list)  # (M, N, 3)
    ok = np.isfinite(stacked).all(axis=(0, 2))
    stacked = stacked[:, ok]
    y = one_hot(np.asarray(results)[ok])
    p_true = (stacked * y).sum(axis=2)  # (M, N)

    def loss(z):
        return -np.log(np.clip(_softmax(z) @ p_true, 1e-12, None)).mean()

    res = minimize(loss, np.zeros(len(prob_list)), method="Nelder-Mead",
                   options={"xatol": 1e-4, "fatol": 1e-7, "maxiter": 2000})
    return _softmax(res.x)


def pool(mats_list: list[np.ndarray], weights) -> np.ndarray:
    """Mischt Score-Matrizen; Modelle ohne Vorhersage (NaN) fallen pro Spiel heraus."""
    stacked = np.stack(mats_list)  # (M, N, G, G)
    available = np.isfinite(stacked).all(axis=(2, 3))  # (M, N)
    w = np.asarray(weights, float)[:, None] * available
    # Haben alle verfügbaren Modelle Gewicht 0 (z. B. nur Quoten gewichtet, aber
    # keine Quoten da), wird gleichmäßig über die verfügbaren Modelle gemischt.
    empty = w.sum(axis=0) == 0
    w[:, empty] = available[:, empty]
    w = w / np.clip(w.sum(axis=0, keepdims=True), 1e-12, None)
    return np.einsum("mn,mnij->nij", w, np.nan_to_num(stacked))


def pooled_probs(mats_list, weights) -> np.ndarray:
    return outcome_probs(pool(mats_list, weights))
