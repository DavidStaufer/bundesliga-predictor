"""Bewertung von Wahrscheinlichkeitsvorhersagen.

Trefferquote allein ist ein schwaches Maß: Ein Modell, das 51 % Heimsieg sagt,
ist genauso "richtig" wie eines, das 90 % sagt. Deshalb zählen vor allem
Log-Loss und RPS, die bewerten, wie gut die Wahrscheinlichkeiten selbst sind.
"""

import numpy as np

from .config import OUTCOMES


def one_hot(results) -> np.ndarray:
    results = np.asarray(results)
    return np.stack([results == o for o in OUTCOMES], axis=1).astype(float)


def log_loss(probs: np.ndarray, results) -> float:
    """Mittlere negative Log-Wahrscheinlichkeit des tatsächlichen Ausgangs."""
    p = (np.clip(probs, 1e-12, 1) * one_hot(results)).sum(axis=1)
    return float(-np.log(p).mean())


def brier(probs: np.ndarray, results) -> float:
    return float(((probs - one_hot(results)) ** 2).sum(axis=1).mean())


def rps(probs: np.ndarray, results) -> float:
    """Ranked Probability Score: berücksichtigt die Reihenfolge Heim < Remis < Gast.

    Ein Remis-Tipp ist bei einem Auswärtssieg "weniger falsch" als ein Heimsieg-Tipp.
    Üblicher Standard für Fußballvorhersagen, kleiner ist besser.
    """
    cum_p = np.cumsum(probs, axis=1)[:, :2]
    cum_y = np.cumsum(one_hot(results), axis=1)[:, :2]
    return float(((cum_p - cum_y) ** 2).sum(axis=1).mean() / 2)


def accuracy(probs: np.ndarray, results) -> float:
    predicted = np.array(OUTCOMES)[probs.argmax(axis=1)]
    return float((predicted == np.asarray(results)).mean())


def evaluate(probs: np.ndarray, results) -> dict:
    return {
        "log_loss": log_loss(probs, results),
        "rps": rps(probs, results),
        "brier": brier(probs, results),
        "accuracy": accuracy(probs, results),
    }
