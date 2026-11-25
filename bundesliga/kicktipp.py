"""Kicktipp-Punkte und der Tipp mit den meisten erwarteten Punkten.

Der wahrscheinlichste Endstand ist nicht automatisch der beste Tipp: Ein 2:1
bringt bei jedem knappen Heimsieg mit einem Tor Unterschied Tordifferenz-Punkte,
ein 1:0 genauso, aber ein 3:1 nur selten. Deshalb wird für jeden möglichen Tipp
der Erwartungswert der Punkte über die ganze Score-Matrix berechnet.
"""

from dataclasses import dataclass

import numpy as np

from .scores import GOALS

MAX_TIP = 5


@dataclass(frozen=True)
class Rules:
    """Kicktipp-Standard: 4 Punkte Ergebnis, 3 Tordifferenz, 2 Tendenz.

    Bei Remis zählt die Tordifferenz in der Standardwertung nicht: Wer 1:1
    tippt und es endet 2:2, bekommt die Tendenzpunkte.
    """

    exact: int = 4
    goal_diff: int = 3
    tendency: int = 2
    goal_diff_for_draws: bool = False


DEFAULT_RULES = Rules()


def points(tip_h, tip_a, hg, ag, rules: Rules = DEFAULT_RULES):
    """Punkte für einen Tipp (funktioniert auch elementweise mit Arrays)."""
    tip_h, tip_a, hg, ag = (np.asarray(v) for v in (tip_h, tip_a, hg, ag))
    tip_diff, real_diff = tip_h - tip_a, hg - ag
    same_tendency = np.sign(tip_diff) == np.sign(real_diff)
    exact = (tip_h == hg) & (tip_a == ag)
    diff_ok = (tip_diff == real_diff) & ((real_diff != 0) | rules.goal_diff_for_draws)
    return np.select(
        [exact, same_tendency & diff_ok, same_tendency],
        [rules.exact, rules.goal_diff, rules.tendency],
        default=0,
    )


def _points_tensor(rules: Rules) -> np.ndarray:
    """P[a, b, i, j] = Punkte für Tipp a:b, wenn es i:j endet."""
    tips = np.arange(MAX_TIP + 1)
    a, b, i, j = np.meshgrid(tips, tips, GOALS, GOALS, indexing="ij")
    return points(a, b, i, j, rules).astype(float)


def expected_points(mats: np.ndarray, rules: Rules = DEFAULT_RULES) -> np.ndarray:
    """(N, MAX_TIP + 1, MAX_TIP + 1): erwartete Punkte für jeden Tipp."""
    return np.einsum("nij,abij->nab", mats, _points_tensor(rules))


def best_tips(mats: np.ndarray, rules: Rules = DEFAULT_RULES):
    """Bester Tipp pro Spiel: (tip_home, tip_away, erwartete Punkte)."""
    ev = expected_points(mats, rules)
    flat = ev.reshape(len(ev), -1)
    best = flat.argmax(axis=1)
    tip_h, tip_a = np.divmod(best, MAX_TIP + 1)
    return tip_h, tip_a, flat[np.arange(len(flat)), best]
