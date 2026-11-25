"""Das Marktmodell: Buchmacherquoten als Wahrscheinlichkeiten.

Quoten enthalten eine Marge (die Kehrwerte summieren sich auf etwas über 1).
Sie wird proportional herausgerechnet. Aus den Wahrscheinlichkeiten für
Heimsieg/Auswärtssieg und über/unter 2,5 Tore werden passende Torerwartungen
geschätzt, damit auch das Marktmodell eine Score-Matrix (und damit einen
Kicktipp-Tipp) liefert.
"""

import numpy as np
import pandas as pd

from ..scores import fit_lambdas, rescale_to_outcomes, score_matrix


def implied_probs(df: pd.DataFrame) -> np.ndarray:
    """(N, 3) Heimsieg/Remis/Auswärtssieg ohne Buchmachermarge, NaN ohne Quoten."""
    inv = 1 / df[["odds_h", "odds_d", "odds_a"]].to_numpy(float)
    return inv / inv.sum(axis=1, keepdims=True)


def implied_over(df: pd.DataFrame) -> np.ndarray:
    inv_over = 1 / df["odds_over"].to_numpy(float)
    inv_under = 1 / df["odds_under"].to_numpy(float)
    return inv_over / (inv_over + inv_under)


def market_matrices(df: pd.DataFrame):
    """Score-Matrizen aus Quoten. Rückgabe: (Matrizen, Maske "Quoten vorhanden")."""
    probs = implied_probs(df)
    valid = np.isfinite(probs).all(axis=1)
    mats = np.full((len(df), *score_matrix([1.0], [1.0]).shape[1:]), np.nan)
    if valid.any():
        lh, la = fit_lambdas(probs[valid, 0], probs[valid, 2], implied_over(df)[valid])
        mats[valid] = rescale_to_outcomes(score_matrix(lh, la), probs[valid])
    return mats, valid
