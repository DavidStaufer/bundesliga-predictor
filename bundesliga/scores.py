"""Ergebniswahrscheinlichkeiten als Score-Matrix.

Jedes Modell liefert pro Spiel eine Matrix M mit M[i, j] = P(Heim i Tore,
Gast j Tore). Daraus folgen die Wahrscheinlichkeiten für Heimsieg, Remis und
Auswärtssieg, aber auch der beste Kicktipp-Tipp.
"""

import numpy as np
from scipy.special import gammaln

from .config import MAX_GOALS

GOALS = np.arange(MAX_GOALS + 1)
_HOME_WIN = GOALS[:, None] > GOALS[None, :]
_DRAW = GOALS[:, None] == GOALS[None, :]
_AWAY_WIN = GOALS[:, None] < GOALS[None, :]


def poisson_pmf(lam: np.ndarray) -> np.ndarray:
    """P(X = k) für k = 0..MAX_GOALS, Form (N, MAX_GOALS + 1)."""
    lam = np.asarray(lam, dtype=float)[:, None]
    return np.exp(GOALS * np.log(lam) - lam - gammaln(GOALS + 1))


def dixon_coles_tau(lh, la, rho) -> np.ndarray:
    """Dixon-Coles-Korrektur für die Ergebnisse 0:0, 1:0, 0:1 und 1:1.

    Reine Poisson-Modelle unterschätzen 0:0 und 1:1 leicht. Mit rho < 0 werden
    diese Remis etwas wahrscheinlicher, 1:0 und 0:1 etwas unwahrscheinlicher.
    """
    lh, la = np.asarray(lh, float), np.asarray(la, float)
    rho = np.broadcast_to(np.asarray(rho, float), lh.shape)
    tau = np.ones((len(lh), 2, 2))
    tau[:, 0, 0] = 1 - lh * la * rho
    tau[:, 0, 1] = 1 + lh * rho
    tau[:, 1, 0] = 1 + la * rho
    tau[:, 1, 1] = 1 - rho
    return np.clip(tau, 1e-6, None)


def score_matrix(lh, la, rho=0.0) -> np.ndarray:
    """Score-Matrizen (N, G, G) aus erwarteten Toren lh (Heim) und la (Gast)."""
    lh, la = np.atleast_1d(lh).astype(float), np.atleast_1d(la).astype(float)
    mats = poisson_pmf(lh)[:, :, None] * poisson_pmf(la)[:, None, :]
    mats[:, :2, :2] *= dixon_coles_tau(lh, la, rho)
    return mats / mats.sum(axis=(1, 2), keepdims=True)


def outcome_probs(mats: np.ndarray) -> np.ndarray:
    """(N, 3) mit den Spalten Heimsieg, Remis, Auswärtssieg."""
    return np.stack([
        (mats * _HOME_WIN).sum(axis=(1, 2)),
        (mats * _DRAW).sum(axis=(1, 2)),
        (mats * _AWAY_WIN).sum(axis=(1, 2)),
    ], axis=1)


def rescale_to_outcomes(mats: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Skaliert die Matrizen so, dass Heimsieg/Remis/Auswärtssieg exakt `probs` ergeben.

    Die Verteilung innerhalb eines Ausgangs (etwa 1:0 gegen 2:1) bleibt erhalten.
    """
    current = outcome_probs(mats)
    factors = probs / np.clip(current, 1e-12, None)
    scale = (
        factors[:, 0, None, None] * _HOME_WIN
        + factors[:, 1, None, None] * _DRAW
        + factors[:, 2, None, None] * _AWAY_WIN
    )
    out = mats * scale
    return out / out.sum(axis=(1, 2), keepdims=True)


def _market_targets(log_lam: np.ndarray) -> np.ndarray:
    """P(Heimsieg), P(Auswärtssieg), P(über 2,5 Tore) bei unabhängigen Poisson-Toren."""
    lam = np.exp(log_lam)
    ph, pa = poisson_pmf(lam[:, 0]), poisson_pmf(lam[:, 1])
    mats = ph[:, :, None] * pa[:, None, :]
    total = GOALS[:, None] + GOALS[None, :]
    under = (mats * (total <= 2)).sum(axis=(1, 2))
    return np.stack([
        (mats * _HOME_WIN).sum(axis=(1, 2)),
        (mats * _AWAY_WIN).sum(axis=(1, 2)),
        1 - under,
    ], axis=1)


def fit_lambdas(p_home, p_away, p_over=None, iterations: int = 40):
    """Erwartete Tore, die zu gegebenen Wahrscheinlichkeiten passen.

    Löst für alle Spiele gleichzeitig ein kleines Kleinste-Quadrate-Problem
    (Gauss-Newton): Welche Torerwartungen erzeugen P(Heimsieg), P(Auswärtssieg)
    und, falls vorhanden, P(über 2,5 Tore)? So lassen sich Buchmacherquoten in
    eine komplette Score-Matrix übersetzen.
    """
    p_home, p_away = np.asarray(p_home, float), np.asarray(p_away, float)
    n = len(p_home)
    p_over = np.full(n, np.nan) if p_over is None else np.asarray(p_over, float)
    targets = np.stack([p_home, p_away, np.nan_to_num(p_over, nan=0.5)], axis=1)
    weights = np.stack([np.ones(n), np.ones(n), (~np.isnan(p_over)).astype(float)], axis=1)

    x = np.log(np.column_stack([np.full(n, 1.5), np.full(n, 1.2)]))
    eps = 1e-5
    for _ in range(iterations):
        r = (_market_targets(x) - targets) * weights
        jac = np.empty((n, 3, 2))
        for k in range(2):
            step = np.zeros(2)
            step[k] = eps
            jac[:, :, k] = ((_market_targets(x + step) - targets) * weights - r) / eps
        jtj = np.einsum("nki,nkj->nij", jac, jac) + 1e-9 * np.eye(2)
        jtr = np.einsum("nki,nk->ni", jac, r)
        dx = -np.linalg.solve(jtj, jtr[:, :, None])[:, :, 0]
        x = np.clip(x + np.clip(dx, -0.5, 0.5), np.log(0.05), np.log(8.0))
        if np.abs(dx).max() < 1e-8:
            break
    lam = np.exp(x)
    return lam[:, 0], lam[:, 1]


def match_loglik(lh, la, rho, hg, ag) -> np.ndarray:
    """Log-Likelihood der tatsächlichen Ergebnisse unter dem Dixon-Coles-Modell."""
    lh, la, hg, ag = (np.asarray(v, float) for v in (lh, la, hg, ag))
    tau = np.ones_like(lh)
    tau = np.where((hg == 0) & (ag == 0), 1 - lh * la * rho, tau)
    tau = np.where((hg == 0) & (ag == 1), 1 + lh * rho, tau)
    tau = np.where((hg == 1) & (ag == 0), 1 + la * rho, tau)
    tau = np.where((hg == 1) & (ag == 1), 1 - rho, tau)
    return (np.log(np.clip(tau, 1e-10, None))
            + hg * np.log(lh) - lh - gammaln(hg + 1)
            + ag * np.log(la) - la - gammaln(ag + 1))
