"""Elo-Ratings für Bundesligavereine.

Jeder Verein hat eine Zahl, die nach jedem Spiel steigt oder fällt, je nachdem,
ob er besser oder schlechter abgeschnitten hat als erwartet. Anders als
Tabellenplatz oder "letzte 5 Spiele" läuft das Rating über Saisons hinweg weiter.

Aufsteiger übernehmen zum Saisonstart den Durchschnitt der Absteiger, die sie
ersetzen. Alle Ratings werden zum Saisonstart ein Stück Richtung Ligamittel
gezogen, weil sich Kader über den Sommer verändern.

Aus der Rating-Differenz werden mit einer Poisson-Regression erwartete Tore
für Heim- und Gastteam geschätzt; daraus folgt die Score-Matrix.
"""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import gammaln

from ..data import transitions
from ..scores import match_loglik, score_matrix


@dataclass
class EloParams:
    k: float = 20.0
    home_adv: float = 70.0
    season_regression: float = 0.2
    init: float = 1500.0

    def to_dict(self):
        return asdict(self)


def _goal_multiplier(goal_diff: int) -> float:
    """Höhere Siege zählen mehr (Variante der World-Football-Elo)."""
    goal_diff = abs(goal_diff)
    if goal_diff <= 1:
        return 1.0
    if goal_diff == 2:
        return 1.5
    return (11 + goal_diff) / 8


def compute_elo(matches: pd.DataFrame, teams_by_season: dict[int, set[str]],
                params: EloParams = EloParams()):
    """Elo vor jedem Spiel berechnen.

    `teams_by_season` darf eine Saison enthalten, für die es noch keine Spiele
    gibt (etwa die kommende): Dann werden am Ende auch deren Saisonstart-Regeln
    angewendet. Rückgabe: (Rating Heim vor dem Spiel, Rating Gast vor dem Spiel,
    Ratings nach dem letzten Spiel).
    """
    if not matches["date"].is_monotonic_increasing:
        raise ValueError("compute_elo erwartet nach Datum sortierte Spiele.")
    ratings: dict[str, float] = {}
    pre_home = np.full(len(matches), np.nan)
    pre_away = np.full(len(matches), np.nan)
    positions = {s: np.flatnonzero(matches["season"].to_numpy() == s) for s in teams_by_season}
    home, away = matches["home"].to_numpy(), matches["away"].to_numpy()
    hg, ag = matches["hg"].to_numpy(), matches["ag"].to_numpy()

    for season in sorted(teams_by_season):
        _start_season(ratings, teams_by_season, season, params)
        for idx in positions[season]:
            h, a = home[idx], away[idx]
            r_h, r_a = ratings[h], ratings[a]
            pre_home[idx], pre_away[idx] = r_h, r_a
            expected = 1 / (1 + 10 ** ((r_a - r_h - params.home_adv) / 400))
            score = 1.0 if hg[idx] > ag[idx] else 0.5 if hg[idx] == ag[idx] else 0.0
            delta = params.k * _goal_multiplier(hg[idx] - ag[idx]) * (score - expected)
            ratings[h] = r_h + delta
            ratings[a] = r_a - delta
    return pre_home, pre_away, ratings


def _start_season(ratings, teams_by_season, season, params):
    teams = teams_by_season[season]
    promoted, relegated = transitions(teams_by_season, season)
    if not ratings:
        for team in teams:
            ratings[team] = params.init
        return
    relegated_known = [ratings[t] for t in relegated if t in ratings]
    replacement = float(np.mean(relegated_known)) if relegated_known else params.init - 100
    for team in promoted:
        ratings[team] = replacement
    for team in teams:
        ratings.setdefault(team, replacement)
    league_mean = float(np.mean([ratings[t] for t in teams]))
    for team in teams:
        ratings[team] = league_mean + (ratings[team] - league_mean) * (1 - params.season_regression)


class EloGoalModel:
    """Übersetzt eine Elo-Differenz in erwartete Tore.

    log(Tore Heim) = a0 + a1 * d,  log(Tore Gast) = b0 + b1 * d,
    mit d = (Elo Heim - Elo Gast) / 100. Heimvorteil steckt in a0 > b0.
    """

    def fit(self, elo_diff, hg, ag):
        d = np.asarray(elo_diff, float) / 100
        self.home_coef = _poisson_regression(d, np.asarray(hg, float))
        self.away_coef = _poisson_regression(d, np.asarray(ag, float))
        lh, la = self.expected_goals(elo_diff)
        self.rho = minimize_scalar(
            lambda r: -match_loglik(lh, la, r, hg, ag).sum(),
            bounds=(-0.2, 0.2), method="bounded",
        ).x
        return self

    def expected_goals(self, elo_diff):
        d = np.asarray(elo_diff, float) / 100
        lh = np.exp(self.home_coef[0] + self.home_coef[1] * d)
        la = np.exp(self.away_coef[0] + self.away_coef[1] * d)
        return lh, la

    def predict(self, elo_diff) -> np.ndarray:
        lh, la = self.expected_goals(elo_diff)
        return score_matrix(lh, la, self.rho)


def _poisson_regression(x, y):
    def nll(beta):
        eta = beta[0] + beta[1] * x
        mu = np.exp(eta)
        value = (mu - y * eta + gammaln(y + 1)).sum()
        grad = np.array([(mu - y).sum(), ((mu - y) * x).sum()])
        return value, grad

    start = np.array([np.log(y.mean()), 0.0])
    return minimize(nll, start, jac=True, method="L-BFGS-B").x
