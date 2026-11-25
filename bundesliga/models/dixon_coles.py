"""Dixon-Coles-Modell (Dixon & Coles, 1997).

Jeder Verein bekommt eine Angriffs- und eine Abwehrstärke. Die erwarteten Tore sind

    log(Tore Heim) = mu + Heimvorteil + Angriff[Heim] + Abwehrschwäche[Gast]
    log(Tore Gast) = mu +              Angriff[Gast] + Abwehrschwäche[Heim]

Die Tore sind Poisson-verteilt, mit einer kleinen Korrektur rho für 0:0, 1:0,
0:1 und 1:1. Die Parameter werden per Maximum Likelihood geschätzt; ältere
Spiele zählen exponentiell weniger (Halbwertszeit).

Zwei Zusätze gegenüber dem Original:
- Ridge-Strafe: Ohne viele Daten bleiben Stärken nahe am Ligadurchschnitt.
- Aufsteiger-Prior: Aufsteiger werden zum Durchschnitt der Absteiger gezogen,
  die sie ersetzen. Ohne diesen Prior hätte ein Verein wie Elversberg, der nie
  Bundesliga gespielt hat, zum Saisonstart Durchschnittsstärke.
"""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..scores import score_matrix


@dataclass
class DCParams:
    half_life_days: float = 365.0
    window_days: int | None = None  # None: vier Halbwertszeiten
    ridge: float = 1.0
    promoted_strength: float = 5.0

    def to_dict(self):
        return asdict(self)


class DixonColes:
    def __init__(self, params: DCParams = DCParams()):
        self.params = params
        self.teams: list[str] = []
        self.x: np.ndarray | None = None

    # ------------------------------------------------------------------ fit
    def fit(self, matches: pd.DataFrame, ref_date, teams=(), promoted=(), relegated=()):
        """Schätzt das Modell mit allen Spielen strikt vor `ref_date`."""
        p = self.params
        ref_date = pd.Timestamp(ref_date)
        max_age = p.window_days or 4 * p.half_life_days
        age = (ref_date - matches["date"]).dt.days.to_numpy()
        in_window = (age > 0) & (age <= max_age)
        window, age = matches[in_window], age[in_window]

        all_teams = sorted(set(teams) | set(window["home"]) | set(window["away"]))
        index = {t: i for i, t in enumerate(all_teams)}
        n = len(all_teams)
        data = {
            "h": window["home"].map(index).to_numpy(),
            "a": window["away"].map(index).to_numpy(),
            "hg": window["hg"].to_numpy(float),
            "ag": window["ag"].to_numpy(float),
            "w": 0.5 ** (age / p.half_life_days),
            "prom": np.array([index[t] for t in promoted if t in index], dtype=int),
            "rel": np.array([index[t] for t in relegated if t in index], dtype=int),
        }
        hg, ag = data["hg"], data["ag"]
        data["m00"] = (hg == 0) & (ag == 0)
        data["m01"] = (hg == 0) & (ag == 1)
        data["m10"] = (hg == 1) & (ag == 0)
        data["m11"] = (hg == 1) & (ag == 1)

        x0 = self._warm_start(all_teams)
        bounds = [(None, None), (None, None), (-0.2, 0.2)] + [(-3, 3)] * (2 * n)
        result = minimize(_objective, x0, args=(data, n, p.ridge, p.promoted_strength),
                          jac=True, method="L-BFGS-B", bounds=bounds)
        self.teams, self.index, self.x = all_teams, index, result.x
        return self

    def _warm_start(self, teams):
        n = len(teams)
        x0 = np.zeros(3 + 2 * n)
        x0[:3] = [np.log(1.4), 0.25, -0.05]
        if self.x is not None:
            old = {t: i for i, t in enumerate(self.teams)}
            old_n = len(self.teams)
            x0[:3] = self.x[:3]
            for i, t in enumerate(teams):
                if t in old:
                    x0[3 + i] = self.x[3 + old[t]]
                    x0[3 + n + i] = self.x[3 + old_n + old[t]]
        return x0

    # ------------------------------------------------------------- predict
    @property
    def rho(self) -> float:
        return float(self.x[2])

    def strengths(self) -> pd.DataFrame:
        n = len(self.teams)
        return pd.DataFrame({
            "attack": self.x[3:3 + n],
            "defence_weakness": self.x[3 + n:],
        }, index=self.teams)

    def expected_goals(self, home, away):
        n = len(self.teams)
        mu, home_adv = self.x[0], self.x[1]
        att, dfn = self.x[3:3 + n], self.x[3 + n:]
        h = np.array([self.index[t] for t in home])
        a = np.array([self.index[t] for t in away])
        lh = np.exp(mu + home_adv + att[h] + dfn[a])
        la = np.exp(mu + att[a] + dfn[h])
        return lh, la

    def predict(self, home, away) -> np.ndarray:
        lh, la = self.expected_goals(home, away)
        return score_matrix(lh, la, self.rho)


def _objective(x, d, n, ridge, kappa):
    """Negative gewichtete Log-Likelihood plus Strafterme, mit Gradient."""
    mu, home_adv, rho = x[0], x[1], x[2]
    att, dfn = x[3:3 + n], x[3 + n:]
    h, a, hg, ag, w = d["h"], d["a"], d["hg"], d["ag"], d["w"]

    log_lh = mu + home_adv + att[h] + dfn[a]
    log_la = mu + att[a] + dfn[h]
    lh, la = np.exp(log_lh), np.exp(log_la)

    # Dixon-Coles-Korrektur tau und Ableitungen von log(tau)
    tau = np.ones_like(lh)
    d_lh, d_la, d_rho = np.zeros_like(lh), np.zeros_like(lh), np.zeros_like(lh)
    m = d["m00"]
    tau[m] = 1 - lh[m] * la[m] * rho
    m = d["m01"]
    tau[m] = 1 + lh[m] * rho
    m = d["m10"]
    tau[m] = 1 + la[m] * rho
    m = d["m11"]
    tau[m] = 1 - rho
    tau = np.clip(tau, 1e-10, None)

    m = d["m00"]
    d_lh[m] = -lh[m] * la[m] * rho / tau[m]
    d_la[m] = d_lh[m]
    d_rho[m] = -lh[m] * la[m] / tau[m]
    m = d["m01"]
    d_lh[m] = lh[m] * rho / tau[m]
    d_rho[m] = lh[m] / tau[m]
    m = d["m10"]
    d_la[m] = la[m] * rho / tau[m]
    d_rho[m] = la[m] / tau[m]
    m = d["m11"]
    d_rho[m] = -1 / tau[m]

    loglik = (w * (np.log(tau) + hg * log_lh - lh + ag * log_la - la)).sum()
    g_h = w * (hg - lh + d_lh)
    g_a = w * (ag - la + d_la)

    grad = np.empty_like(x)
    grad[0] = -(g_h.sum() + g_a.sum())
    grad[1] = -g_h.sum()
    grad[2] = -(w * d_rho).sum()
    grad_att = -(np.bincount(h, g_h, n) + np.bincount(a, g_a, n))
    grad_dfn = -(np.bincount(a, g_h, n) + np.bincount(h, g_a, n))

    value = -loglik + ridge / 2 * (att @ att + dfn @ dfn)
    grad_att += ridge * att
    grad_dfn += ridge * dfn

    prom, rel = d["prom"], d["rel"]
    if len(prom) and len(rel):
        for vec, g in ((att, grad_att), (dfn, grad_dfn)):
            diff = vec[prom] - vec[rel].mean()
            value += kappa / 2 * diff @ diff
            g[prom] += kappa * diff
            g[rel] -= kappa * diff.sum() / len(rel)

    grad[3:3 + n] = grad_att
    grad[3 + n:] = grad_dfn
    return value, grad
