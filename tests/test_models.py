"""Modelltests auf künstlichen Daten mit bekannter Wahrheit."""

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import check_grad

from bundesliga.models.dixon_coles import DCParams, DixonColes, _objective
from bundesliga.models.elo import EloGoalModel, EloParams, compute_elo
from bundesliga.models.ensemble import fit_weights, pool
from bundesliga.scores import outcome_probs, score_matrix

TEAMS = [f"T{i}" for i in range(8)]
STRENGTH = np.linspace(-0.5, 0.5, 8)  # T7 am stärksten


def synthetic_league(seasons=3, seed=0):
    """Doppelrunden mit bekannten Stärken, Poisson-Tore."""
    rng = np.random.default_rng(seed)
    rows, day = [], pd.Timestamp("2020-08-01")
    for season in range(2020, 2020 + seasons):
        for _ in range(2):
            for h in range(8):
                for a in range(8):
                    if h == a:
                        continue
                    lh = np.exp(0.25 + 0.25 + STRENGTH[h] - STRENGTH[a] * 0.8)
                    la = np.exp(0.25 + STRENGTH[a] - STRENGTH[h] * 0.8)
                    hg, ag = rng.poisson(lh), rng.poisson(la)
                    rows.append((season, day, TEAMS[h], TEAMS[a], hg, ag))
                    day += pd.Timedelta(days=1)
    df = pd.DataFrame(rows, columns=["season", "date", "home", "away", "hg", "ag"])
    df["result"] = np.select([df.hg > df.ag, df.hg < df.ag], ["H", "A"], "D")
    return df


def test_dc_gradient_matches_numeric():
    rng = np.random.default_rng(1)
    n, size = 6, 300
    d = {"h": rng.integers(0, n, size), "a": rng.integers(0, n, size),
         "hg": rng.poisson(1.5, size).astype(float), "ag": rng.poisson(1.1, size).astype(float),
         "w": rng.uniform(0.2, 1, size), "prom": np.array([0, 1]), "rel": np.array([4, 5])}
    for key, (i, j) in {"m00": (0, 0), "m01": (0, 1), "m10": (1, 0), "m11": (1, 1)}.items():
        d[key] = (d["hg"] == i) & (d["ag"] == j)
    x = rng.normal(0, 0.2, 3 + 2 * n)
    x[2] = -0.08
    err = check_grad(lambda v: _objective(v, d, n, 1.0, 5.0)[0],
                     lambda v: _objective(v, d, n, 1.0, 5.0)[1], x)
    assert err < 1e-3


def test_dc_recovers_strength_order_and_home_advantage():
    df = synthetic_league()
    dc = DixonColes(DCParams(half_life_days=2000, ridge=0.1)).fit(df, df["date"].max() + pd.Timedelta(days=1))
    s = dc.strengths()
    net = (s["attack"] - s["defence_weakness"]).loc[TEAMS]
    assert np.corrcoef(net, STRENGTH)[0, 1] > 0.9
    assert 0.1 < dc.x[1] < 0.4  # Heimvorteil (wahr: 0,25)


def test_dc_uses_only_past_matches():
    df = synthetic_league(seasons=1)
    cutoff = df["date"].iloc[50]
    a = DixonColes().fit(df, cutoff)
    b = DixonColes().fit(df[df["date"] < cutoff], cutoff)
    assert np.allclose(a.x, b.x)


def test_dc_promoted_team_starts_near_relegated():
    df = synthetic_league(seasons=1)
    df = df[(df.home != "T7") & (df.away != "T7")]  # T7 = Absteiger-Kandidat raus
    weak = DixonColes(DCParams(promoted_strength=50)).fit(
        df, df["date"].max() + pd.Timedelta(days=1),
        teams=set(TEAMS[1:7]) | {"NEW"}, promoted={"NEW"}, relegated={"T0"})
    s = weak.strengths()
    assert abs(s.loc["NEW", "attack"] - s.loc["T0", "attack"]) < 0.1


def test_elo_is_zero_sum_and_rewards_winner():
    df = pd.DataFrame({
        "season": [2020, 2020], "date": pd.to_datetime(["2020-08-01", "2020-08-02"]),
        "home": ["A", "B"], "away": ["B", "A"], "hg": [3, 0], "ag": [0, 0],
    })
    pre_h, pre_a, ratings = compute_elo(df, {2020: {"A", "B"}}, EloParams())
    assert pre_h[0] == pre_a[0] == 1500
    assert ratings["A"] > 1500 > ratings["B"]
    assert ratings["A"] + ratings["B"] == pytest.approx(3000)


def test_elo_promoted_team_inherits_relegated_rating():
    df = pd.DataFrame({
        "season": [2020, 2021], "date": pd.to_datetime(["2020-08-01", "2021-08-01"]),
        "home": ["A", "A"], "away": ["B", "C"], "hg": [4, 1], "ag": [0, 1],
    })
    params = EloParams(season_regression=0.0)
    pre_h, pre_a, _ = compute_elo(df, {2020: {"A", "B"}, 2021: {"A", "C"}}, params)
    rating_b_after = pre_a[0] - (pre_h[1] - pre_h[0])  # B verliert, was A gewinnt
    assert pre_a[1] == pytest.approx(rating_b_after)


def test_elo_requires_sorted_matches():
    df = pd.DataFrame({
        "season": [2020, 2020], "date": pd.to_datetime(["2020-08-02", "2020-08-01"]),
        "home": ["A", "B"], "away": ["B", "A"], "hg": [1, 0], "ag": [0, 0],
    })
    with pytest.raises(ValueError):
        compute_elo(df, {2020: {"A", "B"}})


def test_elo_goal_model_maps_rating_gap_to_goals():
    rng = np.random.default_rng(2)
    diff = rng.normal(0, 150, 3000)
    hg = rng.poisson(np.exp(0.35 + 0.002 * diff))
    ag = rng.poisson(np.exp(0.1 - 0.002 * diff))
    model = EloGoalModel().fit(diff, hg, ag)
    lh, la = model.expected_goals(np.array([200.0]))
    assert lh[0] > la[0]
    assert model.home_coef[1] == pytest.approx(0.2, abs=0.03)


def test_ensemble_prefers_better_model():
    rng = np.random.default_rng(3)
    truth = outcome_probs(score_matrix(rng.uniform(0.6, 2.5, 4000), rng.uniform(0.6, 2.0, 4000)))
    results = np.array(["H", "D", "A"])[[rng.choice(3, p=p) for p in truth]]
    noise = np.full_like(truth, 1 / 3)
    w = fit_weights([truth, noise], results)
    assert w[0] > 0.8


def test_pool_skips_missing_model():
    a, b = score_matrix([1.5], [1.0]), np.full((1, 11, 11), np.nan)
    assert np.allclose(pool([a, b], [0.5, 0.5]), a)
    assert np.allclose(pool([a, b], [0.0, 1.0]), a)  # Gewicht nur auf fehlendem Modell
