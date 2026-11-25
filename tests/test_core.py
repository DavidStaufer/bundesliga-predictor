"""Tests ohne Netzwerkzugriff: Datenaufbereitung, Score-Matrizen, Kicktipp, Metriken."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from bundesliga import data
from bundesliga.backtest import date_blocks
from bundesliga.kicktipp import Rules, best_tips, expected_points, points
from bundesliga.metrics import accuracy, log_loss, rps
from bundesliga.scores import (fit_lambdas, outcome_probs, poisson_pmf,
                               rescale_to_outcomes, score_matrix)
from bundesliga.teams import UnknownTeamError, canonical

# --------------------------------------------------------------------- Teams

@pytest.mark.parametrize("raw, expected", [
    ("FC Bayern München", "Bayern Munich"),
    ("Bayer 04 Leverkusen", "Leverkusen"),
    ("Borussia Mönchengladbach", "M'gladbach"),
    ("1. FC Köln", "FC Koln"),
    ("SV 07 Elversberg", "Elversberg"),
    ("SC Paderborn 07", "Paderborn"),
    ("FC Schalke 04", "Schalke 04"),
    ("Hamburger SV", "Hamburg"),
    ("TSG 1899 Hoffenheim", "Hoffenheim"),
    ("Sport-Club Freiburg", "Freiburg"),
    ("1. FC Heidenheim 1846", "Heidenheim"),
    ("Eintracht Frankfurt", "Ein Frankfurt"),
    ("M'gladbach", "M'gladbach"),
])
def test_team_names(raw, expected):
    assert canonical(raw) == expected


def test_unknown_team_fails_loudly():
    with pytest.raises(UnknownTeamError):
        canonical("FC Barcelona")


# ---------------------------------------------------------------------- Daten

OLD_FORMAT = """Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,B365H,BbAvH,BbAvD,BbAvA,BbAv>2.5,BbAv<2.5
D1,05/08/05,Bayern Munich,M'gladbach,3,0,H,1.3,1.28,4.6,9.0,1.6,2.2
D1,06/08/05,Hertha,Duisburg,1,1,D,1.5,1.55,3.6,5.5,1.9,1.8
,,,,,,,,,,,,
"""

NEW_FORMAT = "\ufeff" + """Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,AvgH,AvgD,AvgA,Avg>2.5,Avg<2.5
D1,22/08/2025,19:30,Bayern Munich,RB Leipzig,6,0,H,1.35,5.5,7.5,1.4,2.9
D1,23/08/2025,15:30,Werder Bremen,Ein Frankfurt,1,4,A,2.9,3.6,2.3,1.6,2.3
"""


def test_parse_old_format():
    df = data.parse_season(OLD_FORMAT, 2005)
    assert len(df) == 2
    assert list(df["result"]) == ["H", "D"]
    assert df["date"].iloc[0] == pd.Timestamp("2005-08-05")
    assert df["odds_h"].iloc[0] == 1.28  # Durchschnittsquote vor Bet365
    assert df["odds_over"].iloc[1] == 1.9


def test_parse_new_format_with_bom():
    df = data.parse_season(NEW_FORMAT, 2025)
    assert list(df["result"]) == ["H", "A"]
    assert df["odds_a"].iloc[1] == 2.3
    assert df["season"].unique().tolist() == [2025]


def test_season_helpers():
    assert data.season_code(2025) == "2526"
    assert data.season_code(1999) == "9900"
    assert data.season_label(2026) == "2026/27"
    assert data.current_season(date(2026, 11, 25)) == 2026
    assert data.current_season(date(2027, 3, 1)) == 2026
    assert data.current_season(date(2027, 7, 1)) == 2027


def test_transitions():
    teams = {2024: {"A", "B", "C"}, 2025: {"A", "B", "D"}}
    promoted, relegated = data.transitions(teams, 2025)
    assert promoted == {"D"} and relegated == {"C"}
    assert data.transitions(teams, 2024) == (set(), set())


def test_date_blocks():
    dates = pd.Series(pd.to_datetime([
        "2025-08-22", "2025-08-23", "2025-08-24",  # Fr–So
        "2025-08-26", "2025-08-27",                # Di–Mi (englische Woche)
        "2025-08-30",                              # Sa
    ]))
    assert date_blocks(dates).tolist() == [0, 0, 0, 1, 1, 2]


# ------------------------------------------------------------- Score-Matrizen

def test_score_matrix_is_distribution():
    mats = score_matrix([1.6, 0.8], [1.1, 2.0], rho=-0.1)
    assert np.allclose(mats.sum(axis=(1, 2)), 1)
    assert np.allclose(outcome_probs(mats).sum(axis=1), 1)
    probs = outcome_probs(mats)
    assert probs[0, 0] > probs[0, 2]  # stärkeres Heimteam
    assert probs[1, 2] > probs[1, 0]  # stärkeres Gastteam


def test_rho_zero_is_independent_poisson():
    mats = score_matrix([1.4], [1.2], rho=0.0)
    expected = poisson_pmf([1.4])[0][:, None] * poisson_pmf([1.2])[0][None, :]
    assert np.allclose(mats[0], expected / expected.sum())


def test_negative_rho_adds_draws():
    plain = outcome_probs(score_matrix([1.3], [1.1], rho=0.0))[0, 1]
    adjusted = outcome_probs(score_matrix([1.3], [1.1], rho=-0.1))[0, 1]
    assert adjusted > plain


def test_fit_lambdas_recovers_goal_expectations():
    lh, la = np.array([2.1, 1.0, 1.4]), np.array([0.6, 1.7, 1.3])
    mats = score_matrix(lh, la)
    probs = outcome_probs(mats)
    over = 1 - np.array([m[np.add.outer(range(11), range(11)) <= 2].sum() for m in mats])
    fit_h, fit_a = fit_lambdas(probs[:, 0], probs[:, 2], over)
    assert np.allclose(fit_h, lh, atol=1e-4) and np.allclose(fit_a, la, atol=1e-4)


def test_rescale_to_outcomes():
    target = np.array([[0.5, 0.3, 0.2]])
    mats = rescale_to_outcomes(score_matrix([1.0], [1.5]), target)
    assert np.allclose(outcome_probs(mats), target)


# ------------------------------------------------------------------- Kicktipp

@pytest.mark.parametrize("tip, result, expected", [
    ((2, 1), (2, 1), 4),  # Ergebnis
    ((2, 1), (3, 2), 3),  # Tordifferenz
    ((2, 1), (3, 0), 2),  # Tendenz
    ((2, 1), (1, 1), 0),
    ((1, 1), (1, 1), 4),
    ((1, 1), (2, 2), 2),  # Remis: in der Standardwertung keine Tordifferenz-Punkte
    ((0, 2), (1, 3), 3),
])
def test_points(tip, result, expected):
    assert points(*tip, *result) == expected


def test_points_draw_rule_configurable():
    assert points(1, 1, 2, 2, Rules(goal_diff_for_draws=True)) == 3


def test_expected_points_by_hand():
    mats = np.zeros((1, 11, 11))
    mats[0, 2, 1], mats[0, 1, 0], mats[0, 1, 1] = 0.5, 0.3, 0.2
    ev = expected_points(mats)
    # Tipp 2:1: 0,5*4 + 0,3*3 + 0,2*0 = 2,9
    assert ev[0, 2, 1] == pytest.approx(2.9)
    tip_h, tip_a, best = best_tips(mats)
    assert (tip_h[0], tip_a[0]) == (2, 1) and best[0] == pytest.approx(2.9)


def test_best_tip_for_clear_favourite():
    tip_h, tip_a, _ = best_tips(score_matrix([3.0], [0.5]))
    assert tip_h[0] > tip_a[0]


# -------------------------------------------------------------------- Metriken

def test_metrics():
    uniform = np.full((4, 3), 1 / 3)
    results = ["H", "D", "A", "H"]
    assert log_loss(uniform, results) == pytest.approx(np.log(3))
    perfect = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]], float)
    assert rps(perfect, results) == pytest.approx(0)
    assert accuracy(perfect, results) == 1
    # Ein Remis-Tipp ist bei Auswärtssieg weniger falsch als ein Heimsieg-Tipp
    assert rps(np.array([[0, 1, 0.]]), ["A"]) < rps(np.array([[1, 0, 0.]]), ["A"])
