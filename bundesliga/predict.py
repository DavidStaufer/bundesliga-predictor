"""Vorhersage für einen Spieltag, inklusive Kicktipp-Tipp."""

import json

import numpy as np
import pandas as pd

from . import hyperparams
from .config import ENSEMBLE_FILE, FIRST_SEASON, PREDICTIONS_DIR
from .data import current_season, season_label, season_teams, transitions
from .fixtures import current_matchday, get_matchday, get_upcoming_odds
from .kicktipp import DEFAULT_RULES, best_tips, points
from .models.dixon_coles import DixonColes
from .models.elo import EloGoalModel, compute_elo
from .models.ensemble import pool
from .models.market import market_matrices
from .scores import GOALS, outcome_probs

MODEL_CHOICES = ("auto", "own", "market", "elo", "dc")
WEEKDAYS = ("Mo", "Di", "Mi", "Do", "Fr", "Sa", "So")

DEFAULT_WEIGHTS = {"own": {"elo": 0.5, "dc": 0.5}, "combined": {"elo": 0.0, "dc": 0.0, "market": 1.0}}


def _load_weights() -> dict:
    if ENSEMBLE_FILE.exists():
        return json.loads(ENSEMBLE_FILE.read_text())["weights"]
    print("Hinweis: Noch kein Backtest gelaufen, nutze Standardgewichte "
          "(python -m bundesliga backtest).")
    return DEFAULT_WEIGHTS


def _attach_odds(fixtures: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Quoten aus den Ergebnisdaten (schon gespielt) oder der Live-Datei (anstehend)."""
    odds_cols = ["odds_h", "odds_d", "odds_a", "odds_over", "odds_under"]
    played = matches[matches["season"] == fixtures["season"].iloc[0]][["home", "away", *odds_cols]]
    upcoming = get_upcoming_odds()[["home", "away", *odds_cols]]
    known = pd.concat([upcoming, played]).drop_duplicates(["home", "away"], keep="first")
    return fixtures.merge(known, on=["home", "away"], how="left")


def _missing_results(history: pd.DataFrame, season: int, matchday: int) -> int:
    """Wie viele Ergebnisse des Vorspieltags football-data.co.uk noch nicht hat."""
    if matchday <= 1:
        return 0
    previous = get_matchday(season, matchday - 1)
    finished = previous.dropna(subset=["hg"])
    known = set(zip(history.loc[history["season"] == season, "home"],
                    history.loc[history["season"] == season, "away"]))
    return sum((h, a) not in known for h, a in zip(finished["home"], finished["away"]))


def predict_matchday(matches: pd.DataFrame, season: int | None = None,
                     matchday: int | None = None, model: str = "auto",
                     rules=DEFAULT_RULES) -> pd.DataFrame:
    season = current_season() if season is None else season
    matchday = current_matchday() if matchday is None else matchday
    fixtures = get_matchday(season, matchday)
    cutoff = fixtures["date"].min()
    history = matches[matches["date"] < cutoff].reset_index(drop=True)

    missing = _missing_results(history, season, matchday)
    if missing:
        print(f"Achtung: {missing} Ergebnis(se) von Spieltag {matchday - 1} fehlen noch in den "
              f"Daten von football-data.co.uk. Die Vorhersage kennt sie nicht.")

    teams_by_season = season_teams(history)
    teams_by_season[season] = (teams_by_season.get(season, set())
                               | set(fixtures["home"]) | set(fixtures["away"]))
    elo_params, dc_params = hyperparams.load()

    # Elo
    pre_h, pre_a, ratings = compute_elo(history, teams_by_season, elo_params)
    past = (history["season"] > FIRST_SEASON).to_numpy()
    goal_model = EloGoalModel().fit((pre_h - pre_a)[past], history["hg"][past], history["ag"][past])
    elo_diff = np.array([ratings[h] - ratings[a] for h, a in zip(fixtures["home"], fixtures["away"])])
    mats = {"elo": goal_model.predict(elo_diff)}

    # Dixon-Coles
    promoted, relegated = transitions(teams_by_season, season)
    dc = DixonColes(dc_params).fit(history, cutoff, teams=teams_by_season[season],
                                   promoted=promoted, relegated=relegated)
    mats["dc"] = dc.predict(fixtures["home"], fixtures["away"])

    # Quoten
    fixtures = _attach_odds(fixtures, matches)
    mats["market"], has_odds = market_matrices(fixtures)

    weights = _load_weights()
    mats["own"] = pool([mats[m] for m in weights["own"]], list(weights["own"].values()))
    combined = pool([mats[m] for m in weights["combined"]], list(weights["combined"].values()))
    mats["combined"] = np.where(has_odds[:, None, None], combined, mats["own"])

    if model == "auto":
        chosen = mats["combined"]
        source = np.where(has_odds, "Quoten", "Modell")
    elif model == "market" and not has_odds.all():
        raise ValueError("Für diesen Spieltag gibt es (noch) nicht für alle Spiele Quoten.")
    else:
        chosen = mats[model]
        source = np.full(len(fixtures), {"own": "Modell", "market": "Quoten", "elo": "Elo",
                                          "dc": "Dixon-Coles"}[model])

    probs = outcome_probs(chosen)
    tip_h, tip_a, exp_pts = best_tips(chosen, rules)
    xg_h = (chosen.sum(axis=2) * GOALS).sum(axis=1)
    xg_a = (chosen.sum(axis=1) * GOALS).sum(axis=1)

    out = pd.DataFrame({
        "Anstoß": [f"{WEEKDAYS[k.weekday()]} {k:%d.%m. %H:%M}" for k in fixtures["kickoff"]],
        "Heim": fixtures["home"],
        "Gast": fixtures["away"],
        "1": probs[:, 0],
        "X": probs[:, 1],
        "2": probs[:, 2],
        "xG": [f"{h:.1f}:{a:.1f}" for h, a in zip(xg_h, xg_a)],
        "Tipp": [f"{h}:{a}" for h, a in zip(tip_h, tip_a)],
        "erw. Punkte": exp_pts,
        "Basis": source,
    })
    finished = fixtures["hg"].notna().to_numpy()
    if finished.any():
        out["Ergebnis"] = [f"{int(h)}:{int(a)}" if ok else "" for h, a, ok
                           in zip(fixtures["hg"], fixtures["ag"], finished)]
        pts = points(tip_h, tip_a, fixtures["hg"].fillna(-1).astype(int), fixtures["ag"].fillna(-1).astype(int), rules)
        out["Punkte"] = np.where(finished, pts, 0)

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = PREDICTIONS_DIR / f"{season}-{(season + 1) % 100:02d}_spieltag_{matchday:02d}.csv"
    out.to_csv(path, index=False)
    out.attrs.update(season=season, matchday=matchday, path=path,
                     odds_share=float(has_odds.mean()))
    return out


def print_prediction(df: pd.DataFrame) -> None:
    season, matchday = df.attrs["season"], df.attrs["matchday"]
    shown = df.copy()
    for col in ("1", "X", "2"):
        shown[col] = (shown[col] * 100).round().astype(int).astype(str) + " %"
    shown["erw. Punkte"] = shown["erw. Punkte"].round(2)
    print(f"\nBundesliga {season_label(season)}, {matchday}. Spieltag\n")
    print(shown.to_string(index=False))
    if df.attrs["odds_share"] < 1:
        print("\nOhne Quoten berechnet (Basis = Modell): football-data.co.uk veröffentlicht die "
              "Quoten meist dienstags und freitags. Später erneut ausführen für die bessere Variante.")
    if "Punkte" in df:
        print(f"\nKicktipp-Punkte (gespielte Partien): {int(df['Punkte'].sum())}")
    print(f"\nGespeichert: {df.attrs['path'].relative_to(df.attrs['path'].parents[1])}")
