"""Spielplan (OpenLigaDB) und aktuelle Quoten (football-data.co.uk)."""

from io import StringIO

import pandas as pd
import requests

from .config import USER_AGENT
from .data import ODDS_COLUMNS, _first_available
from .teams import canonical

OPENLIGA_URL = "https://api.openligadb.de"
ODDS_FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"


def _get_json(url: str):
    response = requests.get(url, timeout=20, headers={"User-Agent": USER_AGENT})
    response.raise_for_status()
    return response.json()


def current_matchday() -> int:
    """Der Spieltag, den OpenLigaDB gerade als aktuell führt."""
    return int(_get_json(f"{OPENLIGA_URL}/getcurrentgroup/bl1")["groupOrderID"])


def get_matchday(season: int, matchday: int) -> pd.DataFrame:
    """Alle Spiele eines Spieltags, inklusive Ergebnis, falls schon gespielt."""
    games = _get_json(f"{OPENLIGA_URL}/getmatchdata/bl1/{season}/{matchday}")
    rows = []
    for game in games:
        final = next(
            (r for r in game.get("matchResults") or [] if r.get("resultTypeID") == 2),
            None,
        )
        finished = bool(game.get("matchIsFinished")) and final is not None
        rows.append({
            "season": season,
            "matchday": matchday,
            "kickoff": pd.Timestamp(game["matchDateTime"]),
            "home": canonical(game["team1"]["teamName"]),
            "away": canonical(game["team2"]["teamName"]),
            "hg": final["pointsTeam1"] if finished else None,
            "ag": final["pointsTeam2"] if finished else None,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"OpenLigaDB kennt keine Spiele für Saison {season}, Spieltag {matchday}.")
    df["date"] = df["kickoff"].dt.normalize()
    return df.sort_values("kickoff").reset_index(drop=True)


def get_upcoming_odds() -> pd.DataFrame:
    """Quoten der anstehenden Bundesliga-Spiele.

    football-data.co.uk aktualisiert die Datei meist dienstags und freitags.
    Fehlen die Quoten (noch), kommt eine leere Tabelle zurück.
    """
    try:
        response = requests.get(ODDS_FIXTURES_URL, timeout=20, headers={"User-Agent": USER_AGENT})
        response.raise_for_status()
    except requests.RequestException:
        return pd.DataFrame(columns=["date", "home", "away", *ODDS_COLUMNS])
    raw = pd.read_csv(StringIO(response.content.decode("utf-8-sig", errors="replace")),
                      on_bad_lines="skip")
    raw.columns = [c.strip().lstrip("﻿") for c in raw.columns]
    raw = raw[raw["Div"] == "D1"]
    df = pd.DataFrame({
        "date": pd.to_datetime(raw["Date"], dayfirst=True, format="mixed"),
        "home": raw["HomeTeam"].map(canonical),
        "away": raw["AwayTeam"].map(canonical),
    })
    for name, candidates in ODDS_COLUMNS.items():
        df[name] = _first_available(raw, candidates)
    return df.reset_index(drop=True)
