"""Ergebnisse und Quoten von football-data.co.uk laden und vereinheitlichen.

Pro Saison gibt es eine CSV-Datei. Abgeschlossene Saisons werden einmal geladen
und in data/raw/ zwischengespeichert, die laufende Saison bei jedem Update neu.
Das Ergebnis ist eine einzige Tabelle data/matches.csv mit diesen Spalten:

season      Startjahr der Saison (2025 = 2025/26)
date        Spieltag als Datum
home, away  Vereinsnamen (football-data-Schreibweise)
hg, ag      Tore Heim / Gast
result      H, D oder A
odds_h/d/a  Durchschnittsquote Heimsieg / Remis / Auswärtssieg
odds_over, odds_under   Durchschnittsquote über / unter 2,5 Tore
"""

from datetime import date
from io import StringIO

import pandas as pd
import requests

from .config import FIRST_SEASON, MATCHES_FILE, RAW_DIR, USER_AGENT
from .teams import canonical

BASE_URL = "https://www.football-data.co.uk/mmz4281/{code}/D1.csv"

# Je Wert die Spaltenvarianten in Prioritätsreihenfolge. Bis 2018/19 heißen die
# Durchschnittsquoten "BbAv…", ab 2019/20 "Avg…". Bet365 ist die Rückfalloption.
ODDS_COLUMNS = {
    "odds_h": ("AvgH", "BbAvH", "B365H"),
    "odds_d": ("AvgD", "BbAvD", "B365D"),
    "odds_a": ("AvgA", "BbAvA", "B365A"),
    "odds_over": ("Avg>2.5", "BbAv>2.5", "B365>2.5"),
    "odds_under": ("Avg<2.5", "BbAv<2.5", "B365<2.5"),
}

COLUMNS = ["season", "date", "home", "away", "hg", "ag", "result", *ODDS_COLUMNS]


def season_code(start_year: int) -> str:
    """2025 -> '2526'"""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def season_label(start_year: int) -> str:
    """2025 -> '2025/26'"""
    return f"{start_year}/{(start_year + 1) % 100:02d}"


def current_season(today: date | None = None) -> int:
    """Startjahr der Saison, in der `today` liegt (Saisonwechsel im Juli)."""
    today = today or date.today()
    return today.year if today.month >= 7 else today.year - 1


def _download(url: str) -> str:
    response = requests.get(url, timeout=30, headers={"User-Agent": USER_AGENT})
    response.raise_for_status()
    text = response.content.decode("utf-8-sig", errors="replace")
    if not text.lstrip().startswith("Div"):
        raise ValueError(f"Unerwartete Antwort von {url}")
    return text


def raw_path(start_year: int):
    return RAW_DIR / f"D1_{season_code(start_year)}.csv"


def fetch_season(start_year: int, force: bool = False) -> str:
    """Rohdaten einer Saison, aus dem Cache oder frisch geladen."""
    path = raw_path(start_year)
    if path.exists() and not force:
        return path.read_text(encoding="utf-8")
    text = _download(BASE_URL.format(code=season_code(start_year)))
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text


def _first_available(raw: pd.DataFrame, candidates: tuple) -> pd.Series:
    out = pd.Series(float("nan"), index=raw.index)
    for col in candidates:
        if col in raw.columns:
            out = out.fillna(pd.to_numeric(raw[col], errors="coerce"))
    return out


def parse_season(text: str, start_year: int) -> pd.DataFrame:
    """Wandelt eine football-data-CSV in das einheitliche Format um."""
    raw = pd.read_csv(StringIO(text), on_bad_lines="skip")
    raw.columns = [c.strip().lstrip("﻿") for c in raw.columns]
    raw = raw.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])

    df = pd.DataFrame({
        "season": start_year,
        "date": pd.to_datetime(raw["Date"], dayfirst=True, format="mixed"),
        "home": raw["HomeTeam"].map(canonical),
        "away": raw["AwayTeam"].map(canonical),
        "hg": raw["FTHG"].astype(int),
        "ag": raw["FTAG"].astype(int),
    })
    df["result"] = "D"
    df.loc[df["hg"] > df["ag"], "result"] = "H"
    df.loc[df["hg"] < df["ag"], "result"] = "A"
    for name, candidates in ODDS_COLUMNS.items():
        df[name] = _first_available(raw, candidates)
    return df[COLUMNS]


def update(first: int = FIRST_SEASON, last: int | None = None, verbose: bool = True) -> pd.DataFrame:
    """Lädt alle Saisons von `first` bis `last` und schreibt data/matches.csv."""
    last = current_season() if last is None else last
    frames = []
    for year in range(first, last + 1):
        is_current = year == current_season()
        try:
            text = fetch_season(year, force=is_current)
        except requests.HTTPError:
            if is_current:  # Neue Saison hat noch kein Spiel -> noch keine Datei
                continue
            raise
        df = parse_season(text, year)
        frames.append(df)
        if verbose:
            print(f"  {season_label(year)}: {len(df):3d} Spiele")

    matches = pd.concat(frames, ignore_index=True)
    matches = matches.sort_values(["date", "home"]).reset_index(drop=True)
    MATCHES_FILE.parent.mkdir(parents=True, exist_ok=True)
    matches.to_csv(MATCHES_FILE, index=False)
    if verbose:
        print(f"  -> {len(matches)} Spiele in {MATCHES_FILE.relative_to(MATCHES_FILE.parents[1])}")
    return matches


def load_matches() -> pd.DataFrame:
    """Lädt data/matches.csv (und erzeugt sie beim ersten Mal)."""
    if not MATCHES_FILE.exists():
        print("Noch keine Daten vorhanden, lade sie herunter …")
        return update()
    df = pd.read_csv(MATCHES_FILE, parse_dates=["date"])
    return df


def season_teams(matches: pd.DataFrame) -> dict[int, set[str]]:
    """Welche Vereine in welcher Saison Bundesliga gespielt haben."""
    return {
        int(season): set(group["home"]) | set(group["away"])
        for season, group in matches.groupby("season")
    }


def transitions(teams_by_season: dict[int, set[str]], season: int) -> tuple[set[str], set[str]]:
    """Aufsteiger und Absteiger beim Wechsel in `season`."""
    current = teams_by_season.get(season, set())
    previous = teams_by_season.get(season - 1)
    if previous is None:
        return set(), set()
    return current - previous, previous - current
