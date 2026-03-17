import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime


def get_next_matchday() -> list:
    """
    Scrapt den nächsten Bundesliga Spieltag von kicker.de
    Gibt Liste von (HomeTeam, AwayTeam, Datum) zurück.
    """

    # Teamnamen-Mapping: kicker Namen → unsere Datenbasis Namen
    TEAM_MAPPING = {
        "Bayern München":       "Bayern Munich",
        "Borussia Dortmund":    "Dortmund",
        "Bayer Leverkusen":     "Leverkusen",
        "RB Leipzig":           "RB Leipzig",
        "Eintracht Frankfurt":  "Ein Frankfurt",
        "VfB Stuttgart":        "Stuttgart",
        "SC Freiburg":          "Freiburg",
        "Union Berlin":         "Union Berlin",
        "Werder Bremen":        "Werder Bremen",
        "VfL Wolfsburg":        "Wolfsburg",
        "Borussia Mönchengladbach": "M'gladbach",
        "TSG Hoffenheim":       "Hoffenheim",
        "FC Augsburg":          "Augsburg",
        "1. FC Köln":           "FC Koln",
        "1. FSV Mainz 05":      "Mainz",
        "1. FC Heidenheim":     "Heidenheim",
        "FC St. Pauli":         "St Pauli",
        "Hamburger SV":         "Hamburg",
        "Holstein Kiel":        "Holstein Kiel",
        "FC St. Pauli 1910":    "St Pauli",
        "SV Werder Bremen":     "Werder Bremen",
        "TSG Hoffenheim":           "Hoffenheim",
    }

    url = "https://www.kicker.de/bundesliga/spieltag"
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except Exception as e:
        print(f"⚠️  Scraping fehlgeschlagen: {e}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    matches = []

    # Spiele finden
    game_rows = soup.find_all("div", class_=lambda c: c and "kick__v100-gameList__gameRow" in c)

    current_date = datetime.today().strftime("%Y-%m-%d")

    for row in game_rows:
        try:
            # Teams
            teams = row.find_all("div", class_=lambda c: c and "kick__v100-gameCell__team__shortname" in c)
            if len(teams) < 2:
                teams = row.find_all("span", class_=lambda c: c and "teamname" in c)
            if len(teams) < 2:
                continue

            home_raw = teams[0].get_text(strip=True)
            away_raw = teams[1].get_text(strip=True)

            home = TEAM_MAPPING.get(home_raw, home_raw)
            away = TEAM_MAPPING.get(away_raw, away_raw)

            # Datum
            date_tag = row.find("span", class_=lambda c: c and "kick__v100-gameList__gameRow__gameTime" in c)
            date_str = current_date
            if date_tag:
                date_str = date_tag.get_text(strip=True)

            matches.append((home, away, date_str))

        except Exception:
            continue

    return matches


def get_next_matchday_manual(matchday_number: int, season: str = "2025-26") -> list:
    """
    Fallback: Holt Spieltag von openligadb.de (zuverlässige kostenlose API)
    """
    TEAM_MAPPING = {
        "FC Bayern München":        "Bayern Munich",
        "Borussia Dortmund":        "Dortmund",
        "Bayer 04 Leverkusen":      "Leverkusen",
        "RB Leipzig":               "RB Leipzig",
        "Eintracht Frankfurt":      "Ein Frankfurt",
        "VfB Stuttgart":            "Stuttgart",
        "Sport-Club Freiburg":      "Freiburg",
        "1. FC Union Berlin":       "Union Berlin",
        "Werder Bremen":            "Werder Bremen",
        "VfL Wolfsburg":            "Wolfsburg",
        "Borussia Mönchengladbach": "M'gladbach",
        "TSG 1899 Hoffenheim":      "Hoffenheim",
        "FC Augsburg":              "Augsburg",
        "1. FC Köln":               "FC Koln",
        "1. FSV Mainz 05":          "Mainz",
        "1. FC Heidenheim 1846":    "Heidenheim",
        "FC St. Pauli":             "St Pauli",
        "Hamburger SV":             "Hamburg",
        "Holstein Kiel":            "Holstein Kiel",
        "SV Werder Bremen":         "Werder Bremen",
        "TSG Hoffenheim":           "Hoffenheim",
        "SC Freiburg":              "Freiburg",
    }

    url = f"https://api.openligadb.de/getmatchdata/bl1/2025/{matchday_number}"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"  API Fehler: {e}")
        return []

    matches = []
    for game in data:
        home_raw = game["team1"]["teamName"]
        away_raw = game["team2"]["teamName"]
        date_raw = game["matchDateTime"][:10]  # "2026-02-28T18:30:00" → "2026-02-28"

        home = TEAM_MAPPING.get(home_raw, home_raw)
        away = TEAM_MAPPING.get(away_raw, away_raw)

        matches.append((home, away, date_raw))
        print(f"  {home} vs {away}  |  {date_raw}")

    return matches


if __name__ == "__main__":
    print("Hole nächsten Spieltag von openligadb.de...")
    # Spieltagnummer anpassen
    matches = get_next_matchday_manual(matchday_number=25)

    if matches:
        print(f"\n {len(matches)} Spiele gefunden")
    else:
        print(" Keine Spiele gefunden")