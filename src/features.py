import pandas as pd
import numpy as np


import pandas as pd
import numpy as np


def calculate_form(df: pd.DataFrame, team: str, date, last_n: int = 5) -> dict:
    date = pd.to_datetime(date)

    home_games = df[(df["HomeTeam"] == team) & (pd.to_datetime(df["Date"], dayfirst=True, format="mixed") < date)].copy()
    away_games = df[(df["AwayTeam"] == team) & (pd.to_datetime(df["Date"], dayfirst=True, format="mixed") < date)].copy()

    home_games["points"] = home_games["Result"].map({"H": 3, "D": 1, "A": 0})
    home_games["gf"]     = home_games["HomeGoals"]
    home_games["ga"]     = home_games["AwayGoals"]

    away_games["points"] = away_games["Result"].map({"A": 3, "D": 1, "H": 0})
    away_games["gf"]     = away_games["AwayGoals"]
    away_games["ga"]     = away_games["HomeGoals"]

    all_games = pd.concat([
        home_games[["Date", "points", "gf", "ga"]],
        away_games[["Date", "points", "gf", "ga"]]
    ]).sort_values("Date", ascending=False)

    last_n_games = all_games.head(last_n)

    # Trend
    trend = 0
    if len(all_games) >= 6:
        recent = all_games.head(3)["points"].mean()
        older  = all_games.iloc[3:6]["points"].mean()
        trend  = round(recent - older, 2)

    # Heimstärke letzte 5 Heimspiele
    last_home = home_games.sort_values("Date", ascending=False).head(5)
    home_ppg  = last_home["points"].mean() if len(last_home) > 0 else 0
    home_gf   = last_home["gf"].mean()     if len(last_home) > 0 else 0
    home_ga   = last_home["ga"].mean()     if len(last_home) > 0 else 0

    # Auswärtsstärke letzte 5 Auswärtsspiele
    last_away = away_games.sort_values("Date", ascending=False).head(5)
    away_ppg  = last_away["points"].mean() if len(last_away) > 0 else 0
    away_gf   = last_away["gf"].mean()     if len(last_away) > 0 else 0
    away_ga   = last_away["ga"].mean()     if len(last_away) > 0 else 0

    # Tage seit letztem Spiel
    if len(all_games) > 0:
        last_date = pd.to_datetime(all_games.iloc[0]["Date"], dayfirst=True, format="mixed")
        days_rest = (date - last_date).days
    else:
        days_rest = 14

    # Stabilitätsindex
    goal_variance = float(all_games["gf"].var()) if len(all_games) > 1 else 0.0

    # Ergebnisserie
    streak = 0
    for pts in all_games["points"].values:
        if pts == 3:
            streak += 1
        elif pts == 0:
            streak -= 1
        else:
            break

    # Auswärtsserie
    away_streak = 0
    for pts in last_away["points"].values:
        if pts == 3:
            away_streak += 1
        elif pts == 0:
            away_streak -= 1
        else:
            break

    # Heimserie
    home_streak = 0
    for pts in last_home["points"].values:
        if pts == 3:
            home_streak += 1
        elif pts == 0:
            home_streak -= 1
        else:
            break

    return {
        "points":        last_n_games["points"].sum()                            if len(last_n_games) > 0 else 0,
        "gf":            last_n_games["gf"].sum()                                if len(last_n_games) > 0 else 0,
        "ga":            last_n_games["ga"].sum()                                if len(last_n_games) > 0 else 0,
        "gd":            last_n_games["gf"].sum() - last_n_games["ga"].sum()     if len(last_n_games) > 0 else 0,
        "trend":         trend,
        "home_ppg":      home_ppg,
        "home_gf":       home_gf,
        "home_ga":       home_ga,
        "away_ppg":      away_ppg,
        "away_gf":       away_gf,
        "away_ga":       away_ga,
        "days_rest":     days_rest,
        "goal_variance": goal_variance,
        "streak":        streak,
        "away_streak":   away_streak,
        "home_streak":   home_streak,
    }


def calculate_h2h(df: pd.DataFrame, home_team: str, away_team: str, date, last_n: int = 5) -> dict:
    date = pd.to_datetime(date)

    h2h = df[
        (((df["HomeTeam"] == home_team) & (df["AwayTeam"] == away_team)) |
         ((df["HomeTeam"] == away_team) & (df["AwayTeam"] == home_team))) &
        (pd.to_datetime(df["Date"], dayfirst=True, format="mixed") < date)
    ].sort_values("Date", ascending=False).head(last_n)

    if len(h2h) == 0:
        return {"h2h_home_wins": 0, "h2h_draws": 0, "h2h_away_wins": 0}

    home_wins = len(h2h[(h2h["HomeTeam"] == home_team) & (h2h["Result"] == "H")]) + \
                len(h2h[(h2h["HomeTeam"] == away_team) & (h2h["Result"] == "A")])
    away_wins = len(h2h[(h2h["HomeTeam"] == away_team) & (h2h["Result"] == "H")]) + \
                len(h2h[(h2h["HomeTeam"] == home_team) & (h2h["Result"] == "A")])
    draws     = len(h2h[h2h["Result"] == "D"])

    return {"h2h_home_wins": home_wins, "h2h_draws": draws, "h2h_away_wins": away_wins}


def calculate_season_position(df: pd.DataFrame, team: str, date, season) -> dict:
    date = pd.to_datetime(date)

    season_games = df[
        (df["Season"].astype(str) == str(season)) &
        (pd.to_datetime(df["Date"], dayfirst=True, format="mixed") < date)
    ].copy()

    if len(season_games) == 0:
        return {"season_points": 0, "season_gd": 0, "season_position": 10}

    table = {}
    for _, row in season_games.iterrows():
        ht, at = row["HomeTeam"], row["AwayTeam"]
        hg, ag = row["HomeGoals"],  row["AwayGoals"]
        res    = row["Result"]

        for t in [ht, at]:
            if t not in table:
                table[t] = {"points": 0, "gd": 0}

        if res == "H":
            table[ht]["points"] += 3
        elif res == "D":
            table[ht]["points"] += 1
            table[at]["points"] += 1
        else:
            table[at]["points"] += 3

        table[ht]["gd"] += (hg - ag)
        table[at]["gd"] += (ag - hg)

    if team not in table:
        return {"season_points": 0, "season_gd": 0, "season_position": 10}

    sorted_table = sorted(table.items(), key=lambda x: (x[1]["points"], x[1]["gd"]), reverse=True)
    position     = next((i + 1 for i, (t, _) in enumerate(sorted_table) if t == team), 10)

    return {
        "season_points":   table[team]["points"],
        "season_gd":       table[team]["gd"],
        "season_position": position
    }


def get_season_phase(date) -> int:
    date  = pd.to_datetime(date)
    month = date.month
    return 1 if month in [8, 9, 10, 11, 12] else 2


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed")
    df = df.sort_values("Date").reset_index(drop=True)

    rows  = []
    total = len(df)

    for i, row in df.iterrows():
        if i % 100 == 0:
            print(f"  Verarbeite Spiel {i}/{total}...")

        date      = row["Date"]
        home_team = row["HomeTeam"]
        away_team = row["AwayTeam"]
        season    = row["Season"]

        home_form = calculate_form(df, home_team, date)
        away_form = calculate_form(df, away_team, date)
        home_pos  = calculate_season_position(df, home_team, date, season)
        away_pos  = calculate_season_position(df, away_team, date, season)
        h2h       = calculate_h2h(df, home_team, away_team, date)

        rows.append({
            "Date":     date,
            "Season":   season,
            "HomeTeam": home_team,
            "AwayTeam": away_team,
            "Result":   row["Result"],

            # Gesamtform
            "home_form_points": home_form["points"],
            "home_form_gf":     home_form["gf"],
            "home_form_ga":     home_form["ga"],
            "home_form_gd":     home_form["gd"],
            "away_form_points": away_form["points"],
            "away_form_gf":     away_form["gf"],
            "away_form_ga":     away_form["ga"],
            "away_form_gd":     away_form["gd"],

            # Differenzen
            "form_points_diff": home_form["points"] - away_form["points"],
            "form_gd_diff":     home_form["gd"]     - away_form["gd"],

            # Trend
            "home_trend":       home_form["trend"],
            "away_trend":       away_form["trend"],
            "trend_diff":       home_form["trend"]  - away_form["trend"],

            # Heimstärke / Auswärtsstärke
            "home_home_ppg":    home_form["home_ppg"],
            "home_home_gf":     home_form["home_gf"],
            "home_home_ga":     home_form["home_ga"],
            "away_away_ppg":    away_form["away_ppg"],
            "away_away_gf":     away_form["away_gf"],
            "away_away_ga":     away_form["away_ga"],
            "home_advantage":   home_form["home_ppg"] - away_form["away_ppg"],

            # Erholung
            "home_days_rest":   home_form["days_rest"],
            "away_days_rest":   away_form["days_rest"],
            "rest_diff":        home_form["days_rest"] - away_form["days_rest"],

            # Tabellenposition
            "home_season_points":   home_pos["season_points"],
            "home_season_gd":       home_pos["season_gd"],
            "home_season_position": home_pos["season_position"],
            "away_season_points":   away_pos["season_points"],
            "away_season_gd":       away_pos["season_gd"],
            "away_season_position": away_pos["season_position"],
            "position_diff":        away_pos["season_position"] - home_pos["season_position"],
            "points_diff_season":   home_pos["season_points"]  - away_pos["season_points"],

            # H2H
            "h2h_home_wins": h2h["h2h_home_wins"],
            "h2h_draws":     h2h["h2h_draws"],
            "h2h_away_wins": h2h["h2h_away_wins"],

            # Saisonphase
            "season_phase": get_season_phase(date),

            # Stabilitätsindex
            "home_goal_variance": home_form["goal_variance"],
            "away_goal_variance": away_form["goal_variance"],

            # Ergebnisserien
            "home_streak":      home_form["streak"],
            "away_streak":      away_form["streak"],
            "streak_diff":      home_form["streak"] - away_form["streak"],
            "home_home_streak": home_form["home_streak"],
            "away_away_streak": away_form["away_streak"],
        })

    features_df = pd.DataFrame(rows)
    features_df.to_csv("data/bundesliga_features.csv", index=False)
    print(f"\nFeatures gespeichert: data/bundesliga_features.csv")
    print(f"   Shape: {features_df.shape}")
    return features_df


if __name__ == "__main__":
    df = pd.read_csv("data/bundesliga_historical.csv")
    print("Baue erweiterte Features...")
    features = build_features(df)
    print("\nFeature-Spalten:", list(features.columns))