import pandas as pd
import numpy as np
import joblib
import sys
sys.path.append(".")
from src.features import calculate_form, calculate_h2h, calculate_season_position, get_season_phase
from src.model import FEATURE_COLS
from datetime import datetime


class BundesligaPredictor:

    def __init__(self):
        print("Lade Modell und Daten")
        self.model      = joblib.load("models/best_model.pkl")
        self.scaler     = joblib.load("models/scaler.pkl")
        self.le         = joblib.load("models/label_encoder.pkl")
        self.threshold  = joblib.load("models/best_threshold.pkl")
        self.draw_idx   = joblib.load("models/draw_idx.pkl")
        self.class_list = joblib.load("models/class_list.pkl")

        train       = pd.read_csv("data/bundesliga_historical.csv")
        current     = pd.read_csv("data/bundesliga_2526_raw.csv")
        self.df     = pd.concat([train, current], ignore_index=True)
        self.df["Date"] = pd.to_datetime(self.df["Date"], dayfirst=True)
        self.teams  = sorted(self.df["HomeTeam"].unique())
        print(f"Bereit! {len(self.teams)} Teams geladen.\n")

    def _build_features(self, home_team, away_team, date):
        home_form = calculate_form(self.df, home_team, date)
        away_form = calculate_form(self.df, away_team, date)
        h2h       = calculate_h2h(self.df, home_team, away_team, date)
        home_pos  = calculate_season_position(self.df, home_team, date, 2526)
        away_pos  = calculate_season_position(self.df, away_team, date, 2526)

        X = pd.DataFrame([{
            "home_form_points": home_form["points"],
            "home_form_gf":     home_form["gf"],
            "home_form_ga":     home_form["ga"],
            "home_form_gd":     home_form["gd"],
            "away_form_points": away_form["points"],
            "away_form_gf":     away_form["gf"],
            "away_form_ga":     away_form["ga"],
            "away_form_gd":     away_form["gd"],
            "form_points_diff": home_form["points"] - away_form["points"],
            "form_gd_diff":     home_form["gd"]     - away_form["gd"],
            "home_trend":       home_form["trend"],
            "away_trend":       away_form["trend"],
            "trend_diff":       home_form["trend"]  - away_form["trend"],
            "home_home_ppg":    home_form["home_ppg"],
            "home_home_gf":     home_form["home_gf"],
            "home_home_ga":     home_form["home_ga"],
            "away_away_ppg":    away_form["away_ppg"],
            "away_away_gf":     away_form["away_gf"],
            "away_away_ga":     away_form["away_ga"],
            "home_advantage":   home_form["home_ppg"] - away_form["away_ppg"],
            "home_days_rest":   home_form["days_rest"],
            "away_days_rest":   away_form["days_rest"],
            "rest_diff":        home_form["days_rest"] - away_form["days_rest"],
            "home_season_points":   home_pos["season_points"],
            "home_season_gd":       home_pos["season_gd"],
            "home_season_position": home_pos["season_position"],
            "away_season_points":   away_pos["season_points"],
            "away_season_gd":       away_pos["season_gd"],
            "away_season_position": away_pos["season_position"],
            "position_diff":        away_pos["season_position"] - home_pos["season_position"],
            "points_diff_season":   home_pos["season_points"]  - away_pos["season_points"],
            "h2h_home_wins": h2h["h2h_home_wins"],
            "h2h_draws":     h2h["h2h_draws"],
            "h2h_away_wins": h2h["h2h_away_wins"],
            "home_goal_variance": home_form["goal_variance"],
            "away_goal_variance": away_form["goal_variance"],
            "home_streak":        home_form["streak"],
            "away_streak":        away_form["streak"],
            "streak_diff":        home_form["streak"] - away_form["streak"],
            "home_home_streak":   home_form["home_streak"],
            "away_away_streak":   away_form["away_streak"],
            "season_phase":  get_season_phase(date),
        }])[FEATURE_COLS]

        return X, home_form, away_form, home_pos, away_pos, h2h

    def predict(self, home_team, away_team, date=None):
        if date is None:
            date = datetime.today().strftime("%Y-%m-%d")

        if home_team not in self.teams:
            print(f"'{home_team}' nicht gefunden.")
            return None
        if away_team not in self.teams:
            print(f"'{away_team}' nicht gefunden.")
            return None

        X, home_form, away_form, home_pos, away_pos, h2h = self._build_features(
            home_team, away_team, date
        )

        X_scaled = self.scaler.transform(X)
        proba    = self.model.predict_proba(X_scaled)[0]

        prob_dict = {}
        for i, cls in enumerate(self.class_list):
            label = self.le.inverse_transform([cls])[0] if isinstance(cls, (int, np.integer)) else cls
            prob_dict[label] = proba[i]

        home_prob = prob_dict.get("H", 0)
        draw_prob = prob_dict.get("D", 0)
        away_prob = prob_dict.get("A", 0)

        if draw_prob >= self.threshold:
            prediction = "Unentschieden"
        elif home_prob >= away_prob:
            prediction = f"{home_team} gewinnt"
        else:
            prediction = f"{away_team} gewinnt"

        print("\n" + "=" * 55)
        print(f"  {home_team} vs {away_team}  |  {date}")
        print("=" * 55)
        print(f"  {home_team} gewinnt:   {home_prob:.1%}")
        print(f"  Unentschieden:        {draw_prob:.1%}")
        print(f"  {away_team} gewinnt:   {away_prob:.1%}")
        print("-" * 55)
        print(f"  Prediction: {prediction}")
        print("=" * 55)
        print(f"  Form (letzte 5):  "
              f"{home_team} {round(home_form['points'],1)}Pts "
              f"(Trend: {'+' if home_form['trend']>0 else ''}{round(home_form['trend'],1)})  |  "
              f"{away_team} {round(away_form['points'],1)}Pts "
              f"(Trend: {'+' if away_form['trend']>0 else ''}{round(away_form['trend'],1)})")
        print(f"  Tabelle:  "
              f"{home_team} Platz {home_pos['season_position']} ({home_pos['season_points']}Pts)  |  "
              f"{away_team} Platz {away_pos['season_position']} ({away_pos['season_points']}Pts)")
        print(f"  H2H (letzte 5):  "
              f"{home_team} {h2h['h2h_home_wins']}S — "
              f"{h2h['h2h_draws']}U — "
              f"{h2h['h2h_away_wins']}S {away_team}\n")

        return {"home_win": home_prob, "draw": draw_prob,
                "away_win": away_prob, "prediction": prediction}

    def predict_matchday(self, matches, date=None):
        if date is None:
            date = datetime.today().strftime("%Y-%m-%d")

        print(f"\n{'='*55}")
        print(f"  SPIELTAG PREDICTIONS  |  {date}")
        print(f"{'='*55}")

        results = []
        for home, away in matches:
            result = self.predict(home, away, date)
            if result:
                results.append({
                    "Heim":          home,
                    "Gast":          away,
                    "Heimsieg":      f"{result['home_win']:.1%}",
                    "Unentschieden": f"{result['draw']:.1%}",
                    "Auswärtssieg":  f"{result['away_win']:.1%}",
                    "Prediction":    result["prediction"]
                })

        if results:
            print("\nZUSAMMENFASSUNG:")
            print(pd.DataFrame(results).to_string(index=False))

        return results


if __name__ == "__main__":
    from src.scraper import get_next_matchday_manual

    predictor = BundesligaPredictor()

    matchday = int(input("Welcher Spieltag soll predicted werden? "))
    matches  = get_next_matchday_manual(matchday_number=matchday)

    if not matches:
        print("Keine Spiele gefunden")
    else:
        date = matches[0][2]
        predictor.predict_matchday(
            [(home, away) for home, away, _ in matches],
            date=date
        )