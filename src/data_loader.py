import pandas as pd
import requests
import os

def download_bundesliga_data(seasons: list, save_path: str = "data/") -> pd.DataFrame:
    """
    Lädt Bundesliga-Daten von football-data.co.uk für mehrere Saisons
    und speichert sie lokal als CSV.
    """
    all_seasons = []

    for season in seasons:
        url = f"https://www.football-data.co.uk/mmz4281/{season}/D1.csv"
        print(f"Lade Saison {season}")

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            # CSV direkt aus dem Response lesen
            from io import StringIO
            df = pd.read_csv(StringIO(response.text), low_memory=False)
            df["Season"] = season
            all_seasons.append(df)

        except Exception as e:
            print(f"Fehler bei Saison {season}: {e}")

    combined = pd.concat(all_seasons, ignore_index=True)

    # Nur relevante Spalten behalten
    cols = ["Season", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
    combined = combined[cols]

    # Bessere Lesbarkeit
    combined.rename(columns={
        "FTHG": "HomeGoals",
        "FTAG": "AwayGoals",
        "FTR":  "Result"   # H = Home Win, D = Draw, A = Away Win
    }, inplace=True)

    # Speichern
    os.makedirs(save_path, exist_ok=True)
    output_file = os.path.join(save_path, "bundesliga_raw.csv")
    combined.to_csv(output_file, index=False)
    print(f"\nDaten gespeichert: {output_file}")
    print(f"   {len(combined)} Spiele geladen aus {len(all_seasons)} Saisons")

    return combined


def update_all_data():
    """
    Lädt alle Saisons inklusive der aktuellen 2025/26.
    Kann jede Woche neu ausgeführt werden um aktuelle Daten zu haben.
    """
    # Historische Trainingsdaten 
    training_seasons = [
        "1415", "1516", "1617", "1718",
        "1819", "1920", "2021", "2122",
        "2223", "2324", "2425" 
    ]

    # Aktuelle Saison separat
    current_season = ["2526"]

    print("Lade historische Trainingsdaten")
    train_df = download_bundesliga_data(training_seasons, save_path="data/")
    train_df.to_csv("data/bundesliga_historical.csv", index=False)

    print("\nLade aktuelle Saison 2025/26")
    try:
        current_df = download_bundesliga_data(current_season, save_path="data/")
        current_df.to_csv("data/bundesliga_2526_raw.csv", index=False)
        print(f"{len(current_df)} Spiele der Saison 2025/26 geladen")
    except Exception as e:
        print(f" Aktuelle Saison konnte nicht geladen werden: {e}")
        current_df = None

    return train_df, current_df


if __name__ == "__main__":
    update_all_data()
    
