"""Kommandozeile: python -m bundesliga <befehl>

  update     Ergebnisse und Quoten neu laden
  tune       Hyperparameter auf den Validierungssaisons wählen
  backtest   Alle Saisons rückwirkend vorhersagen und Bericht schreiben
  predict    Spieltag vorhersagen (Standard: der aktuelle)
"""

import argparse

import pandas as pd
import requests

from . import backtest, data, report
from .predict import MODEL_CHOICES, predict_matchday, print_prediction


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m bundesliga",
                                     description="Bundesliga-Vorhersagen mit Elo, Dixon-Coles und Quoten.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("update", help="Ergebnisse und Quoten neu laden")
    sub.add_parser("tune", help="Hyperparameter wählen (Validierungssaisons 2008/09–2013/14)")
    sub.add_parser("backtest", help="Walk-forward-Backtest und Bericht in reports/")
    p = sub.add_parser("predict", help="Spieltag vorhersagen")
    p.add_argument("--spieltag", type=int, help="Spieltag (Standard: aktueller laut OpenLigaDB)")
    p.add_argument("--saison", type=int, help="Startjahr der Saison, z. B. 2026 für 2026/27")
    p.add_argument("--modell", choices=MODEL_CHOICES, default="auto",
                   help="auto = mit Quoten, wenn verfügbar, sonst eigenes Modell")
    p.add_argument("--offline", action="store_true", help="Daten nicht vorher aktualisieren")
    args = parser.parse_args(argv)

    if args.command == "update":
        data.update()
    elif args.command == "tune":
        backtest.tune(data.load_matches())
    elif args.command == "backtest":
        matches = data.load_matches()
        result = backtest.run(matches)
        with pd.option_context("display.width", 140, "display.precision", 4):
            print("\n", result["summary"].to_string())
        report.write(matches, result)
    elif args.command == "predict":
        if args.offline:
            matches = data.load_matches()
        else:
            try:
                matches = data.update(verbose=False)
            except requests.RequestException as err:
                print(f"Daten konnten nicht aktualisiert werden ({err}); nutze den letzten Stand.")
                matches = data.load_matches()
        df = predict_matchday(matches, season=args.saison, matchday=args.spieltag, model=args.modell)
        print_prediction(df)


if __name__ == "__main__":
    main()
