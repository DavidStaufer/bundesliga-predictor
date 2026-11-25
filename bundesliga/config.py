"""Zentrale Pfade und Konstanten."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MATCHES_FILE = DATA_DIR / "matches.csv"
MODELS_DIR = ROOT / "models"
HYPERPARAMS_FILE = MODELS_DIR / "hyperparams.json"
ENSEMBLE_FILE = MODELS_DIR / "ensemble.json"
REPORTS_DIR = ROOT / "reports"
PREDICTIONS_DIR = ROOT / "predictions"

# 2005/06 ist die erste Saison, für die football-data.co.uk Durchschnittsquoten
# inklusive Über/Unter 2,5 Tore liefert.
FIRST_SEASON = 2005
# Auf diesen Saisons werden Hyperparameter gewählt (nie auf den Testsaisons).
VALIDATION_SEASONS = tuple(range(2008, 2014))
# Ab dieser Saison wird berichtet: echte Out-of-Sample-Vorhersagen.
TEST_FIRST_SEASON = 2014

# Score-Matrizen gehen von 0 bis MAX_GOALS Tore pro Team.
MAX_GOALS = 10

OUTCOMES = ("H", "D", "A")

USER_AGENT = "bundesliga-predictor/2.0"
