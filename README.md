# Bundesliga Match Predictor

A machine learning project that predicts Bundesliga match outcomes (Home Win / Draw / Away Win) using XGBoost and historical match data.

---

## Results

| Metric | Score |
|--------|-------|
| **Overall Accuracy** | ~48% |
| **Baseline (random)** | 33.3% |
| **Improvement over baseline** | +14-15% |
| **Best predicted class** | Away Win (F1: 0.60) |

> Predicting football matches is fundamentally difficult — even professional bookmakers struggle with draws. This model focuses on being realistic rather than overfitted.

---

## Project Structure
```
bundesliga-predictor/
├── data/
│   ├── bundesliga_historical.csv   # 11 seasons training data (2014-2025)
│   ├── bundesliga_2526_raw.csv     # Current season 2025/26
│   └── bundesliga_features.csv     # Engineered features
├── models/
│   ├── best_model.pkl              # Trained XGBoost model
│   ├── scaler.pkl                  # StandardScaler
│   └── label_encoder.pkl           # Label encoder
├── notebooks/
│   └── bundesliga_predictor.ipynb  # Full analysis notebook
├── plots/
│   ├── feature_importance.png
│   ├── confusion_matrix.png
│   └── evaluation.png
├── src/
│   ├── data_loader.py              # Download match data
│   ├── features.py                 # Feature engineering
│   ├── model.py                    # Model training & evaluation
│   ├── predict_future.py           # Match predictor class
│   └── scraper.py                  # Auto-fetch matchday schedule
└── README.md
```

---

## Features Used

| Feature Group | Description |
|---|---|
| **Form (last 5 games)** | Points, goals, goal difference |
| **Home / Away strength** | PPG separately at home and away |
| **Trend** | Last 3 games vs previous 3 |
| **League position** | Points, GD, rank at time of match |
| **Head-to-Head** | Results of last 5 meetings |
| **Rest days** | Days since last match (fatigue) |
| **Stability index** | Goal variance (draw indicator) |
| **Result streak** | Current winning/losing streak |

---

## Getting Started

**1. Clone the repository**
```bash
git clone https://github.com/yourusername/bundesliga-predictor.git
cd bundesliga-predictor
```

**2. Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Download data & train model**
```bash
python src/data_loader.py   # Download match data
python src/model.py         # Train XGBoost model
```

**5. Predict next matchday**
```bash
python src/predict_future.py
# Enter matchday number when prompted (e.g. 25)
```

---

## Example Output
```
=======================================================
  Dortmund vs Bayern Munich  |  2026-02-28
=======================================================
  Dortmund gewinnt:   32.8%
  Unentschieden:        17.0%
  Bayern Munich gewinnt:   50.2%
-------------------------------------------------------
  Prediction: Bayern Munich gewinnt
=======================================================
  Form (letzte 5):  Dortmund 13Pts (Trend: -0.7)  |  Bayern Munich 10Pts (Trend: +1.7)
  Tabelle:  Dortmund Platz 2 (52Pts)  |  Bayern Munich Platz 1 (60Pts)
  H2H (letzte 5):  Dortmund 1S — 2U — 2S Bayern Munich
```

---

## Weekly Update Workflow
```bash
python src/data_loader.py    # Fetch latest results
python src/model.py          # Retrain on updated data
python src/predict_future.py # Predict upcoming matchday
```

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.13-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-red)
![pandas](https://img.shields.io/badge/pandas-2.0-green)

---

## Limitations & Future Improvements

- **Player-level data** — injuries and suspensions are not considered
- **Expected Goals (xG)** — would significantly improve predictions
- **Betting odds** as implicit market signal
- **LSTM / Deep Learning** for better time-series modeling
- **Live data integration** for real-time predictions

---

## Data Source

Match data from [football-data.co.uk](https://www.football-data.co.uk/germany.php) — free historical Bundesliga results.
Matchday schedule from [OpenLigaDB](https://api.openligadb.de) — free open football API.

---

## Author

**David**  
[GitHub](https://github.com/DavidStaufer) · [LinkedIn](https://linkedin.com/in/DavidStaufer)
```

Dann noch die `requirements.txt` updaten damit sie vollständig ist:
```
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
jupyter
requests
openpyxl
joblib
imbalanced-learn
beautifulsoup4
nbformat