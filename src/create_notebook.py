import nbformat as nb
import json

# Installiere nbformat falls nötig
# pip install nbformat

notebook = nb.v4.new_notebook()

cells = [

# ── Zelle 1: Titel ────────────────────────────────────────────
nb.v4.new_markdown_cell("""# Bundesliga Match Predictor
### Machine Learning Projekt — Vorhersage von Bundesliga Spielausgängen

**Ziel:** Vorhersage ob ein Bundesliga Spiel mit Heimsieg, Unentschieden oder Auswärtssieg endet  
**Modell:** XGBoost mit SMOTE & Threshold Tuning  
**Daten:** 12 Saisons Bundesliga (2014/15 — 2025/26), ~3.500+ Spiele  
**Accuracy:** ~48% (Baseline: 33.3% Zufallslevel)

---"""),

# ── Zelle 2: Imports ──────────────────────────────────────────
nb.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append("..")

from sklearn.metrics import classification_report, confusion_matrix
from src.features import build_features
from src.model import FEATURE_COLS

plt.style.use("seaborn-v0_8-whitegrid")
print("Imports erfolgreich")"""),

# ── Zelle 3: Daten laden ──────────────────────────────────────
nb.v4.new_markdown_cell("## 1. Daten laden & verstehen"),

nb.v4.new_code_cell("""# Historische Daten laden
train   = pd.read_csv("../data/bundesliga_historical.csv")
current = pd.read_csv("../data/bundesliga_2526_raw.csv")
df_all  = pd.concat([train, current], ignore_index=True)

print(f"Gesamte Spiele:     {len(df_all):,}")
print(f"Saisons:            {sorted(df_all['Season'].unique())}")
print(f"Teams:              {df_all['HomeTeam'].nunique()}")
print(f"Zeitraum:           {df_all['Date'].min()} bis {df_all['Date'].max()}")
print()
print("Ergebnis-Verteilung:")
print(df_all['Result'].value_counts(normalize=True).round(3).rename({'H':'Heimsieg','D':'Unentschieden','A':'Auswärtssieg'}))"""),

# ── Zelle 4: Ergebnis Verteilung Plot ─────────────────────────
nb.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Ergebnis-Verteilung
counts = df_all['Result'].value_counts()
colors = ['#2196F3', '#FF9800', '#4CAF50']
bars   = axes[0].bar(['Heimsieg', 'Unentschieden', 'Auswärtssieg'],
                      [counts.get('H',0), counts.get('D',0), counts.get('A',0)],
                      color=colors, edgecolor='white', linewidth=1.5)
axes[0].set_title('Ergebnis-Verteilung (alle Saisons)', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Anzahl Spiele')
for bar, count in zip(bars, [counts.get('H',0), counts.get('D',0), counts.get('A',0)]):
    axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 20,
                 f'{count}\\n({count/len(df_all):.1%})', ha='center', fontsize=11)

# Plot 2: Ergebnisse pro Saison
season_results = df_all.groupby(['Season','Result']).size().unstack(fill_value=0)
season_results.plot(kind='bar', ax=axes[1], color=colors, edgecolor='white')
axes[1].set_title('Ergebnisse pro Saison', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Saison')
axes[1].set_ylabel('Anzahl Spiele')
axes[1].legend(['Auswärtssieg', 'Unentschieden', 'Heimsieg'])
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('../plots/data_overview.png', dpi=150, bbox_inches='tight')
plt.show()"""),

# ── Zelle 5: Feature Engineering ─────────────────────────────
nb.v4.new_markdown_cell("""## 2. Feature Engineering

Für jedes Spiel berechnen wir folgende Features **ausschließlich aus Daten die vor dem Spiel verfügbar waren** (kein Data Leakage):

| Feature Gruppe | Features |
|---|---|
| **Form (letzte 5 Spiele)** | Punkte, Tore, Gegentore, Tordifferenz |
| **Heimstärke / Auswärtsstärke** | PPG Heim/Auswärts separat |
| **Trend** | Vergleich letzte 3 vs davor 3 Spiele |
| **Tabellenposition** | Punkte, Tordifferenz, Platz |
| **Head-to-Head** | Siege/Unentschieden in letzten 5 Duellen |
| **Erholung** | Tage seit letztem Spiel |
| **Stabilität** | Tor-Varianz (draw-Indikator) |
| **Ergebnisserie** | Aktuelle Gewinn/Verlustserie |"""),

nb.v4.new_code_cell("""# Features laden
features = pd.read_csv("../data/bundesliga_features.csv")
print(f"Feature Matrix: {features.shape}")
print(f"\\nFeature Spalten ({len(FEATURE_COLS)}):")
for i, col in enumerate(FEATURE_COLS, 1):
    print(f"  {i:2d}. {col}")"""),

# ── Zelle 6: Modell & Ergebnisse ─────────────────────────────
nb.v4.new_markdown_cell("## 3. Modell — XGBoost mit SMOTE & Threshold Tuning"),

nb.v4.new_code_cell("""# Modell laden
model      = joblib.load("../models/best_model.pkl")
scaler     = joblib.load("../models/scaler.pkl")
le         = joblib.load("../models/label_encoder.pkl")
threshold  = joblib.load("../models/best_threshold.pkl")
draw_idx   = joblib.load("../models/draw_idx.pkl")
class_list = joblib.load("../models/class_list.pkl")

print("Modell-Parameter:")
params = model.get_params()
for key in ['n_estimators','max_depth','learning_rate','subsample','colsample_bytree']:
    print(f"  {key}: {params[key]}")
print(f"\\nDraw-Threshold: {threshold}")"""),

# ── Zelle 7: Feature Importance ───────────────────────────────
nb.v4.new_code_cell("""# Feature Importance
importance_df = pd.DataFrame({
    'Feature':    FEATURE_COLS,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=True)

plt.figure(figsize=(10, 8))
colors = ['#2196F3' if imp > importance_df['Importance'].median() else '#90CAF9'
          for imp in importance_df['Importance']]
plt.barh(importance_df['Feature'], importance_df['Importance'], color=colors)
plt.xlabel('Feature Importance (XGBoost)', fontsize=12)
plt.title('Feature Importance — Welche Features sind am wichtigsten?', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../plots/feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()

print("Top 5 wichtigste Features:")
print(importance_df.tail(5)[['Feature','Importance']].iloc[::-1].to_string(index=False))"""),

# ── Zelle 8: Confusion Matrix ─────────────────────────────────
nb.v4.new_markdown_cell("## 4. Evaluation auf Holdout-Set"),

nb.v4.new_code_cell("""# Holdout Evaluation
df_train = features[features['Season'].astype(str) != '2526'].copy()
df_train = df_train.sort_values('Date').reset_index(drop=True)
split_idx  = int(len(df_train) * 0.8)
holdout_df = df_train.iloc[split_idx:]

X_hold = holdout_df[FEATURE_COLS]
y_hold = holdout_df['Result']
X_hold_scaled = scaler.transform(X_hold)
proba  = model.predict_proba(X_hold_scaled)

y_pred = []
for prob_row in proba:
    if prob_row[draw_idx] >= threshold:
        y_pred.append('D')
    else:
        other_idx  = [i for i in range(len(class_list)) if i != draw_idx]
        other_prob = [prob_row[i] for i in other_idx]
        best_other = other_idx[np.argmax(other_prob)]
        y_pred.append(le.inverse_transform([best_other])[0])

print(classification_report(y_hold, y_pred,
      target_names=['Heimsieg', 'Unentschieden', 'Auswärtssieg'], zero_division=0))"""),

nb.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Confusion Matrix
cm = confusion_matrix(y_hold, y_pred, labels=['H','D','A'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Heimsieg','Unentschieden','Auswärtssieg'],
            yticklabels=['Heimsieg','Unentschieden','Auswärtssieg'])
axes[0].set_ylabel('Tatsächlich', fontsize=12)
axes[0].set_xlabel('Vorhergesagt', fontsize=12)
axes[0].set_title('Confusion Matrix — Holdout Set', fontsize=14, fontweight='bold')

# Accuracy vs Threshold
thresholds, accuracies, draw_recalls = [], [], []
for t in np.arange(0.20, 0.45, 0.025):
    yp = []
    for prob_row in proba:
        if prob_row[draw_idx] >= t:
            yp.append('D')
        else:
            other_idx  = [i for i in range(len(class_list)) if i != draw_idx]
            yp.append(le.inverse_transform([other_idx[np.argmax([prob_row[i] for i in other_idx])]])[0])
    from sklearn.metrics import accuracy_score
    thresholds.append(t)
    accuracies.append(accuracy_score(y_hold, yp))
    rep = classification_report(y_hold, yp, output_dict=True, zero_division=0)
    draw_recalls.append(rep.get('D',{}).get('recall',0))

axes[1].plot(thresholds, accuracies,   'b-o', label='Accuracy',     linewidth=2)
axes[1].plot(thresholds, draw_recalls, 'r-o', label='Draw Recall',  linewidth=2)
axes[1].axvline(x=threshold, color='green', linestyle='--', label=f'Gewählter Threshold ({threshold})')
axes[1].set_xlabel('Draw Threshold', fontsize=12)
axes[1].set_ylabel('Score', fontsize=12)
axes[1].set_title('Accuracy vs Draw Recall', fontsize=14, fontweight='bold')
axes[1].legend()

plt.tight_layout()
plt.savefig('../plots/evaluation.png', dpi=150, bbox_inches='tight')
plt.show()"""),

# ── Zelle 9: Beispiel Prediction ─────────────────────────────
nb.v4.new_markdown_cell("## 5. Beispiel — Spieltag Prediction"),

nb.v4.new_code_cell("""from src.predict_future import BundesligaPredictor

predictor = BundesligaPredictor()

# Beispiel Spieltag
predictor.predict_matchday([
    ("Bayern Munich", "Dortmund"),
    ("Leverkusen",    "RB Leipzig"),
    ("Freiburg",      "Stuttgart"),
], date="2026-03-01")"""),

# ── Zelle 10: Fazit ───────────────────────────────────────────
nb.v4.new_markdown_cell("""## 6. Fazit & Erkenntnisse

### Ergebnisse
- **48% Accuracy** auf ungesehenen Daten (vs. 33.3% Zufallslevel)
- **Auswärtssiege** werden am besten erkannt (F1: 0.60)
- **Heimsiege** solide (F1: 0.47)  
- **Unentschieden** schwierigste Klasse (F1: 0.17)

### Wichtigste Erkenntnisse
1. **Tabellenposition** ist der stärkste Prädiktor (`points_diff_season`)
2. **Erholung** zwischen Spielen hat überraschend großen Einfluss (`rest_diff`)
3. **Head-to-Head** Geschichte ist wichtiger als aktuelle Form
4. **Draws** sind fundamental schwer vorherzusagen

### Mögliche Verbesserungen
- Spieler-Level Daten (Verletzungen, Sperren)
- Expected Goals (xG) als Feature
- Wettquoten als implizites Markt-Signal
- Deep Learning Ansatz (LSTM für Zeitreihen)"""),
]

notebook.cells = cells

import os
os.makedirs("notebooks", exist_ok=True)
with open("notebooks/bundesliga_predictor.ipynb", "w") as f:
    nb.write(notebook, f)

print("Notebook erstellt: notebooks/bundesliga_predictor.ipynb")