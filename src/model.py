import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.calibration import CalibratedClassifierCV
import os
import sys
sys.path.append(".")
from src.features import build_features


for folder in ['models', 'plots']:
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Ordner '{folder}' wurde erstellt.")

# Features die wir nutzen
FEATURE_COLS = [
    "home_form_points", "home_form_gf", "home_form_ga", "home_form_gd",
    "away_form_points", "away_form_gf", "away_form_ga", "away_form_gd",
    "form_points_diff", "form_gd_diff",
    "home_trend", "away_trend", "trend_diff",
    "home_home_ppg", "home_home_gf", "home_home_ga",
    "away_away_ppg", "away_away_gf", "away_away_ga",
    "home_advantage",
    "home_days_rest", "away_days_rest", "rest_diff",
    "home_season_points", "home_season_gd", "home_season_position",
    "away_season_points", "away_season_gd", "away_season_position",
    "position_diff", "points_diff_season",
    "h2h_home_wins", "h2h_draws", "h2h_away_wins",
    "season_phase", "home_goal_variance", "away_goal_variance",
    "home_streak", "away_streak", "streak_diff",
    "home_home_streak", "away_away_streak",
]


def load_data(path: str = "data/bundesliga_features.csv"):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def prepare_xy(df: pd.DataFrame):
    X = df[FEATURE_COLS]
    y = df["Result"]  # H, D, A
    return X, y


def train_and_evaluate(df: pd.DataFrame):
    X, y = prepare_xy(df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = xgb.XGBClassifier(
        n_estimators=300,
        random_state=42,
        eval_metric="mlogloss",
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0
    )

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    scores = cross_val_score(model, X_scaled, y_enc, cv=5, scoring="accuracy")

    print("=" * 50)
    print("XGBoost Cross-Validation (5-Fold):")
    print("=" * 50)
    print(f"Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")

    return model, scaler, le


def train_final_model(df: pd.DataFrame, model_name: str, models: dict, scaler: StandardScaler):
    """Trainiert das beste Modell auf allen Trainingsdaten."""
    X, y = prepare_xy(df)
    X_scaled = scaler.fit_transform(X)

    model = models[model_name]

    le = None
    if model_name == "XGBoost":
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        model.fit(X_scaled, y_enc)
    else:
        model.fit(X_scaled, y)

    # Modell speichern
    os.makedirs("models", exist_ok=True)
    joblib.dump(model,  "models/best_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    if le:
        joblib.dump(le, "models/label_encoder.pkl")

    print(f"\nModell gespeichert in models/")
    return model, le


def plot_feature_importance(model, model_name: str):
    """Feature Importance Plot für Random Forest und XGBoost."""
    if model_name not in ["Random Forest", "XGBoost"]:
        print("Feature Importance nur für Random Forest und XGBoost verfügbar.")
        return

    importances = model.feature_importances_
    feat_df = pd.DataFrame({
        "Feature":    FEATURE_COLS,
        "Importance": importances
    }).sort_values("Importance", ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(feat_df["Feature"], feat_df["Importance"], color="steelblue")
    plt.xlabel("Importance")
    plt.title(f"Feature Importance — {model_name}")
    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/feature_importance.png", dpi=150)
    plt.show()
    print("Plot gespeichert: plots/feature_importance.png")


def plot_confusion_matrix(model, scaler, df: pd.DataFrame, le=None):
    """Confusion Matrix auf den Trainingsdaten."""
    X, y = prepare_xy(df)
    X_scaled = scaler.transform(X)

    if le:
        y_enc = le.transform(y)
        y_pred_enc = model.predict(X_scaled)
        y_pred = le.inverse_transform(y_pred_enc)
    else:
        y_pred = model.predict(X_scaled)

    cm = confusion_matrix(y, y_pred, labels=["H", "D", "A"])

    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Home Win", "Draw", "Away Win"],
                yticklabels=["Home Win", "Draw", "Away Win"])
    plt.ylabel("Tatsächlich")
    plt.xlabel("Vorhergesagt")
    plt.title("Confusion Matrix (Trainingsdaten)")
    plt.tight_layout()
    plt.savefig("plots/confusion_matrix.png", dpi=150)
    plt.show()
    print("Plot gespeichert: plots/confusion_matrix.png")

    print("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=["Home Win", "Draw", "Away Win"]))


if __name__ == "__main__":
    import pandas as pd
    from src.features import build_features

    train   = pd.read_csv("data/bundesliga_historical.csv")
    current = pd.read_csv("data/bundesliga_2526_raw.csv")
    combined = pd.concat([train, current], ignore_index=True)
    combined = combined.dropna(subset=["Result"])

    features = build_features(combined)
    features.to_csv("data/bundesliga_features.csv", index=False)

    df = features[features["Season"].astype(str) != "2526"].copy()
    df = df.sort_values("Date").reset_index(drop=True)

    split_idx  = int(len(df) * 0.8)
    train_df   = df.iloc[:split_idx]
    holdout_df = df.iloc[split_idx:]

    print(f"   Training:  {len(train_df)} Spiele")
    print(f"   Holdout:   {len(holdout_df)} Spiele\n")

    # Cross-Validation
    model, scaler, le = train_and_evaluate(train_df)

    # SMOTE
    X_train, y_train = prepare_xy(train_df)
    X_hold,  y_hold  = prepare_xy(holdout_df)

    scaler       = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_hold_scaled  = scaler.transform(X_hold)

    print("\nWende SMOTE an...")
    y_train_enc = le.fit_transform(y_train)
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train_enc)

    # Finales Modell trainieren
    final_model = xgb.XGBClassifier(
        n_estimators=300,
        random_state=42,
        eval_metric="mlogloss",
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0
    )
    final_model.fit(X_resampled, y_resampled)

    # Threshold Tuning
    proba = final_model.predict_proba(X_hold_scaled)
    classes    = list(le.classes_)
    draw_idx   = classes.index("D")
    class_list = classes

    print(f"\n{'Threshold':>10} {'Accuracy':>10} {'Draw-Recall':>12} {'Draw-Precision':>15}")
    print("-" * 52)

    best_threshold = 0.400
    for threshold in np.arange(0.20, 0.45, 0.025):
        y_pred_tuned = []
        for prob_row in proba:
            if prob_row[draw_idx] >= threshold:
                y_pred_tuned.append("D")
            else:
                other_idx  = [i for i in range(len(class_list)) if i != draw_idx]
                other_prob = [prob_row[i] for i in other_idx]
                best_other = other_idx[np.argmax(other_prob)]
                y_pred_tuned.append(le.inverse_transform([best_other])[0])

        from sklearn.metrics import accuracy_score
        acc     = accuracy_score(y_hold, y_pred_tuned)
        report  = classification_report(y_hold, y_pred_tuned,
                      labels=["H","D","A"], output_dict=True, zero_division=0)
        dr      = report.get("D", {}).get("recall", 0)
        dp      = report.get("D", {}).get("precision", 0)
        print(f"{threshold:>10.3f} {acc:>10.3f} {dr:>12.3f} {dp:>15.3f}")

    # Finale Evaluation mit Threshold 0.325
    print("\n" + "=" * 50)
    print("Finale Evaluation (Threshold=0.325):")
    print("=" * 50)

    y_final = []
    for prob_row in proba:
        if prob_row[draw_idx] >= best_threshold:
            y_final.append("D")
        else:
            other_idx  = [i for i in range(len(class_list)) if i != draw_idx]
            other_prob = [prob_row[i] for i in other_idx]
            best_other = other_idx[np.argmax(other_prob)]
            y_final.append(le.inverse_transform([best_other])[0])

    print(classification_report(y_hold, y_final,
          target_names=["Home Win", "Draw", "Away Win"], zero_division=0))

    # Speichern
    os.makedirs("models", exist_ok=True)
    joblib.dump(final_model,   "models/best_model.pkl")
    joblib.dump(scaler,        "models/scaler.pkl")
    joblib.dump(le,            "models/label_encoder.pkl")
    joblib.dump(best_threshold,"models/best_threshold.pkl")
    joblib.dump(draw_idx,      "models/draw_idx.pkl")
    joblib.dump(class_list,    "models/class_list.pkl")
    print("Modell gespeichert!")

    # Plots
    plot_feature_importance(final_model, "XGBoost")

    cm = confusion_matrix(y_hold, y_final, labels=["H", "D", "A"])
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Home Win", "Draw", "Away Win"],
                yticklabels=["Home Win", "Draw", "Away Win"])
    plt.ylabel("Tatsächlich")
    plt.xlabel("Vorhergesagt")
    plt.title("Confusion Matrix — Holdout Set (XGBoost)")
    plt.tight_layout()
    plt.savefig("plots/confusion_matrix.png", dpi=150)
    plt.show()
    print("Plots gespeichert!")