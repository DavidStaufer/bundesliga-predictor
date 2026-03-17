import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import sys
sys.path.append(".")
from src.model import FEATURE_COLS


def predict_season(features_path: str = "data/bundesliga_2425_features.csv"):
    model      = joblib.load("models/best_model.pkl")
    scaler     = joblib.load("models/scaler.pkl")
    le         = joblib.load("models/label_encoder.pkl")
    threshold  = joblib.load("models/best_threshold.pkl")
    draw_idx   = joblib.load("models/draw_idx.pkl")
    class_list = joblib.load("models/class_list.pkl")

    print(f"Threshold: {threshold}")
    print(f"Draw idx:  {draw_idx}")
    print(f"Classes:   {class_list}")

    df = pd.read_csv(features_path)
    print(f"\n{len(df)} Spiele der Saison 2024/25 geladen")

    X      = df[FEATURE_COLS]
    y_true = df["Result"]

    X_scaled = scaler.transform(X)
    proba    = model.predict_proba(X_scaled)

    print(f"\nModell Klassen: {model.classes_}")
    print(f"Beispiel Proba (erstes Spiel): {proba[0]}")

    def apply_threshold(proba, t):
        y_pred = []
        for prob_row in proba:
            if prob_row[draw_idx] >= t:
                y_pred.append("D")
            else:
                other_idx  = [i for i in range(len(class_list)) if i != draw_idx]
                other_prob = [prob_row[i] for i in other_idx]
                best_other = other_idx[np.argmax(other_prob)]
                y_pred.append(le.inverse_transform([best_other])[0])
        return y_pred

    # Threshold Tabelle
    print(f"\n{'Threshold':>10} {'Accuracy':>10} {'Draw-Recall':>12} {'Draw-Prec':>10}")
    print("-" * 46)
    for t in np.arange(0.20, 0.45, 0.025):
        yp     = apply_threshold(proba, t)
        acc    = accuracy_score(y_true, yp)
        report = classification_report(y_true, yp, output_dict=True, zero_division=0)
        dr     = report.get("D", {}).get("recall", 0)
        dp     = report.get("D", {}).get("precision", 0)
        print(f"{t:>10.3f} {acc:>10.3f} {dr:>12.3f} {dp:>10.3f}")

    # Finale Evaluation
    y_pred = apply_threshold(proba, threshold)

    print(f"\n{'='*55}")
    print("FINALER TEST — Bundesliga Saison 2024/25")
    print(f"(Threshold: {threshold})")
    print(f"{'='*55}")
    print(classification_report(y_true, y_pred,
          target_names=["Home Win", "Draw", "Away Win"], zero_division=0))

    correct  = sum(a == b for a, b in zip(y_true, y_pred))
    accuracy = correct / len(y_true)
    print(f"Richtig predicted: {correct}/{len(y_true)} ({accuracy:.1%})")

    # Plots
    cm = confusion_matrix(y_true, y_pred, labels=["H", "D", "A"])
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=["Home Win", "Draw", "Away Win"],
                yticklabels=["Home Win", "Draw", "Away Win"])
    plt.ylabel("Tatsächlich")
    plt.xlabel("Vorhergesagt")
    plt.title(f"Confusion Matrix — Saison 2024/25\nAccuracy: {accuracy:.1%}")
    plt.tight_layout()
    plt.savefig("plots/confusion_matrix_2425.png", dpi=150)
    plt.show()

    df["Predicted"] = y_pred
    df["Correct"]   = df["Result"] == df["Predicted"]
    df["Date"]      = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    df["rolling_acc"] = df["Correct"].rolling(window=10, min_periods=1).mean()

    plt.figure(figsize=(12, 5))
    plt.plot(df["Date"], df["rolling_acc"], color="steelblue", linewidth=2)
    plt.axhline(y=accuracy, color="red",  linestyle="--", label=f"Gesamt: {accuracy:.1%}")
    plt.axhline(y=0.333,    color="gray", linestyle=":",  label="Zufallslevel (33.3%)")
    plt.xlabel("Datum")
    plt.ylabel("Rolling Accuracy (10 Spiele)")
    plt.title("Modell-Performance über die Saison 2024/25")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/rolling_accuracy_2425.png", dpi=150)
    plt.show()

    df[["Date","HomeTeam","AwayTeam","Result","Predicted","Correct"]].to_csv(
        "data/predictions_2425.csv", index=False)
    print("✅ Predictions gespeichert: data/predictions_2425.csv")

    return df


if __name__ == "__main__":
    predict_season()