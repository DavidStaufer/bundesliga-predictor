"""Backtest-Bericht: Markdown-Tabellen und zwei Diagramme in reports/."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .backtest import kicktipp_points  # noqa: E402
from .config import REPORTS_DIR, TEST_FIRST_SEASON  # noqa: E402
from .metrics import one_hot  # noqa: E402
from .scores import outcome_probs  # noqa: E402

# Referenzpalette (validiert: Helligkeitsband, Farbsehschwäche, Abstand)
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def _spread_labels(values, min_gap):
    """Schiebt Endbeschriftungen auseinander, damit sie sich nicht überlappen."""
    order = np.argsort(values)
    placed = np.array(values, float)
    for prev, cur in zip(order, order[1:]):
        placed[cur] = max(placed[cur], placed[prev] + min_gap)
    return placed


def plot_log_loss(by_season: pd.DataFrame, path, partial_last: bool = False):
    series = [("Eigenes Ensemble (Elo + DC)", "Eigenes Modell"), ("Buchmacher", "Buchmacher")]
    fig, ax = plt.subplots(figsize=(9, 4.4), facecolor=SURFACE)
    _style(ax)
    x = np.arange(len(by_season))
    ends = []
    for (col, label), color in zip(series, SERIES):
        y = by_season[col].to_numpy(float)
        ax.plot(x, y, color=color, linewidth=2, solid_capstyle="round", label=label, zorder=3)
        ax.scatter(x, y, s=36, color=color, edgecolor=SURFACE, linewidth=2, zorder=4)
        ends.append(y[-1])
    span = np.ptp(by_season[[c for c, _ in series]].to_numpy(float))
    for (_, label), y_text, y_end in zip(series, _spread_labels(ends, span * 0.07), ends):
        ax.annotate(label, (x[-1], y_end), xytext=(x[-1] + 0.25, y_text), textcoords="data",
                    va="center", fontsize=9, color=TEXT)
    ticks = list(by_season.index)
    if partial_last:
        ticks[-1] += "*"
    ax.set_xticks(x, ticks, fontsize=8)
    ax.set_xlim(-0.4, len(x) + 1.2)
    ax.set_ylabel("Log-Loss (niedriger ist besser)", color=TEXT_SECONDARY, fontsize=9)
    ax.set_title("Log-Loss pro Saison: eigenes Modell gegen Buchmacher",
                 loc="left", fontsize=12, color=TEXT, pad=30)
    ax.legend(frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0, 1.0),
              labelcolor=TEXT, ncols=2, borderaxespad=0.2)
    note = "Zum Vergleich: Raten (je 1/3) läge bei 1,099."
    if partial_last:
        note += "  * laufende Saison"
    fig.text(0.01, 0.01, note, fontsize=8, color=TEXT_SECONDARY)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_calibration(probs: np.ndarray, results, path, bins: int = 10):
    """Zuverlässigkeitsdiagramm: Wenn das Modell 40 % sagt, tritt es in 40 % der Fälle ein?"""
    y = one_hot(results)
    fig, ax = plt.subplots(figsize=(5.6, 5.2), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="x", color=GRID, linewidth=1)
    ax.plot([0, 1], [0, 1], color=TEXT_SECONDARY, linewidth=1, zorder=2)
    edges = np.linspace(0, 1, bins + 1)
    for k, (label, color) in enumerate(zip(("Heimsieg", "Remis", "Auswärtssieg"), SERIES)):
        which = np.clip(np.digitize(probs[:, k], edges) - 1, 0, bins - 1)
        pred, obs = [], []
        for b in range(bins):
            sel = which == b
            if sel.sum() >= 30:  # zu dünn besetzte Klassen weglassen
                pred.append(probs[sel, k].mean())
                obs.append(y[sel, k].mean())
        ax.plot(pred, obs, color=color, linewidth=2, label=label, zorder=3)
        ax.scatter(pred, obs, s=36, color=color, edgecolor=SURFACE, linewidth=2, zorder=4)
    ax.set_xlim(0, 0.9)
    ax.set_ylim(0, 0.9)
    ax.set_xlabel("Vorhergesagte Wahrscheinlichkeit", color=TEXT_SECONDARY, fontsize=9)
    ax.set_ylabel("Tatsächliche Häufigkeit", color=TEXT_SECONDARY, fontsize=9)
    ax.set_title("Kalibrierung des eigenen Modells", loc="left", fontsize=12, color=TEXT, pad=12)
    ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=TEXT)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _md_table(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    cols = [df.index.name or ""] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        cells = [str(idx)] + [format(v, floatfmt) if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write(matches: pd.DataFrame, result: dict) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    mask = result["test_mask"]
    test = matches[mask]
    preds = result["preds"]

    by_season_raw = result["log_loss_by_season"]
    plot_log_loss(by_season_raw, REPORTS_DIR / "log_loss_per_season.png",
                  partial_last=by_season_raw["Spiele"].iloc[-1] < 306)
    own_probs = outcome_probs(preds["own"][mask])
    plot_calibration(own_probs, test["result"], REPORTS_DIR / "calibration.png")

    # Vorhersagen je Spiel (eigenes Modell und Quoten-Ensemble)
    out = test[["season", "date", "home", "away", "hg", "ag", "result"]].copy()
    for key in ("own", "combined"):
        probs = outcome_probs(preds[key][mask])
        tip_h, tip_a, pts = kicktipp_points(preds[key][mask], test)
        out[f"{key}_p_h"], out[f"{key}_p_d"], out[f"{key}_p_a"] = probs.T.round(4)
        out[f"{key}_tip"] = [f"{h}:{a}" for h, a in zip(tip_h, tip_a)]
        out[f"{key}_points"] = pts
    out.to_csv(REPORTS_DIR / "backtest_predictions.csv", index=False)

    summary = result["summary"].rename(columns={
        "log_loss": "Log-Loss", "rps": "RPS", "brier": "Brier",
        "accuracy": "Trefferquote", "kicktipp": "Kicktipp-Punkte/Spiel",
    })
    summary.index.name = "Modell"
    by_season = result["log_loss_by_season"].copy()
    by_season.index.name = "Saison"
    by_season["Spiele"] = by_season["Spiele"].astype(int)
    kicktipp = result["kicktipp_by_season"].astype(int)
    kicktipp.index.name = "Saison"

    first, last = TEST_FIRST_SEASON, int(matches["season"].max())
    text = f"""# Backtest-Bericht

Testzeitraum {first}/{(first + 1) % 100:02d} bis {last}/{(last + 1) % 100:02d}, {int(mask.sum())} Spiele.
Jeder Spieltag wurde nur mit Daten vorhergesagt, die vorher bekannt waren.

## Gesamt

{_md_table(summary, ".4f")}

Log-Loss, RPS und Brier: niedriger ist besser. Raten (je 1/3) hätte einen Log-Loss von {np.log(3):.4f}.

## Log-Loss pro Saison

![Log-Loss pro Saison](log_loss_per_season.png)

{_md_table(by_season)}

## Kicktipp-Punkte pro Saison (4/3/2, bester Tipp pro Spiel)

{_md_table(kicktipp, "d")}

## Kalibrierung

![Kalibrierung](calibration.png)

## Ensemble-Gewichte für Live-Vorhersagen

{result["weights"]}
"""
    (REPORTS_DIR / "backtest.md").write_text(text, encoding="utf-8")
    print("Bericht geschrieben: reports/backtest.md")
