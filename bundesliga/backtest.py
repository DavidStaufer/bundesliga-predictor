"""Walk-forward-Backtest: so tun, als würde man jede Saison live tippen.

Für jeden Spieltag werden die Modelle nur mit Spielen trainiert, die vorher
stattgefunden haben, und sagen dann diesen Spieltag vorher. Hyperparameter
werden ausschließlich auf den Validierungssaisons gewählt (2008/09–2013/14),
berichtet wird auf den Testsaisons ab 2014/15. Die Ensemble-Gewichte für eine
Saison werden nur aus Vorhersagen früherer Saisons gelernt.
"""

import itertools
import json
import time

import numpy as np
import pandas as pd

from . import hyperparams
from .config import (ENSEMBLE_FILE, FIRST_SEASON, MAX_GOALS, TEST_FIRST_SEASON,
                     VALIDATION_SEASONS)
from .data import season_teams, transitions
from .kicktipp import DEFAULT_RULES, best_tips, points
from .metrics import evaluate, log_loss
from .models.dixon_coles import DCParams, DixonColes
from .models.elo import EloGoalModel, EloParams, compute_elo
from .models.ensemble import fit_weights, pool
from .models.market import market_matrices
from .scores import outcome_probs

MODELS = {
    "freq": "Häufigkeit (Basis)",
    "elo": "Elo",
    "dc": "Dixon-Coles",
    "own": "Eigenes Ensemble (Elo + DC)",
    "market": "Buchmacher",
    "combined": "Ensemble + Quoten",
}
OWN_MEMBERS = ("elo", "dc")
COMBINED_MEMBERS = ("elo", "dc", "market")


def date_blocks(dates: pd.Series) -> np.ndarray:
    """Fasst aufeinanderfolgende Spieltage zu Blöcken zusammen (Lücke > 1 Tag = neuer Block).

    Fr/Sa/So bilden einen Block, ein Spieltag unter der Woche einen eigenen.
    Innerhalb eines Blocks wird nicht nachtrainiert, wie beim echten Tippen.
    """
    gaps = dates.diff().dt.days.fillna(0).to_numpy()
    return np.cumsum(gaps > 1)


def frequency_matrix(train: pd.DataFrame) -> np.ndarray:
    """Empirische Verteilung aller Endstände (mit etwas Glättung)."""
    counts = np.full((MAX_GOALS + 1, MAX_GOALS + 1), 0.1)
    hg = train["hg"].clip(upper=MAX_GOALS).to_numpy()
    ag = train["ag"].clip(upper=MAX_GOALS).to_numpy()
    np.add.at(counts, (hg, ag), 1)
    return counts / counts.sum()


def base_predictions(matches: pd.DataFrame, seasons, elo_params: EloParams,
                     dc_params: DCParams, models=("freq", "elo", "dc", "market"),
                     verbose: bool = False) -> dict[str, np.ndarray]:
    """Score-Matrizen der Einzelmodelle für alle Spiele der angegebenen Saisons."""
    n, g = len(matches), MAX_GOALS + 1
    teams_by_season = season_teams(matches)
    season_col = matches["season"].to_numpy()
    home, away = matches["home"].to_numpy(), matches["away"].to_numpy()
    preds = {k: np.full((n, g, g), np.nan) for k in models}

    if "elo" in models:
        pre_h, pre_a, _ = compute_elo(matches, teams_by_season, elo_params)
        elo_diff = pre_h - pre_a
    if "market" in models:
        preds["market"], _ = market_matrices(matches)

    for season in seasons:
        start = time.time()
        idx = np.flatnonzero(season_col == season)
        past = (season_col > FIRST_SEASON) & (season_col < season)  # erste Saison = Elo-Einschwingphase

        if "freq" in models:
            preds["freq"][idx] = frequency_matrix(matches[season_col < season])
        if "elo" in models:
            goal_model = EloGoalModel().fit(elo_diff[past], matches["hg"][past], matches["ag"][past])
            preds["elo"][idx] = goal_model.predict(elo_diff[idx])
        if "dc" in models:
            promoted, relegated = transitions(teams_by_season, season)
            dc = DixonColes(dc_params)
            blocks = date_blocks(matches["date"].iloc[idx])
            for block in np.unique(blocks):
                rows = idx[blocks == block]
                dc.fit(matches, matches["date"].iloc[rows[0]], teams=teams_by_season[season],
                       promoted=promoted, relegated=relegated)
                preds["dc"][rows] = dc.predict(home[rows], away[rows])
        if verbose:
            print(f"  Saison {season}/{(season + 1) % 100:02d}: {len(idx)} Spiele "
                  f"({time.time() - start:.1f}s)")
    return preds


def add_ensembles(matches: pd.DataFrame, preds: dict, seasons) -> dict:
    """Ergänzt "own" und "combined". Gewichte nur aus früheren Saisons."""
    season_col = matches["season"].to_numpy()
    results = matches["result"].to_numpy()
    probs = {k: outcome_probs(np.nan_to_num(v, nan=1.0)) for k, v in preds.items()}
    for k, v in preds.items():
        probs[k][~np.isfinite(v).all(axis=(1, 2))] = np.nan

    shape = next(iter(preds.values())).shape
    preds["own"] = np.full(shape, np.nan)
    preds["combined"] = np.full(shape, np.nan)
    history = []
    for season in seasons:
        idx = np.flatnonzero(season_col == season)
        hist = np.flatnonzero((season_col >= seasons[0]) & (season_col < season))
        entry = {"season": int(season)}
        for name, members in (("own", OWN_MEMBERS), ("combined", COMBINED_MEMBERS)):
            if len(hist):
                w = fit_weights([probs[m][hist] for m in members], results[hist])
            else:
                w = np.full(len(members), 1 / len(members))
            preds[name][idx] = pool([preds[m][idx] for m in members], w)
            entry[name] = dict(zip(members, np.round(w, 3).tolist()))
        history.append(entry)
    return {"history": history}


def final_weights(matches: pd.DataFrame, preds: dict, seasons) -> dict:
    """Gewichte für Live-Vorhersagen: gelernt aus allen Out-of-Sample-Vorhersagen."""
    rows = np.flatnonzero(matches["season"].isin(seasons).to_numpy())
    results = matches["result"].to_numpy()[rows]
    out = {}
    for name, members in (("own", OWN_MEMBERS), ("combined", COMBINED_MEMBERS)):
        w = fit_weights([outcome_probs(preds[m][rows]) for m in members], results)
        out[name] = dict(zip(members, np.round(w, 4).tolist()))
    return out


def kicktipp_points(mats: np.ndarray, matches: pd.DataFrame, rules=DEFAULT_RULES):
    tip_h, tip_a, _ = best_tips(mats, rules)
    pts = points(tip_h, tip_a, matches["hg"].to_numpy(), matches["ag"].to_numpy(), rules)
    return tip_h, tip_a, pts


def summarize(matches: pd.DataFrame, preds: dict, mask: np.ndarray) -> pd.DataFrame:
    sub = matches[mask]
    rows = []
    for key, label in MODELS.items():
        mats = preds[key][mask]
        metrics = evaluate(outcome_probs(mats), sub["result"])
        _, _, pts = kicktipp_points(mats, sub)
        rows.append({"Modell": label, **metrics, "kicktipp": pts.mean()})
    return pd.DataFrame(rows).set_index("Modell")


def per_season(matches: pd.DataFrame, preds: dict, seasons, metric="log_loss") -> pd.DataFrame:
    table = {}
    for season in seasons:
        mask = (matches["season"] == season).to_numpy()
        sub = matches[mask]
        row = {}
        for key, label in MODELS.items():
            mats = preds[key][mask]
            if metric == "kicktipp":
                row[label] = kicktipp_points(mats, sub)[2].sum()
            else:
                row[label] = log_loss(outcome_probs(mats), sub["result"])
        row["Spiele"] = int(mask.sum())
        table[f"{season}/{(season + 1) % 100:02d}"] = row
    return pd.DataFrame(table).T


# --------------------------------------------------------------------- Tuning
ELO_GRID = {
    "k": (8, 10, 12, 15, 20, 25),
    "home_adv": (20, 35, 50, 70),
    "season_regression": (0.1, 0.2, 0.33),
}
DC_GRID = {
    "half_life_days": (180, 270, 365, 540),
    "promoted_strength": (5.0, 10.0, 20.0, 40.0),
    "ridge": (1.0, 3.0, 10.0),
}


def tune(matches: pd.DataFrame, verbose: bool = True) -> tuple[EloParams, DCParams]:
    """Rastersuche auf den Validierungssaisons (Kriterium: Log-Loss)."""
    val_mask = matches["season"].isin(VALIDATION_SEASONS).to_numpy()
    results = matches["result"].to_numpy()[val_mask]

    def score(preds, key):
        return log_loss(outcome_probs(preds[key][val_mask]), results)

    best_elo, best_elo_loss = None, np.inf
    for values in itertools.product(*ELO_GRID.values()):
        params = EloParams(**dict(zip(ELO_GRID, values)))
        loss = score(base_predictions(matches, VALIDATION_SEASONS, params, DCParams(),
                                      models=("elo",)), "elo")
        if loss < best_elo_loss:
            best_elo, best_elo_loss = params, loss
    if verbose:
        print(f"  Elo:         {best_elo}  Log-Loss {best_elo_loss:.4f}")

    best_dc, best_dc_loss = None, np.inf
    for values in itertools.product(*DC_GRID.values()):
        params = DCParams(**dict(zip(DC_GRID, values)))
        loss = score(base_predictions(matches, VALIDATION_SEASONS, EloParams(), params,
                                      models=("dc",)), "dc")
        if loss < best_dc_loss:
            best_dc, best_dc_loss = params, loss
    if verbose:
        print(f"  Dixon-Coles: {best_dc}  Log-Loss {best_dc_loss:.4f}")

    hyperparams.save(best_elo, best_dc, {
        "tuned_on_seasons": list(VALIDATION_SEASONS),
        "validation_log_loss": {"elo": round(best_elo_loss, 5), "dixon_coles": round(best_dc_loss, 5)},
    })
    return best_elo, best_dc


# ------------------------------------------------------------------- Ablauf
def run(matches: pd.DataFrame, verbose: bool = True) -> dict:
    """Kompletter Backtest. Gibt Tabellen und Vorhersagen zurück."""
    elo_params, dc_params = hyperparams.load()
    last = int(matches["season"].max())
    seasons = list(range(VALIDATION_SEASONS[0], last + 1))
    if verbose:
        print(f"Elo: {elo_params}\nDixon-Coles: {dc_params}")

    preds = base_predictions(matches, seasons, elo_params, dc_params, verbose=verbose)
    weight_info = add_ensembles(matches, preds, seasons)

    complete = [s for s in seasons if (matches["season"] == s).sum() >= 306]
    weights = final_weights(matches, preds, complete)
    ENSEMBLE_FILE.parent.mkdir(parents=True, exist_ok=True)
    ENSEMBLE_FILE.write_text(json.dumps({
        "weights": weights,
        "fitted_on_seasons": complete,
        "history": weight_info["history"],
    }, indent=2))

    test_mask = (matches["season"] >= TEST_FIRST_SEASON).to_numpy()
    return {
        "preds": preds,
        "seasons": seasons,
        "test_mask": test_mask,
        "summary": summarize(matches, preds, test_mask),
        "log_loss_by_season": per_season(matches, preds, [s for s in seasons if s >= TEST_FIRST_SEASON]),
        "kicktipp_by_season": per_season(matches, preds, [s for s in seasons if s >= TEST_FIRST_SEASON],
                                         metric="kicktipp"),
        "weights": weights,
    }
