"""Gespeicherte Hyperparameter laden und schreiben (models/hyperparams.json)."""

import json

from .config import HYPERPARAMS_FILE
from .models.dixon_coles import DCParams
from .models.elo import EloParams


def load() -> tuple[EloParams, DCParams]:
    """Getunte Werte, falls vorhanden, sonst die Standardwerte."""
    if not HYPERPARAMS_FILE.exists():
        return EloParams(), DCParams()
    saved = json.loads(HYPERPARAMS_FILE.read_text())
    return EloParams(**saved["elo"]), DCParams(**saved["dixon_coles"])


def save(elo: EloParams, dc: DCParams, info: dict) -> None:
    HYPERPARAMS_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {"elo": elo.to_dict(), "dixon_coles": dc.to_dict(), **info}
    HYPERPARAMS_FILE.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
