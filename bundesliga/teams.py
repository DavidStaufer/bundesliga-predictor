"""Teamnamen vereinheitlichen.

Die Ergebnisse und Quoten kommen von football-data.co.uk, der Spielplan von
OpenLigaDB. Beide schreiben Vereinsnamen unterschiedlich. Intern gelten immer
die football-data-Namen. Unbekannte Namen führen zu einem klaren Fehler statt
eines still übersprungenen Spiels.
"""

import re
import unicodedata

# Alle Bundesligisten seit 2005/06 in der Schreibweise von football-data.co.uk.
CANONICAL = (
    "Aachen", "Augsburg", "Bayern Munich", "Bielefeld", "Bochum", "Braunschweig",
    "Cottbus", "Darmstadt", "Dortmund", "Duisburg", "Ein Frankfurt", "Elversberg",
    "FC Koln", "Fortuna Dusseldorf", "Freiburg", "Greuther Furth", "Hamburg",
    "Hannover", "Hansa Rostock", "Heidenheim", "Hertha", "Hoffenheim",
    "Holstein Kiel", "Ingolstadt", "Kaiserslautern", "Karlsruhe", "Leverkusen",
    "M'gladbach", "Mainz", "Nurnberg", "Paderborn", "RB Leipzig", "Schalke 04",
    "St Pauli", "Stuttgart", "Union Berlin", "Werder Bremen", "Wolfsburg",
)

# Namen, die sich nicht über den Kern-Schlüssel (siehe _core_key) zuordnen lassen.
ALIASES = {
    "fc bayern munchen": "Bayern Munich",
    "bayern munchen": "Bayern Munich",
    "eintracht frankfurt": "Ein Frankfurt",
    "borussia monchengladbach": "M'gladbach",
    "monchengladbach": "M'gladbach",
    "hamburger sv": "Hamburg",
    "hertha bsc": "Hertha",
    "hertha bsc berlin": "Hertha",
    "arminia bielefeld": "Bielefeld",
    "dsc arminia bielefeld": "Bielefeld",
    "eintracht braunschweig": "Braunschweig",
    "karlsruher sc": "Karlsruhe",
    "energie cottbus": "Cottbus",
    "fc energie cottbus": "Cottbus",
    "alemannia aachen": "Aachen",
    "rasenballsport leipzig": "RB Leipzig",
    "1 fc koln": "FC Koln",
}

# Füllwörter in Vereinsnamen, die für die Zuordnung keine Rolle spielen.
_NOISE = {
    "1", "fc", "sv", "vfl", "vfb", "tsg", "sc", "fsv", "msv", "spvgg", "bayer",
    "borussia", "sport", "club", "04", "05", "07", "96", "98", "1846", "1899", "1910",
}


def _key(name: str) -> str:
    """Kleinschreibung, ohne Umlaute, Akzente und Satzzeichen."""
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    ascii_name = re.sub(r"[^a-z0-9 ]", " ", ascii_name.lower())
    return " ".join(ascii_name.split())


def _core_key(name: str) -> str:
    return " ".join(t for t in _key(name).split() if t not in _NOISE)


_BY_KEY = {_key(n): n for n in CANONICAL}
_BY_CORE = {_core_key(n): n for n in CANONICAL}


class UnknownTeamError(ValueError):
    pass


def canonical(name: str) -> str:
    """Bildet einen beliebigen Vereinsnamen auf den football-data-Namen ab."""
    key = _key(name)
    if key in _BY_KEY:
        return _BY_KEY[key]
    if key in ALIASES:
        return ALIASES[key]
    core = _core_key(name)
    if core in _BY_CORE:
        return _BY_CORE[core]
    raise UnknownTeamError(
        f"Unbekannter Vereinsname: {name!r}. Bitte in bundesliga/teams.py "
        f"in CANONICAL oder ALIASES eintragen."
    )
