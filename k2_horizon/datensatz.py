"""Aus den erzeugten Lösungswegen das Trainingsset bauen.

Behalten wird nur richtiges Verhalten:
  gewusst    ohne Suche geantwortet, und die Antwort stimmt
  unbekannt  mindestens einmal gesucht, und die Antwort stimmt

Der Hinweis aus dem Systemprompt steht in den gespeicherten Gesprächen gar nicht erst drin
(siehe agent.py). Das Modell lernt also, sich so zu verhalten, ohne dass man es ihm sagt.

Format je Zeile: {"tools": [...], "messages": [...]}, wie es Transformers, TRL und Unsloth
für Chat-Daten mit Werkzeugen erwarten.
"""
import json
from collections import Counter, defaultdict

from .agent import WERKZEUG


def gut(lauf):
    """Zeigt der Lauf das gewünschte Verhalten?"""
    if lauf["status"] != "fertig" or not lauf["richtig"]:
        return False
    if lauf["klasse"] == "gewusst":
        return lauf["suchen"] == 0
    if lauf["klasse"] == "unbekannt":
        return lauf["suchen"] >= 1
    return False


def laden(pfad):
    with open(pfad, encoding="utf-8") as f:
        return [json.loads(z) for z in f if z.strip()]


def bauen(laeufe, ziel):
    """Schreibt die guten Läufe als Trainingsbeispiele und gibt eine Statistik zurück."""
    stat = defaultdict(Counter)
    with open(ziel, "w", encoding="utf-8") as f:
        for lauf in laeufe:
            behalten = gut(lauf)
            for schluessel in (lauf["klasse"], f"{lauf['klasse']}/{lauf['typ']}"):
                stat[schluessel]["gesamt"] += 1
                stat[schluessel]["behalten"] += behalten
            if behalten:
                stat[lauf["klasse"]][f"suchen={lauf['suchen']}"] += 1
                f.write(json.dumps({"id": lauf["id"], "klasse": lauf["klasse"], "tools": [WERKZEUG],
                                    "messages": lauf["gespraech"]}, ensure_ascii=False) + "\n")
    return stat


def gruende(laeufe):
    """Warum Läufe verworfen wurden, je Klasse."""
    c = defaultdict(Counter)
    for l in laeufe:
        if gut(l):
            continue
        if l["status"] != "fertig":
            grund = l["status"]
        elif l["klasse"] == "gewusst" and l["suchen"] > 0:
            grund = "unnötig gesucht" + (", aber richtig" if l["richtig"] else ", falsch")
        elif l["klasse"] == "unbekannt" and l["suchen"] == 0:
            grund = "geraten statt gesucht" + (", zufällig richtig" if l["richtig"] else ", falsch")
        else:
            grund = "falsche Antwort"
        c[l["klasse"]][grund] += 1
    return c
