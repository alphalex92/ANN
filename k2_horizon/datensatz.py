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


def schlecht(lauf):
    """Zeigt der Lauf genau das Verhalten, das wir abgewöhnen wollen?

    gewusst    gesucht (unnötig), oder ohne Suche falsch
    unbekannt  geraten statt gesucht, nach zu vielen Suchen abgebrochen, oder gesucht und falsch
    Abgeschnittene Läufe (Tokenlimit im Denkteil) taugen weder als gutes noch als schlechtes Beispiel.
    """
    if lauf["status"] == "abgeschnitten" or lauf["klasse"] not in ("gewusst", "unbekannt"):
        return False
    return not gut(lauf)


def _frage_id(lauf):
    return str(lauf["id"]).split("#")[0]


def paare(laeufe, je_frage=2, seed=0):
    """Bildet Vergleichspaare (gut gegen schlecht) aus mehreren Läufen derselben Frage.

    Jedes Paar teilt die Frage als Prompt. `chosen` und `rejected` sind die Fortsetzungen danach:
    Gedanken, Suchaufrufe, Treffer, Antwort. Pro Frage höchstens `je_frage` Paare, damit leichte
    Fragen mit vielen Läufen nicht dominieren. Jeder Lauf kommt höchstens in einem Paar vor.
    """
    import random
    rng = random.Random(seed)
    gruppen = defaultdict(list)
    for l in laeufe:
        gruppen[_frage_id(l)].append(l)
    ergebnis = []
    for fid in sorted(gruppen):
        g = gruppen[fid]
        gute, schlechte = [l for l in g if gut(l)], [l for l in g if schlecht(l)]
        rng.shuffle(gute)
        rng.shuffle(schlechte)
        for c, r in list(zip(gute, schlechte))[:je_frage]:
            ergebnis.append({
                "id": fid, "klasse": c["klasse"], "typ": c["typ"], "tools": [WERKZEUG],
                "prompt": c["gespraech"][:1], "chosen": c["gespraech"][1:], "rejected": r["gespraech"][1:],
                "grund": grund(r),
            })
    return ergebnis


def grund(lauf):
    """Kurzbeschreibung, warum ein schlechter Lauf schlecht ist."""
    if lauf["status"] == "zu_viele_suchen":
        return "abgebrochen"
    if lauf["klasse"] == "gewusst":
        return "unnötig gesucht" if lauf["suchen"] else "ohne Suche falsch"
    return "gesucht, falsch" if lauf["suchen"] else "geraten"
