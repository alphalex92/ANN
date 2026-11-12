"""Messen, ob das Modell richtig zwischen Wissen und Nachschlagen entscheidet.

Das Modell beantwortet Validierungsfragen mit Zugriff auf die Suche, aber ohne den Hinweis im
Systemprompt. So verhält es sich auch nach dem Training. Gezählt wird je Wissensklasse:

  gewusst    direkt richtig (gut) · unnötig gesucht · ohne Suche falsch
  unbekannt  gesucht und richtig (gut) · geraten statt gesucht · gesucht, aber falsch · abgebrochen
  unklar     wie oft gesucht wird und wie oft die Antwort stimmt, ohne Wertung
"""
from collections import Counter, defaultdict


def auswahl(zeilen, teil="val", unbekannt=200, seed=0):
    """Alle gewussten und unklaren Fragen des Teils plus eine feste Stichprobe unbekannter."""
    import random
    zeilen = [z for z in zeilen if z["teil"] == teil]
    rest = [z for z in zeilen if z["klasse"] == "unbekannt"]
    return [z for z in zeilen if z["klasse"] != "unbekannt"] + random.Random(seed).sample(rest, min(unbekannt, len(rest)))


def ergebnis(lauf):
    """Ordnet einen Lauf einer Kategorie zu."""
    if lauf["status"] != "fertig":
        return "abgebrochen"
    gesucht = lauf["suchen"] > 0
    if lauf["klasse"] == "gewusst":
        if gesucht:
            return "unnötig gesucht"
        return "direkt richtig" if lauf["richtig"] else "ohne Suche falsch"
    if lauf["klasse"] == "unbekannt":
        if not gesucht:
            return "geraten, zufällig richtig" if lauf["richtig"] else "geraten, falsch"
        return "gesucht, richtig" if lauf["richtig"] else "gesucht, falsch"
    return ("gesucht" if gesucht else "direkt") + (", richtig" if lauf["richtig"] else ", falsch")


def auswerten(laeufe):
    """Gibt den Bericht als Text zurück."""
    zeilen = []
    for klasse in ("gewusst", "unbekannt", "unklar"):
        teil = [l for l in laeufe if l["klasse"] == klasse]
        if not teil:
            continue
        n = len(teil)
        c = Counter(ergebnis(l) for l in teil)
        gesucht = sum(l["suchen"] > 0 for l in teil)
        richtig = sum(l["richtig"] for l in teil)
        zeilen.append(f"{klasse} ({n}): gesucht {gesucht / n:.0%}, richtig {richtig / n:.0%}")
        for k, v in c.most_common():
            zeilen.append(f"    {k:28s} {v:4d}  {v / n:4.0%}")
        je_typ = defaultdict(list)
        for l in teil:
            je_typ[l["typ"]].append(l)
        for typ, ls in sorted(je_typ.items(), key=lambda x: -len(x[1])):
            m = len(ls)
            zeilen.append(f"      {typ:16s} {m:4d}  gesucht {sum(l['suchen'] > 0 for l in ls) / m:4.0%}"
                          f"  richtig {sum(l['richtig'] for l in ls) / m:4.0%}")
    n = len(laeufe)
    if not n:
        return "Keine Läufe."
    zeilen.append(f"gesamt ({n}): richtig {sum(l['richtig'] for l in laeufe) / n:.0%}, "
                  f"Suchen je Frage {sum(l['suchen'] for l in laeufe) / n:.2f}")
    return "\n".join(zeilen)
