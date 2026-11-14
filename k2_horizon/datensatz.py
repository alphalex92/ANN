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
            ergebnis.append(_paar(c, r))
    return ergebnis


def grund(lauf):
    """Kurzbeschreibung, warum ein schlechter Lauf schlecht ist."""
    if lauf["status"] == "zu_viele_suchen":
        return "abgebrochen"
    if lauf["klasse"] == "gewusst":
        return "unnötig gesucht" if lauf["suchen"] else "ohne Suche falsch"
    return "gesucht, falsch" if lauf["suchen"] else "geraten"


_SUCH_ABSICHT = ("search", "look", "check", "find", "tool", "query", "anfrage", "verify", "browse", "confirm",
                 "not sure", "don't know", "do not know", "uncertain", "not certain", "unable", "cannot", "can't")


def echt_geraten(lauf):
    """Eine Antwort ohne Suche taugt nur dann als schlechtes Beispiel, wenn das Modell wirklich eine
    falsche Antwort behauptet. Wollte es eigentlich suchen und schreibt wegen der Sperre nun
    „I'll look that up“ oder einen Aufruf als Text, ist das kein Raten, sondern ein Artefakt."""
    if lauf["klasse"] != "unbekannt" or lauf["status"] != "fertig" or lauf["suchen"] or lauf["richtig"]:
        return False
    antwort = (lauf["antwort"] or "").strip()
    if len(antwort) < 3 or len(antwort) > 120 or "<" in antwort or "{" in antwort:
        return False
    text = antwort.lower().replace("’", "'")
    return not any(w in text for w in _SUCH_ABSICHT)


def ausgewogene_paare(laeufe, erzwungen, seed=0):
    """Gleich viele Paare in beide Richtungen: „nicht unnötig suchen“ und „suchen statt raten“.

    Unbekannte Fragen: je Frage ein guter Suchlauf gegen eine echte Rate-Antwort, erzwungen
    (`erzwungen`) oder natürlich (in `laeufe`). Dazu ein Paar gegen einen anderen schlechten Lauf
    (Abbruch, gesucht und falsch). Gewusste Fragen: wie in `paare`, zufällig auf dieselbe Anzahl
    gekürzt, damit keine Richtung überwiegt.
    """
    import random
    rng = random.Random(seed)
    gute = defaultdict(list)
    for l in laeufe:
        if l["klasse"] == "unbekannt" and gut(l):
            gute[_frage_id(l)].append(l)
    rate = {}
    for r in laeufe + erzwungen:
        fid = _frage_id(r)
        if echt_geraten(r) and fid in gute and fid not in rate:
            rate[fid] = _paar(rng.choice(gute[fid]), r)
    unbekannt = list(rate.values()) + paare(
        [l for l in laeufe if l["klasse"] == "unbekannt" and (gut(l) or l["suchen"])], je_frage=2, seed=seed)
    gewusst = paare([l for l in laeufe if l["klasse"] == "gewusst"], seed=seed)
    rng.shuffle(gewusst)
    rng.shuffle(unbekannt)
    n = min(len(gewusst), len(unbekannt))
    return gewusst[:n] + unbekannt[:n]


def _paar(c, r):
    return {"id": _frage_id(c), "klasse": c["klasse"], "typ": c["typ"], "tools": [WERKZEUG],
            "prompt": c["gespraech"][:1], "chosen": c["gespraech"][1:], "rejected": r["gespraech"][1:],
            "grund": grund(r)}


def _sucht_zuerst(lauf):
    return len(lauf["gespraech"]) > 1 and bool(lauf["gespraech"][1].get("tool_calls"))


def _erster_zug(c, r, richtung):
    return {"id": _frage_id(c), "klasse": c["klasse"], "typ": c["typ"], "tools": [WERKZEUG],
            "prompt": c["gespraech"][:1], "chosen": c["gespraech"][1:2], "rejected": r["gespraech"][1:2],
            "grund": richtung}


def entscheidungs_paare(laeufe, je_frage=2, seed=0):
    """Paare nur für den ersten Zug: suchen oder direkt antworten. Was danach passiert, zählt nicht.

    mehr suchen   unbekannte Frage: erster Zug mit Suchaufruf (gut) gegen echte Rate-Antwort
    weniger suchen  gewusste Frage: direkt richtig (gut) gegen ersten Zug mit Suchaufruf

    So lernt DPO nur die Entscheidung. Abgebrochene Suchläufe zählen bei unbekannten Fragen als
    gut, denn die Entscheidung zu suchen war richtig; schlecht war erst das Auswerten der Treffer.
    Beide Richtungen bekommen gleich viele Paare.
    """
    import random
    rng = random.Random(seed)
    gruppen = defaultdict(list)
    for l in laeufe:
        gruppen[_frage_id(l)].append(l)
    mehr, weniger = [], []
    for fid in sorted(gruppen):
        g = gruppen[fid]
        if g[0]["klasse"] == "unbekannt":
            gute = [l for l in g if _sucht_zuerst(l)]
            schlechte = [l for l in g if echt_geraten(l)]
            ziel = mehr
        elif g[0]["klasse"] == "gewusst":
            gute = [l for l in g if gut(l)]
            schlechte = [l for l in g if _sucht_zuerst(l)]
            ziel = weniger
        else:
            continue
        rng.shuffle(gute)
        rng.shuffle(schlechte)
        if ziel is mehr:
            # Echte Rate-Antworten sind selten. Jede bekommt deshalb bis zu zwei verschiedene
            # Suchzüge als Gegenstück; das Paar bleibt trotzdem ein anderes.
            kombis = [(c, r) for r in schlechte for c in gute[:2]][:2 * je_frage]
        else:
            kombis = list(zip(gute, schlechte))[:je_frage]
        richtung = "mehr suchen" if ziel is mehr else "weniger suchen"
        ziel += [_erster_zug(c, r, richtung) for c, r in kombis]
    rng.shuffle(mehr)
    rng.shuffle(weniger)
    n = min(len(mehr), len(weniger))
    return mehr[:n] + weniger[:n]


def stellen(laeufe, je_frage=1):
    """Stellen zum Neu-Weiterschreiben: gescheiterte Suchläufe zu unbekannten Fragen, bei denen die
    Antwort schon in den Suchtreffern stand. Die Stelle ist das Gespräch bis einschließlich der
    ersten Treffer mit der Antwort. Je Frage höchstens `je_frage` Stellen, in Dateireihenfolge."""
    from .bewertung import richtig

    gezaehlt, ergebnis = Counter(), []
    for lauf in laeufe:
        if lauf["klasse"] != "unbekannt" or lauf["richtig"] or lauf["status"] == "abgeschnitten":
            continue
        fid = _frage_id(lauf)
        if gezaehlt[fid] >= je_frage:
            continue
        g = lauf["gespraech"]
        for i, m in enumerate(g):
            if m["role"] == "tool" and richtig(m["content"], lauf["loesungen"]):
                anfang = g[:i + 1]
                ergebnis.append({"id": fid, "frage": lauf["frage"], "typ": lauf["typ"],
                                 "loesungen": lauf["loesungen"], "klasse": lauf["klasse"], "teil": lauf["teil"],
                                 "quelle": lauf["id"], "gespraech": anfang,
                                 "suchen": sum(x["role"] == "tool" for x in anfang)})
                gezaehlt[fid] += 1
                break
    return ergebnis


_AUSWEICHEND = ("specify", "clarify", "which one do you mean", "which do you mean", "could you tell",
                "please provide", "more context", "more information")


def ausweichend(antwort):
    """Weicht die Antwort aus, statt etwas zu nennen (z. B. "Please specify which film you mean")?"""
    a = (antwort or "").lower()
    return any(w in a for w in _AUSWEICHEND)


def fortsetzungs_set(eigene, lehrer, je_stelle=1, seed=0):
    """Trainingsbeispiele aus fortgesetzten Stellen.

    Je Frage zuerst K2s eigene richtige Fortsetzungen (höchstens `je_stelle`), nur wenn es keine
    gibt, die richtige Fortsetzung des Lehrers. Ausweichende Antworten fallen weg. `lernen_ab` ist
    der Index der ersten neuen Nachricht: Nur ab dort wird gelernt, der Anfang ist Kontext.
    """
    import random

    def brauchbar(l):
        return l["status"] == "fertig" and l["richtig"] and not ausweichend(l["antwort"])

    rng, je_frage = random.Random(seed), defaultdict(list)
    for l in eigene:
        if brauchbar(l):
            je_frage[_frage_id(l)].append(l)
    beispiele = []
    for fid in sorted(je_frage):
        ls = je_frage[fid]
        rng.shuffle(ls)
        beispiele += [dict(l, quelle_art="k2") for l in ls[:je_stelle]]
    for l in lehrer:
        if _frage_id(l) not in je_frage and brauchbar(l):
            beispiele.append(dict(l, quelle_art="lehrer"))
    return [{"id": l["id"], "klasse": l["klasse"], "quelle": l["quelle_art"], "tools": [WERKZEUG],
             "messages": l["gespraech"], "lernen_ab": l["start"]} for l in beispiele]


def such_beispiele(laeufe, anzahl, seed=0):
    """Ganze gute Suchläufe zu unbekannten Fragen (je Frage einer), als Gegengewicht zum
    Fortsetzungs-Set: Sie zeigen, dass bei Unbekanntem zuerst gesucht wird."""
    import random

    je_frage = {}
    for l in laeufe:
        if l["klasse"] == "unbekannt" and gut(l) and not ausweichend(l["antwort"]):
            je_frage.setdefault(_frage_id(l), l)
    auswahl = random.Random(seed).sample(sorted(je_frage.values(), key=lambda l: l["id"]), min(anzahl, len(je_frage)))
    return [{"id": l["id"], "klasse": l["klasse"], "quelle": "suchlauf", "tools": [WERKZEUG],
             "messages": l["gespraech"]} for l in auswahl]
