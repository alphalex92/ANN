"""Faktenfragen laden und Antworten gegen die erlaubten Lösungen prüfen. Ohne torch, damit schnell testbar."""
import csv
import json
import random
import re
import unicodedata

# Fragetypen, bei denen Raten selten trifft. Weggelassen: country (33 % "USA"),
# sport (60 % Fußball), occupation (26 % Politiker), religion (40 % katholisch),
# color (74 % rot) und genre (wenige, oft erratbare Kategorien).
TYPEN = ("director", "screenwriter", "producer", "author", "composer", "capital",
         "place of birth", "father", "mother", "capital of")

STEUER = re.compile(r"<\|?/?ifm\|[^>]*>")


def normalisieren(text):
    """Kleinbuchstaben, ohne Akzente, Satzzeichen und Artikel am Anfang."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(z for z in text if not unicodedata.combining(z)).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"^(the|a|an)\s+", "", text.strip())
    return " ".join(text.split())


def richtig(antwort, loesungen):
    """True, wenn eine Lösung als ganzes Wort in der Antwort steht oder ihr genau entspricht.

    Lösungen mit weniger als 3 Zeichen zählen nur bei genauer Übereinstimmung,
    sonst würde z. B. "la" in fast jeder Antwort stecken.
    """
    a = normalisieren(STEUER.sub(" ", antwort))
    if not a:
        return False
    for l in loesungen:
        l = normalisieren(l)
        if not l:
            continue
        if a == l or (len(l) >= 3 and re.search(rf"\b{re.escape(l)}\b", a)):
            return True
    return False


def fragen_laden(pfad, typen=TYPEN):
    """Liest die PopQA-TSV und gibt nur die gewünschten Fragetypen zurück."""
    with open(pfad, encoding="utf-8") as f:
        zeilen = list(csv.DictReader(f, delimiter="\t"))
    return [{
        "id": z["id"], "frage": z["question"], "typ": z["prop"],
        "loesungen": json.loads(z["possible_answers"]), "beliebtheit": int(z["s_pop"]),
        "wiki_titel": z["s_wiki_title"],
    } for z in zeilen if z["prop"] in typen]


def stichprobe(fragen, anzahl, seed=0, anteile=(0.8, 0.1, 0.1)):
    """Zufällige Auswahl, aufgeteilt in train/val/test. Die Aufteilung hängt nur vom Seed ab."""
    rng = random.Random(seed)
    auswahl = rng.sample(fragen, min(anzahl, len(fragen)))
    grenze1 = round(len(auswahl) * anteile[0])
    grenze2 = grenze1 + round(len(auswahl) * anteile[1])
    for i, f in enumerate(auswahl):
        f["teil"] = "train" if i < grenze1 else "val" if i < grenze2 else "test"
    return auswahl


def einteilen(treffer, versuche, gewusst_ab=0.8, unbekannt_bis=0.2):
    """'gewusst', 'unbekannt' oder 'unklar' aus dem Anteil richtiger Versuche."""
    anteil = treffer / versuche
    if anteil >= gewusst_ab:
        return "gewusst"
    if anteil <= unbekannt_bis:
        return "unbekannt"
    return "unklar"
