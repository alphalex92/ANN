"""Wissensprüfung: Welche Faktenfragen beantwortet das Modell ohne Suche zuverlässig richtig?

Jede Frage wird `versuche`-mal mit Sampling gestellt (temperature 1.0). Ein Fakt gilt als
gewusst, wenn fast alle Versuche stimmen. Erfindet das Modell bei jedem Versuch etwas anderes,
weiß es die Antwort nicht. Das Ergebnis landet zeilenweise in einer JSONL-Datei. Ein
abgebrochener Lauf setzt beim nächsten Start dort fort.
"""
import json
import os
import time

import torch

from .bewertung import STEUER, einteilen, richtig
from .modell import aufteilen

ANWEISUNG = "Answer with only the answer, no explanation.\nQuestion: {frage}"
PAD = 1  # <|ifm|endoftext|>


def _stapel(tok, model, fragen, versuche, aufwand, max_neu):
    prompts = [tok.apply_chat_template(
        [{"role": "user", "content": ANWEISUNG.format(frage=f["frage"])}],
        add_generation_prompt=True, tokenize=False, reasoning_effort=aufwand,
    ) for f in fragen]
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    with torch.no_grad():
        aus = model.generate(**enc, max_new_tokens=max_neu, do_sample=True, temperature=1.0,
                             top_p=0.95, num_return_sequences=versuche, pad_token_id=PAD)
    neu = aus[:, enc["input_ids"].shape[1]:]
    texte = [tok.decode(x, skip_special_tokens=False) for x in neu]
    antworten = [STEUER.sub("", aufteilen(t)[1]).strip() for t in texte]
    return [antworten[i * versuche:(i + 1) * versuche] for i in range(len(fragen))]


def pruefen(tok, model, fragen, ziel, versuche=5, aufwand="low", stapel=16, max_neu=96):
    """Prüft alle Fragen und hängt die Ergebnisse an `ziel` an. Bereits geprüfte werden übersprungen."""
    tok.padding_side = "left"
    fertig = set()
    if os.path.exists(ziel):
        with open(ziel, encoding="utf-8") as f:
            fertig = {json.loads(z)["id"] for z in f if z.strip()}
    offen = [f for f in fragen if f["id"] not in fertig]
    print(f"{len(fertig)} schon geprüft, {len(offen)} offen")

    start = time.perf_counter()
    with open(ziel, "a", encoding="utf-8") as datei:
        for i in range(0, len(offen), stapel):
            teil = offen[i:i + stapel]
            for f, antworten in zip(teil, _stapel(tok, model, teil, versuche, aufwand, max_neu)):
                treffer = sum(richtig(a, f["loesungen"]) for a in antworten)
                zeile = dict(f, antworten=antworten, treffer=treffer, versuche=versuche,
                             aufwand=aufwand, klasse=einteilen(treffer, versuche))
                datei.write(json.dumps(zeile, ensure_ascii=False) + "\n")
            datei.flush()
            erledigt = i + len(teil)
            rate = erledigt / (time.perf_counter() - start)
            print(f"{erledigt}/{len(offen)} Fragen, {rate:.1f}/s, "
                  f"noch etwa {(len(offen) - erledigt) / rate / 60:.0f} min", flush=True)


def suche_messen(index_ordner, wissen_datei, anzahl=500, teil="val", seed=0):
    """Misst die Suche mit der Frage als Anfrage.

    Artikel gefunden: Die Wikipedia-Seite des Fragegegenstands ist unter den 20 Artikeln aus Stufe 1.
    Antwort in Treffern: Eine erlaubte Lösung steht in den 5 zurückgegebenen Abschnitten.
    """
    import random
    from collections import defaultdict

    from .suche import Suche, als_text

    with open(wissen_datei, encoding="utf-8") as f:
        zeilen = [json.loads(z) for z in f if z.strip()]
    zeilen = [z for z in zeilen if z["teil"] == teil]
    zeilen = random.Random(seed).sample(zeilen, min(anzahl, len(zeilen)))

    start = time.perf_counter()
    index = Suche(index_ordner)
    print(f"Index geladen in {time.perf_counter() - start:.0f} s, {len(index.titel):,} Artikel")

    zahlen = defaultdict(lambda: [0, 0, 0])  # Anzahl, Artikel gefunden, Antwort in Treffern
    start = time.perf_counter()
    for z in zeilen:
        kandidaten = [index.titel[d] for d in index.artikel_finden(z["frage"])]
        antwort_da = richtig(als_text(index.suchen(z["frage"])), z["loesungen"])
        for schluessel in ("alle", z["klasse"], z["typ"]):
            zahlen[schluessel][0] += 1
            zahlen[schluessel][1] += z["wiki_titel"] in kandidaten
            zahlen[schluessel][2] += antwort_da
    dauer = (time.perf_counter() - start) / len(zeilen)

    print(f"{len(zeilen)} Fragen aus '{teil}', {dauer * 1000:.0f} ms pro Suche\n")
    print(f"{'':16s} {'Anzahl':>6s} {'Artikel gefunden':>17s} {'Antwort in Treffern':>20s}")
    for schluessel, (n, artikel, antwort) in sorted(zahlen.items(), key=lambda x: -x[1][0]):
        print(f"{schluessel:16s} {n:6d} {artikel / n:17.0%} {antwort / n:20.0%}")


def zusammenfassen(ziel):
    """Zählt Klassen gesamt und je Fragetyp."""
    from collections import Counter, defaultdict
    with open(ziel, encoding="utf-8") as f:
        zeilen = [json.loads(z) for z in f if z.strip()]
    gesamt = Counter(z["klasse"] for z in zeilen)
    je_typ = defaultdict(Counter)
    for z in zeilen:
        je_typ[z["typ"]][z["klasse"]] += 1
    return len(zeilen), gesamt, je_typ
