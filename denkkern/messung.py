"""Messung: das Modell frei laufen lassen, die Umgebung beantwortet seine Suchen.

Gemessen wird je Hop-Zahl und je Schleifen-Zahl, damit man zwei Fragen getrennt sieht:
  - Generalisiert es auf längere Ketten als im Training?
  - Hilft dabei mehr Rechnen, also mehr Durchläufe durch den Kern?

Im Modus werkzeug gibt es dieselben Kategorien wie bei K2 Horizon: richtig, gesucht aber
falsch, geraten (geantwortet, ohne genug gesucht zu haben) und abgebrochen.
"""

import random
from collections import Counter

import torch

from . import welt


@torch.no_grad()
def ausrollen(modell, episoden, modus, schleifen):
    """Alle Episoden gleichzeitig, gierig (immer das wahrscheinlichste Token).

    Die Folgen dürfen verschieden lang sein: Sie stehen linksbündig in einem Tensor, rechts
    mit PAD aufgefüllt, und jede Folge liest ihre Logits an ihrer eigenen letzten Stelle.
    Weil die Attention kausal ist, stört das PAD dahinter nicht. Fertige Episoden laufen
    einfach mit, ihr Ergebnis zählt nicht mehr.
    """
    modell.eval()
    geraet = next(modell.parameters()).device
    B = len(episoden)
    folgen = [welt.anfang(ep, modus) for ep in episoden]
    platz = modell.konfig.kontext
    folge = torch.full((B, platz), welt.PAD, dtype=torch.long, device=geraet)
    for i, f in enumerate(folgen):
        folge[i, : len(f)] = torch.tensor(f)
    laenge = torch.tensor([len(f) for f in folgen], device=geraet)
    zeilen = torch.arange(B, device=geraet)
    verlaeufe = [welt.Verlauf() for _ in episoden]
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if geraet.type == "cuda" else torch.autocast("cpu", enabled=False)

    for _ in range(max(welt.max_schritte(ep, modus) for ep in episoden)):
        for i, v in enumerate(verlaeufe):
            if not v.fertig and laenge[i].item() + welt.AKTION_LAENGE > platz:
                v.ungueltig = True  # kein Platz mehr im Kontext: gilt als abgebrochen
        offen = [i for i, v in enumerate(verlaeufe) if not v.fertig]
        if not offen:
            break
        aktiv = torch.zeros(B, dtype=torch.bool, device=geraet)
        aktiv[offen] = True
        for _ in range(welt.AKTION_LAENGE):
            bis = laenge[aktiv].max().item()
            with autocast:
                logits = modell(folge[:, :bis], schleifen)
            stelle = (laenge - 1).clamp(max=bis - 1)
            naechstes = logits[zeilen, stelle].argmax(dim=-1)
            folge[zeilen[aktiv], laenge[aktiv]] = naechstes[aktiv]
            laenge = laenge + aktiv
        aktionen = {i: folge[i, laenge[i] - welt.AKTION_LAENGE : laenge[i]].tolist() for i in offen}
        for i in offen:
            antwort = welt.schritt(episoden[i], modus, verlaeufe[i], aktionen[i])
            if modus == "kompakt" and antwort:
                # Verlauf wegwerfen: frischer Notizzettel statt Anhängen
                antwort = welt.notiz(episoden[i], antwort)
                folge[i] = welt.PAD
                laenge[i] = 0
            n = len(antwort)
            if n and laenge[i].item() + n <= platz:
                folge[i, laenge[i] : laenge[i] + n] = torch.tensor(antwort, device=geraet)
                laenge[i] += n
            elif n:
                verlaeufe[i].ungueltig = True
    return verlaeufe, [folge[i, : laenge[i]].tolist() for i in range(B)]


def messen(modell, modus, hops_liste, schleifen_liste, anzahl=500, seed=12345, stapel=250):
    """Ergebnis: {(hops, schleifen): {"kategorien": Counter, "weg_exakt": int, "anzahl": n}}."""
    ergebnis = {}
    for hops in hops_liste:
        # Dieselben Episoden für alle Schleifen-Zahlen, damit der Vergleich fair ist.
        rng = random.Random(seed + hops)
        episoden = [welt.neue_episode(rng, hops) for _ in range(anzahl)]
        for schleifen in schleifen_liste:
            kategorien = Counter()
            weg_exakt = 0
            for i in range(0, anzahl, stapel):
                teil = episoden[i : i + stapel]
                verlaeufe, _ = ausrollen(modell, teil, modus, schleifen)
                for ep, v in zip(teil, verlaeufe):
                    kategorien[welt.einordnen(ep, modus, v)] += 1
                    weg_exakt += v.weg_exakt and v.suchen == ep.hops
            ergebnis[(hops, schleifen)] = {"kategorien": kategorien, "weg_exakt": weg_exakt, "anzahl": anzahl}
    return ergebnis


def tabelle_richtig(ergebnis, hops_liste, schleifen_liste, trainiert_hops=None):
    """Richtig in % als Tabelle: Zeilen Hops, Spalten Schleifen."""
    kopf = "| Hops | " + " | ".join(f"{s} Schl." for s in schleifen_liste) + " |"
    zeilen = [kopf, "|---" * (len(schleifen_liste) + 1) + "|"]
    for h in hops_liste:
        marke = ""
        if trainiert_hops and not trainiert_hops[0] <= h <= trainiert_hops[1]:
            marke = " (neu)"
        zellen = []
        for s in schleifen_liste:
            r = ergebnis[(h, s)]
            zellen.append(f"{100 * r['kategorien']['richtig'] / r['anzahl']:.0f} %")
        zeilen.append(f"| {h}{marke} | " + " | ".join(zellen) + " |")
    return "\n".join(zeilen)


def tabelle_kategorien(ergebnis, modus, hops_liste, schleifen):
    kategorien = welt.KATEGORIEN[modus]
    spalten = kategorien + (["Suchweg exakt"] if modus in welt.MIT_WERKZEUG else [])
    zeilen = ["| Hops | " + " | ".join(spalten) + " |", "|---" * (len(spalten) + 1) + "|"]
    for h in hops_liste:
        r = ergebnis[(h, schleifen)]
        n = r["anzahl"]
        zellen = [f"{100 * r['kategorien'][k] / n:.0f} %" for k in kategorien]
        if modus in welt.MIT_WERKZEUG:
            zellen.append(f"{100 * r['weg_exakt'] / n:.0f} %")
        zeilen.append(f"| {h} | " + " | ".join(zellen) + " |")
    return "\n".join(zeilen)
