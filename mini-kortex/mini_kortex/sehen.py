"""Visuelles Sudoku: Die Vorgaben sind handgeschriebene Ziffern (MNIST), keine Zahlen.

Bisher bekam der Mini-Kortex das Rätsel fehlerfrei als Zahlen. Jetzt muss ein **Auge** (kleines
CNN) die Ziffern erst erkennen, und das Auge irrt sich manchmal. Ein einziger Lesefehler macht
das ganze Rätsel falsch. Schlimmer: Der Regelprüfer prüft nur gegen das, was das Auge gesehen hat.
Ist das gelesene Rätsel lösbar, bestätigt er eine Lösung des falschen Rätsels.

Verglichen werden drei Abläufe:

    KETTE        Erst erkennen, dann lösen. Jedes Modul arbeitet blind für das andere.
    ARBEITSRAUM  Erkennen und Lösen tauschen sich aus:
                   1. Das Werkzeug findet Vorgaben, die sich widersprechen (zwei gleiche Ziffern
                      in einer Zeile, Spalte oder einem Kasten). Mindestens eine davon ist falsch gelesen.
                   2. Findet der Denkkern keine regelkonforme Lösung, ist vermutlich etwas falsch gelesen.
                   3. Dann "schaut das System nochmal hin": Es liest die verdächtigsten Felder
                      anders, und zwar so, wie das Auge sie am zweitliebsten gelesen hätte. Verdächtig
                      ist, was im Widerspruch steht oder wo das Auge unsicher war.
                   4. Jede Lesart wird durchgedacht. Von den regelkonformen Lösungen gewinnt die,
                      die am besten zu den Bildern passt (höchste Wahrscheinlichkeit laut Auge).
    ORAKEL       Das Rätsel als fehlerfreie Zahlen, wie bisher. Die Obergrenze.
"""

import gzip
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .sudoku import feld_fuer
from .weltmodell import erlaubte_ziffern, nachdenken, probieren

# Die vier MNIST-Originaldateien (*-ubyte.gz). Standard: Ordner `mnist` im aktuellen Verzeichnis.
MNIST_ORDNER = Path(os.environ.get("MINI_KORTEX_MNIST", "mnist"))


def mnist_laden(ordner=MNIST_ORDNER, teil="train"):
    """Liest die MNIST-Originaldateien (idx-Format, gzip). Rückgabe: Bilder (N, 28, 28) uint8, Ziffern (N,)."""
    vorsilbe = "train" if teil == "train" else "t10k"
    if not (Path(ordner) / f"{vorsilbe}-images-idx3-ubyte.gz").exists():
        raise FileNotFoundError(f"MNIST fehlt in {ordner}. Herunterladen von "
                                "https://storage.googleapis.com/cvdf-datasets/mnist/ oder MINI_KORTEX_MNIST setzen.")
    with gzip.open(Path(ordner) / f"{vorsilbe}-images-idx3-ubyte.gz") as f:
        bilder = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)
    with gzip.open(Path(ordner) / f"{vorsilbe}-labels-idx1-ubyte.gz") as f:
        ziffern = np.frombuffer(f.read(), np.uint8, offset=8).astype(np.int64)
    return bilder, ziffern


class Auge(nn.Module):
    """Kleines CNN: 28×28-Bild → 10 Klassen (0 = leeres Feld, 1–9 = Ziffer)."""

    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 3)
        self.c2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, bilder):
        h = bilder.float().unsqueeze(1) / 255.0
        h = F.max_pool2d(F.relu(self.c1(h)), 2)
        h = F.max_pool2d(F.relu(self.c2(h)), 2)
        return self.fc2(F.relu(self.fc1(h.flatten(1))))

    @property
    def geraet(self):
        return next(self.parameters()).device


def _ohne_null(bilder, ziffern):
    """Sudoku kennt nur 1–9: Die MNIST-Nullen fallen weg."""
    behalten = ziffern > 0
    return bilder[behalten], ziffern[behalten]


def auge_trainieren(epochen=1, batch=128, lernrate=1e-3, seed=0, geraet="cpu", protokoll=print):
    """Trainiert das Auge auf den MNIST-Trainingsbildern (1–9) plus leeren Feldern."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    bilder, ziffern = _ohne_null(*mnist_laden(teil="train"))
    leer = len(bilder) // 9  # etwa so viele leere Felder wie Bilder einer Ziffer
    bilder = np.concatenate([bilder, np.zeros((leer, 28, 28), np.uint8)])
    ziffern = np.concatenate([ziffern, np.zeros(leer, np.int64)])
    auge = Auge().to(geraet)
    opt = torch.optim.Adam(auge.parameters(), lernrate)
    b_t, z_t = torch.as_tensor(bilder).to(geraet), torch.as_tensor(ziffern).to(geraet)
    for epoche in range(epochen):
        reihenfolge = torch.as_tensor(rng.permutation(len(bilder))).to(geraet)
        for start in range(0, len(bilder), batch):
            idx = reihenfolge[start:start + batch]
            loss = F.cross_entropy(auge(b_t[idx]), z_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        protokoll(f"Epoche {epoche + 1}/{epochen}: Fehler auf dem Testteil {1 - auge_genauigkeit(auge):.2%}")
    return auge.eval()


@torch.no_grad()
def auge_genauigkeit(auge, bilder=None, ziffern=None):
    if bilder is None:
        bilder, ziffern = _ohne_null(*mnist_laden(teil="test"))
    auge.eval()
    vorhersage = torch.cat([auge(torch.as_tensor(bilder[i:i + 4096]).to(auge.geraet)).argmax(1).cpu()
                            for i in range(0, len(bilder), 4096)]).numpy()
    auge.train()
    return float((vorhersage == ziffern).mean())


def auge_speichern(auge, pfad):
    torch.save({k: v.cpu() for k, v in auge.state_dict().items()}, pfad)


def auge_laden(pfad, geraet="cpu"):
    auge = Auge()
    auge.load_state_dict(torch.load(pfad, map_location="cpu"))
    return auge.to(geraet).eval()


def bilder_sudoku(raetsel, rng, bilder=None, ziffern=None):
    """Ersetzt jede Vorgabe durch ein zufälliges Test-Bild dieser Ziffer, leere Felder bleiben schwarz.

    Rückgabe: (B, 81, 28, 28) uint8. Es werden nur Bilder aus dem MNIST-Testteil verwendet, die das Auge
    beim Training nie gesehen hat.
    """
    if bilder is None:
        bilder, ziffern = _ohne_null(*mnist_laden(teil="test"))
    raetsel = np.asarray(raetsel)
    nach_ziffer = [np.flatnonzero(ziffern == d) for d in range(10)]
    ergebnis = np.zeros((*raetsel.shape, 28, 28), np.uint8)
    for d in range(1, 10):
        wo = raetsel == d
        ergebnis[wo] = bilder[rng.choice(nach_ziffer[d], wo.sum())]
    return ergebnis


@torch.no_grad()
def sehen(auge, sudoku_bilder):
    """Wahrscheinlichkeiten des Auges pro Feld: (B, 81, 10), Klasse 0 = leer."""
    b, felder = sudoku_bilder.shape[:2]
    flach = sudoku_bilder.reshape(-1, 28, 28)
    p = torch.cat([torch.softmax(auge(torch.as_tensor(flach[i:i + 8192]).to(auge.geraet)), -1).cpu()
                   for i in range(0, len(flach), 8192)])
    return p.numpy().reshape(b, felder, 10)


def widersprueche(raetsel):
    """(B, 81) bool: Vorgaben, die mit einer anderen Vorgabe derselben Gruppe übereinstimmen."""
    raetsel = np.asarray(raetsel)
    f = feld_fuer(raetsel)
    ergebnis = np.zeros(raetsel.shape, bool)
    for e in f.einheiten:
        werte = raetsel[:, e]                                   # (B, n)
        gleich = (werte[:, :, None] == werte[:, None, :]) & (werte[:, :, None] > 0)
        doppelt = gleich.sum(-1) > 1
        ergebnis[:, e] |= doppelt
    return ergebnis


def _plausibilitaet(p, gelesen_leer, antwort):
    """Wie gut passt eine Lösung zu den Bildern? Summe der log-Wahrscheinlichkeiten auf den Vorgabe-Feldern."""
    lp = np.log(np.clip(np.take_along_axis(p, antwort[..., None], -1)[..., 0], 1e-12, None))
    return np.where(gelesen_leer, 0.0, lp).sum(-1)


def kette(modell, p, **loeser):
    """Erst erkennen (wahrscheinlichste Ziffer pro Feld), dann lösen. Rückgabe wie `probieren`."""
    gelesen = p.argmax(-1)
    e = probieren(modell, gelesen, **loeser)
    e["gelesen"] = gelesen
    return e


def zweite_loesung(modell, raetsel, antwort, schritte=16):
    """Gegenprobe: Gibt es neben `antwort` noch eine andere Lösung? Rückgabe: (gefunden (B,), Schritte (B,)).

    Ein echtes Sudoku hat genau eine Lösung. Für jedes leere Feld und jede dort erlaubte Ziffer, die
    nicht in `antwort` steht, entsteht ein Gedanken-Rätsel mit dieser Ziffer als zusätzlicher
    Vorgabe. Löst das Netz eines davon regelkonform, hat das Rätsel mindestens zwei Lösungen.
    Findet es keines, heißt das nicht sicher, dass es keine gibt (das Netz kann scheitern).
    """
    raetsel, antwort = np.asarray(raetsel), np.asarray(antwort)
    erlaubt = erlaubte_ziffern(raetsel)
    erlaubt[np.arange(len(raetsel))[:, None], np.arange(raetsel.shape[1])[None], np.clip(antwort - 1, 0, None)] = False
    wem, feld, ziffer = np.nonzero(erlaubt)
    gefunden = np.zeros(len(raetsel), bool)
    kosten = np.zeros(len(raetsel), np.int64)
    if len(wem):
        kinder = raetsel[wem].copy()
        kinder[np.arange(len(kinder)), feld] = ziffer + 1
        ok, *_rest, verbraucht = nachdenken(modell, kinder, schritte)
        np.logical_or.at(gefunden, wem, ok)
        np.add.at(kosten, wem, verbraucht)
    return gefunden, kosten


def arbeitsraum(modell, p, kandidaten=10, paare=5, verdacht=0.99, mindestens=1e-3, gegenprobe=False, **loeser):
    """Erkennen und Lösen im Austausch (siehe Moduldoku).

    kandidaten: so viele Umdeutungen einzelner Felder werden höchstens ausprobiert
    paare:      aus den besten so vielen Umdeutungen auch alle Paare (für zwei Lesefehler)
    verdacht:   nur Vorgaben umdeuten, die im Widerspruch stehen oder bei denen das Auge weniger
                sicher war als dieser Wert (None: alle Vorgaben)
    mindestens: eine andere Ziffer kommt nur in Frage, wenn das Auge ihr mindestens so viel zutraut
    gegenprobe: eine Lösung zählt nur, wenn das Netz keine zweite Lösung derselben Lesart findet
                (siehe `zweite_loesung`). Eine Lesart mit zwei Lösungen ist vermutlich falsch gelesen.

    Nochmal hingeschaut wird nur, wenn die erste Lesart nicht aufgeht (oder mit Gegenprobe
    mehrdeutig ist). Geht sie auf, passt sie ohnehin am besten zu den Bildern, denn sie nimmt
    überall die wahrscheinlichste Ziffer.
    """
    gelesen = p.argmax(-1)
    leer = gelesen == 0
    b = len(gelesen)
    konflikt = widersprueche(gelesen)
    erst = probieren(modell, gelesen, **loeser)
    geloest, antwort = erst["geloest"] & ~konflikt.any(1), erst["antwort"].copy()
    schritte = erst["schritte"].copy()
    mehrdeutig = np.zeros(b, bool)
    if gegenprobe and geloest.any():
        i = np.flatnonzero(geloest)
        zweite, extra = zweite_loesung(modell, gelesen[i], antwort[i])
        schritte[i] += extra
        mehrdeutig[i] = zweite
        geloest[i[zweite]] = False
    nochmal = np.flatnonzero(~geloest)
    plausibel = np.where(geloest, _plausibilitaet(p, leer, antwort), -np.inf)

    # Umdeutungen: jede Vorgabe zu jeder anderen Ziffer 1–9. Je wahrscheinlicher laut Auge, desto eher.
    # Vorgaben im Widerspruch kommen zuerst dran.
    lp = np.log(np.clip(p[..., 1:], 1e-12, None))
    abstand = lp - np.take_along_axis(lp, np.clip(gelesen - 1, 0, 8)[..., None], -1)
    abstand += np.where(konflikt, 50.0, 0.0)[..., None]
    abstand[leer] = -np.inf
    if verdacht is not None:
        abstand[~konflikt & (p.max(-1) >= verdacht)] = -np.inf
    abstand[p[..., 1:] < mindestens] = -np.inf
    abstand[np.arange(b)[:, None], np.arange(gelesen.shape[1])[None], np.clip(gelesen - 1, 0, 8)] = -np.inf
    lesarten, wem = [], []
    for i in nochmal:
        rang = np.argsort(-abstand[i], axis=None)[:kandidaten]
        einzeln = [np.unravel_index(r, abstand[i].shape) for r in rang if np.isfinite(abstand[i].flat[r])]
        aenderungen = [[e] for e in einzeln]
        aenderungen += [[x, y] for k, x in enumerate(einzeln[:paare]) for y in einzeln[k + 1:paare] if x[0] != y[0]]
        for aenderung in aenderungen:
            r = gelesen[i].copy()
            for feld, ziffer in aenderung:
                r[feld] = ziffer + 1
            lesarten.append(r)
            wem.append(i)
    extra_aufrufe = np.zeros(b, np.int64)
    if lesarten:
        lesarten, wem = np.array(lesarten), np.array(wem)
        # Das Werkzeug sortiert vorab aus: Lesarten, deren Vorgaben sich widersprechen, kann man sich sparen.
        ok = ~widersprueche(lesarten).any(1)
        lesarten, wem = lesarten[ok], wem[ok]
        np.add.at(extra_aufrufe, wem, 1)
        if len(lesarten):
            e = probieren(modell, lesarten, **loeser)
            np.add.at(schritte, wem, e["schritte"])
            if gegenprobe and e["geloest"].any():
                j = np.flatnonzero(e["geloest"])
                zweite, extra = zweite_loesung(modell, lesarten[j], e["antwort"][j])
                np.add.at(schritte, wem[j], extra)
                e["geloest"][j[zweite]] = False
            punkte = np.where(e["geloest"], _plausibilitaet(p[wem], lesarten == 0, e["antwort"]), -np.inf)
            for j in np.argsort(-punkte):
                i = wem[j]
                if punkte[j] > plausibel[i]:
                    plausibel[i], antwort[i], geloest[i] = punkte[j], e["antwort"][j], True
    return {"geloest": geloest, "antwort": antwort, "schritte": schritte, "gelesen": gelesen,
            "lesarten": extra_aufrufe, "mehrdeutig": mehrdeutig}


def auswerten(e, raetsel, loesung):
    """Kennzahlen gegen die Wahrheit: richtig, geprüft-aber-falsch (falsches Rätsel gelöst), aufgegeben."""
    richtig = e["geloest"] & (e["antwort"] == loesung).all(1)
    return {"richtig": float(richtig.mean()),
            "falsch_geprueft": float((e["geloest"] & ~richtig).mean()),
            "aufgegeben": float((~e["geloest"]).mean()),
            "schritte": float(e["schritte"].mean())}


def lesefehler(gelesen, raetsel):
    """Pro Rätsel: wie viele Felder hat das Auge falsch gelesen?"""
    return (gelesen != raetsel).sum(1)
