"""Sudoku in zwei Größen: Rätsel erzeugen, exakt prüfen, Schwierigkeit messen.

    6×6: Kästen aus 2 Zeilen × 3 Spalten, Ziffern 1–6, 36 Felder
    9×9: Kästen aus 3 Zeilen × 3 Spalten, Ziffern 1–9, 81 Felder

In jeder Zeile, jeder Spalte und jedem Kasten kommt jede Ziffer genau einmal vor.
Eine 0 steht für ein leeres Feld. Ein Gitter ist ein flaches Array mit n·n Einträgen;
die Größe ergibt sich aus der Länge (36 → 6×6, 81 → 9×9).

Dieses Modul enthält kein neuronales Netz. Es ist das **symbolische Werkzeug** der
Architektur: Regeln, die exakt gelten und die das Netz nicht "glauben" muss, sondern
nachprüfen kann.
"""

from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache

import numpy as np

KASTEN = {6: (2, 3), 9: (3, 3)}  # Größe → (Zeilen, Spalten) eines Kastens


class Feld:
    """Die Geometrie eines Spielfelds: welche Felder gehören zusammen?"""

    def __init__(self, n):
        if n not in KASTEN:
            raise ValueError(f"Unbekannte Größe {n}, möglich: {sorted(KASTEN)}")
        self.n = n
        self.felder = n * n
        self.kz, self.ks = KASTEN[n]
        self.alle = (1 << n) - 1  # Bitmaske: alle Ziffern noch möglich
        zeilen = [[r * n + c for c in range(n)] for r in range(n)]
        spalten = [[r * n + c for r in range(n)] for c in range(n)]
        kaesten = [
            [(br + r) * n + bc + c for r in range(self.kz) for c in range(self.ks)]
            for br in range(0, n, self.kz)
            for bc in range(0, n, self.ks)
        ]
        self.einheiten = zeilen + spalten + kaesten
        self.einheiten_array = np.array(self.einheiten)
        # Für jedes Feld: Nummer seiner Zeile, Spalte und seines Kastens.
        self.zeile = [i // n for i in range(self.felder)]
        self.spalte = [i % n for i in range(self.felder)]
        self.kasten = [(i // n) // self.kz * (n // self.ks) + (i % n) // self.ks for i in range(self.felder)]


@lru_cache(maxsize=None)
def feld(n):
    return Feld(n)


def feld_fuer(gitter):
    """Das passende Spielfeld zu einem Gitter (oder einer Batch von Gittern)."""
    laenge = np.asarray(gitter).shape[-1]
    n = int(round(laenge ** 0.5))
    if n * n != laenge:
        raise ValueError(f"Gitter mit {laenge} Einträgen ist kein Quadrat")
    return feld(n)


# ---------------------------------------------------------------------------
# Werkzeug: exakte Regelprüfung
# ---------------------------------------------------------------------------

def ist_gueltige_loesung(gitter, raetsel=None):
    """True, wenn das Gitter vollständig und regelkonform ist (und zu den Vorgaben passt)."""
    g = np.asarray(gitter).reshape(-1)
    f = feld_fuer(g)
    if not np.all((g >= 1) & (g <= f.n)):
        return False
    if any(len(set(g[e].tolist())) != f.n for e in f.einheiten):
        return False
    if raetsel is not None:
        r = np.asarray(raetsel).reshape(-1)
        if np.any((r != 0) & (r != g)):
            return False
    return True


def gueltig_batch(gitter, raetsel):
    """Wie `ist_gueltige_loesung`, aber für viele Gitter auf einmal (Form (B, n·n)) → (B,) bool."""
    g = np.asarray(gitter)
    r = np.asarray(raetsel)
    f = feld_fuer(g)
    gruppen = g[:, f.einheiten_array]  # (B, 3n, n)
    one_hot = gruppen[..., None] == np.arange(1, f.n + 1)  # (B, 3n, n, n)
    jede_ziffer_einmal = (one_hot.sum(2) == 1).all((1, 2))
    passt_zu_vorgaben = ((r == 0) | (r == g)).all(1)
    return jede_ziffer_einmal & passt_zu_vorgaben


def regelverstoesse(gitter):
    """Anzahl doppelter Ziffern über alle Gruppen. 0 heißt: kein Widerspruch (Leerfelder zählen nicht)."""
    g = np.asarray(gitter).reshape(-1)
    fehler = 0
    for e in feld_fuer(g).einheiten:
        werte = g[e][g[e] > 0]
        fehler += len(werte) - len(set(werte.tolist()))
    return fehler


# ---------------------------------------------------------------------------
# Klassischer Löser (nur zum Erzeugen eindeutiger Rätsel, nicht für das Netz)
# ---------------------------------------------------------------------------

def _masken(g, f):
    """Welche Ziffern sind in jeder Zeile, Spalte, jedem Kasten schon belegt? (als Bitmasken)"""
    zm, sm, km = [0] * f.n, [0] * f.n, [0] * f.n
    for i, v in enumerate(g):
        if v:
            bit = 1 << (v - 1)
            zm[f.zeile[i]] |= bit
            sm[f.spalte[i]] |= bit
            km[f.kasten[i]] |= bit
    return zm, sm, km


def anzahl_loesungen(raetsel, grenze=2):
    """Zählt Lösungen per Backtracking, bricht bei `grenze` ab.

    Belegte Ziffern werden pro Zeile, Spalte und Kasten als Bitmaske mitgeführt. So kostet
    die Frage "welche Ziffern passen hier noch?" nur drei ODER-Verknüpfungen.
    """
    g = [int(v) for v in np.asarray(raetsel).reshape(-1)]
    f = feld_fuer(g)
    zm, sm, km = _masken(g, f)
    leer = [i for i in range(f.felder) if g[i] == 0]
    Z, S, K, ALLE = f.zeile, f.spalte, f.kasten, f.alle

    def suche():
        # Das leere Feld mit den wenigsten Kandidaten zuerst: das hält den Suchbaum klein.
        bestes, beste_maske, beste_zahl = -1, 0, 99
        for i in leer:
            if g[i] == 0:
                m = ALLE & ~(zm[Z[i]] | sm[S[i]] | km[K[i]])
                k = m.bit_count()
                if k < beste_zahl:
                    bestes, beste_maske, beste_zahl = i, m, k
                    if k <= 1:
                        break
        if bestes < 0:
            return 1
        gefunden = 0
        z, s, k = Z[bestes], S[bestes], K[bestes]
        m = beste_maske
        while m:
            bit = m & -m
            m ^= bit
            g[bestes] = bit.bit_length()
            zm[z] |= bit
            sm[s] |= bit
            km[k] |= bit
            gefunden += suche()
            zm[z] ^= bit
            sm[s] ^= bit
            km[k] ^= bit
            g[bestes] = 0
            if gefunden >= grenze:
                break
        return gefunden

    return suche()


def denktiefe(raetsel):
    """Wie viele Runden logischen Schließens braucht ein Mensch (mindestens)?

    Pro Runde werden gleichzeitig alle Felder ausgefüllt, die sich direkt ergeben:
    - "nackte Einer": In dem Feld ist nur noch eine Ziffer möglich.
    - "versteckte Einer": Eine Ziffer passt in einer Gruppe nur noch an eine Stelle.
    Die Zahl der Runden ist ein gutes Maß dafür, wie lang die Schlusskette ist.
    Gibt 0 zurück, wenn diese einfachen Regeln nicht reichen und man probieren müsste.
    """
    g = [int(v) for v in np.asarray(raetsel).reshape(-1)]
    f = feld_fuer(g)
    runden = 0
    while 0 in g:
        zm, sm, km = _masken(g, f)
        kand = {i: f.alle & ~(zm[f.zeile[i]] | sm[f.spalte[i]] | km[f.kasten[i]])
                for i in range(f.felder) if g[i] == 0}
        neu = {}
        for i, m in kand.items():
            if m and m & (m - 1) == 0:
                neu[i] = m.bit_length()
        for e in f.einheiten:
            for d in range(f.n):
                bit = 1 << d
                orte = [i for i in e if i in kand and kand[i] & bit]
                if len(orte) == 1 and not any(g[i] == d + 1 for i in e):
                    neu[orte[0]] = d + 1
        if not neu:
            return 0
        for i, d in neu.items():
            g[i] = d
        runden += 1
    return runden


# ---------------------------------------------------------------------------
# Symmetrien: aus einem Rätsel viele gleichwertige machen
# ---------------------------------------------------------------------------

def zufaellige_symmetrie(n, rng):
    """Eine zufällige Umordnung, die die Sudoku-Regeln erhält.

    Erlaubt sind: Ziffern umbenennen, Zeilen innerhalb eines Bandes tauschen, Bänder tauschen,
    Spalten innerhalb eines Stapels tauschen, Stapel tauschen und (bei quadratischen Kästen)
    spiegeln an der Diagonale. Rückgabe: (Feld-Reihenfolge, Ziffern-Tabelle).
    """
    f = feld(n)
    baender = rng.permutation(n // f.kz)
    zeilen = np.array([b * f.kz + z for b in baender for z in rng.permutation(f.kz)])
    stapel = rng.permutation(n // f.ks)
    spalten = np.array([s * f.ks + z for s in stapel for z in rng.permutation(f.ks)])
    reihenfolge = (zeilen[:, None] * n + spalten[None, :])
    if f.kz == f.ks and rng.random() < 0.5:
        reihenfolge = reihenfolge.T
    ziffern = np.concatenate([[0], rng.permutation(n) + 1])  # 0 (leer) bleibt 0
    return reihenfolge.reshape(-1), ziffern


def varianten(raetsel, loesungen, rng):
    """Wendet auf jedes Paar (Rätsel, Lösung) eine eigene zufällige Symmetrie an."""
    raetsel, loesungen = np.asarray(raetsel), np.asarray(loesungen)
    n = feld_fuer(raetsel).n
    neu_r, neu_l = np.empty_like(raetsel), np.empty_like(loesungen)
    for i in range(len(raetsel)):
        reihenfolge, ziffern = zufaellige_symmetrie(n, rng)
        neu_r[i] = ziffern[raetsel[i][reihenfolge]]
        neu_l[i] = ziffern[loesungen[i][reihenfolge]]
    return neu_r, neu_l


# ---------------------------------------------------------------------------
# Rätsel erzeugen
# ---------------------------------------------------------------------------

def zufaelliges_gitter(rng, n=6):
    """Ein zufälliges vollständiges Gitter: festes gültiges Muster + zufällige Symmetrie."""
    f = feld(n)
    r = np.arange(n)[:, None]
    c = np.arange(n)[None, :]
    muster = ((f.ks * (r % f.kz) + r // f.kz + c) % n + 1).reshape(-1)
    reihenfolge, ziffern = zufaellige_symmetrie(n, rng)
    return ziffern[muster[reihenfolge]]


def raetsel_erzeugen(rng, leer_ziel, n=6):
    """Entfernt Ziffern, solange das Rätsel eindeutig lösbar bleibt (höchstens `leer_ziel` Stück)."""
    loesung = zufaelliges_gitter(rng, n)
    raetsel = loesung.copy()
    leer = 0
    for i in rng.permutation(n * n):
        if leer >= leer_ziel:
            break
        alt = raetsel[i]
        raetsel[i] = 0
        if anzahl_loesungen(raetsel) == 1:
            leer += 1
        else:
            raetsel[i] = alt
    return raetsel, loesung


# Standard-Schwierigkeit: Spanne der leeren Felder pro Größe.
LEER_STANDARD = {6: (12, 28), 9: (30, 58)}


def _datensatz_teil(args):
    anzahl, seed, n, leer_min, leer_max = args
    rng = np.random.default_rng(seed)
    raetsel, loesungen, tiefe = [], [], []
    for _ in range(anzahl):
        r, l = raetsel_erzeugen(rng, int(rng.integers(leer_min, leer_max + 1)), n)
        raetsel.append(r)
        loesungen.append(l)
        tiefe.append(denktiefe(r))
    return raetsel, loesungen, tiefe


def datensatz(anzahl, rng, n=6, leer_min=None, leer_max=None, prozesse=1):
    """`anzahl` Rätsel mit gleichmäßig verteilter Zahl leerer Felder.

    Mit `prozesse` > 1 wird auf mehrere CPU-Kerne verteilt (lohnt sich bei 9×9).
    Rückgabe: raetsel (anzahl, n·n), loesungen (anzahl, n·n), tiefe (anzahl,)
    """
    std_min, std_max = LEER_STANDARD[n]
    leer_min = std_min if leer_min is None else leer_min
    leer_max = std_max if leer_max is None else leer_max
    stuecke = max(1, min(prozesse, anzahl)) if prozesse > 1 else 1
    groessen = [anzahl // stuecke + (i < anzahl % stuecke) for i in range(stuecke)]
    seeds = rng.integers(0, 2**63, stuecke)
    auftraege = [(g, int(s), n, leer_min, leer_max) for g, s in zip(groessen, seeds)]
    if stuecke == 1:
        teile = [_datensatz_teil(auftraege[0])]
    else:
        with ProcessPoolExecutor(stuecke) as pool:
            teile = list(pool.map(_datensatz_teil, auftraege))
    raetsel = [r for t in teile for r in t[0]]
    loesungen = [l for t in teile for l in t[1]]
    tiefe = [d for t in teile for d in t[2]]
    form = (len(raetsel), n * n)
    return (np.array(raetsel, np.int64).reshape(form), np.array(loesungen, np.int64).reshape(form),
            np.array(tiefe, np.int64))


def als_text(gitter):
    """Hübsche Textdarstellung mit Kastenlinien."""
    g = np.asarray(gitter).reshape(-1)
    f = feld_fuer(g)
    g = g.reshape(f.n, f.n)
    zeilen = []
    for r in range(f.n):
        if r and r % f.kz == 0:
            zeilen.append("+".join(["-" * (2 * f.ks)] * (f.n // f.ks)))
        teile = []
        for c in range(f.n):
            if c and c % f.ks == 0:
                teile.append("|")
            teile.append(str(g[r, c]) if g[r, c] else "·")
        zeilen.append(" ".join(teile))
    return "\n".join(zeilen)
