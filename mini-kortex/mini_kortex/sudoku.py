"""6×6-Sudoku: Rätsel erzeugen, exakt prüfen, Schwierigkeit messen.

Spielfeld: 6 Zeilen × 6 Spalten, Kästen aus 2 Zeilen × 3 Spalten, Ziffern 1–6.
In jeder Zeile, jeder Spalte und jedem Kasten kommt jede Ziffer genau einmal vor.
Eine 0 steht für ein leeres Feld.

Dieses Modul enthält kein neuronales Netz. Es ist das **symbolische Werkzeug** der
Architektur: Regeln, die exakt gelten und die das Netz nicht "glauben" muss, sondern
nachprüfen kann.
"""

import numpy as np

N = 6
KASTEN_ZEILEN, KASTEN_SPALTEN = 2, 3
ALLE = (1 << N) - 1  # Bitmaske 0b111111: alle Ziffern noch möglich


def _einheiten():
    """Die 18 Gruppen (6 Zeilen, 6 Spalten, 6 Kästen) als Listen von Feldnummern 0–35."""
    zeilen = [[r * N + c for c in range(N)] for r in range(N)]
    spalten = [[r * N + c for r in range(N)] for c in range(N)]
    kaesten = [
        [(br + r) * N + bc + c for r in range(KASTEN_ZEILEN) for c in range(KASTEN_SPALTEN)]
        for br in range(0, N, KASTEN_ZEILEN)
        for bc in range(0, N, KASTEN_SPALTEN)
    ]
    return zeilen + spalten + kaesten


EINHEITEN = _einheiten()
# Für jedes Feld: alle anderen Felder, die in einer gemeinsamen Gruppe liegen.
NACHBARN = [sorted({f for e in EINHEITEN if i in e for f in e} - {i}) for i in range(N * N)]


# ---------------------------------------------------------------------------
# Werkzeug: exakte Regelprüfung
# ---------------------------------------------------------------------------

def ist_gueltige_loesung(gitter, raetsel=None):
    """True, wenn das Gitter vollständig und regelkonform ist (und zu den Vorgaben passt)."""
    g = np.asarray(gitter).reshape(-1)
    if g.shape != (N * N,) or not np.all((g >= 1) & (g <= N)):
        return False
    if any(len(set(g[e].tolist())) != N for e in EINHEITEN):
        return False
    if raetsel is not None:
        r = np.asarray(raetsel).reshape(-1)
        if np.any((r != 0) & (r != g)):
            return False
    return True


def regelverstoesse(gitter):
    """Anzahl doppelter Ziffern über alle Gruppen. 0 heißt: kein Widerspruch (Leerfelder zählen nicht)."""
    g = np.asarray(gitter).reshape(-1)
    fehler = 0
    for e in EINHEITEN:
        werte = g[e][g[e] > 0]
        fehler += len(werte) - len(set(werte.tolist()))
    return fehler


# ---------------------------------------------------------------------------
# Klassischer Löser (nur zum Erzeugen eindeutiger Rätsel, nicht für das Netz)
# ---------------------------------------------------------------------------

def _kandidaten(g, i):
    belegt = 0
    for j in NACHBARN[i]:
        if g[j]:
            belegt |= 1 << (g[j] - 1)
    return ALLE & ~belegt


def anzahl_loesungen(raetsel, grenze=2):
    """Zählt Lösungen per Backtracking, bricht bei `grenze` ab."""
    g = [int(v) for v in np.asarray(raetsel).reshape(-1)]

    def suche():
        # Das leere Feld mit den wenigsten Kandidaten zuerst: das hält den Suchbaum klein.
        bestes, beste_maske, beste_zahl = -1, 0, N + 1
        for i in range(N * N):
            if g[i] == 0:
                m = _kandidaten(g, i)
                k = bin(m).count("1")
                if k < beste_zahl:
                    bestes, beste_maske, beste_zahl = i, m, k
                    if k <= 1:
                        break
        if bestes < 0:
            return 1
        gefunden = 0
        for d in range(N):
            if beste_maske >> d & 1:
                g[bestes] = d + 1
                gefunden += suche()
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
    runden = 0
    while 0 in g:
        neu = {}
        kand = {i: _kandidaten(g, i) for i in range(N * N) if g[i] == 0}
        for i, m in kand.items():
            if m and m & (m - 1) == 0:
                neu[i] = m.bit_length()
        for e in EINHEITEN:
            for d in range(N):
                orte = [i for i in e if i in kand and kand[i] >> d & 1]
                if len(orte) == 1 and not any(g[i] == d + 1 for i in e):
                    neu[orte[0]] = d + 1
        if not neu:
            return 0
        for i, d in neu.items():
            g[i] = d
        runden += 1
    return runden


# ---------------------------------------------------------------------------
# Rätsel erzeugen
# ---------------------------------------------------------------------------

def zufaelliges_gitter(rng):
    """Ein zufälliges vollständiges 6×6-Gitter.

    Wir starten mit einem festen gültigen Muster und mischen es mit Operationen, die die
    Regeln erhalten: Ziffern umbenennen, Zeilen innerhalb eines Bandes tauschen, Bänder
    tauschen, Spalten innerhalb eines Stapels tauschen, Stapel tauschen.
    """
    r = np.arange(N)[:, None]
    c = np.arange(N)[None, :]
    g = (KASTEN_SPALTEN * (r % KASTEN_ZEILEN) + r // KASTEN_ZEILEN + c) % N + 1

    baender = rng.permutation(N // KASTEN_ZEILEN)
    zeilen = [b * KASTEN_ZEILEN + z for b in baender for z in rng.permutation(KASTEN_ZEILEN)]
    stapel = rng.permutation(N // KASTEN_SPALTEN)
    spalten = [s * KASTEN_SPALTEN + z for s in stapel for z in rng.permutation(KASTEN_SPALTEN)]
    g = g[zeilen][:, spalten]
    ziffern = np.concatenate([[0], rng.permutation(N) + 1])
    return ziffern[g].reshape(-1)


def raetsel_erzeugen(rng, leer_ziel):
    """Entfernt Ziffern, solange das Rätsel eindeutig lösbar bleibt (höchstens `leer_ziel` Stück)."""
    loesung = zufaelliges_gitter(rng)
    raetsel = loesung.copy()
    leer = 0
    for i in rng.permutation(N * N):
        if leer >= leer_ziel:
            break
        alt = raetsel[i]
        raetsel[i] = 0
        if anzahl_loesungen(raetsel) == 1:
            leer += 1
        else:
            raetsel[i] = alt
    return raetsel, loesung


def datensatz(anzahl, rng, leer_min=12, leer_max=28):
    """`anzahl` Rätsel mit gleichmäßig verteilter Zahl leerer Felder.

    Rückgabe: raetsel (anzahl, 36), loesungen (anzahl, 36), tiefe (anzahl,)
    """
    raetsel, loesungen, tiefe = [], [], []
    for _ in range(anzahl):
        r, l = raetsel_erzeugen(rng, int(rng.integers(leer_min, leer_max + 1)))
        raetsel.append(r)
        loesungen.append(l)
        tiefe.append(denktiefe(r))
    return np.array(raetsel, np.int64), np.array(loesungen, np.int64), np.array(tiefe, np.int64)


def als_text(gitter):
    """Hübsche Textdarstellung mit Kastenlinien."""
    g = np.asarray(gitter).reshape(N, N)
    zeilen = []
    for r in range(N):
        if r and r % KASTEN_ZEILEN == 0:
            zeilen.append("+".join(["-" * (2 * KASTEN_SPALTEN)] * (N // KASTEN_SPALTEN)))
        teile = []
        for c in range(N):
            if c and c % KASTEN_SPALTEN == 0:
                teile.append("|")
            teile.append(str(g[r, c]) if g[r, c] else "·")
        zeilen.append(" ".join(teile))
    return "\n".join(zeilen)


_EINHEITEN_ARRAY = np.array(EINHEITEN)  # (18, 6)


def gueltig_batch(gitter, raetsel):
    """Wie `ist_gueltige_loesung`, aber für viele Gitter auf einmal (Form (B, 36)) → (B,) bool."""
    g = np.asarray(gitter)
    r = np.asarray(raetsel)
    gruppen = g[:, _EINHEITEN_ARRAY]  # (B, 18, 6)
    one_hot = gruppen[..., None] == np.arange(1, N + 1)  # (B, 18, 6, 6)
    jede_ziffer_einmal = (one_hot.sum(2) == 1).all((1, 2))
    passt_zu_vorgaben = ((r == 0) | (r == g)).all(1)
    return jede_ziffer_einmal & passt_zu_vorgaben
