"""Trainingsdaten: 8-Byte-Speicherslots mit bekanntem Inhalt.

Jedes Beispiel ist ein 8 Byte großer, 8-Byte-ausgerichteter Slot, wie ihn ein
Speicher-Scanner sieht (Little Endian, wie auf x86). Wir erzeugen die Bytes selbst,
deshalb kennen wir die richtige Antwort und können beliebig viele Beispiele bauen.
"""

import numpy as np

KLASSEN = ["i64", "f64", "2×i32", "2×f32", "zeiger", "text", "zufall"]


def _vorzeichen(rng, n, anteil_negativ):
    return np.where(rng.random(n) < anteil_negativ, -1, 1)


def _i64(rng, n):
    # Zähler, Geld, Punkte: meist positiv, Größenordnung gleichverteilt (1 … 10¹²).
    werte = (10 ** rng.uniform(0, 12, n)).astype(np.int64)
    werte *= _vorzeichen(rng, n, 0.1)
    return werte.astype("<i8").view(np.uint8).reshape(n, 8)


def _f64(rng, n):
    # Idle-Spiele lieben riesige Doubles; dazu "glatte" Werte wie 250.0 oder 0.5.
    werte = 10 ** rng.uniform(-3, 30, n)
    glatt = rng.random(n) < 0.3
    werte[glatt] = rng.integers(1, 10_000, glatt.sum()) * rng.choice([0.25, 0.5, 1.0], glatt.sum())
    werte *= _vorzeichen(rng, n, 0.2)
    return werte.astype("<f8").view(np.uint8).reshape(n, 8)


def _2x_i32(rng, n):
    # Zwei kleine Ganzzahlen nebeneinander, z. B. Level und Erfahrung.
    werte = (10 ** rng.uniform(0, 9, (n, 2))).astype(np.int32)
    werte *= _vorzeichen(rng, (n, 2), 0.1)
    return werte.astype("<i4").view(np.uint8).reshape(n, 8)


def _2x_f32(rng, n):
    # Zwei Floats, z. B. eine Position (x, y).
    werte = (10 ** rng.uniform(-3, 6, (n, 2))) * _vorzeichen(rng, (n, 2), 0.3)
    return werte.astype("<f4").view(np.uint8).reshape(n, 8)


def _zeiger(rng, n):
    # Typische Adressbereiche: Linux-Programm/Heap, Linux-Bibliotheken, Windows/Proton-Heap.
    basis = rng.choice(np.array([0x5555_0000_0000, 0x7F00_0000_0000, 0x0001_0000_0000], dtype=np.uint64), n)
    versatz = rng.integers(0, 2**36, n, dtype=np.uint64) & ~np.uint64(7)  # auf 8 ausgerichtet
    return (basis + versatz).astype("<u8").view(np.uint8).reshape(n, 8)


def _text(rng, n):
    # 3 bis 8 druckbare ASCII-Zeichen, der Rest mit Nullbytes aufgefüllt.
    zeichen = rng.integers(32, 127, (n, 8), dtype=np.uint8)
    laenge = rng.integers(3, 9, n)
    zeichen[np.arange(8) >= laenge[:, None]] = 0
    return zeichen


def _zufall(rng, n):
    # Komprimierte oder verschlüsselte Daten sehen aus wie Rauschen.
    return rng.integers(0, 256, (n, 8), dtype=np.uint8)


_ERZEUGER = [_i64, _f64, _2x_i32, _2x_f32, _zeiger, _text, _zufall]


def erzeugen(n_pro_klasse, rng):
    """Gibt (bytes, klassen) zurück: bytes hat die Form (N, 8), klassen (N,), gemischt."""
    teile = [erzeuger(rng, n_pro_klasse) for erzeuger in _ERZEUGER]
    X = np.concatenate(teile)
    y = np.repeat(np.arange(len(KLASSEN)), n_pro_klasse)
    reihenfolge = rng.permutation(len(y))
    return X[reihenfolge], y[reihenfolge]


def als_bits(X):
    """8 Bytes → 64 Eingänge mit 0 oder 1. So sieht das Netz jedes einzelne Bit."""
    return np.unpackbits(X, axis=1).astype(np.float64)
