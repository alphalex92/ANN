"""Trainingsschleife und Auswertung."""

import numpy as np

from .daten import KLASSEN, als_bits, erzeugen
from .netz import Netz


def genauigkeit(netz, X, y):
    return (netz.vorwaerts(X).argmax(axis=1) == y).mean()


def trainieren(epochen=20, versteckt=64, lernrate=0.2, batch=64, n_pro_klasse=3000, seed=0, ausgabe=print):
    rng = np.random.default_rng(seed)
    # Trainings- und Testdaten getrennt erzeugen: Die Testdaten sieht das Netz beim Lernen nie.
    X_train, y_train = erzeugen(n_pro_klasse, rng)
    X_test, y_test = erzeugen(n_pro_klasse // 3, rng)
    X_train, X_test = als_bits(X_train), als_bits(X_test)

    netz = Netz(X_train.shape[1], versteckt, len(KLASSEN), rng)

    for epoche in range(1, epochen + 1):
        # Eine Epoche = einmal durch alle Trainingsdaten, in zufälliger Reihenfolge,
        # in kleinen Häppchen (Mini-Batches). Nach jedem Häppchen ein Lernschritt.
        reihenfolge = rng.permutation(len(y_train))
        verluste = []
        for start in range(0, len(reihenfolge), batch):
            idx = reihenfolge[start : start + batch]
            verlust, gradienten = netz.verlust_und_gradienten(X_train[idx], y_train[idx])
            netz.schritt(gradienten, lernrate)
            verluste.append(verlust)
        ausgabe(
            f"Epoche {epoche:3d}  Verlust {np.mean(verluste):.4f}  "
            f"Training {genauigkeit(netz, X_train, y_train):6.1%}  Test {genauigkeit(netz, X_test, y_test):6.1%}"
        )

    return netz, X_test, y_test


def verwechslungsmatrix(netz, X, y):
    """Zeile = richtige Klasse, Spalte = Vermutung des Netzes."""
    vermutung = netz.vorwaerts(X).argmax(axis=1)
    matrix = np.zeros((len(KLASSEN), len(KLASSEN)), dtype=int)
    np.add.at(matrix, (y, vermutung), 1)
    return matrix


def matrix_als_text(matrix):
    breite = max(len(k) for k in KLASSEN) + 2
    zeilen = [" " * breite + "".join(f"{k:>{breite}}" for k in KLASSEN)]
    for name, zeile in zip(KLASSEN, matrix):
        zeilen.append(f"{name:<{breite}}" + "".join(f"{wert:>{breite}}" for wert in zeile))
    return "\n".join(zeilen)
