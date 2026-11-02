"""Ein kleines neuronales Netz (MLP), komplett von Hand in numpy.

Aufbau:  Eingabe (64 Bits) → versteckte Schicht (ReLU) → Ausgabe (eine Zahl pro Klasse) → Softmax

Kein Framework: Vorwärtsrechnung, Verlust und Backpropagation stehen hier ausgeschrieben,
damit man jede Zeile nachvollziehen kann.
"""

import numpy as np


def softmax(z):
    # Rohwerte → Wahrscheinlichkeiten, die zusammen 1 ergeben.
    # Das Maximum abziehen ändert das Ergebnis nicht, verhindert aber Überläufe bei exp().
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class Netz:
    def __init__(self, eingaenge, versteckt, ausgaenge, rng):
        # He-Initialisierung: Zufallsgewichte, skaliert nach der Zahl der Eingänge.
        # So bleiben die Werte von Schicht zu Schicht ungefähr gleich groß.
        self.W1 = rng.normal(0, np.sqrt(2 / eingaenge), (eingaenge, versteckt))
        self.b1 = np.zeros(versteckt)
        self.W2 = rng.normal(0, np.sqrt(2 / versteckt), (versteckt, ausgaenge))
        self.b2 = np.zeros(ausgaenge)

    def parameter(self):
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}

    def vorwaerts(self, X):
        """X: (N, 64) → Wahrscheinlichkeiten (N, Klassen)."""
        return self._vorwaerts(X)[-1]

    def _vorwaerts(self, X):
        z1 = X @ self.W1 + self.b1  # gewichtete Summe jedes versteckten Neurons
        h = np.maximum(z1, 0)  # ReLU: negative Werte werden 0
        z2 = h @ self.W2 + self.b2  # Rohwert ("Logit") pro Klasse
        return z1, h, softmax(z2)

    def verlust_und_gradienten(self, X, y):
        """Kreuzentropie-Verlust und seine Ableitung nach jedem Parameter.

        Der Verlust ist klein, wenn das Netz der richtigen Klasse eine hohe
        Wahrscheinlichkeit gibt. Die Gradienten sagen, in welche Richtung jedes
        Gewicht wandern muss, damit der Verlust steigt – wir gehen also in die Gegenrichtung.
        """
        n = len(y)
        z1, h, p = self._vorwaerts(X)
        verlust = -np.log(p[np.arange(n), y] + 1e-12).mean()

        # Backpropagation: Kettenregel, von hinten nach vorne.
        # Für Softmax + Kreuzentropie ist die Ableitung nach z2 einfach: p minus "richtige Antwort".
        dz2 = p.copy()
        dz2[np.arange(n), y] -= 1
        dz2 /= n

        dW2 = h.T @ dz2
        db2 = dz2.sum(axis=0)
        dh = dz2 @ self.W2.T
        dz1 = dh * (z1 > 0)  # ReLU lässt den Gradienten nur durch, wo sie aktiv war
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0)

        return verlust, {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    def schritt(self, gradienten, lernrate):
        """Gradientenabstieg: jedes Gewicht ein kleines Stück gegen seinen Gradienten bewegen."""
        for name, wert in self.parameter().items():
            wert -= lernrate * gradienten[name]

    def speichern(self, pfad):
        np.savez(pfad, **self.parameter())

    @classmethod
    def laden(cls, pfad):
        daten = np.load(pfad)
        netz = cls.__new__(cls)
        for name in ("W1", "b1", "W2", "b2"):
            setattr(netz, name, daten[name])
        return netz
