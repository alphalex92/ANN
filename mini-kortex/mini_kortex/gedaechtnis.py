"""Episodisches Gedächtnis als modernes Hopfield-Netz (das "Hippocampus"-Modul).

Ein klassisches neuronales Netz lernt langsam: Es braucht tausende Beispiele und viele
Gradientenschritte. Das Gehirn hat zusätzlich ein schnelles System, das ein einzelnes
Erlebnis sofort speichert. Genau das macht dieses Modul:

- **Speichern** ist ein einziger Schritt, ganz ohne Training: Schlüssel (das Rätsel) und
  Wert (die geprüfte Lösung) werden einfach angehängt.
- **Abrufen** funktioniert über Ähnlichkeit, wie bei Attention:

      Gewichte = softmax(β · Schlüssel · Anfrage)
      Ergebnis = Gewichte · Werte

  Ein großes β macht den Abruf "scharf": Es gewinnt fast nur die ähnlichste Erinnerung.
  Diese Formel ist die Update-Regel moderner Hopfield-Netze (Ramsauer, Hochreiter u. a. 2020)
  und gleichzeitig genau die Attention-Formel eines Transformers.

Weil die Ähnlichkeit auf den vorgegebenen Ziffern beruht, findet das Gedächtnis auch
*verwandte* Rätsel wieder, etwa dasselbe Sudoku mit ein paar zusätzlichen Vorgaben.
"""

import numpy as np

from .sudoku import N


def _kodieren(gitter):
    """Gitter (…, 36) → One-Hot-Vektor (…, 216); leere Felder sind ein Nullvektor."""
    g = np.asarray(gitter)
    one_hot = (g[..., None] == np.arange(1, N + 1)).astype(np.float32)
    return one_hot.reshape(*g.shape[:-1], -1)


class HopfieldGedaechtnis:
    def __init__(self, beta=40.0):
        self.beta = beta
        self.schluessel = np.zeros((0, N * N * N), np.float32)
        self.werte = np.zeros((0, N * N * N), np.float32)

    def __len__(self):
        return len(self.schluessel)

    def speichern(self, raetsel, loesung):
        k = _kodieren(raetsel)
        k /= np.linalg.norm(k) + 1e-9
        self.schluessel = np.vstack([self.schluessel, k])
        self.werte = np.vstack([self.werte, _kodieren(loesung)])

    def abrufen(self, raetsel):
        """Gibt (Lösungsvorschlag, Ähnlichkeit 0–1 der besten Erinnerung) zurück, oder (None, 0)."""
        if len(self) == 0:
            return None, 0.0
        q = _kodieren(raetsel)
        q /= np.linalg.norm(q) + 1e-9
        aehnlichkeit = self.schluessel @ q
        gewichte = np.exp(self.beta * (aehnlichkeit - aehnlichkeit.max()))
        gewichte /= gewichte.sum()
        mischung = (gewichte @ self.werte).reshape(N * N, N)
        return mischung.argmax(-1) + 1, float(aehnlichkeit.max())
