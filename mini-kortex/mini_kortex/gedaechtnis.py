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

from .sudoku import feld_fuer


def _kodieren(gitter):
    """Gitter (…, n·n) → One-Hot-Vektor (…, n·n·n); leere Felder sind ein Nullvektor."""
    g = np.asarray(gitter)
    one_hot = (g[..., None] == np.arange(1, feld_fuer(g).n + 1)).astype(np.float32)
    return one_hot.reshape(*g.shape[:-1], -1)


class HopfieldGedaechtnis:
    def __init__(self, beta=40.0):
        self.beta = beta
        self.schluessel = None  # (Episoden, n·n·n), normiert
        self.werte = None       # (Episoden, n·n·n)
        # Pro Episode zusätzlich: das Rätsel, die Lösung und wie viele Denkschritte sie gekostet hat.
        # Das braucht die Konsolidierung, um zu entscheiden, was im "Schlaf" wiederholt wird.
        self.raetsel, self.loesungen, self.schritte = [], [], []

    def __len__(self):
        return len(self.raetsel)

    def speichern(self, raetsel, loesung, schritte=0):
        k = _kodieren(raetsel)
        k /= np.linalg.norm(k) + 1e-9
        v = _kodieren(loesung)
        self.schluessel = k[None] if self.schluessel is None else np.vstack([self.schluessel, k])
        self.werte = v[None] if self.werte is None else np.vstack([self.werte, v])
        self.raetsel.append(np.asarray(raetsel).copy())
        self.loesungen.append(np.asarray(loesung).copy())
        self.schritte.append(int(schritte))

    def episoden(self, mindestens_schritte=0):
        """Alle gespeicherten Episoden, die mindestens so viele Denkschritte gekostet haben."""
        auswahl = [i for i, s in enumerate(self.schritte) if s >= mindestens_schritte]
        n = len(self.raetsel[0]) if self.raetsel else 0
        r = np.array([self.raetsel[i] for i in auswahl], np.int64).reshape(-1, n)
        l = np.array([self.loesungen[i] for i in auswahl], np.int64).reshape(-1, n)
        return r, l, np.array([self.schritte[i] for i in auswahl], np.int64)

    def abrufen(self, raetsel):
        """Gibt (Lösungsvorschlag, Ähnlichkeit 0–1 der besten Erinnerung) zurück, oder (None, 0)."""
        if len(self) == 0:
            return None, 0.0
        q = _kodieren(raetsel)
        q /= np.linalg.norm(q) + 1e-9
        aehnlichkeit = self.schluessel @ q
        gewichte = np.exp(self.beta * (aehnlichkeit - aehnlichkeit.max()))
        gewichte /= gewichte.sum()
        felder = len(raetsel)
        mischung = (gewichte @ self.werte).reshape(felder, -1)
        return mischung.argmax(-1) + 1, float(aehnlichkeit.max())
