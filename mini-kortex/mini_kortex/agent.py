"""Der Agent: verbindet alle Module zu einem Denkablauf.

Ablauf für jedes Rätsel:

    1. GEDÄCHTNIS   Kenne ich das (oder etwas sehr Ähnliches) schon?
                    → Vorschlag vom Werkzeug prüfen lassen. Passt er, fertig, ohne Nachdenken.
    2. DENKEN       Sonst: Denkschritt für Denkschritt den Entwurf im Arbeitsraum verbessern.
    3. METAKOGNITION Nach jedem Schritt: Wie sicher bin ich? Erst wenn ich sicher genug bin,
                    lohnt es sich, die Arbeit abzugeben bzw. prüfen zu lassen.
    4. WERKZEUG     Exakte Regelprüfung. Ist der Entwurf falsch, wird weitergedacht,
                    statt eine falsche Antwort abzugeben.
    5. LERNEN       Jede geprüfte Lösung wandert sofort ins Gedächtnis.

Jedes Modul lässt sich einzeln abschalten. So sieht man, was es zum Ganzen beiträgt.
"""

from dataclasses import dataclass

import numpy as np
import torch

from .gedaechtnis import HopfieldGedaechtnis
from .sudoku import gueltig_batch


@dataclass
class Ergebnis:
    antwort: np.ndarray        # (B, 36) Ziffern
    richtig: np.ndarray        # (B,) stimmt die Antwort wirklich? (nur zur Auswertung)
    schritte: np.ndarray       # (B,) verbrauchte Denkschritte (0 = aus dem Gedächtnis)
    quelle: np.ndarray         # (B,) "gedaechtnis" oder "denken"
    geprueft: np.ndarray       # (B,) hat das Werkzeug die Antwort bestätigt?
    werkzeug_aufrufe: np.ndarray  # (B,) wie oft wurde geprüft?
    zuversicht: np.ndarray     # (B,) letzte Selbsteinschätzung 0–1


class Agent:
    def __init__(self, modell, metakognition=True, werkzeug=True, gedaechtnis=True,
                 max_schritte=16, schwelle=0.5):
        self.modell = modell.eval()
        self.metakognition = metakognition
        self.werkzeug = werkzeug
        self.gedaechtnis = HopfieldGedaechtnis() if gedaechtnis else None
        self.max_schritte = max_schritte
        self.schwelle = schwelle

    def _aus_gedaechtnis(self, raetsel):
        """Versucht, jedes Rätsel aus dem Gedächtnis zu beantworten. Gibt (Antworten, Treffer) zurück."""
        b = len(raetsel)
        antworten = np.zeros_like(raetsel)
        treffer = np.zeros(b, bool)
        if self.gedaechtnis is None or len(self.gedaechtnis) == 0:
            return antworten, treffer
        for i in range(b):
            vorschlag, aehnlichkeit = self.gedaechtnis.abrufen(raetsel[i])
            if vorschlag is None:
                continue
            if self.werkzeug:
                ok = gueltig_batch(vorschlag[None], raetsel[i][None])[0]
            else:
                # Ohne Werkzeug bleibt nur das Bauchgefühl: nur bei fast identischen Erinnerungen trauen.
                ok = aehnlichkeit > 0.95 and bool(((raetsel[i] == 0) | (raetsel[i] == vorschlag)).all())
            if ok:
                antworten[i], treffer[i] = vorschlag, True
        return antworten, treffer

    @torch.no_grad()
    def loesen(self, raetsel, loesung=None):
        raetsel = np.asarray(raetsel, np.int64)
        b = len(raetsel)
        antwort, aus_gedaechtnis = self._aus_gedaechtnis(raetsel)

        schritte = np.zeros(b, np.int64)
        geprueft = aus_gedaechtnis & self.werkzeug
        aufrufe = aus_gedaechtnis.astype(np.int64) * int(self.werkzeug)
        zuversicht = aus_gedaechtnis.astype(np.float64)
        aktiv = ~aus_gedaechtnis

        if aktiv.any():
            idx = np.flatnonzero(aktiv)
            r = torch.as_tensor(raetsel[idx])
            noch_aktiv = np.ones(len(idx), bool)
            for s, (y, _z, _roh, logit) in enumerate(self.modell.denken(r, self.max_schritte), start=1):
                entwurf = self.modell.ablesen(r, y).numpy()
                sicher = torch.sigmoid(logit).numpy()
                letzter = s == self.max_schritte

                # Metakognition: Bin ich sicher genug, um abzugeben bzw. prüfen zu lassen?
                will_abgeben = sicher >= self.schwelle if self.metakognition else np.full(len(idx), letzter)
                will_abgeben &= noch_aktiv

                if self.werkzeug:
                    # Mit Werkzeug: Ohne Metakognition wird nach jedem Schritt geprüft (teuer),
                    # mit Metakognition nur, wenn das Modell sich sicher fühlt.
                    pruefen = will_abgeben if self.metakognition else noch_aktiv.copy()
                    ok = gueltig_batch(entwurf, raetsel[idx]) & pruefen
                    aufrufe[idx[pruefen]] += 1
                    fertig = ok | (noch_aktiv & letzter)
                    geprueft[idx[ok]] = True
                else:
                    fertig = will_abgeben | (noch_aktiv & letzter)

                antwort[idx[fertig]] = entwurf[fertig]
                schritte[idx[fertig]] = s
                zuversicht[idx[fertig]] = sicher[fertig]
                noch_aktiv &= ~fertig
                if not noch_aktiv.any():
                    break

        # Lernen: jede vom Werkzeug bestätigte, neu erdachte Lösung sofort speichern.
        if self.gedaechtnis is not None:
            for i in np.flatnonzero(geprueft & ~aus_gedaechtnis):
                self.gedaechtnis.speichern(raetsel[i], antwort[i])

        richtig = (antwort == loesung).all(1) if loesung is not None else gueltig_batch(antwort, raetsel)
        quelle = np.where(aus_gedaechtnis, "gedaechtnis", "denken")
        return Ergebnis(antwort, richtig, schritte, quelle, geprueft, aufrufe, zuversicht)
