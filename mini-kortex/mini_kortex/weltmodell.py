"""Probieren in Gedanken: das Netz als Weltmodell.

Bei manchen Rätseln reichen einfache Schlüsse nicht, man muss eine Annahme treffen ("angenommen,
hier steht eine 5") und schauen, wohin sie führt. Genau dort ist das Modell am schwächsten.

Ein **Weltmodell** sagt voraus, was aus einer Handlung folgt. Hier übernimmt das vorhandene Netz
diese Rolle: Man setzt eine Ziffer probeweise ein und lässt das Netz die Folgen durchdenken. Ist
der Entwurf danach regelkonform, ist das Rätsel gelöst. Sonst sagt die Metakognition, wie
aussichtsreich die Annahme ist, und die beste wird weiterverfolgt.

Pro Ebene:

    1. WÄHLEN    Das Werkzeug kennt die erlaubten Ziffern pro Feld. Gewählt wird das leere Feld mit
                 den wenigsten erlaubten Ziffern (mindestens zwei), wie es auch Menschen tun.
    2. VORSTELLEN Für die `breite` wahrscheinlichsten Ziffern (laut Netz) je ein Gedanken-Rätsel
                 mit dieser Ziffer als Vorgabe. Das Netz denkt jedes `schritte` Schritte durch.
    3. PRÜFEN    Ist ein Entwurf regelkonform, fertig. Führt eine Annahme sofort in einen
                 Widerspruch (ein Feld ohne erlaubte Ziffer), wird sie verworfen.
    4. BEWERTEN  Sonst geht es mit der Annahme weiter, der die Metakognition am meisten zutraut.

Das ist eine gierige Suche (Strahlbreite 1) mit höchstens `tiefe` Annahmen hintereinander.
"""

import numpy as np
import torch

from .modell import regel_tensor
from .sudoku import _kandidaten, _masken, feld_fuer, gueltig_batch


def erlaubte_ziffern(raetsel, diagonal=False):
    """(B, n·n, n) bool: Welche Ziffern sind in jedem leeren Feld nach den Regeln noch erlaubt?
    Vorgegebene Felder sind überall False."""
    raetsel = np.asarray(raetsel)
    f = feld_fuer(raetsel, diagonal)
    erlaubt = np.zeros((*raetsel.shape, f.n), bool)
    for b, g in enumerate(raetsel):
        g = [int(v) for v in g]
        zm, sm, km, dm = _masken(g, f)
        for i, v in enumerate(g):
            if v == 0:
                m = _kandidaten(i, zm, sm, km, dm, f)
                erlaubt[b, i] = [(m >> d) & 1 for d in range(f.n)]
    return erlaubt


@torch.no_grad()
def nachdenken(modell, raetsel, schritte, diagonal=False, regel=None, batch=1024):
    """Lässt das Netz `schritte` Schritte nachdenken und prüft nach jedem Schritt mit dem Werkzeug.

    Rückgabe: geloest (B,), antwort (B, n·n), zuversicht (B,), wahrscheinlichkeiten (B, n·n, n),
    verbrauchte Schritte (B,). Die Antwort ist der erste regelkonforme Entwurf, sonst der letzte.
    """
    modell.eval()
    raetsel = np.asarray(raetsel, np.int64)
    b, felder = raetsel.shape
    n = feld_fuer(raetsel).n
    geloest = np.zeros(b, bool)
    antwort = raetsel.copy()
    zuversicht = np.zeros(b)
    wahrsch = np.zeros((b, felder, n), np.float32)
    verbraucht = np.full(b, schritte)
    for start in range(0, b, batch):
        sl = slice(start, start + batch)
        r_np = raetsel[sl]
        r = torch.as_tensor(r_np).to(modell.geraet)
        offen = np.ones(len(r_np), bool)
        for s, (y, _z, roh, logit) in enumerate(modell.denken(r, schritte, regel_tensor(regel, len(r), modell.geraet)),
                                                start=1):
            entwurf = modell.ablesen(r, y).cpu().numpy()
            ok = gueltig_batch(entwurf, r_np, diagonal) & offen
            idx = np.arange(start, start + len(r_np))
            antwort[idx[ok | (offen & (s == schritte))]] = entwurf[ok | (offen & (s == schritte))]
            geloest[idx[ok]] = True
            verbraucht[idx[ok]] = s
            offen &= ~ok
            if s == schritte or not offen.any():
                zuversicht[sl] = torch.sigmoid(logit).cpu().numpy()
                wahrsch[sl] = torch.softmax(roh, -1).cpu().numpy()
                break
    return geloest, antwort, zuversicht, wahrsch, verbraucht


def probieren(modell, raetsel, basis_schritte=32, tiefe=3, breite=3, schritte=16, diagonal=False, regel=None,
              ordnung="netz", bewertung="metakognition", seed=0):
    """Erst normal nachdenken, dann für alles Ungelöste Annahmen treffen und durchdenken.

    Zum Vergleich (Ablation) lassen sich die beiden Beiträge des Netzes abschalten:
    ordnung="zufall":    welche erlaubten Ziffern ausprobiert werden, entscheidet der Zufall statt der Intuition
    bewertung="zufall":  mit welcher Annahme es weitergeht, entscheidet der Zufall statt der Metakognition

    Rückgabe: dict mit geloest (B,), antwort (B, n·n), schritte (B,) = alle Denkschritte, die für das
    Rätsel verbraucht wurden (über alle Annahmen), annahmen (B,) = wie viele Annahmen nötig waren
    (0 = ohne Probieren gelöst, -1 = nicht gelöst).
    """
    raetsel = np.asarray(raetsel, np.int64)
    rng = np.random.default_rng(seed)
    geloest, antwort, zuversicht, wahrsch, verbraucht = nachdenken(modell, raetsel, basis_schritte, diagonal, regel)
    gesamt = verbraucht.astype(np.int64)
    annahmen = np.where(geloest, 0, -1)
    offen = np.flatnonzero(~geloest)
    zustand = raetsel[offen].copy()          # Rätsel samt bisheriger Annahmen
    w_zustand = wahrsch[offen]
    for ebene in range(1, tiefe + 1):
        if len(offen) == 0:
            break
        erlaubt = erlaubte_ziffern(zustand, diagonal)
        anzahl = erlaubt.sum(-1)
        # 1. Wählen: das Feld mit den wenigsten erlaubten Ziffern (mindestens zwei).
        kosten = np.where(anzahl >= 2, anzahl, 99)
        feld = kosten.argmin(1)
        # 2. Vorstellen: die wahrscheinlichsten erlaubten Ziffern laut Netz.
        intuition = w_zustand[np.arange(len(offen)), feld]
        if ordnung == "zufall":
            intuition = rng.random(intuition.shape)
        p = np.where(erlaubt[np.arange(len(offen)), feld], intuition, -1.0)
        ziffern = np.argsort(-p, 1)[:, :breite] + 1
        gueltige_wahl = np.take_along_axis(p, ziffern - 1, 1) >= 0
        kinder = np.repeat(zustand, breite, 0)
        kinder[np.arange(len(kinder)), np.repeat(feld, breite)] = ziffern.reshape(-1)
        k_ok, k_antwort, k_zuversicht, k_wahrsch, k_verbraucht = nachdenken(modell, kinder, schritte, diagonal, regel)
        # Annahmen, die gar nicht erlaubt waren oder sofort ein Feld ohne Möglichkeit lassen, fallen weg.
        widerspruch = ((erlaubte_ziffern(kinder, diagonal).sum(-1) == 0) & (kinder == 0)).any(1)
        nutzbar = gueltige_wahl.reshape(-1) & ~widerspruch
        k_ok &= nutzbar
        if bewertung == "zufall":
            k_zuversicht = rng.random(k_zuversicht.shape)
        k_zuversicht = np.where(nutzbar, k_zuversicht, -1.0)
        k_verbraucht = np.where(gueltige_wahl.reshape(-1), k_verbraucht, 0)
        form = (len(offen), breite)
        gesamt[offen] += k_verbraucht.reshape(form).sum(1)
        # 3. Prüfen: ist eine Annahme aufgegangen?
        ok = k_ok.reshape(form)
        treffer = ok.any(1)
        erstes = ok.argmax(1)
        loes = np.flatnonzero(treffer)
        geloest[offen[loes]] = True
        antwort[offen[loes]] = k_antwort.reshape(*form, -1)[loes, erstes[loes]]
        annahmen[offen[loes]] = ebene
        # 4. Bewerten: mit der aussichtsreichsten Annahme weiter (gierig).
        weiter = np.flatnonzero(~treffer & (k_zuversicht.reshape(form).max(1) >= 0))
        beste = k_zuversicht.reshape(form).argmax(1)
        zustand = kinder.reshape(*form, -1)[weiter, beste[weiter]]
        w_zustand = k_wahrsch.reshape(*form, *k_wahrsch.shape[1:])[weiter, beste[weiter]]
        offen = offen[weiter]
    return {"geloest": geloest, "antwort": antwort, "schritte": gesamt, "annahmen": annahmen}
