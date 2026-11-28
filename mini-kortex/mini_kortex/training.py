"""Training des Denkmodells mit tiefer Überwachung (deep supervision)."""

import math
import time
from pathlib import Path

import numpy as np
import torch

from .modell import DenkModell, regel_tensor, verlust
from .sudoku import datensatz, varianten


def datenpfad(n):
    return Path(f"daten_sudoku{n}.npz")


def daten_laden(n, anzahl_training, anzahl_test, seed=0, prozesse=4, protokoll=print):
    """Erzeugt die Rätsel einmal und legt sie als Datei ab. Danach geht das Laden sofort.

    Rückgabe: (training, test), jeweils (raetsel, loesungen, tiefe).
    """
    pfad = datenpfad(n)
    if pfad.exists():
        d = np.load(pfad)
        if len(d["tr_r"]) >= anzahl_training and len(d["te_r"]) >= anzahl_test:
            return ((d["tr_r"][:anzahl_training], d["tr_l"][:anzahl_training], d["tr_t"][:anzahl_training]),
                    (d["te_r"][:anzahl_test], d["te_l"][:anzahl_test], d["te_t"][:anzahl_test]))
    protokoll(f"Erzeuge {anzahl_training} + {anzahl_test} Rätsel ({n}×{n}, einmalig) …")
    # Getrennte Zufallsströme: Die Testrätsel sind dieselben, egal wie viele Trainingsrätsel es gibt.
    tr = datensatz(anzahl_training, np.random.default_rng(seed), n=n, prozesse=prozesse)
    te = datensatz(anzahl_test, np.random.default_rng(seed + 1_000_000), n=n, prozesse=prozesse)
    np.savez_compressed(pfad, tr_r=tr[0], tr_l=tr[1], tr_t=tr[2], te_r=te[0], te_l=te[1], te_t=te[2])
    return tr, te


@torch.no_grad()
def genauigkeit_pro_schritt(modell, raetsel, loesung, schritte, batch=500, regel=None):
    """Anteil komplett gelöster Rätsel nach 1, 2, …, `schritte` Denkschritten.
    `regel`: Regel-Hinweis für alle Rätsel (0 oder 1), nur bei Modellen mit Hinweis-Eingang."""
    modell.eval()
    geraet = modell.geraet
    treffer = torch.zeros(schritte, device=geraet)
    for i in range(0, len(raetsel), batch):
        r = torch.as_tensor(raetsel[i:i + batch]).to(geraet)
        l = torch.as_tensor(loesung[i:i + batch]).to(geraet)
        for s, (y, *_rest) in enumerate(modell.denken(r, schritte, regel_tensor(regel, len(r), geraet))):
            treffer[s] += (modell.ablesen(r, y) == l).all(1).sum()
    modell.train()
    return treffer.cpu().numpy() / max(1, len(raetsel))


def ziehen_aus(raetsel, loesungen, batch, symmetrien=True):
    """Gibt eine Funktion zurück, die zufällige Batches zieht, auf Wunsch als Symmetrie-Varianten.

    Durch Ziffern umbenennen und Zeilen/Spalten tauschen wird aus jedem Rätsel ein neues,
    gleich schweres. Das Netz sieht so dasselbe Rätsel praktisch nie zweimal.
    """
    def ziehen(rng):
        idx = rng.integers(0, len(raetsel), batch)
        r, l = raetsel[idx], loesungen[idx]
        if symmetrien:
            r, l = varianten(r, l, rng)
        return r, l
    return ziehen


def trainingslauf(modell, ziehen, laeufe, schritte=8, lernrate=1e-3, aufwaermen=200, seed=0,
                  pruefen=None, alle=250, protokoll=print, zwischenstand=None, strafe=None):
    """Die eigentliche Trainingsschleife. `ziehen(rng)` liefert pro Lauf einen Batch (Rätsel, Lösungen),
    bei Modellen mit Regel-Hinweis auch (Rätsel, Lösungen, Regel pro Rätsel).

    `pruefen(lauf)` darf einen Text zurückgeben, der alle `alle` Läufe ins Protokoll geschrieben wird.
    `strafe(modell)`: zusätzlicher Verlust (Tensor), der zu jedem Teilverlust addiert wird, etwa EWC.
    `zwischenstand`: Datei, in die alle `alle` Läufe der komplette Trainingszustand geschrieben wird.
    Existiert sie beim Start schon, geht das Training dort weiter. So kostet ein Absturz oder
    Neustart des Rechners höchstens `alle` Läufe.
    """
    opt = torch.optim.AdamW(modell.parameters(), lr=lernrate, weight_decay=0.01)

    # Lernrate: kurz hochfahren, dann sanft (Kosinus) absenken.
    def plan(lauf):
        if lauf < aufwaermen:
            return (lauf + 1) / aufwaermen
        return 0.5 * (1 + math.cos(math.pi * (lauf - aufwaermen) / max(1, laeufe - aufwaermen)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, plan)
    rng = np.random.default_rng(seed)
    erster = 1
    if zwischenstand is not None and Path(zwischenstand).exists():
        stand = torch.load(zwischenstand, weights_only=False, map_location=modell.geraet)  # eigene Datei, enthält auch den Zufallszustand
        modell.load_state_dict(stand["modell"])
        opt.load_state_dict(stand["optimierer"])
        sched.load_state_dict(stand["plan"])
        rng.bit_generator.state = stand["zufall"]
        erster = stand["lauf"] + 1
        protokoll(f"Setze Training nach Lauf {stand['lauf']} fort (Zwischenstand {zwischenstand})")
    modell.train()
    geraet = modell.geraet
    start = time.time()
    for lauf in range(erster, laeufe + 1):
        r, l, *regel = ziehen(rng)
        r, l = torch.as_tensor(r).to(geraet, non_blocking=True), torch.as_tensor(l).to(geraet, non_blocking=True)
        regel = regel_tensor(regel[0], len(r), geraet) if regel else None
        letzter = 0.0
        for teil, richtig in verlust(modell, r, l, schritte, regel):
            opt.zero_grad()
            if strafe is not None:
                teil = teil + strafe(modell)
            teil.backward()
            torch.nn.utils.clip_grad_norm_(modell.parameters(), 1.0)
            opt.step()
            letzter = richtig
        sched.step()
        if pruefen and (lauf % alle == 0 or lauf == laeufe):
            protokoll(f"Lauf {lauf:5d}  gelöst (Training, letzter Schritt): {float(letzter):5.1%}   "
                      f"{pruefen(lauf)}   ({time.time() - start:.0f} s)")
        if zwischenstand is not None and lauf % alle == 0 and lauf < laeufe:
            # Erst in eine Hilfsdatei schreiben, dann umbenennen: So ist der Zwischenstand nie halb geschrieben.
            tmp = Path(str(zwischenstand) + ".tmp")
            torch.save({"modell": modell.state_dict(), "optimierer": opt.state_dict(), "plan": sched.state_dict(),
                        "zufall": rng.bit_generator.state, "lauf": lauf}, tmp)
            tmp.replace(zwischenstand)
    modell.eval()
    return modell


def ewc_strafe(modell, raetsel, loesungen, staerke, schritte=8, batch=100, regel=None):
    """Elastic Weight Consolidation (Kirkpatrick u. a. 2017): wichtige Gewichte festhalten.

    Wie wichtig ein Gewicht für das Alte ist, schätzt die Fisher-Information: das mittlere Quadrat
    seines Gradienten auf alten Rätseln. Die Strafe wächst mit dem Quadrat der Abweichung vom alten
    Wert, gewichtet mit dieser Wichtigkeit. Unwichtige Gewichte dürfen sich frei bewegen.
    Gibt `strafe(modell)` für `trainingslauf` zurück.
    """
    alt = {n: p.detach().clone() for n, p in modell.named_parameters()}
    fisher = {n: torch.zeros_like(p) for n, p in modell.named_parameters()}
    geraet = modell.geraet
    modell.train()
    anzahl = 0
    for i in range(0, len(raetsel), batch):
        r = torch.as_tensor(raetsel[i:i + batch]).to(geraet)
        l = torch.as_tensor(loesungen[i:i + batch]).to(geraet)
        for teil, _ in verlust(modell, r, l, schritte, regel_tensor(regel, len(r), geraet)):
            modell.zero_grad()
            teil.backward()
            for n, p in modell.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.detach() ** 2
            anzahl += 1
    modell.zero_grad()
    fisher = {n: f / anzahl for n, f in fisher.items()}

    def strafe(m):
        return staerke * sum((fisher[n] * (p - alt[n]) ** 2).sum() for n, p in m.named_parameters())
    return strafe


def trainieren(n=6, laeufe=3000, batch=128, schritte=8, d=64, innere=3, lernrate=1e-3,
               anzahl_training=40_000, anzahl_test=2_000, leer_max=None, seed=0, protokoll=print,
               zwischenstand=None, geraet="cpu", block="mixer"):
    """Trainiert ein neues Modell. Mit `leer_max` nur auf Rätseln mit höchstens so vielen Leerfeldern.
    `block`: "mixer" (Standard) oder "attention" für den Denkkern."""
    torch.manual_seed(seed)
    (tr_r, tr_l, _), (te_r, te_l, te_t) = daten_laden(n, anzahl_training, anzahl_test, seed, protokoll=protokoll)
    if leer_max is not None:
        auswahl = (tr_r == 0).sum(1) <= leer_max
        tr_r, tr_l = tr_r[auswahl], tr_l[auswahl]
        protokoll(f"Training nur auf Rätseln mit höchstens {leer_max} leeren Feldern: {len(tr_r)} Stück")
    modell = DenkModell(n=n, d=d, innere=innere, block=block).to(geraet)
    protokoll(f"Modell: {modell.anzahl_parameter():,} Parameter · Batch {batch} · rechnet auf {geraet}".replace(",", "."))

    def pruefen(_lauf):
        acc = genauigkeit_pro_schritt(modell, te_r[:500], te_l[:500], schritte)
        return (f"Test nach 1/{schritte // 2}/{schritte} Schritten: "
                f"{acc[0]:5.1%} / {acc[schritte // 2 - 1]:5.1%} / {acc[-1]:5.1%}")

    trainingslauf(modell, ziehen_aus(tr_r, tr_l, batch), laeufe, schritte, lernrate, seed=seed,
                  pruefen=pruefen, protokoll=protokoll, zwischenstand=zwischenstand)
    return modell, (te_r, te_l, te_t)


def speichern(modell, pfad, info=None):
    """`info`: zusätzliche Angaben zum Training, z. B. auf welchen Rätseln trainiert wurde.
    Gespeichert wird immer in CPU-Form, damit die Datei auch auf Rechnern ohne GPU lädt."""
    gewichte = {k: v.cpu() for k, v in modell.state_dict().items()}
    torch.save({"konfig": modell.konfig, "gewichte": gewichte, "info": info or {}}, pfad)


def laden(pfad, nur_info=False, geraet="cpu"):
    daten = torch.load(pfad, weights_only=True, map_location="cpu")
    if nur_info:
        return daten.get("info", {})
    konfig = {"n": 6, **daten["konfig"]}  # ältere Modelle kannten nur 6×6
    modell = DenkModell(**konfig)
    modell.load_state_dict(daten["gewichte"])
    modell.to(geraet).eval()
    return modell
