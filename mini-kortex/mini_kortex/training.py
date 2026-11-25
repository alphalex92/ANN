"""Training des Denkmodells mit tiefer Überwachung (deep supervision)."""

import math
import time
from pathlib import Path

import numpy as np
import torch

from .modell import DenkModell, verlust
from .sudoku import datensatz

DATEN = Path("daten_sudoku6.npz")


def daten_laden(anzahl_training, anzahl_test, seed=0):
    """Erzeugt die Rätsel einmal und legt sie als Datei ab. Danach geht das Laden sofort."""
    if DATEN.exists():
        d = np.load(DATEN)
        if len(d["tr_r"]) >= anzahl_training and len(d["te_r"]) >= anzahl_test:
            return ((d["tr_r"][:anzahl_training], d["tr_l"][:anzahl_training], d["tr_t"][:anzahl_training]),
                    (d["te_r"][:anzahl_test], d["te_l"][:anzahl_test], d["te_t"][:anzahl_test]))
    print(f"Erzeuge {anzahl_training} + {anzahl_test} Rätsel (einmalig) …")
    # Getrennte Zufallsströme: Die Testrätsel sind dieselben, egal wie viele Trainingsrätsel es gibt.
    tr = datensatz(anzahl_training, np.random.default_rng(seed))
    te = datensatz(anzahl_test, np.random.default_rng(seed + 1_000_000))
    np.savez_compressed(DATEN, tr_r=tr[0], tr_l=tr[1], tr_t=tr[2], te_r=te[0], te_l=te[1], te_t=te[2])
    return tr, te


@torch.no_grad()
def genauigkeit_pro_schritt(modell, raetsel, loesung, schritte, batch=500):
    """Anteil komplett gelöster Rätsel nach 1, 2, …, `schritte` Denkschritten."""
    modell.eval()
    treffer = np.zeros(schritte)
    for i in range(0, len(raetsel), batch):
        r = torch.as_tensor(raetsel[i:i + batch])
        l = torch.as_tensor(loesung[i:i + batch])
        for s, (y, *_rest) in enumerate(modell.denken(r, schritte)):
            treffer[s] += (modell.ablesen(r, y) == l).all(1).sum().item()
    modell.train()
    return treffer / len(raetsel)


def trainieren(laeufe=4000, batch=128, schritte=8, d=64, innere=3, lernrate=1e-3,
               anzahl_training=40_000, anzahl_test=2_000, seed=0, protokoll=print):
    torch.manual_seed(seed)
    (tr_r, tr_l, _), (te_r, te_l, te_t) = daten_laden(anzahl_training, anzahl_test, seed)
    modell = DenkModell(d=d, innere=innere)
    protokoll(f"Modell: {modell.anzahl_parameter():,} Parameter".replace(",", "."))
    opt = torch.optim.AdamW(modell.parameters(), lr=lernrate, weight_decay=0.01)

    # Lernrate: kurz hochfahren, dann sanft (Kosinus) absenken.
    def plan(lauf):
        if lauf < 200:
            return (lauf + 1) / 200
        return 0.5 * (1 + math.cos(math.pi * (lauf - 200) / max(1, laeufe - 200)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, plan)
    rng = np.random.default_rng(seed)
    start = time.time()
    for lauf in range(1, laeufe + 1):
        idx = rng.integers(0, len(tr_r), batch)
        r, l = torch.as_tensor(tr_r[idx]), torch.as_tensor(tr_l[idx])
        letzter = 0.0
        for teil, richtig in verlust(modell, r, l, schritte):
            opt.zero_grad()
            teil.backward()
            torch.nn.utils.clip_grad_norm_(modell.parameters(), 1.0)
            opt.step()
            letzter = richtig
        sched.step()
        if lauf % 250 == 0 or lauf == laeufe:
            acc = genauigkeit_pro_schritt(modell, te_r[:500], te_l[:500], schritte)
            protokoll(f"Lauf {lauf:5d}  gelöst (Training, letzter Schritt): {letzter:5.1%}   "
                      f"Test nach 1/{schritte // 2}/{schritte} Schritten: "
                      f"{acc[0]:5.1%} / {acc[schritte // 2 - 1]:5.1%} / {acc[-1]:5.1%}   "
                      f"({time.time() - start:.0f} s)")
    return modell, (te_r, te_l, te_t)


def speichern(modell, pfad):
    torch.save({"konfig": modell.konfig, "gewichte": modell.state_dict()}, pfad)


def laden(pfad):
    daten = torch.load(pfad, weights_only=True)
    modell = DenkModell(**daten["konfig"])
    modell.load_state_dict(daten["gewichte"])
    modell.eval()
    return modell
