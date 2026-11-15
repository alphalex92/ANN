"""Befehle:

    python -m denkkern beispiel  --modus werkzeug --hops 3
    python -m denkkern trainieren --modus werkzeug --hops 1-4 --schleifen 1-4 --minuten 5 --name w1
    python -m denkkern messen    --name w1 --hops 1-12 --schleifen 1,2,4,8
"""

import argparse
import json
import random
from pathlib import Path

import torch

from . import messung, welt
from .modell import SchleifenGPT
from .training import bereich, trainieren

LAEUFE = Path(__file__).resolve().parent.parent / "laeufe"


def liste(text):
    """'1-4' → [1,2,3,4], '1,2,8' → [1,2,8]."""
    if "," in text:
        return [int(t) for t in text.split(",")]
    lo, hi = bereich(text)
    return list(range(lo, hi + 1))


def main():
    p = argparse.ArgumentParser(prog="denkkern", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    unter = p.add_subparsers(dest="befehl", required=True)

    b = unter.add_parser("beispiel", help="eine vorbildliche Lösung zeigen")
    b.add_argument("--modus", choices=welt.MODI, default="werkzeug")
    b.add_argument("--hops", type=int, default=3)
    b.add_argument("--seed", type=int, default=0)

    t = unter.add_parser("trainieren")
    t.add_argument("--modus", choices=welt.MODI, required=True)
    t.add_argument("--name", required=True, help="Ordner unter laeufe/")
    t.add_argument("--hops", default="1-4")
    t.add_argument("--schleifen", default="1-4", help="pro Stapel zufällig aus diesem Bereich")
    t.add_argument("--minuten", type=float, default=5.0)
    t.add_argument("--stapel", type=int, default=256)
    t.add_argument("--lernrate", type=float, default=1e-3)
    t.add_argument("--positionen", choices=["rope", "keine"], default="rope")
    t.add_argument("--breite", type=int, default=256)
    t.add_argument("--kern", type=int, default=2, help="Blöcke in der Schleife")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--mess-hops", type=int, default=12, help="so lange Ketten soll der Kontext später fassen")

    m = unter.add_parser("messen")
    m.add_argument("--name", required=True)
    m.add_argument("--hops", default="1-12")
    m.add_argument("--schleifen", default="1,2,4,8")
    m.add_argument("--anzahl", type=int, default=500)

    a = p.parse_args()
    if a.befehl == "beispiel":
        ep = welt.neue_episode(random.Random(a.seed), a.hops)
        for tokens, eigen in welt.folgen(ep, a.modus):
            print(welt.lesbar(tokens))
            print("  vom Modell geschrieben:", welt.lesbar([t for t, e in zip(tokens, eigen) if e]))
    elif a.befehl == "trainieren":
        trainieren(a.modus, a.hops, a.schleifen, a.minuten, a.stapel, a.lernrate, a.positionen,
                   a.breite, a.kern, a.seed, pfad=LAEUFE / a.name / "modell.pt", mess_hops=a.mess_hops)
    elif a.befehl == "messen":
        modell, info = SchleifenGPT.laden(LAEUFE / a.name / "modell.pt")
        geraet = "cuda" if torch.cuda.is_available() else "cpu"
        modell.to(geraet)
        hops, schleifen = liste(a.hops), liste(a.schleifen)
        ergebnis = messung.messen(modell, info["modus"], hops, schleifen, a.anzahl)
        print(f"Lauf {a.name}: {info}\n")
        print("Richtig:\n")
        print(messung.tabelle_richtig(ergebnis, hops, schleifen, info["hops"]))
        s_train = info["schleifen"][1]
        s_zeigen = s_train if s_train in schleifen else schleifen[-1]
        print(f"\nAufgeschlüsselt bei {s_zeigen} Schleifen:\n")
        print(messung.tabelle_kategorien(ergebnis, info["modus"], hops, s_zeigen))
        roh = {f"{h},{s}": {"kategorien": dict(r["kategorien"]), "weg_exakt": r["weg_exakt"], "anzahl": r["anzahl"]}
               for (h, s), r in ergebnis.items()}
        (LAEUFE / a.name / "messung.json").write_text(json.dumps({"info": info, "ergebnis": roh}, indent=1))


if __name__ == "__main__":
    main()
