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

from . import gedaechtnis, messung, welt
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

    k = unter.add_parser("kindheit", help="Gedächtnis-Experiment: Kern mit Gedächtnisschicht vortrainieren")
    k.add_argument("--name", required=True)
    k.add_argument("--minuten", type=float, default=10.0)
    k.add_argument("--plaetze", type=int, default=64, help="Gedächtnis mit plaetze² Plätzen")
    k.add_argument("--seed", type=int, default=0)
    k.add_argument("--stapel", type=int, default=256)
    k.add_argument("--kern-faktor", type=float, default=1.0, help="Lernrate des Denkens relativ zu 1e-3")
    k.add_argument("--welten", type=int, default=1, help="so viele Kindheitswelten, Gedächtnis je Welt gelöscht")
    k.add_argument("--gpu-anteil", type=float, default=None, help="höchstens so viel GPU-Speicher nutzen")

    l = unter.add_parser("leben", help="Gedächtnis-Experiment: Tage mit Recherche, Sichten und Schlaf")
    l.add_argument("--name", required=True)
    l.add_argument("--tage", type=int, default=8)
    l.add_argument("--fragen", type=int, default=200, help="Fragen je Tag")
    l.add_argument("--festigen", choices=["gedaechtnis", "ganz", "keiner"], default="gedaechtnis")
    l.add_argument("--sichten", choices=["pruefen", "alles"], default="pruefen")
    l.add_argument("--schlaf", type=int, default=300, help="Übungsschritte je Nacht")
    l.add_argument("--wandel-tag", type=int, default=None, help="an diesem Morgen ändert sich ein Teil der Welt")
    l.add_argument("--gpu-anteil", type=float, default=None)

    a = p.parse_args()
    if getattr(a, "gpu_anteil", None) and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(a.gpu_anteil)
    if a.befehl == "kindheit":
        modell, info = gedaechtnis.kindheit(gedaechtnis.neue_welt(a.seed), a.minuten, plaetze=a.plaetze, seed=a.seed,
                                              stapel=a.stapel, welten=a.welten,
                                              kern_faktor=a.kern_faktor)
        kwelt = gedaechtnis.neue_welt(a.seed)
        info |= {"seed": a.seed, "test": {
            art: gedaechtnis.messen(modell, art, dict(kwelt.kindheit)) for art in ("kindheit", "neu")}}
        print(json.dumps(info["test"], indent=1))
        modell.speichern(LAEUFE / a.name / "kindheit.pt", **info)
        return
    if a.befehl == "leben":
        modell, info = SchleifenGPT.laden(LAEUFE / a.name / "kindheit.pt")
        modell.to("cuda" if torch.cuda.is_available() else "cpu")
        variante = f"{a.festigen}-{a.sichten}" + (f"-wandel{a.wandel_tag}" if a.wandel_tag else "")
        print(f"Leben {a.name}, Variante {variante}")
        tagebuch = gedaechtnis.leben(modell, gedaechtnis.neue_welt(info["seed"]), a.tage, a.fragen, a.festigen,
                                     a.sichten, a.schlaf, a.wandel_tag)
        (LAEUFE / a.name / f"leben-{variante}.json").write_text(json.dumps(tagebuch, indent=1))
        return
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
