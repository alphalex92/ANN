"""Kommandozeile: python -m denkkern trainieren | auswerten | loesen | beispiele"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from .agent import Agent
from .sudoku import als_text, datensatz, gueltig_batch
from .training import daten_laden, genauigkeit_pro_schritt, laden, speichern, trainieren

MODELL = Path("denkkern.pt")


def _modell():
    if not MODELL.exists():
        sys.exit(f"{MODELL} fehlt – erst `python -m denkkern trainieren` ausführen")
    return laden(MODELL)


def befehl_trainieren(args):
    torch.set_num_threads(args.threads or torch.get_num_threads())
    modell, _ = trainieren(laeufe=args.laeufe, schritte=args.schritte, d=args.d, lernrate=args.lernrate)
    speichern(modell, MODELL)
    print(f"\nGespeichert in {MODELL}")


def _mittel(werte):
    return float(np.mean(werte)) if len(werte) else float("nan")


def befehl_auswerten(args):
    modell = _modell()
    _, (raetsel, loesung, tiefe) = daten_laden(0, args.anzahl)
    raetsel, loesung, tiefe = raetsel[:args.anzahl], loesung[:args.anzahl], tiefe[:args.anzahl]
    print(f"Modell: {modell.anzahl_parameter():,} Parameter · {len(raetsel)} Testrätsel\n".replace(",", "."))

    # 1) Hilft längeres Nachdenken? Genauigkeit nach festen Schrittzahlen, getrennt nach Denktiefe.
    # Ab Tiefe 6 gibt es nur wenige Rätsel, deshalb fassen wir sie zusammen (6 = "6 und mehr").
    tiefe = np.where(tiefe >= 6, 6, tiefe)
    pro_schritt = {t: genauigkeit_pro_schritt(modell, raetsel[tiefe == t], loesung[tiefe == t], 16)
                   for t in np.unique(tiefe)}
    meta = Agent(modell, metakognition=True, werkzeug=False, gedaechtnis=False).loesen(raetsel, loesung)
    print("Gelöst nach festen Denkschritten, nach Denktiefe des Rätsels:")
    print(f"  {'Tiefe':>9} {'Anzahl':>7} {'1 Schritt':>10} {'4 Schritte':>11} {'16 Schritte':>12} {'Ø Schritte (Metakognition)':>28}")
    tiefe_tabelle = []
    # Tiefe 0 heißt "einfache Regeln reichen nicht": das sind die schwersten, also ans Ende.
    for t, acc in sorted(pro_schritt.items(), key=lambda kv: kv[0] if kv[0] else 99):
        n = int((tiefe == t).sum())
        schnitt = _mittel(meta.schritte[tiefe == t])
        name = {0: "Probieren", 6: "6+"}.get(int(t), str(t))
        print(f"  {name:>9} {n:>7} {acc[0]:>10.1%} {acc[3]:>11.1%} {acc[15]:>12.1%} {schnitt:>28.1f}")
        tiefe_tabelle.append([name, n, f"{acc[0]:.1%}", f"{acc[3]:.1%}", f"{acc[15]:.1%}", f"{schnitt:.1f}"])

    # 2) Was bringt jedes Modul? Gleiches Modell, Module schrittweise zugeschaltet.
    konfigurationen = [
        ("Nur Denkkern, 1 Schritt", dict(metakognition=False, werkzeug=False, gedaechtnis=False, max_schritte=1),
         "Ein Durchlauf, wie ein Netz ohne Nachdenken"),
        ("Denkkern, immer 16 Schritte", dict(metakognition=False, werkzeug=False, gedaechtnis=False),
         "Nachdenken hilft, kostet aber bei jedem Rätsel gleich viel"),
        ("+ Metakognition", dict(metakognition=True, werkzeug=False, gedaechtnis=False),
         "Hört bei leichten Rätseln früh auf, gibt aber auch mal falsch ab"),
        ("+ Werkzeug (Regelprüfer)", dict(metakognition=True, werkzeug=True, gedaechtnis=False),
         "Falsche Entwürfe werden erkannt, das Modell denkt weiter"),
        ("Werkzeug ohne Metakognition", dict(metakognition=False, werkzeug=True, gedaechtnis=False),
         "Prüft nach jedem Schritt. Hier kaum teurer, weil fast alles nach Schritt 1 stimmt"),
    ]
    print("\nWas jedes Modul beiträgt:")
    print(f"  {'Konfiguration':<30} {'gelöst':>7} {'Ø Schritte':>11} {'Ø Prüfungen':>12}")
    ablation = []
    for name, kw, text in konfigurationen:
        e = Agent(modell, **kw).loesen(raetsel, loesung)
        zeile = [name, f"{e.richtig.mean():.1%}", f"{e.schritte.mean():.2f}", f"{e.werkzeug_aufrufe.mean():.2f}", text]
        ablation.append(zeile)
        print(f"  {zeile[0]:<30} {zeile[1]:>7} {zeile[2]:>11} {zeile[3]:>12}")

    # 3) Gedächtnis: dieselben Rätsel ein zweites Mal, dann verwandte Varianten.
    agent = Agent(modell, metakognition=True, werkzeug=True, gedaechtnis=True)
    erst = agent.loesen(raetsel, loesung)
    zweit = agent.loesen(raetsel, loesung)
    rng = np.random.default_rng(7)
    varianten = raetsel.copy()
    for i in range(len(varianten)):
        leer = np.flatnonzero(varianten[i] == 0)
        extra = rng.choice(leer, size=min(2, len(leer)), replace=False)
        varianten[i, extra] = loesung[i, extra]
    verwandt = agent.loesen(varianten, loesung)
    for name, e, text in [
        ("Mit Gedächtnis, 1. Durchgang", erst, "Gedächtnis ist leer, gelöste Rätsel werden gespeichert"),
        ("Mit Gedächtnis, 2. Durchgang", zweit, f"{(zweit.quelle == 'gedaechtnis').mean():.0%} sofort aus dem Gedächtnis"),
        ("Verwandte Rätsel (+2 Vorgaben)", verwandt, f"{(verwandt.quelle == 'gedaechtnis').mean():.0%} über Ähnlichkeit wiedererkannt"),
    ]:
        zeile = [name, f"{e.richtig.mean():.1%}", f"{e.schritte.mean():.2f}", f"{e.werkzeug_aufrufe.mean():.2f}", text]
        ablation.append(zeile)
        print(f"  {zeile[0]:<30} {zeile[1]:>7} {zeile[2]:>11} {zeile[3]:>12}   {text}")

    print(f"\nGedächtnis enthält jetzt {len(agent.gedaechtnis)} Episoden.")
    falsch_abgegeben = int((~erst.richtig & erst.geprueft).sum())
    print(f"Vom Werkzeug bestätigt, aber falsch: {falsch_abgegeben} (muss 0 sein)")

    if args.json:
        Path(args.json).write_text(json.dumps({"ablation": ablation, "tiefe": tiefe_tabelle,
                                               "parameter": modell.anzahl_parameter()}, ensure_ascii=False, indent=1))
        print(f"Ergebnisse als JSON in {args.json}")


def befehl_loesen(args):
    """Zeigt Schritt für Schritt, wie das Modell ein Rätsel durchdenkt."""
    modell = _modell()
    rng = np.random.default_rng(args.seed)
    raetsel, loesung, tiefe = datensatz(1, rng, leer_min=args.leer, leer_max=args.leer)
    print(f"Rätsel ({int((raetsel[0] == 0).sum())} leere Felder, Denktiefe {tiefe[0]}):\n")
    print(als_text(raetsel[0]))
    r = torch.as_tensor(raetsel)
    with torch.no_grad():
        for s, (y, _z, _roh, logit) in enumerate(modell.denken(r, args.schritte), start=1):
            entwurf = modell.ablesen(r, y).numpy()
            falsch = int((entwurf[0] != loesung[0]).sum())
            ok = gueltig_batch(entwurf, raetsel)[0]
            print(f"\nDenkschritt {s:2d}: {falsch:2d} Felder falsch · Zuversicht {torch.sigmoid(logit)[0]:.0%}"
                  f" · Regelprüfer: {'gültig ✔' if ok else 'ungültig'}")
            if ok or s == args.schritte:
                print(als_text(entwurf[0]))
                break


def befehl_beispiele(args):
    rng = np.random.default_rng(args.seed)
    for leer in (10, 18, 26):
        r, _, t = datensatz(1, rng, leer_min=leer, leer_max=leer)
        print(f"{leer} leere Felder, Denktiefe {t[0]}:\n{als_text(r[0])}\n")


def main():
    p = argparse.ArgumentParser(prog="python -m denkkern", description=__doc__)
    sub = p.add_subparsers(dest="befehl", required=True)

    t = sub.add_parser("trainieren", help="Modell trainieren und als denkkern.pt speichern")
    t.add_argument("--laeufe", type=int, default=3000, help="Anzahl Trainings-Batches")
    t.add_argument("--schritte", type=int, default=8, help="Denkschritte pro Rätsel beim Training")
    t.add_argument("--d", type=int, default=64, help="Breite der Vektoren im Arbeitsraum")
    t.add_argument("--lernrate", type=float, default=1e-3)
    t.add_argument("--threads", type=int, default=0)
    t.set_defaults(f=befehl_trainieren)

    a = sub.add_parser("auswerten", help="Module einzeln zu- und abschalten und vergleichen")
    a.add_argument("--anzahl", type=int, default=2000)
    a.add_argument("--json", help="Ergebnisse zusätzlich als JSON speichern")
    a.set_defaults(f=befehl_auswerten)

    l = sub.add_parser("loesen", help="Einem Rätsel beim Nachdenken zuschauen")
    l.add_argument("--leer", type=int, default=24, help="Zahl leerer Felder (8–28)")
    l.add_argument("--schritte", type=int, default=16)
    l.add_argument("--seed", type=int, default=None)
    l.set_defaults(f=befehl_loesen)

    b = sub.add_parser("beispiele", help="Ein paar erzeugte Rätsel zeigen")
    b.add_argument("--seed", type=int, default=None)
    b.set_defaults(f=befehl_beispiele)

    args = p.parse_args()
    args.f(args)


if __name__ == "__main__":
    main()
