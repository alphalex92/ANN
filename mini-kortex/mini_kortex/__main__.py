"""Kommandozeile: python -m mini_kortex [--groesse 6|9] trainieren | auswerten | konsolidieren | umlernen | loesen | beispiele"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from .agent import Agent
from .konsolidierung import SPALTEN, experiment, experiment_daten
from .sudoku import LEER_STANDARD, als_text, datensatz, gueltig_batch
from .training import daten_laden, genauigkeit_pro_schritt, laden, speichern, trainieren

# Pro Größe: wie lange höchstens nachgedacht wird, und auf welchen Rätseln standardmäßig trainiert wird.
MAX_SCHRITTE = {6: 16, 9: 32}
LEER_TRAINING = {6: 28, 9: 46}  # 9×9: die schwersten Rätsel bleiben für die Konsolidierung übrig


def modellpfad(n, zusatz=""):
    return Path(f"mini_kortex_{n}{zusatz}.pt")


def geraet_waehlen(wunsch):
    """'auto' nimmt die GPU, wenn PyTorch eine findet, sonst die CPU."""
    if wunsch == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if wunsch == "cuda" and not torch.cuda.is_available():
        sys.exit("--geraet cuda: PyTorch findet keine GPU (CUDA). Mit --geraet cpu läuft alles auf der CPU.")
    return wunsch


def _modell(n, geraet="cpu"):
    pfad = modellpfad(n)
    if not pfad.exists():
        sys.exit(f"{pfad} fehlt – erst `python -m mini_kortex --groesse {n} trainieren` ausführen")
    return laden(pfad, geraet=geraet)


def _tiefe_gruppen(n):
    """Gruppen der Denktiefe für die Auswertung. Tiefe 0 heißt: einfache Regeln reichen nicht."""
    if n == 6:
        gruppen = [(str(t), lambda x, t=t: x == t) for t in range(1, 6)] + [("6+", lambda x: x >= 6)]
    else:
        gruppen = [("1–3", lambda x: (x >= 1) & (x <= 3)), ("4–6", lambda x: (x >= 4) & (x <= 6)),
                   ("7–9", lambda x: (x >= 7) & (x <= 9)), ("10+", lambda x: x >= 10)]
    return gruppen + [("Probieren", lambda x: x == 0)]


def befehl_trainieren(args):
    n = args.groesse
    if args.threads:
        torch.set_num_threads(args.threads)
    leer_max = args.leer_max if args.leer_max is not None else LEER_TRAINING[n]
    zwischenstand = modellpfad(n, ".zwischenstand")
    modell, _ = trainieren(n=n, laeufe=args.laeufe, batch=args.batch, schritte=args.schritte, d=args.d,
                           lernrate=args.lernrate, leer_max=leer_max, zwischenstand=zwischenstand,
                           geraet=args.geraet, seed=args.seed)
    speichern(modell, modellpfad(n), info={"leer_max": leer_max, "schritte": args.schritte})
    zwischenstand.unlink(missing_ok=True)
    print(f"\nGespeichert in {modellpfad(n)}")


def _mittel(werte):
    return float(np.mean(werte)) if len(werte) else float("nan")


def befehl_auswerten(args):
    n = args.groesse
    modell = _modell(n, args.geraet)
    lang = MAX_SCHRITTE[n]
    mittel = lang // 4
    _, (raetsel, loesung, tiefe) = daten_laden(n, 0, args.anzahl)
    raetsel, loesung, tiefe = raetsel[:args.anzahl], loesung[:args.anzahl], tiefe[:args.anzahl]
    print(f"Modell: {modell.anzahl_parameter():,} Parameter · {len(raetsel)} Testrätsel ({n}×{n})\n".replace(",", "."))

    # 1) Hilft längeres Nachdenken? Genauigkeit nach festen Schrittzahlen, getrennt nach Denktiefe.
    meta = Agent(modell, metakognition=True, werkzeug=False, gedaechtnis=False,
                 max_schritte=lang).loesen(raetsel, loesung)
    print("Gelöst nach festen Denkschritten, nach Denktiefe des Rätsels:")
    print(f"  {'Tiefe':>9} {'Anzahl':>7} {'1 Schritt':>10} {f'{mittel} Schritte':>11} {f'{lang} Schritte':>12}"
          f" {'Ø Schritte (Metakognition)':>28}")
    tiefe_tabelle = []
    for name, passt in _tiefe_gruppen(n):
        auswahl = passt(tiefe)
        anzahl = int(auswahl.sum())
        if not anzahl:
            continue
        acc = genauigkeit_pro_schritt(modell, raetsel[auswahl], loesung[auswahl], lang)
        schnitt = _mittel(meta.schritte[auswahl])
        print(f"  {name:>9} {anzahl:>7} {acc[0]:>10.1%} {acc[mittel - 1]:>11.1%} {acc[-1]:>12.1%} {schnitt:>28.1f}")
        tiefe_tabelle.append([name, anzahl, f"{acc[0]:.1%}", f"{acc[mittel - 1]:.1%}", f"{acc[-1]:.1%}", f"{schnitt:.1f}"])

    # 2) Was bringt jedes Modul? Gleiches Modell, Module schrittweise zugeschaltet.
    konfigurationen = [
        ("Nur Denkkern, 1 Schritt", dict(metakognition=False, werkzeug=False, gedaechtnis=False, max_schritte=1),
         "Ein Durchlauf, wie ein Netz ohne Nachdenken"),
        (f"Denkkern, immer {lang} Schritte", dict(metakognition=False, werkzeug=False, gedaechtnis=False),
         "Nachdenken hilft, kostet aber bei jedem Rätsel gleich viel"),
        ("+ Metakognition", dict(metakognition=True, werkzeug=False, gedaechtnis=False),
         "Hört bei leichten Rätseln früh auf, gibt aber auch mal falsch ab"),
        ("+ Werkzeug (Regelprüfer)", dict(metakognition=True, werkzeug=True, gedaechtnis=False),
         "Falsche Entwürfe werden erkannt, das Modell denkt weiter"),
        ("Werkzeug ohne Metakognition", dict(metakognition=False, werkzeug=True, gedaechtnis=False),
         "Prüft nach jedem Schritt statt nur bei Zuversicht"),
    ]
    print("\nWas jedes Modul beiträgt:")
    print(f"  {'Konfiguration':<30} {'gelöst':>7} {'Ø Schritte':>11} {'Ø Prüfungen':>12}")
    ablation = []
    for name, kw, text in konfigurationen:
        kw.setdefault("max_schritte", lang)
        e = Agent(modell, **kw).loesen(raetsel, loesung)
        zeile = [name, f"{e.richtig.mean():.1%}", f"{e.schritte.mean():.2f}", f"{e.werkzeug_aufrufe.mean():.2f}", text]
        ablation.append(zeile)
        print(f"  {zeile[0]:<30} {zeile[1]:>7} {zeile[2]:>11} {zeile[3]:>12}")

    # 3) Gedächtnis: dieselben Rätsel ein zweites Mal, dann verwandte Varianten.
    agent = Agent(modell, metakognition=True, werkzeug=True, gedaechtnis=True, max_schritte=lang)
    erst = agent.loesen(raetsel, loesung)
    zweit = agent.loesen(raetsel, loesung)
    rng = np.random.default_rng(7)
    verwandte = raetsel.copy()
    for i in range(len(verwandte)):
        leer = np.flatnonzero(verwandte[i] == 0)
        extra = rng.choice(leer, size=min(2, len(leer)), replace=False)
        verwandte[i, extra] = loesung[i, extra]
    verwandt = agent.loesen(verwandte, loesung)
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


def befehl_konsolidieren(args):
    n = args.groesse
    modell = _modell(n, args.geraet)
    info = laden(modellpfad(n), nur_info=True)
    leer_basis = info.get("leer_max", LEER_TRAINING[n])
    # Im Schlaf so viele Denkschritte üben wie im Training, sonst verlernt das Modell die hinteren.
    schritte = args.schritte or info.get("schritte", 8)
    print(f"Modell wurde auf Rätseln mit höchstens {leer_basis} leeren Feldern trainiert. "
          f"Im Schlaf: {schritte} Denkschritte pro Rätsel.")
    print(f"Erzeuge Rätsel: {args.anzahl_wach} zum Erleben (schwerer), je {args.anzahl_test} zum Testen …")
    wach_daten, schwer, leicht = experiment_daten(n, leer_basis, args.anzahl_wach, args.anzahl_test)
    (tr_r, tr_l, _), _ = daten_laden(n, 40_000, 0)
    auswahl = (tr_r == 0).sum(1) <= leer_basis
    alte = (tr_r[auswahl], tr_l[auswahl])
    print(f"Testmengen: schwer = mehr als {leer_basis} leere Felder, leicht = höchstens {leer_basis}.")
    if args.lehrer:
        print("Lehrer-Modus: nachts alle erlebten Rätsel mit den echten Lösungen üben.")
    print()
    konsolidiert, zeilen, wachphasen = experiment(modell, alte, wach_daten, schwer, leicht, zyklen=args.zyklen,
                                                  laeufe=args.laeufe, max_schritte=MAX_SCHRITTE[n], schritte=schritte,
                                                  lehrer=args.lehrer)
    pfad = modellpfad(n, "_lehrer" if args.lehrer else "_konsolidiert")
    speichern(konsolidiert, pfad, info={"leer_max": leer_basis, "schritte": schritte})
    print(f"\nKonsolidiertes Modell gespeichert in {pfad}")
    if args.json:
        Path(args.json).write_text(json.dumps({
            "spalten": [titel for _, titel, _ in SPALTEN],
            "zeilen": [[name] + [float(w[k]) for k, _, _ in SPALTEN] for name, w in zeilen],
            "wachphasen": wachphasen}, ensure_ascii=False, indent=1))
        print(f"Ergebnisse als JSON in {args.json}")


def befehl_umlernen(args):
    """Neue Regel (Diagonal-Sudoku): vier Bedingungen vom selben Ausgangsmodell."""
    from . import umlernen
    n = args.groesse
    if n != 9:
        sys.exit("umlernen gibt es nur für 9×9")
    modell = _modell(n, args.geraet)
    info = laden(modellpfad(n), nur_info=True)
    leer_basis = info.get("leer_max", LEER_TRAINING[n])
    schritte = args.schritte or info.get("schritte", 8)
    leer_min = LEER_STANDARD[n][0]
    print(f"Diagonal-Rätsel mit {leer_min}–{leer_basis} leeren Feldern (wie im Training). "
          f"Im Schlaf: {schritte} Denkschritte pro Rätsel.")
    art = "Diagonal-Rätsel, die klassisch mehrdeutig sind," if args.mehrdeutig else "Diagonal-Rätsel"
    print(f"Erzeuge {args.anzahl_wach} {art} zum Erleben und je {args.anzahl_test} zum Testen …")
    wach_daten, diag, klass = umlernen.umlernen_daten(n, args.anzahl_wach, args.anzahl_test, leer_min, leer_basis,
                                                      seed=args.seed, mehrdeutig=args.mehrdeutig)
    (tr_r, tr_l, _), _ = daten_laden(n, 40_000, 0)
    auswahl = (tr_r == 0).sum(1) <= leer_basis
    alte = (tr_r[auswahl], tr_l[auswahl])
    ergebnis = umlernen.experiment(modell, alte, wach_daten, diag, klass, zyklen=args.zyklen, laeufe=args.laeufe,
                                   schritte=schritte, max_schritte=MAX_SCHRITTE[n], seed=args.seed)
    if args.json:
        def zahlen(w):
            return {k: float(v) for k, v in w.items()}
        Path(args.json).write_text(json.dumps({
            "vorher": zahlen(ergebnis["vorher"]),
            "bedingungen": {name: {"zeilen": [[z, zahlen(w)] for z, w in b["zeilen"]], "wachphasen": b["wachphasen"]}
                            for name, b in ergebnis["bedingungen"].items()}}, ensure_ascii=False, indent=1))
        print(f"Ergebnisse als JSON in {args.json}")


def befehl_loesen(args):
    """Zeigt Schritt für Schritt, wie das Modell ein Rätsel durchdenkt."""
    n = args.groesse
    modell = laden(Path(args.modell), geraet=args.geraet) if args.modell else _modell(n, args.geraet)
    rng = np.random.default_rng(args.seed)
    leer = args.leer if args.leer is not None else LEER_STANDARD[n][1] - 4
    raetsel, loesung, tiefe = datensatz(1, rng, n=n, leer_min=leer, leer_max=leer)
    print(f"Rätsel ({int((raetsel[0] == 0).sum())} leere Felder, Denktiefe {tiefe[0]}):\n")
    print(als_text(raetsel[0]))
    r = torch.as_tensor(raetsel).to(modell.geraet)
    schritte = args.schritte or MAX_SCHRITTE[n]
    with torch.no_grad():
        for s, (y, _z, _roh, logit) in enumerate(modell.denken(r, schritte), start=1):
            entwurf = modell.ablesen(r, y).cpu().numpy()
            falsch = int((entwurf[0] != loesung[0]).sum())
            ok = gueltig_batch(entwurf, raetsel)[0]
            print(f"\nDenkschritt {s:2d}: {falsch:2d} Felder falsch · Zuversicht {torch.sigmoid(logit)[0]:.0%}"
                  f" · Regelprüfer: {'gültig ✔' if ok else 'ungültig'}")
            if ok or s == schritte:
                print(als_text(entwurf[0]))
                break


def befehl_beispiele(args):
    n = args.groesse
    rng = np.random.default_rng(args.seed)
    lo, hi = LEER_STANDARD[n]
    for leer in (lo, (lo + hi) // 2, hi):
        r, _, t = datensatz(1, rng, n=n, leer_min=leer, leer_max=leer)
        print(f"{int((r[0] == 0).sum())} leere Felder, Denktiefe {t[0]}:\n{als_text(r[0])}\n")


def main():
    p = argparse.ArgumentParser(prog="python -m mini_kortex", description=__doc__)
    p.add_argument("--groesse", type=int, choices=[6, 9], default=9, help="6×6 oder 9×9 (Standard: 9)")
    p.add_argument("--geraet", choices=["auto", "cpu", "cuda"], default="auto",
                   help="worauf gerechnet wird (Standard: auto = GPU, falls vorhanden)")
    sub = p.add_subparsers(dest="befehl", required=True)

    t = sub.add_parser("trainieren", help="Modell trainieren und als mini_kortex_<groesse>.pt speichern "
                       "(setzt nach einem Abbruch beim letzten Zwischenstand fort)")
    t.add_argument("--laeufe", type=int, default=3000, help="Anzahl Trainings-Batches")
    t.add_argument("--batch", type=int, default=128, help="Rätsel pro Trainings-Batch")
    t.add_argument("--schritte", type=int, default=8, help="Denkschritte pro Rätsel beim Training")
    t.add_argument("--d", type=int, default=64, help="Breite der Vektoren im Arbeitsraum")
    t.add_argument("--lernrate", type=float, default=1e-3)
    t.add_argument("--leer-max", type=int, default=None,
                   help="nur auf Rätseln mit höchstens so vielen Leerfeldern trainieren (Standard: 28 bei 6×6, 46 bei 9×9)")
    t.add_argument("--seed", type=int, default=0,
                   help="Startwerte und Reihenfolge der Batches; die Rätsel bleiben dieselben")
    t.add_argument("--threads", type=int, default=0)
    t.set_defaults(f=befehl_trainieren)

    a = sub.add_parser("auswerten", help="Module einzeln zu- und abschalten und vergleichen")
    a.add_argument("--anzahl", type=int, default=2000)
    a.add_argument("--json", help="Ergebnisse zusätzlich als JSON speichern")
    a.set_defaults(f=befehl_auswerten)

    k = sub.add_parser("konsolidieren", help="Wach-Schlaf-Zyklen: aus eigener Erfahrung lernen")
    k.add_argument("--zyklen", type=int, default=3, help="Anzahl Tage (Wachphase + Nacht)")
    k.add_argument("--laeufe", type=int, default=300, help="Trainingsläufe pro Nacht")
    k.add_argument("--schritte", type=int, default=None,
                   help="Denkschritte pro Rätsel im Schlaf (Standard: wie im Training, sonst 8)")
    k.add_argument("--anzahl-wach", type=int, default=3000, help="schwere Rätsel zum Erleben, verteilt auf die Tage")
    k.add_argument("--anzahl-test", type=int, default=1000, help="Rätsel pro Testmenge (schwer und leicht)")
    k.add_argument("--lehrer", action="store_true",
                   help="Vergleich: nachts alle erlebten Rätsel mit den echten Lösungen üben")
    k.add_argument("--json", help="Ergebnisse zusätzlich als JSON speichern")
    k.set_defaults(f=befehl_konsolidieren)

    u = sub.add_parser("umlernen", help="Neue Regel (Diagonal-Sudoku): selbst/Lehrer × mit/ohne Replay")
    u.add_argument("--zyklen", type=int, default=3)
    u.add_argument("--laeufe", type=int, default=300, help="Trainingsläufe pro Nacht")
    u.add_argument("--schritte", type=int, default=None, help="Denkschritte im Schlaf (Standard: wie im Training)")
    u.add_argument("--anzahl-wach", type=int, default=3000)
    u.add_argument("--anzahl-test", type=int, default=1000)
    u.add_argument("--seed", type=int, default=0, help="andere Diagonal-Rätsel und andere Reihenfolge im Schlaf")
    u.add_argument("--mehrdeutig", action="store_true",
                   help="nur Diagonal-Rätsel, die nach den klassischen Regeln mehrere Lösungen haben")
    u.add_argument("--json", help="Ergebnisse zusätzlich als JSON speichern")
    u.set_defaults(f=befehl_umlernen)

    l = sub.add_parser("loesen", help="Einem Rätsel beim Nachdenken zuschauen")
    l.add_argument("--leer", type=int, default=None, help="Zahl leerer Felder")
    l.add_argument("--schritte", type=int, default=None)
    l.add_argument("--seed", type=int, default=None)
    l.add_argument("--modell", help="anderes Modell laden, z. B. mini_kortex_9_konsolidiert.pt")
    l.set_defaults(f=befehl_loesen)

    b = sub.add_parser("beispiele", help="Ein paar erzeugte Rätsel zeigen")
    b.add_argument("--seed", type=int, default=None)
    b.set_defaults(f=befehl_beispiele)

    args = p.parse_args()
    args.geraet = geraet_waehlen(args.geraet)
    args.f(args)


if __name__ == "__main__":
    main()
