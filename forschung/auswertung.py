"""Gesperrte Auswertung für die Forschungsschleife. Der Agent darf diese Datei nicht ändern.

Misst, wie gut der aktuelle Code aus bytes_erkenner/ in einem festen Zeitbudget lernt:

    python -m forschung.auswertung              # Ergebnis für ergebnisse.tsv
    python -m forschung.auswertung --kontrolle  # zurückgehaltene Kontrollmenge (nur für dich)

Ablauf: Für jeden Seed startet ein eigener Prozess. Er ruft
`bytes_erkenner.training.fuer_auswertung(sekunden, seed)` auf und bewertet die
zurückgegebene Ratefunktion auf einem festen Testset. Gemeldet wird der Mittelwert
über alle Seeds, denn ein einzelner Lauf schwankt um mehrere Prozentpunkte.

Schutz gegen Schummeln (wie im Weco-Artikel, in drei Schichten):
1. program.md verbietet es.
2. Hart: Weicht eine gesperrte Datei vom Tag `forschung-start` ab oder überschreitet
   das Training sein Zeitbudget, gibt es kein Ergebnis.
3. Statistisch: Die Kontrollmenge nutzt einen anderen Seed als das Testset. Steigt nur das
   Test-Ergebnis über viele Versuche, aber nicht die Kontrolle, hat sich der Agent ans
   Testset angepasst.
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

SEKUNDEN = 10.0  # Trainingsbudget pro Seed
SEEDS = (1, 2, 3)
TEST_SEED = 424242
TEST_PRO_KLASSE = 2000
KULANZ = 1.1  # 10 % Spielraum auf das Zeitbudget

GESPERRT = ["bytes_erkenner/daten.py", "forschung/", "tests/"]
START_TAG = "forschung-start"


def gesperrte_aenderungen():
    """Gesperrte Dateien, die vom Start-Tag abweichen (auch ungespeicherte Änderungen)."""
    if subprocess.run(["git", "rev-parse", "-q", "--verify", START_TAG], capture_output=True).returncode != 0:
        sys.exit(f"Tag {START_TAG} fehlt. Zu Beginn der Schleife: git tag {START_TAG}")
    geaendert = subprocess.run(
        ["git", "diff", "--name-only", START_TAG, "--", *GESPERRT], capture_output=True, text=True, check=True
    ).stdout.split()
    neu = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "--", *GESPERRT], capture_output=True, text=True, check=True
    ).stdout.split()
    return geaendert + neu


def einzellauf(seed, test_seed):
    """Läuft im Kindprozess: trainieren, Zeit messen, auf dem festen Testset bewerten."""
    from bytes_erkenner.daten import erzeugen
    from bytes_erkenner.training import fuer_auswertung

    start = time.perf_counter()
    raten = fuer_auswertung(SEKUNDEN, seed)
    dauer = time.perf_counter() - start

    X, y = erzeugen(TEST_PRO_KLASSE, np.random.default_rng(test_seed))
    vermutung = np.asarray(raten(X.copy()))
    genauigkeit = float((vermutung == y).mean()) if vermutung.shape == y.shape else 0.0
    print(json.dumps({"seed": seed, "genauigkeit": genauigkeit, "sekunden": dauer}))


def auswerten(test_seed):
    ergebnisse = []
    for seed in SEEDS:
        try:
            lauf = subprocess.run(
                [sys.executable, "-m", "forschung.auswertung", "--einzel", str(seed), "--test-seed", str(test_seed)],
                capture_output=True,
                text=True,
                timeout=SEKUNDEN * KULANZ + 30,  # Import und Testset dürfen etwas dauern
            )
        except subprocess.TimeoutExpired:
            return {"status": "crash", "grund": f"Seed {seed}: Zeitlimit des Prozesses überschritten"}
        if lauf.returncode != 0:
            letzte_zeile = (lauf.stderr.strip().splitlines() or ["Absturz ohne Meldung"])[-1]
            return {"status": "crash", "grund": f"Seed {seed}: {letzte_zeile}"}
        ergebnis = json.loads(lauf.stdout.strip().splitlines()[-1])
        if ergebnis["sekunden"] > SEKUNDEN * KULANZ:
            return {"status": "ungueltig", "grund": f"Seed {seed}: {ergebnis['sekunden']:.1f} s statt höchstens {SEKUNDEN:.0f} s"}
        ergebnisse.append(ergebnis)

    werte = np.array([e["genauigkeit"] for e in ergebnisse])
    return {
        "status": "ok",
        "genauigkeit": round(float(werte.mean()), 4),
        "streuung": round(float(werte.std()), 4),
        "sekunden": round(max(e["sekunden"] for e in ergebnisse), 1),
        "einzeln": [round(float(w), 4) for w in werte],
    }


def main():
    parser = argparse.ArgumentParser(prog="python -m forschung.auswertung")
    parser.add_argument("--kontrolle", action="store_true", help="auf der zurückgehaltenen Kontrollmenge messen")
    parser.add_argument("--einzel", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--test-seed", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.einzel is not None:
        einzellauf(args.einzel, args.test_seed)
        return

    if verletzt := gesperrte_aenderungen():
        print(json.dumps({"status": "ungueltig", "grund": "gesperrte Dateien geändert: " + ", ".join(verletzt)}, ensure_ascii=False))
        sys.exit(1)

    # Die Kontrollmenge nimmt einen eigenen Seed aus der Umgebung, den der Agent nicht kennt.
    test_seed = int(os.environ.get("KONTROLL_SEED", "7")) if args.kontrolle else TEST_SEED
    ergebnis = auswerten(test_seed)
    print(json.dumps(ergebnis, ensure_ascii=False))
    sys.exit(0 if ergebnis["status"] == "ok" else 1)


if __name__ == "__main__":
    main()
