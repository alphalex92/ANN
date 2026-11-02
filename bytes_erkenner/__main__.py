"""Kommandozeile: python -m bytes_erkenner trainieren | raten | beispiele"""

import argparse
import sys
from pathlib import Path

import numpy as np

from .daten import KLASSEN, als_bits, erzeugen
from .netz import Netz
from .training import matrix_als_text, trainieren, verwechslungsmatrix

MODELL = Path("modell.npz")


def hex_zu_bytes(text):
    roh = "".join(text).replace(" ", "").removeprefix("0x")
    if len(roh) != 16:
        sys.exit(f"Bitte genau 8 Bytes (16 Hex-Ziffern) angeben, bekommen: {len(roh)} Ziffern")
    return np.frombuffer(bytes.fromhex(roh), dtype=np.uint8)


def deutungen(b):
    """Was die 8 Bytes als jeder Typ bedeuten würden – zum Vergleich mit der Vermutung."""
    text = "".join(chr(c) if 32 <= c < 127 else "·" for c in b)
    return {
        "i64": int(b.view("<i8")[0]),
        "f64": float(b.view("<f8")[0]),
        "2×i32": tuple(int(v) for v in b.view("<i4")),
        "2×f32": tuple(float(v) for v in b.view("<f4")),
        "zeiger": hex(int(b.view("<u8")[0])),
        "text": repr(text),
    }


def befehl_trainieren(args):
    netz, X_test, y_test = trainieren(epochen=args.epochen, versteckt=args.versteckt, lernrate=args.lernrate)
    print("\nVerwechslungsmatrix auf den Testdaten (Zeile = richtig, Spalte = Vermutung):")
    print(matrix_als_text(verwechslungsmatrix(netz, X_test, y_test)))
    netz.speichern(MODELL)
    print(f"\nGespeichert in {MODELL}")


def befehl_raten(args):
    if not MODELL.exists():
        sys.exit(f"{MODELL} fehlt – erst `python -m bytes_erkenner trainieren` ausführen")
    netz = Netz.laden(MODELL)
    b = hex_zu_bytes(args.hex)
    p = netz.vorwaerts(als_bits(b[None, :]))[0]
    print("Bytes:", " ".join(f"{c:02x}" for c in b))
    for i in np.argsort(p)[::-1][:3]:
        print(f"  {KLASSEN[i]:<7} {p[i]:6.1%}")
    print("\nSo würden die Bytes gelesen:")
    for name, wert in deutungen(b).items():
        print(f"  als {name:<7} {wert}")


def befehl_beispiele(args):
    X, y = erzeugen(args.anzahl, np.random.default_rng())
    for b, k in sorted(zip(X, y), key=lambda paar: paar[1]):
        print(f"{KLASSEN[k]:<7} {' '.join(f'{c:02x}' for c in b)}")


def main():
    parser = argparse.ArgumentParser(prog="python -m bytes_erkenner", description="Erkennt den Datentyp von 8 Speicher-Bytes")
    unter = parser.add_subparsers(required=True)

    p = unter.add_parser("trainieren", help="Netz trainieren und als modell.npz speichern")
    p.add_argument("--epochen", type=int, default=20)
    p.add_argument("--versteckt", type=int, default=64, help="Neuronen in der versteckten Schicht")
    p.add_argument("--lernrate", type=float, default=0.2)
    p.set_defaults(befehl=befehl_trainieren)

    p = unter.add_parser("raten", help="Typ von 8 Bytes schätzen, z. B. 00 00 00 00 00 40 8f 40")
    p.add_argument("hex", nargs="+", help="8 Bytes in Speicher-Reihenfolge, als Hex")
    p.set_defaults(befehl=befehl_raten)

    p = unter.add_parser("beispiele", help="Ein paar erzeugte Trainingsbeispiele zeigen")
    p.add_argument("--anzahl", type=int, default=2, help="Beispiele pro Klasse")
    p.set_defaults(befehl=befehl_beispiele)

    args = parser.parse_args()
    args.befehl(args)


if __name__ == "__main__":
    main()
