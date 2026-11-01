"""Kommandozeile: python -m sprachmodell vorbereiten | tokens | trainieren | erzeugen | messen"""

import argparse
from pathlib import Path

MODELL = Path("sprachmodell.pt")


def befehl_vorbereiten(args):
    from .daten import vorbereiten

    vorbereiten()


def befehl_tokens(args):
    """Zeigt, wie der Tokenizer einen Text zerlegt."""
    from .daten import tokenizer_laden

    tok = tokenizer_laden()
    ids = tok.kodieren(args.text)
    print(" | ".join(tok.vokabular[i].decode("utf-8", errors="replace") for i in ids))
    print(f"{len(ids)} Tokens für {len(args.text.encode('utf-8'))} Bytes, IDs: {ids}")


def befehl_trainieren(args):
    from .training import trainieren

    trainieren(sekunden=args.sekunden, pfad=MODELL)


def befehl_erzeugen(args):
    import torch

    from .daten import tokenizer_laden
    from .modell import GPT

    tok = tokenizer_laden()
    modell = GPT.laden(MODELL).to("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    for _ in range(args.anzahl):
        # Jeder Artikel im Training beginnt mit dem Sondertoken, also beginnen wir auch so.
        anfang = [tok.dok_id] + tok.kodieren(args.text)
        ids = modell.erzeugen(anfang, args.laenge, temperatur=args.temperatur, top_k=args.top_k)
        print(tok.dekodieren(ids[1:]).split("\n\n\n")[0].strip())
        print("─" * 60)


def befehl_messen(args):
    import torch

    from .daten import val_bpb
    from .modell import GPT

    geraet = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"val_bpb {val_bpb(GPT.laden(MODELL).to(geraet), geraet=geraet):.4f}")


def main():
    parser = argparse.ArgumentParser(prog="python -m sprachmodell")
    befehle = parser.add_subparsers(required=True)

    p = befehle.add_parser("vorbereiten", help="Wikipedia laden, Tokenizer lernen, Daten kodieren")
    p.set_defaults(los=befehl_vorbereiten)

    p = befehle.add_parser("tokens", help="zeigen, wie ein Text in Tokens zerfällt")
    p.add_argument("text")
    p.set_defaults(los=befehl_tokens)

    p = befehle.add_parser("trainieren", help="ein Modell mit festem Zeitbudget trainieren")
    p.add_argument("--sekunden", type=float, default=300)
    p.set_defaults(los=befehl_trainieren)

    p = befehle.add_parser("erzeugen", help="Text fortsetzen lassen")
    p.add_argument("text", nargs="?", default="")
    p.add_argument("--laenge", type=int, default=200, help="Anzahl neuer Tokens")
    p.add_argument("--temperatur", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--anzahl", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(los=befehl_erzeugen)

    p = befehle.add_parser("messen", help="val_bpb des gespeicherten Modells")
    p.set_defaults(los=befehl_messen)

    args = parser.parse_args()
    args.los(args)


if __name__ == "__main__":
    main()
