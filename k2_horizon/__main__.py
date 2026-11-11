"""Kommandozeile: python -m k2_horizon {fragen,tempo,lora-speicher}"""
import argparse
import os
import time

# Vermeidet Speicherzersplitterung bei wechselnden Sequenzlängen. Muss vor dem ersten CUDA-Zugriff stehen.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def main():
    p = argparse.ArgumentParser(prog="python -m k2_horizon")
    unter = p.add_subparsers(dest="befehl", required=True)

    f = unter.add_parser("fragen", help="eine Frage stellen")
    f.add_argument("text")
    f.add_argument("--aufwand", default="high", choices=["low", "medium", "high"])
    f.add_argument("--max-neu", type=int, default=4000)
    f.add_argument("--gedanken", action="store_true", help="auch den Denkteil zeigen")

    t = unter.add_parser("tempo", help="Erzeugungsgeschwindigkeit messen")
    t.add_argument("--tokens", type=int, default=256)

    l = unter.add_parser("lora-speicher", help="VRAM und Durchsatz beim LoRA-Training messen")
    l.add_argument("--rang", type=int, default=16)

    a = p.parse_args()

    import torch
    from .modell import fragen, laden

    tok, model = laden()
    print(f"Geladen: {sum(p.numel() for p in model.parameters()) / 1e9:.2f} Mrd. Parameter, "
          f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB VRAM")

    if a.befehl == "fragen":
        antwort, gedanken, n, s = fragen(tok, model, a.text, a.aufwand, a.max_neu)
        if a.gedanken and gedanken:
            print("--- Gedanken ---\n" + gedanken + "\n--- Antwort ---")
        print(antwort)
        print(f"\n({n} Tokens in {s:.1f} s = {n / s:.1f} Tokens/s)")

    elif a.befehl == "tempo":
        fragen(tok, model, "Hallo", "low", 20)  # Aufwärmen, der erste Lauf ist viel langsamer
        eingabe = tok("Die Geschichte der Eisenbahn begann", return_tensors="pt").to(model.device)
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            model.generate(**eingabe, max_new_tokens=a.tokens, min_new_tokens=a.tokens, do_sample=False)
        torch.cuda.synchronize()
        dauer = time.perf_counter() - start
        print(f"{a.tokens} Tokens in {dauer:.1f} s = {a.tokens / dauer:.1f} Tokens/s")

    elif a.befehl == "lora-speicher":
        from .lora import lora_vorbereiten, speicher_messen

        model = lora_vorbereiten(model, a.rang)
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"LoRA-Rang {a.rang}: {n / 1e6:.1f} Mio. trainierbare Parameter")
        for laenge, spitze, sek in speicher_messen(model):
            if spitze is None:
                print(f"Länge {laenge:6d}: Speicher reicht nicht")
            else:
                print(f"Länge {laenge:6d}: Spitze {spitze:5.1f} GiB | {sek:.2f} s/Schritt | "
                      f"{laenge / sek:,.0f} Tokens/s")


if __name__ == "__main__":
    main()
