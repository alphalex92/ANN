"""Kommandozeile: python -m k2_horizon {fragen,tempo,lora-speicher,wissen,suche,suche-messen,laeufe,messen}"""
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

    w = unter.add_parser("wissen", help="prüfen, welche Faktenfragen das Modell ohne Suche weiß")
    w.add_argument("--fragen", default="/mnt/nvme2/daten/popqa/test.tsv", help="PopQA-TSV")
    w.add_argument("--ziel", default="/mnt/nvme2/daten/k2/wissen.jsonl")
    w.add_argument("--anzahl", type=int, default=3000)
    w.add_argument("--versuche", type=int, default=5)
    w.add_argument("--aufwand", default="low", choices=["low", "medium", "high"])
    w.add_argument("--stapel", type=int, default=16, help="Fragen pro Durchlauf")
    w.add_argument("--max-neu", type=int, default=96)
    w.add_argument("--seed", type=int, default=0)
    w.add_argument("--nur-zusammenfassung", action="store_true")

    s = unter.add_parser("suche", help="die lokale Wikipedia-Suche ausprobieren")
    s.add_argument("anfrage")
    s.add_argument("--index", default="/mnt/nvme2/daten/k2/wiki_index")

    m = unter.add_parser("suche-messen", help="wie oft steckt die Antwort in den Suchtreffern?")
    m.add_argument("--index", default="/mnt/nvme2/daten/k2/wiki_index")
    m.add_argument("--wissen", default="/mnt/nvme2/daten/k2/wissen.jsonl")
    m.add_argument("--anzahl", type=int, default=500)
    m.add_argument("--teil", default="val")

    e = unter.add_parser("laeufe", help="Lösungswege mit Suche erzeugen")
    e.add_argument("--wissen", default="/mnt/nvme2/daten/k2/wissen.jsonl")
    e.add_argument("--index", default="/mnt/nvme2/daten/k2/wiki_index")
    e.add_argument("--ziel", default="/mnt/nvme2/daten/k2/laeufe.jsonl")
    e.add_argument("--teil", default="train")
    e.add_argument("--unbekannt", type=int, default=1600, help="so viele unbekannte Fragen, alle gewussten")
    e.add_argument("--aufwand", default="medium", choices=["low", "medium", "high"])
    e.add_argument("--ohne-hinweis", action="store_true")
    e.add_argument("--stapel", type=int, default=16)
    e.add_argument("--seed", type=int, default=0)

    v = unter.add_parser("messen", help="Wissen oder nachschlagen? Ohne Hinweis auf Validierungsfragen")
    v.add_argument("--wissen", default="/mnt/nvme2/daten/k2/wissen.jsonl")
    v.add_argument("--index", default="/mnt/nvme2/daten/k2/wiki_index")
    v.add_argument("--ziel", default="/mnt/nvme2/daten/k2/messung_basis.jsonl")
    v.add_argument("--teil", default="val")
    v.add_argument("--unbekannt", type=int, default=200, help="so viele unbekannte, alle gewussten und unklaren")
    v.add_argument("--adapter", help="Ordner mit LoRA-Adapter, sonst das Grundmodell")
    v.add_argument("--aufwand", default="medium", choices=["low", "medium", "high"])
    v.add_argument("--stapel", type=int, default=16)
    v.add_argument("--nur-auswertung", action="store_true")

    a = p.parse_args()

    if a.befehl == "messen" and a.nur_auswertung:
        from .datensatz import laden as laeufe_laden
        from .messung import auswerten
        print(auswerten(laeufe_laden(a.ziel)))
        return

    if a.befehl == "suche":
        from .suche import Suche, als_text
        start = time.perf_counter()
        index = Suche(a.index)
        print(f"(Index geladen in {time.perf_counter() - start:.0f} s)\n")
        print(als_text(index.suchen(a.anfrage)))
        return

    if a.befehl == "suche-messen":
        from .wissen import suche_messen
        suche_messen(a.index, a.wissen, a.anzahl, a.teil)
        return

    if a.befehl == "wissen" and a.nur_zusammenfassung:
        from .wissen import zusammenfassen
        n, gesamt, je_typ = zusammenfassen(a.ziel)
        print(f"{n} Fragen: " + ", ".join(f"{k} {v} ({v / n:.0%})" for k, v in gesamt.most_common()))
        for typ, c in sorted(je_typ.items(), key=lambda x: -sum(x[1].values())):
            s = sum(c.values())
            print(f"  {typ:15s} {s:5d}  gewusst {c['gewusst'] / s:4.0%}  unklar {c['unklar'] / s:4.0%}"
                  f"  unbekannt {c['unbekannt'] / s:4.0%}")
        return

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

    elif a.befehl == "wissen":
        from .bewertung import fragen_laden, stichprobe
        from .wissen import pruefen

        os.makedirs(os.path.dirname(a.ziel), exist_ok=True)
        auswahl = stichprobe(fragen_laden(a.fragen), a.anzahl, a.seed)
        pruefen(tok, model, auswahl, a.ziel, a.versuche, a.aufwand, a.stapel, a.max_neu)

    elif a.befehl == "laeufe":
        import json
        import random

        from .agent import Agent, laeufe_erzeugen
        from .suche import Suche

        with open(a.wissen, encoding="utf-8") as f:
            zeilen = [json.loads(z) for z in f if z.strip()]
        zeilen = [z for z in zeilen if z["teil"] == a.teil]
        rng = random.Random(a.seed)
        unbekannt = [z for z in zeilen if z["klasse"] == "unbekannt"]
        auswahl = [z for z in zeilen if z["klasse"] == "gewusst"] + rng.sample(unbekannt, min(a.unbekannt, len(unbekannt)))
        rng.shuffle(auswahl)
        agent = Agent(tok, model, Suche(a.index), aufwand=a.aufwand)
        laeufe_erzeugen(agent, auswahl, a.ziel, hinweis=not a.ohne_hinweis, stapel=a.stapel)

    elif a.befehl == "messen":
        import json

        from .agent import Agent, laeufe_erzeugen
        from .datensatz import laden as laeufe_laden
        from .messung import auswahl, auswerten
        from .suche import Suche

        if a.adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, a.adapter).eval()
            print(f"Adapter geladen: {a.adapter}")
        with open(a.wissen, encoding="utf-8") as f:
            fragen_ = auswahl([json.loads(z) for z in f if z.strip()], a.teil, a.unbekannt)
        agent = Agent(tok, model, Suche(a.index), aufwand=a.aufwand)
        laeufe_erzeugen(agent, fragen_, a.ziel, hinweis=False, stapel=a.stapel)
        print(auswerten(laeufe_laden(a.ziel)))

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
