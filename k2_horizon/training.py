"""LoRA-Training auf den ausgewählten Lösungswegen (überwachtes Feintuning).

Jedes Beispiel ist ein ganzes Gespräch: Frage, Gedanken, Suchaufrufe, Suchtreffer, Antwort.
Gelernt wird nur, was das Modell selbst schreibt. Die Chat-Vorlage markiert diese Teile mit
{% generation %}, Transformers liefert daraus `assistant_masks`. Frage und Suchtreffer bekommen
das Label -100 und zählen nicht zum Verlust.

Pro Schritt geht ein Gespräch durch das Modell (die Längen schwanken zwischen 300 und 5700
Tokens, Auffüllen wäre Verschwendung). Erst nach `sammeln` Gesprächen werden die Gewichte
angepasst. Das entspricht einer Batchgröße von `sammeln`.
"""
import json
import math
import os
import random
import time

import torch

from .lora import lora_vorbereiten, verlust

IGNORIEREN = -100


def beispiele_laden(tok, pfad, max_laenge=8192):
    """Liest das Trainingsset und gibt [(ids, labels, zeile)] zurück, zu lange Gespräche fallen weg."""
    beispiele = []
    with open(pfad, encoding="utf-8") as f:
        for z in f:
            if not z.strip():
                continue
            d = json.loads(z)
            e = tok.apply_chat_template(d["messages"], tools=d["tools"], tokenize=True, return_dict=True,
                                        return_assistant_tokens_mask=True)
            ids = e["input_ids"]
            if len(ids) > max_laenge:
                continue
            labels = [t if m else IGNORIEREN for t, m in zip(ids, e["assistant_masks"])]
            beispiele.append((ids, labels, d))
    return beispiele


def aufteilen(beispiele, anteil=0.05, seed=0):
    """Hält einen festen Teil zurück, um Überanpassung zu erkennen."""
    reihe = list(range(len(beispiele)))
    random.Random(seed).shuffle(reihe)
    n = max(1, round(len(beispiele) * anteil))
    zurueck = set(reihe[:n])
    return ([b for i, b in enumerate(beispiele) if i not in zurueck],
            [b for i, b in enumerate(beispiele) if i in zurueck])


def lernrate(schritt, gesamt, spitze, aufwaermen):
    """Linear hoch über `aufwaermen` Schritte, dann Kosinus bis 0."""
    if schritt < aufwaermen:
        return spitze * (schritt + 1) / aufwaermen
    rest = (schritt - aufwaermen) / max(1, gesamt - aufwaermen)
    return spitze * 0.5 * (1 + math.cos(math.pi * min(1.0, rest)))


def _tensor(x, model):
    return torch.tensor([x], device=model.device)


@torch.no_grad()
def bewerten(model, beispiele):
    """Mittlerer Verlust je gelerntem Token auf zurückgehaltenen Beispielen."""
    model.eval()
    summe = anzahl = 0.0
    for ids, labels, _ in beispiele:
        n = sum(l != IGNORIEREN for l in labels[1:])
        summe += verlust(model, _tensor(ids, model), _tensor(labels, model)).item() * n
        anzahl += n
    model.train()
    return summe / max(1, anzahl)


def trainieren(tok, model, daten, ziel, epochen=2, sammeln=8, spitze=1e-4, rang=16,
               max_schritte=None, seed=0):
    """Trainiert einen LoRA-Adapter und speichert ihn nach jeder Epoche unter ziel/epoche-N."""
    torch.manual_seed(seed)
    beispiele = beispiele_laden(tok, daten)
    train, halt = aufteilen(beispiele, seed=seed)
    tokens = sum(len(b[0]) for b in train)
    print(f"{len(beispiele)} Beispiele: {len(train)} Training, {len(halt)} zurückgehalten, "
          f"{tokens:,} Tokens je Epoche", flush=True)

    model = lora_vorbereiten(model, rang)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=spitze, weight_decay=0.0)
    gesamt = epochen * math.ceil(len(train) / sammeln)
    if max_schritte:
        gesamt = min(gesamt, max_schritte)
    aufwaermen = max(1, gesamt // 20)
    print(f"{gesamt} Optimierungsschritte, Lernrate {spitze} (Aufwärmen {aufwaermen})", flush=True)
    print(f"Vorher: Verlust zurückgehalten {bewerten(model, halt):.4f}", flush=True)

    schritt, gesehen, rng, start = 0, 0, random.Random(seed), time.perf_counter()
    for epoche in range(1, epochen + 1):
        reihe = train[:]
        rng.shuffle(reihe)
        laufend = []
        for i in range(0, len(reihe), sammeln):
            if schritt >= gesamt:
                break
            gruppe = reihe[i:i + sammeln]
            for ids, labels, _ in gruppe:
                v = verlust(model, _tensor(ids, model), _tensor(labels, model))
                (v / len(gruppe)).backward()
                laufend.append(v.item())
                gesehen += len(ids)
            for g in opt.param_groups:
                g["lr"] = lernrate(schritt, gesamt, spitze, aufwaermen)
            norm = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
            opt.step()
            opt.zero_grad(set_to_none=True)
            schritt += 1
            if schritt % 10 == 0 or schritt == gesamt:
                dauer = time.perf_counter() - start
                print(f"Epoche {epoche} Schritt {schritt}/{gesamt}: Verlust {sum(laufend) / len(laufend):.4f}, "
                      f"Gradnorm {norm:.2f}, {gesehen / dauer:,.0f} Tokens/s, "
                      f"noch etwa {(gesamt - schritt) * dauer / schritt / 60:.0f} min", flush=True)
                laufend = []
        ordner = os.path.join(ziel, f"epoche-{epoche}")
        model.save_pretrained(ordner)
        print(f"Epoche {epoche} fertig: Verlust zurückgehalten {bewerten(model, halt):.4f}, "
              f"gespeichert in {ordner}", flush=True)
        if schritt >= gesamt:
            break
    return model
