"""Präferenztraining (DPO) auf Vergleichspaaren: guter gegen schlechten Lauf derselben Frage.

DPO verschiebt die Wahrscheinlichkeiten so, dass der gute Lauf im Vergleich zum Ausgangsmodell
wahrscheinlicher wird als der schlechte:

    z       = β · [(log p(gut) − log p_ref(gut)) − (log p(schlecht) − log p_ref(schlecht))]
    Verlust = −log σ(z)  +  α · mittlerer Kreuzentropie-Verlust auf dem guten Lauf

Der zweite Teil (RPO) hält das Modell nah an echten Läufen, damit es bei wenigen Paaren nicht
einfach beide Läufe unwahrscheinlicher macht.

Gezählt werden nur die Tokens, die das Modell selbst schreibt (wie beim SFT). Zwei Tricks halten
den Speicher klein:
  * Das Referenzmodell ist das Grundmodell. Seine log p werden einmal vorab berechnet, eine
    zweite Kopie im Speicher ist nicht nötig.
  * Gut und schlecht laufen nacheinander durch das Modell, nicht gemeinsam. Dazu werden erst
    beide log p ohne Gradient berechnet, daraus das Gewicht σ(−z), und dann jede Seite einzeln
    mit ihrem Anteil am Gradienten zurückgerechnet. Das ergibt exakt denselben Gradienten.
"""
import json
import math
import os
import random
import time

import torch
import torch.nn.functional as F

from .lora import lora_vorbereiten, verlust
from .training import IGNORIEREN, aufteilen, lernrate

BETA = 0.1
ALPHA = 1.0


def kodieren(tok, nachrichten, tools):
    e = tok.apply_chat_template(nachrichten, tools=tools, tokenize=True, return_dict=True,
                                return_assistant_tokens_mask=True)
    ids = e["input_ids"]
    return ids, [t if m else IGNORIEREN for t, m in zip(ids, e["assistant_masks"])]


def paare_laden(tok, pfad, max_laenge=8192):
    paare = []
    with open(pfad, encoding="utf-8") as f:
        for z in f:
            if not z.strip():
                continue
            d = json.loads(z)
            gut = kodieren(tok, d["prompt"] + d["chosen"], d["tools"])
            schlecht = kodieren(tok, d["prompt"] + d["rejected"], d["tools"])
            if max(len(gut[0]), len(schlecht[0])) <= max_laenge:
                paare.append({"gut": gut, "schlecht": schlecht, "zeile": d})
    return paare


def _anzahl(labels):
    return sum(l != IGNORIEREN for l in labels[1:])


def logp(model, seite):
    """Summe der log-Wahrscheinlichkeiten über die gelernten Tokens (mit Gradient, falls aktiv)."""
    ids, labels = seite
    t = lambda x: torch.tensor([x], device=model.device)
    mittel = verlust(model, t(ids), t(labels))
    return -mittel * _anzahl(labels), mittel


def z_wert(lp_gut, ref_gut, lp_schlecht, ref_schlecht, beta=BETA):
    return beta * ((lp_gut - ref_gut) - (lp_schlecht - ref_schlecht))


@torch.no_grad()
def referenz_berechnen(model, paare):
    model.eval()
    for p in paare:
        p["ref_gut"] = logp(model, p["gut"])[0].item()
        p["ref_schlecht"] = logp(model, p["schlecht"])[0].item()
    model.train()


@torch.no_grad()
def bewerten(model, paare, beta=BETA):
    """Anteil der Paare, bei denen der gute Lauf relativ zur Referenz vorne liegt, und mittleres z."""
    model.eval()
    zs = [z_wert(logp(model, p["gut"])[0].item(), p["ref_gut"],
                 logp(model, p["schlecht"])[0].item(), p["ref_schlecht"], beta) for p in paare]
    model.train()
    return sum(z > 0 for z in zs) / len(zs), sum(zs) / len(zs)


def dpo_schritt(model, p, anteil, beta=BETA, alpha=ALPHA):
    """Rechnet den Gradienten eines Paares zurück (skaliert mit `anteil`) und gibt (Verlust, z) zurück."""
    with torch.no_grad():
        lp_g = logp(model, p["gut"])[0].item()
        lp_s = logp(model, p["schlecht"])[0].item()
    z = z_wert(lp_g, p["ref_gut"], lp_s, p["ref_schlecht"], beta)
    gewicht = torch.sigmoid(torch.tensor(-z)).item()  # Ableitung von −log σ(z) nach z, mit Minus
    # Gut: −β·w·log p(gut) + α·Kreuzentropie(gut)
    lp, mittel = logp(model, p["gut"])
    ((-beta * gewicht * lp + alpha * mittel) * anteil).backward()
    # Schlecht: +β·w·log p(schlecht)
    lp, _ = logp(model, p["schlecht"])
    (beta * gewicht * lp * anteil).backward()
    return -F.logsigmoid(torch.tensor(z)).item(), z


def trainieren(tok, model, daten, ziel, epochen=2, sammeln=8, spitze=5e-5, rang=16, beta=BETA,
               alpha=ALPHA, max_schritte=None, seed=0):
    torch.manual_seed(seed)
    paare = paare_laden(tok, daten)
    train, halt = aufteilen(paare, seed=seed)
    tokens = sum(len(p["gut"][0]) + len(p["schlecht"][0]) for p in train)
    print(f"{len(paare)} Paare: {len(train)} Training, {len(halt)} zurückgehalten, {tokens:,} Tokens je Epoche",
          flush=True)

    # Erst LoRA anhängen: Die Adapter starten bei null, das Modell rechnet also noch genau wie das
    # Grundmodell. `verlust` erwartet die PEFT-Hülle.
    model = lora_vorbereiten(model, rang)
    start = time.perf_counter()
    referenz_berechnen(model, paare)
    print(f"Referenz berechnet in {time.perf_counter() - start:.0f} s", flush=True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=spitze, weight_decay=0.0)
    gesamt = epochen * math.ceil(len(train) / sammeln)
    if max_schritte:
        gesamt = min(gesamt, max_schritte)
    aufwaermen = max(1, gesamt // 10)
    print(f"{gesamt} Optimierungsschritte, Lernrate {spitze}, β {beta}, α {alpha}", flush=True)

    schritt, gesehen, rng, start = 0, 0, random.Random(seed), time.perf_counter()
    for epoche in range(1, epochen + 1):
        reihe = train[:]
        rng.shuffle(reihe)
        verluste, zs = [], []
        for i in range(0, len(reihe), sammeln):
            if schritt >= gesamt:
                break
            gruppe = reihe[i:i + sammeln]
            for p in gruppe:
                v, z = dpo_schritt(model, p, 1 / len(gruppe), beta, alpha)
                verluste.append(v)
                zs.append(z)
                gesehen += len(p["gut"][0]) + len(p["schlecht"][0])
            for g in opt.param_groups:
                g["lr"] = lernrate(schritt, gesamt, spitze, aufwaermen)
            norm = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
            opt.step()
            opt.zero_grad(set_to_none=True)
            schritt += 1
            if schritt % 5 == 0 or schritt == gesamt:
                dauer = time.perf_counter() - start
                print(f"Epoche {epoche} Schritt {schritt}/{gesamt}: DPO-Verlust {sum(verluste) / len(verluste):.3f}, "
                      f"gut vorne {sum(z > 0 for z in zs) / len(zs):.0%}, z {sum(zs) / len(zs):+.2f}, "
                      f"Gradnorm {norm:.2f}, {gesehen / dauer:,.0f} Tokens/s, "
                      f"noch etwa {(gesamt - schritt) * dauer / schritt / 60:.0f} min", flush=True)
                verluste, zs = [], []
        ordner = os.path.join(ziel, f"epoche-{epoche}")
        model.save_pretrained(ordner)
        anteil, z = bewerten(model, halt, beta)
        print(f"Epoche {epoche} fertig: zurückgehalten gut vorne {anteil:.0%}, z {z:+.2f}, gespeichert in {ordner}",
              flush=True)
        if schritt >= gesamt:
            break
    return model
