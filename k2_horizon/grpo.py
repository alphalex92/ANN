"""GRPO: Das Modell übt im Harness und lernt aus dem Vergleich seiner eigenen Läufe.

Je Schritt:
  1. Für `je_schritt` Trainingsfragen erzeugt das Modell je `gruppe` Läufe, mit Suche und
     Notizzettel, genau so wie bei der Messung.
  2. Jeder Lauf bekommt eine Belohnung: 1 für eine richtige Antwort, 0 sonst, -0,2 wenn der
     Denkteil nicht fertig wurde.
  3. Vorteil = (Belohnung − Mittel der Gruppe) / Streuung der Gruppe. Ein Lauf, der besser ist
     als seine Geschwister zur selben Frage, wird wahrscheinlicher gemacht, ein schlechterer
     unwahrscheinlicher. Gruppen, in denen alle Läufe gleich gut sind, tragen nichts bei.
  4. Gelernt wird jeder Zug so, wie das Modell ihn erzeugt hat: mit genau dem Prompt, den es
     sah (Notizzettel), und genau den Token-IDs, die es schrieb (`Agent(mitschnitt=True)`).

Weil nach jedem Schritt neu erzeugt wird, ist jeder Lauf „on-policy“. Das Verhältnis aus neuer
und alter Wahrscheinlichkeit ist beim einzigen Update 1, Clipping wie bei PPO wirkt also nie
und fehlt hier. Einen KL-Strafterm gibt es auch nicht (wie bei DAPO). LoRA mit kleiner
Lernrate hält das Modell ohnehin nahe am Ausgangspunkt.

Verlust je Lauf i: −A_i · (1/T_i) · Σ log p(Token), summiert über alle Tokens aller Züge des
Laufs, T_i = Anzahl dieser Tokens. Der blockweise Kreuzentropie-Verlust liefert −mittleres
log p je Zug, daraus wird die Summe zurückgerechnet.
"""
import json
import os
import random
import statistics
import time
from collections import Counter

import torch

from .agent import Agent
from .bewertung import richtig
from .lora import ZIELE, lora_vorbereiten, verlust

IGNORIEREN = -100
# Qwen3.5 hat neben normaler Attention lineare Attention-Schichten mit eigenen Namen.
ZIELE_QWEN = ZIELE + ["in_proj_qkv", "in_proj_z", "out_proj"]


def belohnung(lauf):
    if lauf["status"] == "abgeschnitten":
        return -0.2
    return 1.0 if lauf["richtig"] else 0.0


def vorteile(belohnungen, gruppe):
    """Normiert je Gruppe von `gruppe` aufeinanderfolgenden Läufen. Ohne Streuung: alles 0."""
    ergebnis = []
    for i in range(0, len(belohnungen), gruppe):
        g = belohnungen[i:i + gruppe]
        mittel = statistics.fmean(g)
        streuung = statistics.pstdev(g)
        ergebnis += [0.0 if streuung < 1e-6 else (b - mittel) / (streuung + 1e-4) for b in g]
    return ergebnis


def zug_beispiele(tok, lauf):
    """Je Zug (ids, labels): Prompt-Tokens zählen nicht, die selbst geschriebenen schon."""
    beispiele = []
    for zug in lauf.get("zuege", []):
        if not zug["ids"]:
            continue
        p = tok(zug["prompt"], add_special_tokens=False)["input_ids"]
        beispiele.append((p + zug["ids"], [IGNORIEREN] * len(p) + zug["ids"]))
    return beispiele


def _schreiben(datei, zeile):
    datei.write(json.dumps(zeile, ensure_ascii=False) + "\n")
    datei.flush()


def trainieren_grpo(tok, model, suche, fragen, ziel, schritte=100, je_schritt=16, gruppe=8,
                    lernrate=2e-5, rang=16, notiz="stand", denkanstoss=None, speichern_alle=20,
                    token_budget=96_000, max_neu=1024, max_laenge=6144, seed=0):
    os.makedirs(ziel, exist_ok=True)
    rng = random.Random(seed)
    torch.manual_seed(seed)
    model = lora_vorbereiten(model, rang, ziele=ZIELE_QWEN)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"LoRA: {sum(p.numel() for p in params) / 1e6:.1f} M trainierbare Parameter", flush=True)
    opt = torch.optim.AdamW(params, lr=lernrate, weight_decay=0.0)
    agent = Agent(tok, model, suche, notiz=notiz, denkanstoss=denkanstoss, token_budget=token_budget,
                  max_neu=max_neu, mitschnitt=True)
    aufwaermen = 5

    with open(os.path.join(ziel, "verlauf.jsonl"), "a", encoding="utf-8") as protokoll:
        for schritt in range(1, schritte + 1):
            t0 = time.perf_counter()
            auswahl = rng.sample(fragen, je_schritt)

            model.eval()
            model.config.use_cache = True
            laeufe = agent.beantworten([f["frage"] for f in auswahl for _ in range(gruppe)])
            model.config.use_cache = False
            model.train()
            t_erzeugen = time.perf_counter() - t0

            for k, lauf in enumerate(laeufe):
                f = auswahl[k // gruppe]
                lauf["richtig"] = lauf["antwort"] is not None and richtig(lauf["antwort"], f["loesungen"])
            b = [belohnung(l) for l in laeufe]
            a = vorteile(b, gruppe)
            mit_signal = [i for i, x in enumerate(a) if x != 0.0]

            tokens = 0
            if mit_signal:
                for i in mit_signal:
                    beispiele = [(ids, lab) for ids, lab in zug_beispiele(tok, laeufe[i]) if len(ids) <= max_laenge]
                    gesamt = sum(sum(l != IGNORIEREN for l in lab) for _, lab in beispiele)
                    for ids, lab in beispiele:
                        n = sum(l != IGNORIEREN for l in lab)
                        ce = verlust(model, torch.tensor([ids], device=model.device),
                                     torch.tensor([lab], device=model.device))
                        # ce = −mittleres log p; Gradientenabstieg auf A·ce erhöht log p bei A > 0
                        (a[i] * ce * n / gesamt / len(mit_signal)).backward()
                        tokens += n
                for g in opt.param_groups:
                    g["lr"] = lernrate * min(1.0, schritt / aufwaermen)
                norm = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
                opt.step()
                opt.zero_grad(set_to_none=True)
            else:
                norm = 0.0

            gruppen_mit_signal = len({i // gruppe for i in mit_signal})
            zeile = {
                "schritt": schritt,
                "richtig": sum(l["richtig"] for l in laeufe) / len(laeufe),
                "belohnung": statistics.fmean(b),
                "gruppen_mit_signal": gruppen_mit_signal,
                "suchen": statistics.fmean(l["suchen"] for l in laeufe),
                "status": dict(Counter(l["status"] for l in laeufe)),
                "tokens": tokens, "gradnorm": norm,
                "sek_erzeugen": round(t_erzeugen), "sek_gesamt": round(time.perf_counter() - t0),
            }
            _schreiben(protokoll, zeile)
            print(f"Schritt {schritt}/{schritte}: richtig {zeile['richtig']:.0%}, Signal in "
                  f"{gruppen_mit_signal}/{je_schritt} Gruppen, Suchen {zeile['suchen']:.2f}, "
                  f"{zeile['status']}, {zeile['sek_erzeugen']} s erzeugen, {zeile['sek_gesamt']} s gesamt",
                  flush=True)
            if schritt % speichern_alle == 0 or schritt == schritte:
                ordner = os.path.join(ziel, f"schritt-{schritt}")
                model.save_pretrained(ordner)
                print(f"gespeichert: {ordner}", flush=True)
    return model
