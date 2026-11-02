"""Trainingsschleife mit festem Zeitbudget.

Statt "N Epochen" trainieren wir eine feste Zeit lang (wie bei autoresearch). So sind alle
Varianten fair vergleichbar: Ein größeres Modell lernt pro Schritt mehr, schafft aber
weniger Schritte. Was am Ende in derselben Zeit besser ist, gewinnt.

Ein Schritt:
  1. Zufällige Textausschnitte holen (x) und dieselben um ein Token verschoben (y).
  2. Vorwärts: Verlust = wie schlecht das Modell an jeder Stelle das nächste Token vorhersagt.
  3. Rückwärts: PyTorch rechnet die Gradienten aus (Backpropagation, wie im Bytes-Erkenner,
     nur automatisch).
  4. AdamW verschiebt jedes Gewicht ein Stück gegen seinen Gradienten.
"""

import time

import numpy as np
import torch

from .daten import KONTEXT, tokenizer_laden, tokens_laden, val_bpb, zufalls_batch
from .modell import GPT, Konfig

KONFIG = Konfig(vokabular=4096, kontext=KONTEXT, schichten=8, koepfe=8, breite=512)
BATCH = 64  # Textausschnitte pro Schritt, also 64 · 512 = 32 768 Tokens
LERNRATE = 1e-3
AUFWAERMEN = 0.03  # Anteil des Zeitbudgets, in dem die Lernrate von 0 hochfährt
GEWICHTSZERFALL = 0.1


def lernrate(fortschritt):
    """Erst sanft hochfahren (am Anfang sind die Gradienten chaotisch), dann linear auf 0.

    `fortschritt` ist der verbrauchte Anteil des Zeitbudgets, 0 … 1. Am Ende kleine
    Schritte zu machen, lässt das Modell in einer guten Stelle "zur Ruhe kommen".
    """
    if fortschritt < AUFWAERMEN:
        return LERNRATE * fortschritt / AUFWAERMEN
    return LERNRATE * max(0.0, 1 - fortschritt) / (1 - AUFWAERMEN)


def optimierer(modell):
    # Gewichtszerfall (zieht Gewichte sanft Richtung 0, gegen Überanpassung) nur für die
    # großen Matrizen, nicht für LayerNorm-Parameter.
    matrizen = [p for p in modell.parameters() if p.dim() >= 2]
    rest = [p for p in modell.parameters() if p.dim() < 2]
    gruppen = [{"params": matrizen, "weight_decay": GEWICHTSZERFALL}, {"params": rest, "weight_decay": 0.0}]
    return torch.optim.AdamW(gruppen, lr=LERNRATE, betas=(0.9, 0.95), fused=torch.cuda.is_available())


def trainieren(sekunden=300, seed=0, pfad=None, messen=True, ausgabe=print):
    """Trainiert `sekunden` lang und gibt das Modell und seinen val_bpb zurück.

    Der erste Schritt zählt nicht zum Budget: Darin übersetzt torch.compile das Modell in
    schnelle GPU-Kerne, und das dauert unabhängig vom Modell um die 30 Sekunden.
    """
    assert torch.cuda.is_available(), "Das Training braucht eine CUDA-GPU."
    geraet = "cuda"
    torch.manual_seed(seed)
    generator = np.random.default_rng(seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    tokens = tokens_laden("train")
    modell = GPT(KONFIG).to(geraet)
    ausgabe(f"{modell.anzahl_parameter() / 1e6:.1f} Mio. Parameter, {len(tokens) / 1e6:.0f} Mio. Trainings-Tokens")
    schnell = torch.compile(modell)
    opt = optimierer(modell)

    schritt, gesehen, start, verlauf = 0, 0, None, []
    while True:
        fortschritt = 0.0 if start is None else (time.perf_counter() - start) / sekunden
        if fortschritt >= 1:
            break
        for gruppe in opt.param_groups:
            gruppe["lr"] = lernrate(fortschritt)

        x, y = zufalls_batch(tokens, BATCH, KONTEXT, generator, geraet)
        # bfloat16: halbe Genauigkeit für die großen Matrixmultiplikationen. Doppelt so schnell,
        # und für das Training eines Netzes genau genug.
        with torch.autocast(geraet, dtype=torch.bfloat16):
            verlust = schnell(x, y)
        verlust.backward()
        # Gradienten-Clipping: Wird ein Gradient einmal riesig, wird er auf Länge 1 gestutzt,
        # damit ein einzelner Ausreißer-Batch das Modell nicht zerschießt.
        torch.nn.utils.clip_grad_norm_(modell.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

        schritt += 1
        gesehen += x.numel()
        if start is None:
            torch.cuda.synchronize()
            start = time.perf_counter()  # Ab jetzt läuft die Uhr
        if schritt % 100 == 0:
            wert = verlust.item()
            verlauf.append(wert)
            vergangen = time.perf_counter() - start
            ausgabe(
                f"Schritt {schritt:5d}  Verlust {wert:.3f}  lr {lernrate(fortschritt):.1e}  "
                f"{gesehen / max(vergangen, 1e-9) / 1e3:5.0f}k Tokens/s  {vergangen:5.0f}/{sekunden:.0f} s"
            )

    ausgabe(f"Fertig: {schritt} Schritte, {gesehen / 1e6:.0f} Mio. Tokens gesehen")
    bpb = val_bpb(modell, token_bytes=tokenizer_laden().token_bytes(), geraet=geraet) if messen else None
    if messen:
        ausgabe(f"val_bpb {bpb:.4f}")
    if pfad is not None:
        modell.speichern(pfad)
        ausgabe(f"Gespeichert in {pfad}")
    return modell, bpb


def fuer_auswertung(sekunden, seed):
    """Vertrag für eine spätere Forschungsschleife (wie bytes_erkenner.training.fuer_auswertung).

    Höchstens `sekunden` lang trainieren und das Modell zurückgeben; gemessen wird außerhalb
    mit daten.val_bpb.
    """
    modell, _ = trainieren(sekunden=sekunden, seed=seed, messen=False, ausgabe=lambda *_: None)
    return modell
