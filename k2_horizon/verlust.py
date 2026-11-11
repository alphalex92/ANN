"""Blockweiser Kreuzentropie-Verlust für Modelle mit großem Wortschatz.

K2 Horizon hat 250 624 Tokens im Wortschatz. Für jede Position einer Sequenz entsteht ein
Logit-Vektor dieser Länge. Bei 8192 Positionen sind das in float32 schon 8 GB, und der normale
Weg über Transformers legt davon noch Kopien für Softmax und Gradienten an.

Hier wird der Verlust in Blöcken berechnet. Jeder Block läuft unter Checkpointing: Im Forward
werden seine Logits berechnet und gleich wieder verworfen, im Backward neu erzeugt. Es liegen
also nie mehr Logits gleichzeitig im Speicher als für einen Block.
"""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _block_summe(h, y, lm_head, ignorieren):
    return F.cross_entropy(lm_head(h).float(), y, reduction="sum", ignore_index=ignorieren)


def blockweiser_verlust(versteckt, labels, lm_head, block=512, ignorieren=-100):
    """Mittlere Kreuzentropie für die nächste-Token-Vorhersage.

    versteckt: (batch, länge, dim), die letzten Hidden States des Transformers
    labels:    (batch, länge), die Eingabe-Tokens; Positionen mit `ignorieren` zählen nicht
    lm_head:   die Ausgabeschicht, die Hidden States auf den Wortschatz abbildet

    Wie bei Transformers sagt Position i das Token i+1 voraus.
    """
    h = versteckt[:, :-1].reshape(-1, versteckt.shape[-1])
    y = labels[:, 1:].reshape(-1)
    anzahl = (y != ignorieren).sum()
    summe = versteckt.new_zeros((), dtype=torch.float32)
    for i in range(0, y.numel(), block):
        summe = summe + checkpoint(
            _block_summe, h[i:i + block], y[i:i + block], lm_head, ignorieren, use_reentrant=False
        )
    return summe / anzahl.clamp(min=1)
