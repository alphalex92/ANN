"""Daten vorbereiten, Batches liefern und die feste Messung val_bpb.

Diese Datei ist das Gegenstück zu `prepare.py` bei autoresearch: Sie legt die Aufgabe fest
(welcher Text, welcher Tokenizer, welche Validierungsmenge, wie gemessen wird). Eine
spätere Forschungsschleife darf sie nicht ändern, sonst wären Ergebnisse nicht vergleichbar.

    python -m sprachmodell vorbereiten

1. Zwei Teile der deutschen Wikipedia (Stand 2023-11-01) von Hugging Face laden.
2. Artikel mit festem Seed mischen; die ersten VAL_ARTIKEL sind die Validierungsmenge.
   Das Modell sieht sie beim Training nie.
3. Den BPE-Tokenizer auf einem Ausschnitt der Trainingsartikel lernen.
4. Alles kodieren und als lange Token-Folgen speichern (train.bin, val.bin).

Alles landet in ~/.cache/ann-sprachmodell (oder in $ANN_DATEN).
"""

import math
import os
import random
import urllib.request
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .tokenizer import Tokenizer

ORDNER = Path(os.environ.get("ANN_DATEN", Path.home() / ".cache" / "ann-sprachmodell"))
QUELLE = "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.de/"
TEILE = ["train-00009-of-00020.parquet", "train-00010-of-00020.parquet"]

VOKABULAR = 4096  # 256 Bytes + 3839 Verschmelzungen + 1 Sondertoken
TOKENIZER_TEXT = 40_000_000  # so viele Zeichen Trainingstext reichen für den Tokenizer
VAL_ARTIKEL = 3000
KONTEXT = 512  # Länge der Token-Folgen, die das Modell auf einmal sieht
VAL_TOKENS = 1_048_576  # so viele Tokens umfasst die Messung (2048 Folgen à 512)


# ---------------------------------------------------------------------------
# Vorbereiten (einmalig)
# ---------------------------------------------------------------------------


def _herunterladen(ausgabe):
    roh = ORDNER / "roh"
    roh.mkdir(parents=True, exist_ok=True)
    for name in TEILE:
        ziel = roh / name
        if not ziel.exists():
            ausgabe(f"Lade {name} (~230 MB) …")
            urllib.request.urlretrieve(QUELLE + name, ziel.with_suffix(".teil"))
            ziel.with_suffix(".teil").rename(ziel)
    return [roh / name for name in TEILE]


def _artikel_lesen(dateien):
    import pyarrow.parquet as pq

    artikel = []
    for datei in dateien:
        artikel.extend(pq.read_table(datei, columns=["text"]).column("text").to_pylist())
    return [a.strip() for a in artikel if len(a) > 200]  # Fast leere Stubs weglassen


_tok = None  # Tokenizer im Arbeitsprozess


def _arbeiter_start(pfad):
    global _tok
    _tok = Tokenizer.laden(pfad)


def _artikel_kodieren(stapel):
    # Jeder Artikel beginnt mit dem Sondertoken. So lernt das Modell, wo Artikel anfangen,
    # und wir können später "von null" einen neuen Artikel erzeugen lassen.
    ids = []
    for text in stapel:
        ids.append(_tok.dok_id)
        ids.extend(_tok.kodieren(text))
    return np.array(ids, dtype=np.uint16)


def _kodieren_parallel(artikel, tok_pfad):
    stapel = [artikel[i : i + 500] for i in range(0, len(artikel), 500)]
    with Pool(os.cpu_count(), initializer=_arbeiter_start, initargs=(tok_pfad,)) as pool:
        return np.concatenate(pool.map(_artikel_kodieren, stapel, chunksize=4))


def vorbereiten(ausgabe=print):
    dateien = _herunterladen(ausgabe)
    ausgabe("Lese Artikel …")
    artikel = _artikel_lesen(dateien)
    random.Random(1234).shuffle(artikel)
    val, train = artikel[:VAL_ARTIKEL], artikel[VAL_ARTIKEL:]
    ausgabe(f"{len(train)} Trainingsartikel, {len(val)} Validierungsartikel")

    tok_pfad = ORDNER / "tokenizer.json"
    if not tok_pfad.exists():
        ausgabe(f"Lerne BPE-Tokenizer mit {VOKABULAR} Tokens …")
        probe, laenge = [], 0
        for text in train:
            probe.append(text)
            laenge += len(text)
            if laenge > TOKENIZER_TEXT:
                break
        Tokenizer.trainieren("\n\n".join(probe), VOKABULAR, ausgabe).speichern(tok_pfad)

    for name, teil in (("val", val), ("train", train)):
        ausgabe(f"Kodiere {name} …")
        ids = _kodieren_parallel(teil, tok_pfad)
        ids.tofile(ORDNER / f"{name}.bin")
        n_bytes = sum(len(t.encode("utf-8")) for t in teil)
        ausgabe(f"  {len(ids):,} Tokens, {n_bytes / len(ids):.2f} Bytes pro Token")


# ---------------------------------------------------------------------------
# Laden und Batches
# ---------------------------------------------------------------------------


def tokenizer_laden():
    return Tokenizer.laden(ORDNER / "tokenizer.json")


def tokens_laden(teil):
    """Die ganze Token-Folge von 'train' oder 'val', als memmap (liegt auf der Platte)."""
    pfad = ORDNER / f"{teil}.bin"
    if not pfad.exists():
        raise SystemExit(f"{pfad} fehlt. Erst: python -m sprachmodell vorbereiten")
    return np.memmap(pfad, dtype=np.uint16, mode="r")


def zufalls_batch(tokens, batch, laenge, generator, geraet):
    """`batch` zufällige Ausschnitte der Länge `laenge` + 1.

    x ist der Ausschnitt, y derselbe um eins verschoben: An jeder Stelle soll das Modell
    das *nächste* Token vorhersagen. Ein Ausschnitt liefert so `laenge` Aufgaben auf einmal.
    """
    import torch

    start = generator.integers(0, len(tokens) - laenge - 1, size=batch)
    stuecke = np.stack([tokens[s : s + laenge + 1] for s in start]).astype(np.int64)
    stuecke = torch.from_numpy(stuecke).pin_memory().to(geraet, non_blocking=True)
    return stuecke[:, :-1], stuecke[:, 1:]


# ---------------------------------------------------------------------------
# Die feste Messung
# ---------------------------------------------------------------------------


def val_bpb(modell, tokens=None, token_bytes=None, batch=32, geraet="cuda"):
    """Bits pro Byte auf der Validierungsmenge. Kleiner ist besser.

    Der Verlust (Kreuzentropie) sagt, wie überrascht das Modell vom richtigen nächsten
    Token ist, in "nats" pro Token. Das hängt aber vom Tokenizer ab: Ein Tokenizer mit
    längeren Tokens hat weniger, dafür schwerere Vorhersagen. Darum rechnen wir um auf
    *Bits pro Byte Text*:

        bpb = Summe aller Verluste / ln(2) / Anzahl der vorhergesagten Bytes

    Das ist die Länge, auf die das Modell den Text komprimieren könnte. 8 hieße: gar
    nichts gelernt (ein Byte hat 8 Bits); gzip schafft auf Wikipedia ungefähr 2.5.

    Gemessen wird immer auf denselben ersten VAL_TOKENS Tokens, in Folgen der Länge KONTEXT.
    `modell(x)` muss Logits der Form (B, T, Vokabular) liefern.
    """
    import torch
    import torch.nn.functional as F

    if tokens is None:
        tokens = tokens_laden("val")
    if token_bytes is None:
        token_bytes = tokenizer_laden().token_bytes()
    token_bytes = torch.tensor(token_bytes, device=geraet)

    n_folgen = min(VAL_TOKENS, len(tokens) - 1) // KONTEXT
    alle = torch.from_numpy(tokens[: n_folgen * KONTEXT + 1].astype(np.int64)).to(geraet)
    summe_nats, summe_bytes = 0.0, 0
    war_training = modell.training
    modell.eval()
    with torch.no_grad():
        for anfang in range(0, n_folgen, batch):
            idx = torch.arange(anfang, min(anfang + batch, n_folgen), device=geraet)
            pos = idx[:, None] * KONTEXT + torch.arange(KONTEXT + 1, device=geraet)
            x, y = alle[pos[:, :-1]], alle[pos[:, 1:]]
            with torch.autocast(geraet, dtype=torch.bfloat16, enabled=geraet == "cuda"):
                logits = modell(x)
            verlust = F.cross_entropy(logits.float().flatten(0, 1), y.flatten(), reduction="none")
            n = token_bytes[y.flatten()]
            # Das Sondertoken steht für 0 Bytes Text und zählt nicht mit.
            summe_nats += (verlust * (n > 0)).sum().item()
            summe_bytes += n.sum().item()
    modell.train(war_training)
    return summe_nats / math.log(2) / summe_bytes
