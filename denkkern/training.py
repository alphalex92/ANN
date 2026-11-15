"""Training: jede Stapel-Ladung frisch ausgewürfelte Welten, gelernt wird nur das eigene Schreiben.

Pro Schritt bekommt jede Folge eine zufällige Hop-Zahl aus dem Trainingsbereich. Pro Stapel
wird außerdem die Zahl der Schleifen zufällig gezogen, damit der Kern lernt, mit wenigen
wie mit vielen Durchläufen etwas Sinnvolles zu tun.

Die Lernrate steigt kurz an und fällt dann linear über das Zeitbudget auf 0, wie beim
Bytes-Erkenner.
"""

import random
import time

import torch

from . import welt
from .modell import Konfig, SchleifenGPT


def stapeln(episoden, modus, geraet):
    """Episoden zu Eingabe x und Ziel y (B, T). Nicht gelernte Stellen haben Ziel -100."""
    folgen = [f for ep in episoden for f in welt.folgen(ep, modus)]
    laenge = max(len(t) for t, _ in folgen)
    x = torch.full((len(folgen), laenge - 1), welt.PAD, dtype=torch.long)
    y = torch.full((len(folgen), laenge - 1), -100, dtype=torch.long)
    for i, (tokens, eigen) in enumerate(folgen):
        t = torch.tensor(tokens)
        n = len(tokens) - 1
        x[i, :n] = t[:-1]
        ziel = t[1:].clone()
        ziel[~torch.tensor(eigen[1:])] = -100
        y[i, :n] = ziel
    return x.to(geraet), y.to(geraet)


def bereich(text):
    """'1-4' → (1, 4), '3' → (3, 3)."""
    teile = [int(t) for t in str(text).split("-")]
    if len(teile) == 1:
        return teile[0], teile[0]
    if len(teile) != 2 or teile[0] > teile[1]:
        raise ValueError(f"Bereich wie '1-4' erwartet, nicht {text!r}")
    return teile[0], teile[1]


def kontext_fuer(modus, max_hops):
    """Kontextlänge, die für Ketten bis max_hops reicht, mit Luft fürs Ausrollen."""
    frage = 3 + max_hops
    if modus == "werkzeug":
        # bis zu max_hops + 3 Aktionen mit je 3 + 2 Tokens
        return frage + 5 * (max_hops + 3) + 1
    if modus == "kompakt":
        # ein Notizzettel mit voller Restkette plus eine Aktion
        return frage + 3 + max_hops + 3 + 1
    if modus == "restkette":
        # dazu je Treffer „|“ und höchstens max_hops - 1 offene Glieder
        return frage + (6 + max_hops) * (max_hops + 3) + 1
    return 1 + 4 * welt.FAKTEN_IM_KONTEXT + frage + 5


def trainieren(
    modus,
    hops="1-4",
    schleifen="1-4",
    minuten=5.0,
    stapel=256,
    lernrate=1e-3,
    positionen="rope",
    breite=256,
    kern=2,
    seed=0,
    pfad=None,
    geraet=None,
    mess_hops=12,
    melden=print,
):
    geraet = geraet or ("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(seed)
    torch.manual_seed(seed)
    h_lo, h_hi = bereich(hops)
    s_lo, s_hi = bereich(schleifen)

    konfig = Konfig(
        vokabular=welt.VOKABULAR,
        kontext=kontext_fuer(modus, max(h_hi, mess_hops)),
        breite=breite,
        kern=kern,
        positionen=positionen,
    )
    modell = SchleifenGPT(konfig).to(geraet)
    melden(f"{modell.anzahl_parameter() / 1e6:.2f} M Parameter, Modus {modus}, Hops {h_lo}-{h_hi}, "
           f"Schleifen {s_lo}-{s_hi}, Positionen {positionen}, auf {geraet}")

    opt = torch.optim.AdamW(modell.parameters(), lr=lernrate, betas=(0.9, 0.95), weight_decay=0.1)
    budget = minuten * 60
    aufwaermen = 100
    beginn = time.time()
    letzte_meldung = beginn
    schritte = 0
    verlust_glatt = None
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if geraet == "cuda" else torch.autocast("cpu", enabled=False)

    modell.train()
    while True:
        vergangen = time.time() - beginn
        if vergangen >= budget:
            break
        faktor = min(1.0, (schritte + 1) / aufwaermen) * max(0.0, 1 - vergangen / budget)
        for gruppe in opt.param_groups:
            gruppe["lr"] = lernrate * faktor

        episoden = [welt.neue_episode(rng, rng.randint(h_lo, h_hi)) for _ in range(stapel)]
        x, y = stapeln(episoden, modus, geraet)
        n_schleifen = rng.randint(s_lo, s_hi)
        with autocast:
            verlust = modell(x, n_schleifen, y)
        opt.zero_grad(set_to_none=True)
        verlust.backward()
        torch.nn.utils.clip_grad_norm_(modell.parameters(), 1.0)
        opt.step()
        schritte += 1

        v = verlust.item()
        verlust_glatt = v if verlust_glatt is None else 0.98 * verlust_glatt + 0.02 * v
        if time.time() - letzte_meldung > 30:
            letzte_meldung = time.time()
            melden(f"  {vergangen / 60:5.1f} min  Schritt {schritte:6d}  Verlust {verlust_glatt:.4f}")

    melden(f"fertig: {schritte} Schritte in {(time.time() - beginn) / 60:.1f} min, Verlust {verlust_glatt:.4f}")
    info = {"modus": modus, "hops": [h_lo, h_hi], "schleifen": [s_lo, s_hi], "schritte": schritte,
            "minuten": minuten, "seed": seed, "verlust": verlust_glatt}
    if pfad is not None:
        modell.speichern(pfad, **info)
    return modell, info
