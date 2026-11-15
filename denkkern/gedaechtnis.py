"""Wachsendes Gedächtnis: erst recherchieren, nachts sichten, träumen und festigen.

Die Frage: Kann ein Kern, der in der Kindheit nur das Nachschlagen gelernt hat, im Leben
Wissen dazugewinnen, ohne dabei mehr zu halluzinieren? Und wohin gehört dieses Wissen,
in eine eigene Gedächtnisschicht oder in das ganze Modell?

Die Welt hat 16 Beziehungen und 64 Dinge. Ein Teil der Beziehungen ist fest, der Rest
wird wie bisher in jeder Episode neu ausgewürfelt („flüchtig“, wie Tagesnachrichten):

    r0–r3    fest seit der Kindheit. Die Hälfte dieser Fakten lernt das Modell in der
             Kindheit auswendig („gelernt“), die andere Hälfte muss es nachschlagen.
    r8–r11   fest, aber erst im Leben. In der Kindheit waren sie flüchtig, das Modell hat
             also gelernt, sie nachzuschlagen. Diese Fakten soll es im Leben dazulernen.
    sonst    immer flüchtig: Man darf sie nie auswendig wissen.

Gearbeitet wird mit dem Notizzettel (Modus kompakt). Neben Suchen und Antworten gibt es
eine dritte Aktion: Wissen.

    <s> ? e5 r9 r13 = e5 | r9 r13     →  W e40 ·       „r9(e5) weiß ich: e40“
    <s> ? e5 r9 r13 = e40 | r13       →  S e40 r13     „r13 ist flüchtig, nachschlagen“
    <s> ? e5 r9 r13 = e2 |            →  A e2 </s>

Die Umgebung prüft ein „W“ nicht, sie schreibt den behaupteten Stand einfach auf den Zettel,
genau wie ein Modell, das etwas zu wissen glaubt. Ein falsches W ist eine Halluzination.

Ein Lebenstag:
  1. Morgens messen, auf festen Testfragen.
  2. Tagsüber Fragen beantworten. Alles, was bei einer Suche herauskommt, wird notiert.
  3. Sichten. „pruefen“: Ein Fakt gilt erst als gesichert, wenn die letzten zwei Suchen
     dasselbe ergaben. Widersprechen sie sich, wird er verworfen. „alles“: Jeder gesehene
     Fakt wird übernommen, auch die flüchtigen.
  4. Träumen und festigen. Aus den gesicherten Fakten werden Übungsketten erfunden, gemischt
     mit Ketten wie in der Kindheit, damit nichts verlernt wird. Gelernt wird entweder nur
     die Gedächtnisschicht („gedaechtnis“, das Denken bleibt eingefroren) oder alles („ganz“).
"""

import random
import time
from collections import defaultdict
from dataclasses import dataclass, field

import torch

from .modell import Konfig, SchleifenGPT

ENTITAETEN = 64
RELATIONEN = 16
SONDER = ["·", "<s>", "?", "=", "|", "S", "W", "A", "</s>"]
PAD, BOS, FRAGE, TREFFER, REST, SUCHE, WEISS, ANTWORT, ENDE = range(len(SONDER))
REL0 = len(SONDER)
ENT0 = REL0 + RELATIONEN
VOKABULAR = ENT0 + ENTITAETEN
AKTION_LAENGE = 3
MAX_HOPS = 4
KONTEXT = 32


def rel(i):
    return REL0 + i


def ent(i):
    return ENT0 + i


def ist_rel(t):
    return REL0 <= t < ENT0


def ist_ent(t):
    return ENT0 <= t < VOKABULAR


ALLE = [rel(i) for i in range(RELATIONEN)]
KINDHEIT_FEST = [rel(i) for i in range(4)]
LEBEN_FEST = [rel(i) for i in range(8, 12)]
ALTE = [rel(i) for i in range(8)]  # Testfragen „alt“: Kindheitswissen und Suchen
NEUE = [rel(i) for i in range(8, 16)]  # Fragen im Leben: halb fest (zu lernen), halb flüchtig
KINDHEIT_SONST = [r for r in ALLE if r not in KINDHEIT_FEST]


def lesbar(tokens):
    teile = []
    for t in tokens:
        if t < REL0:
            teile.append(SONDER[t])
        elif ist_rel(t):
            teile.append(f"r{t - REL0}")
        else:
            teile.append(f"e{t - ENT0}")
    return " ".join(teile)


# --- Die Welt -------------------------------------------------------------------------


def feste_fakten(relationen, seed):
    rng = random.Random(seed)
    return {(ent(e), r): ent(rng.randrange(ENTITAETEN)) for r in relationen for e in range(ENTITAETEN)}


@dataclass
class Welt:
    kindheit: dict  # (Ding, Beziehung) → Ding, auf r0–r3
    gelernt: set  # die Hälfte davon, die in der Kindheit auswendig gelernt wird
    leben: dict  # auf r8–r11, erst im Leben fest

    def fest(self):
        return {**self.kindheit, **self.leben}


def neue_welt(seed=0):
    kindheit = feste_fakten(KINDHEIT_FEST, seed)
    gelernt = set(random.Random(seed + 1).sample(sorted(kindheit), len(kindheit) // 2))
    return Welt(kindheit, gelernt, feste_fakten(LEBEN_FEST, seed + 2))


def wandeln(welt, anteil, seed):
    """Ein Teil der Lebensfakten ändert sich (die Welt bleibt nicht stehen). Gibt die Paare zurück."""
    rng = random.Random(seed)
    paare = rng.sample(sorted(welt.leben), int(anteil * len(welt.leben)))
    for p in paare:
        alt = welt.leben[p]
        welt.leben[p] = ent(rng.choice([i for i in range(ENTITAETEN) if ent(i) != alt]))
    return paare


@dataclass
class Episode:
    start: int
    kette: list
    fest: dict = field(repr=False)  # feste Fakten; alles andere würfelt die Episode selbst
    rng: random.Random = field(repr=False)
    _fluechtig: dict = field(default_factory=dict, repr=False)

    @property
    def hops(self):
        return len(self.kette)

    def nachschlagen(self, e, r):
        if (e, r) in self.fest:
            return self.fest[(e, r)]
        if (e, r) not in self._fluechtig:
            self._fluechtig[(e, r)] = ent(self.rng.randrange(ENTITAETEN))
        return self._fluechtig[(e, r)]

    def weg(self):
        dinge = [self.start]
        for r in self.kette:
            dinge.append(self.nachschlagen(dinge[-1], r))
        return dinge

    @property
    def antwort(self):
        return self.weg()[-1]


def episode(rng, fest, relationen, hops=None, start=None, kette=None):
    eigen = random.Random(rng.getrandbits(64))
    hops = hops or eigen.randint(1, MAX_HOPS)
    start = start if start is not None else ent(eigen.randrange(ENTITAETEN))
    kette = kette if kette is not None else [eigen.choice(relationen) for _ in range(hops)]
    return Episode(start, list(kette), fest, eigen)


def kindheitsrelationen(rng):
    """Halb aus den festen Kindheitsbeziehungen, halb aus allen anderen."""
    return KINDHEIT_FEST if rng.random() < 0.5 else KINDHEIT_SONST


def kindheitsepisode(rng, fest):
    eigen = random.Random(rng.getrandbits(64))
    kette = [eigen.choice(kindheitsrelationen(eigen)) for _ in range(eigen.randint(1, MAX_HOPS))]
    return episode(eigen, fest, ALLE, kette=kette)


# --- Notizzettel, Vorbild und Training ------------------------------------------------


def notiz(ep, stand, offen):
    return [BOS, FRAGE, ep.start, *ep.kette, TREFFER, stand, REST, *offen]


def folgen(ep, bekannt):
    """Je Schritt (Notizzettel + vorbildliche Aktion, eigen). Wissen nur, was bekannt ist."""
    weg = ep.weg()
    ergebnis = []
    for i in range(ep.hops + 1):
        zettel = notiz(ep, weg[i], ep.kette[i:])
        if i == ep.hops:
            aktion = [ANTWORT, weg[i], ENDE]
        elif (weg[i], ep.kette[i]) in bekannt:
            aktion = [WEISS, weg[i + 1], PAD]
        else:
            aktion = [SUCHE, weg[i], ep.kette[i]]
        ergebnis.append((zettel + aktion, [False] * len(zettel) + [True] * len(aktion)))
    return ergebnis


def stapeln(beispiele, geraet):
    folgen_ = [f for ep, bekannt in beispiele for f in folgen(ep, bekannt)]
    laenge = max(len(t) for t, _ in folgen_)
    x = torch.full((len(folgen_), laenge - 1), PAD, dtype=torch.long)
    y = torch.full((len(folgen_), laenge - 1), -100, dtype=torch.long)
    for i, (tokens, eigen) in enumerate(folgen_):
        t = torch.tensor(tokens)
        n = len(tokens) - 1
        x[i, :n] = t[:-1]
        ziel = t[1:].clone()
        ziel[~torch.tensor(eigen[1:])] = -100
        y[i, :n] = ziel
    return x.to(geraet), y.to(geraet)


def _autocast(geraet):
    if geraet.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast("cpu", enabled=False)


def ueben(modell, ziehen, params, lernrate, schritte=None, minuten=None, stapel=256, seed=0,
          schleifen=(1, 4), gewichtszerfall=0.0, melden=print):
    """Gradientenabstieg auf Beispielen aus `ziehen(rng) → (Episode, bekannt)`.

    `params`: Liste von Tensoren oder von Gruppen {"params": [...], "faktor": f}; eine Gruppe
    lernt mit lernrate·f. Entweder `schritte` oder `minuten`. Lernrate: kurz aufwärmen, dann
    linear auf 0.
    """
    geraet = next(modell.parameters()).device
    rng = random.Random(seed)
    gruppen = params if params and isinstance(params[0], dict) else [{"params": list(params), "faktor": 1.0}]
    alle = [p for g in gruppen for p in g["params"]]
    opt = torch.optim.AdamW([dict(g) for g in gruppen], lr=lernrate, betas=(0.9, 0.95),
                            weight_decay=gewichtszerfall)
    aufwaermen = 20 if schritte else 100
    beginn = letzte = time.time()
    schritt = 0
    glatt = None
    modell.train()
    while True:
        vergangen = time.time() - beginn
        anteil = schritt / schritte if schritte else vergangen / (minuten * 60)
        if anteil >= 1:
            break
        for g in opt.param_groups:
            g["lr"] = lernrate * g["faktor"] * min(1.0, (schritt + 1) / aufwaermen) * (1 - anteil)
        x, y = stapeln([ziehen(rng) for _ in range(stapel)], geraet)
        with _autocast(geraet):
            verlust = modell(x, rng.randint(*schleifen), y)
        opt.zero_grad(set_to_none=True)
        verlust.backward()
        torch.nn.utils.clip_grad_norm_(alle, 1.0)
        opt.step()
        schritt += 1
        glatt = verlust.item() if glatt is None else 0.98 * glatt + 0.02 * verlust.item()
        if melden and time.time() - letzte > 30:
            letzte = time.time()
            melden(f"  {vergangen / 60:5.1f} min  Schritt {schritt:6d}  Verlust {glatt:.4f}")
    return schritt, glatt


# --- Ausrollen und Messen -------------------------------------------------------------


@dataclass
class Verlauf:
    antwort: int | None = None
    ungueltig: bool = False
    suchen: int = 0
    gewusst: int = 0  # so oft „W“
    falsch_gewusst: int = 0  # davon falsch: Halluzinationen
    neu_gewusst: int = 0  # richtig gewusst auf den Lebensbeziehungen r8–r11
    beobachtet: list = field(default_factory=list)  # (Ding, Beziehung, Treffer) aus den Suchen

    @property
    def fertig(self):
        return self.antwort is not None or self.ungueltig


@torch.no_grad()
def gierig(modell, zettel, schleifen):
    """Für jeden Notizzettel die nächsten 3 Tokens, immer das wahrscheinlichste."""
    geraet = next(modell.parameters()).device
    B = len(zettel)
    breite = max(len(z) for z in zettel) + AKTION_LAENGE
    folge = torch.full((B, breite), PAD, dtype=torch.long, device=geraet)
    for i, z in enumerate(zettel):
        folge[i, : len(z)] = torch.tensor(z)
    laenge = torch.tensor([len(z) for z in zettel], device=geraet)
    zeilen = torch.arange(B, device=geraet)
    for _ in range(AKTION_LAENGE):
        bis = laenge.max().item()
        with _autocast(geraet):
            logits = modell(folge[:, :bis], schleifen)
        folge[zeilen, laenge] = logits[zeilen, laenge - 1].argmax(dim=-1)
        laenge = laenge + 1
    return [folge[i, laenge[i] - AKTION_LAENGE : laenge[i]].tolist() for i in range(B)]


def schritt(ep, zustand, v, aktion):
    """Eine Aktion verarbeiten. zustand = [Stand, Zahl der erledigten Glieder]."""
    a, b, c = aktion
    stand, i = zustand
    if a == ANTWORT and ist_ent(b):
        v.antwort = b
    elif a == WEISS and ist_ent(b) and i < ep.hops:
        v.gewusst += 1
        if b != ep.nachschlagen(stand, ep.kette[i]):
            v.falsch_gewusst += 1
        elif ep.kette[i] in LEBEN_FEST:
            v.neu_gewusst += 1
        zustand[:] = [b, i + 1]
    elif a == SUCHE and ist_ent(b) and ist_rel(c):
        y = ep.nachschlagen(b, c)
        v.suchen += 1
        v.beobachtet.append((b, c, y))
        zustand[:] = [y, i + 1 if i < ep.hops and (b, c) == (stand, ep.kette[i]) else i]
    else:
        v.ungueltig = True


def ausrollen(modell, episoden, schleifen=4, stapel=1024):
    modell.eval()
    zustaende = [[ep.start, 0] for ep in episoden]
    verlaeufe = [Verlauf() for _ in episoden]
    for _ in range(MAX_HOPS + 3):
        offen = [j for j, v in enumerate(verlaeufe) if not v.fertig]
        if not offen:
            break
        for k in range(0, len(offen), stapel):
            teil = offen[k : k + stapel]
            zettel = [notiz(episoden[j], zustaende[j][0], episoden[j].kette[zustaende[j][1] :]) for j in teil]
            for j, aktion in zip(teil, gierig(modell, zettel, schleifen)):
                schritt(episoden[j], zustaende[j], verlaeufe[j], aktion)
    return verlaeufe


def auswerten(episoden, verlaeufe):
    n = len(episoden)
    neue_feste_hops = sum(sum(r in LEBEN_FEST for r in ep.kette) for ep in episoden)
    return {
        "richtig": sum(v.antwort == ep.antwort for ep, v in zip(episoden, verlaeufe)) / n,
        "suchen": sum(v.suchen for v in verlaeufe) / n,
        "gewusst": sum(v.gewusst for v in verlaeufe) / n,
        "falsch_gewusst": sum(v.falsch_gewusst for v in verlaeufe) / n,
        # Anteil der Schritte auf r8–r11, die das Modell richtig aus dem Gedächtnis wusste
        "neu_gewusst": sum(v.neu_gewusst for v in verlaeufe) / max(1, neue_feste_hops),
        "abgebrochen": sum(v.antwort is None for v in verlaeufe) / n,
    }


def testfragen(art, fest, anzahl=500, seed=777):
    rng = random.Random(seed + {"neu": 0, "alt": 1, "kindheit": 2}[art])
    if art == "kindheit":
        return [kindheitsepisode(rng, fest) for _ in range(anzahl)]
    return [episode(rng, fest, NEUE if art == "neu" else ALTE) for _ in range(anzahl)]


def messen(modell, art, fest, anzahl=500):
    episoden = testfragen(art, fest, anzahl)
    return auswerten(episoden, ausrollen(modell, episoden))


# --- Kindheit ---------------------------------------------------------------------------


def kindheit(welt, minuten=10.0, breite=256, kern=2, plaetze=64, seed=0, stapel=256, welten=1,
             kern_faktor=1.0, geraet=None, melden=print):
    """Das Modell lernt Nachschlagen, Notizzettel und Wissen-oder-Suchen.

    welten=1: eine Kindheitswelt, alles lernt gemeinsam. Dann legt das Modell die Fakten
    dort ab, wo es am bequemsten ist, meist in den normalen Schichten, und die
    Gedächtnisschicht bleibt ungenutzt.

    welten>1: Die Kindheit besteht aus vielen kurzen Abschnitten mit je einer anderen Welt.
    Zu Beginn jedes Abschnitts wird der Inhalt des Gedächtnisses (die Werte) gelöscht, und
    das Gedächtnis lernt 10-mal schneller als der Rest. Fakten in den normalen Schichten
    lohnen sich so nicht mehr: Die Welt ist bald eine andere, und das Gedächtnis ist
    schneller. Das Denken lernt stattdessen, im Gedächtnis nachzusehen. Der letzte
    Abschnitt spielt in `welt`, der Welt, in der das Leben dann weitergeht.
    """
    geraet = torch.device(geraet or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(seed)
    konfig = Konfig(vokabular=VOKABULAR, kontext=KONTEXT, breite=breite, kern=kern, positionen="keine",
                    gedaechtnis=plaetze)
    modell = SchleifenGPT(konfig).to(geraet)
    werte = modell.gedaechtnis.werte.weight
    melden(f"{modell.anzahl_parameter() / 1e6:.2f} M Parameter, davon Gedächtniswerte "
           f"{werte.numel() / 1e6:.2f} M, auf {geraet}, {welten} Kindheitswelt(en)")
    gruppen = [{"params": [p for p in modell.parameters() if p is not werte], "faktor": kern_faktor},
               {"params": [werte], "faktor": 1.0 if welten == 1 else 10.0}]
    schritte_gesamt = 0
    for i in range(welten):
        w = welt if i == welten - 1 else neue_welt(seed * 1000 + 17 * (i + 1))
        if welten > 1:
            with torch.no_grad():
                werte.normal_(0.0, 0.02)
        fest = dict(w.kindheit)
        schritte, verlust = ueben(modell, lambda rng, fest=fest, w=w: (kindheitsepisode(rng, fest), w.gelernt),
                                  gruppen, 1e-3, minuten=minuten / welten, seed=seed + i, stapel=stapel,
                                  melden=melden if welten == 1 else None)
        schritte_gesamt += schritte
        if welten > 1 and ((i + 1) % max(1, welten // 20) == 0 or i == welten - 1):
            m = messen(modell, "kindheit", fest, anzahl=200)
            melden(f"Welt {i + 1}/{welten}: Verlust {verlust:.4f}, richtig {m['richtig']:.0%}, "
                   f"gewusst {m['gewusst']:.2f}, falsch gewusst {m['falsch_gewusst']:.3f}")
    melden(f"fertig: {schritte_gesamt} Schritte, Verlust {verlust:.4f}")
    return modell, {"schritte": schritte_gesamt, "verlust": verlust, "minuten": minuten, "welten": welten,
                    "kern_faktor": kern_faktor}


# --- Leben ------------------------------------------------------------------------------


def sichten(beobachtet, art):
    """Aus allen Beobachtungen: gesicherte Fakten und verworfene Paare."""
    gesichert, verworfen = {}, set()
    for paar, treffer in beobachtet.items():
        if art == "alles":
            gesichert[paar] = treffer[-1]
        elif len(treffer) >= 2 and treffer[-1] == treffer[-2]:
            gesichert[paar] = treffer[-1]
        elif len(treffer) >= 2:
            verworfen.add(paar)
    return gesichert, verworfen


def traumziehen(welt, gesichert, verworfen):
    """Übungsketten für die Nacht: halb um gesicherte Fakten herum erfunden, halb wie früher.

    Die Traumwelt kennt nur, was das Modell zu wissen glaubt: Kindheitsfakten und Gesichertes.
    Alles andere würfelt der Traum aus. Das ist kein Problem, denn Treffer sind nur Kontext,
    gelernt wird bloß die Entscheidung und das, was man weiß.
    """
    fest = {**welt.kindheit, **gesichert}
    bekannt = welt.gelernt | set(gesichert)
    nach_ding = defaultdict(list)
    for e, r in gesichert:
        nach_ding[e].append(r)
    paare = sorted(gesichert)
    zweifel = sorted(verworfen)

    def ziehen(rng):
        wurf = rng.random()
        if paare and wurf < 0.5:
            e, r = rng.choice(paare)
        elif zweifel and wurf < 0.6:
            e, r = rng.choice(zweifel)  # hier soll das Modell wieder suchen
        else:
            return kindheitsepisode(rng, fest), bekannt
        kette = [r]
        stand = fest.get((e, r))
        for _ in range(rng.randint(1, MAX_HOPS) - 1):
            if stand in nach_ding and rng.random() < 0.6:
                kette.append(rng.choice(nach_ding[stand]))
            else:
                kette.append(rng.choice(ALLE))
            stand = fest.get((stand, kette[-1]))
        return episode(rng, fest, ALLE, start=e, kette=kette), bekannt

    return ziehen


def leben(modell, welt, tage=8, fragen=200, festigen="gedaechtnis", sichten_art="pruefen",
          schlaf_schritte=300, wandel_tag=None, wandel_anteil=0.25, seed=0, melden=print):
    """Ein Leben aus `tage` Tagen. Gibt das Tagebuch zurück: je Morgen die Messungen."""
    beobachtet = defaultdict(list)
    gesichert, verworfen = {}, set()
    tagebuch = []
    for tag in range(1, tage + 2):
        if wandel_tag and tag == wandel_tag:
            geaendert = wandeln(welt, wandel_anteil, seed + 99)
            melden(f"Tag {tag}: {len(geaendert)} Lebensfakten haben sich geändert")
        fest = welt.fest()
        eintrag = {
            "tag": tag,
            "gesichert": len(gesichert),
            "gesichert_falsch": sum(fest.get(p) != y for p, y in gesichert.items()),
            "neu": messen(modell, "neu", fest),
            "alt": messen(modell, "alt", fest),
        }
        if tag <= tage:
            rng = random.Random(seed * 1000 + tag)
            heute = [episode(rng, fest, NEUE) for _ in range(fragen)]
            verlaeufe = ausrollen(modell, heute)
            eintrag["tagsueber"] = auswerten(heute, verlaeufe)
            for v in verlaeufe:
                for e, r, y in v.beobachtet:
                    beobachtet[(e, r)].append(y)
            gesichert, verworfen = sichten(beobachtet, sichten_art)
            if festigen != "keiner":
                schlafen(modell, welt, gesichert, verworfen, festigen, schlaf_schritte, seed * 1000 + tag)
        tagebuch.append(eintrag)
        n, a = eintrag["neu"], eintrag["alt"]
        melden(f"Morgen {tag}: neu richtig {n['richtig']:.0%}, Suchen {n['suchen']:.2f}, gewusst "
               f"{n['neu_gewusst']:.0%}, falsch gewusst {n['falsch_gewusst']:.3f} | alt richtig "
               f"{a['richtig']:.0%}, Suchen {a['suchen']:.2f} | gesichert {eintrag['gesichert']} "
               f"(davon falsch {eintrag['gesichert_falsch']})")
    return tagebuch


def schlafen(modell, welt, gesichert, verworfen, festigen, schritte, seed):
    if festigen == "gedaechtnis":
        params, lernrate = [modell.gedaechtnis.werte.weight], 1e-2  # wie in der Kindheit: nur die Werte, schnell
    elif festigen == "ganz":
        params, lernrate = list(modell.parameters()), 3e-4
    else:
        raise ValueError(f"festigen muss 'gedaechtnis', 'ganz' oder 'keiner' sein, nicht {festigen!r}")
    for p in modell.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
    try:
        ueben(modell, traumziehen(welt, gesichert, verworfen), params, lernrate, schritte=schritte, stapel=64,
              seed=seed, melden=None)
    finally:
        for p in modell.parameters():
            p.requires_grad_(True)
