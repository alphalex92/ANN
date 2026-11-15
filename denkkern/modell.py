"""Ein GPT mit Schleife: wenige Blöcke, mehrfach durchlaufen.

    Tokens ─► Einbettung ─► vorne ─► e ─┐
                                         ▼
                              ┌─► einspeisen(h, e) ─► Kern ─┐   × schleifen
                              └──────────────────────────────┘
                                         │ h
                                         ▼
                                      hinten ─► LayerNorm ─► Logits

Der Kern hat eigene Gewichte nur einmal, läuft aber `schleifen`-mal. Mehr Durchläufe heißt
mehr Rechnen bei gleich vielen Parametern. Vor jedem Durchlauf wird die Eingabe `e` neu
eingespeist (wie bei Huginn/„recurrent depth“), damit das Modell die Frage nicht vergisst,
während es denkt.

Trainiert man mit zufällig vielen Durchläufen, lernt der Kern einen Schritt, der sich
wiederholen lässt. Dann kann man beim Messen mehr Durchläufe geben als im Training, und
die spannende Frage ist: Löst das Modell dann längere Ketten?

Positionen: Gelernte Positions-Einbettungen kennen nur die Längen aus dem Training und
würden längere Ketten von vornherein verbauen. Darum gibt es hier RoPE (relative
Positionen, in die Attention eingedreht) oder gar keine (NoPE: Die kausale Maske allein
verrät dem Modell schon etwas über die Reihenfolge).
"""

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Konfig:
    vokabular: int
    kontext: int = 256
    breite: int = 256
    koepfe: int = 4
    vorne: int = 1  # Blöcke vor der Schleife
    kern: int = 2  # Blöcke in der Schleife
    hinten: int = 1  # Blöcke nach der Schleife
    positionen: str = "rope"  # "rope" oder "keine"


def rope_tabellen(laenge, kopfbreite, geraet):
    frequenzen = 1.0 / (10000 ** (torch.arange(0, kopfbreite, 2, device=geraet).float() / kopfbreite))
    winkel = torch.outer(torch.arange(laenge, device=geraet).float(), frequenzen)
    return winkel.cos(), winkel.sin()


def rope_drehen(x, cos, sin):
    """Paare von Kanälen um einen Winkel drehen, der mit der Position wächst.

    Das Skalarprodukt zweier gedrehter Vektoren hängt dann nur vom *Abstand* der beiden
    Positionen ab, nicht von der absoluten Stelle.
    """
    x1, x2 = x[..., 0::2], x[..., 1::2]
    gedreht = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return gedreht.flatten(-2).type_as(x)


class SelbstAufmerksamkeit(nn.Module):
    def __init__(self, k: Konfig):
        super().__init__()
        assert k.breite % k.koepfe == 0
        self.koepfe = k.koepfe
        self.qkv = nn.Linear(k.breite, 3 * k.breite, bias=False)
        self.ausgang = nn.Linear(k.breite, k.breite, bias=False)

    def forward(self, x, rope):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.koepfe, C // self.koepfe).transpose(1, 2) for t in (q, k, v))
        if rope is not None:
            q, k = rope_drehen(q, *rope), rope_drehen(k, *rope)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.ausgang(y.transpose(1, 2).contiguous().view(B, T, C))


class Block(nn.Module):
    def __init__(self, k: Konfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(k.breite)
        self.aufmerksamkeit = SelbstAufmerksamkeit(k)
        self.norm2 = nn.LayerNorm(k.breite)
        self.hoch = nn.Linear(k.breite, 4 * k.breite, bias=False)
        self.runter = nn.Linear(4 * k.breite, k.breite, bias=False)

    def forward(self, x, rope):
        x = x + self.aufmerksamkeit(self.norm1(x), rope)
        return x + self.runter(F.gelu(self.hoch(self.norm2(x))))


class SchleifenGPT(nn.Module):
    def __init__(self, k: Konfig):
        super().__init__()
        if k.positionen not in ("rope", "keine"):
            raise ValueError(f"positionen muss 'rope' oder 'keine' sein, nicht {k.positionen!r}")
        self.konfig = k
        self.token_einbettung = nn.Embedding(k.vokabular, k.breite)
        self.vorne = nn.ModuleList(Block(k) for _ in range(k.vorne))
        self.kern = nn.ModuleList(Block(k) for _ in range(k.kern))
        self.hinten = nn.ModuleList(Block(k) for _ in range(k.hinten))
        # Einspeisen: aus [normierter Zustand h, Eingabe e] wird der neue Zustand.
        self.zustand_norm = nn.LayerNorm(k.breite)
        self.einspeisen = nn.Linear(2 * k.breite, k.breite, bias=False)
        self.norm = nn.LayerNorm(k.breite)
        self.kopf = nn.Linear(k.breite, k.vokabular, bias=False)
        self.kopf.weight = self.token_einbettung.weight

        self.apply(self._initialisieren)
        tiefe = k.vorne + k.kern + k.hinten
        for name, p in self.named_parameters():
            if name.endswith("ausgang.weight") or name.endswith("runter.weight"):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * tiefe))
        # Start als „h + e“: So fließt am Anfang beides ungestört weiter.
        with torch.no_grad():
            eins = torch.eye(k.breite)
            self.einspeisen.weight.copy_(torch.cat([eins, eins], dim=1))

    @staticmethod
    def _initialisieren(modul):
        if isinstance(modul, (nn.Linear, nn.Embedding)):
            nn.init.normal_(modul.weight, std=0.02)

    def anzahl_parameter(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, schleifen, ziele=None):
        """idx: (B, T). Ohne `ziele` → Logits (B, T, V), sonst der mittlere Verlust.

        In `ziele` steht -100 an Stellen, die nicht gelernt werden.
        """
        B, T = idx.shape
        if T > self.konfig.kontext:
            raise ValueError(f"Folge mit {T} Tokens ist länger als der Kontext {self.konfig.kontext}")
        rope = None
        if self.konfig.positionen == "rope":
            rope = rope_tabellen(T, self.konfig.breite // self.konfig.koepfe, idx.device)

        e = self.token_einbettung(idx)
        for block in self.vorne:
            e = block(e, rope)
        h = e
        for _ in range(schleifen):
            h = self.einspeisen(torch.cat([self.zustand_norm(h), e], dim=-1))
            for block in self.kern:
                h = block(h, rope)
        for block in self.hinten:
            h = block(h, rope)
        logits = self.kopf(self.norm(h))
        if ziele is None:
            return logits
        return F.cross_entropy(logits.float().flatten(0, 1), ziele.flatten(), ignore_index=-100)

    def speichern(self, pfad, **info):
        pfad.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"konfig": asdict(self.konfig), "gewichte": self.state_dict(), "info": info}, pfad)

    @classmethod
    def laden(cls, pfad):
        if not pfad.exists():
            raise SystemExit(f"{pfad} fehlt. Erst: python -m denkkern trainieren")
        stand = torch.load(pfad, map_location="cpu")
        modell = cls(Konfig(**stand["konfig"]))
        modell.load_state_dict(stand["gewichte"])
        return modell, stand["info"]
