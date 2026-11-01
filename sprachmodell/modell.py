"""Ein kleines GPT: Decoder-only Transformer.

Aufgabe: Gegeben die Tokens t_1 … t_n, sage für jede Stelle das nächste Token voraus.

    Token-IDs (B, T)
        │  Token-Einbettung + Positions-Einbettung
        ▼
    Vektoren (B, T, C)  ──►  Block  ──►  Block  ──►  …  ──►  LayerNorm
                                                               │  Ausgabeschicht
                                                               ▼
                                         Logits (B, T, Vokabular): eine Bewertung
                                         für jedes mögliche nächste Token

B = Batch (mehrere Textausschnitte gleichzeitig), T = Zeit (Position im Text),
C = Kanäle (Breite der Vektoren).

Ein Block besteht aus zwei Teilen, jeweils mit "Abkürzung" (Residualverbindung):

    x = x + Attention(LayerNorm(x))   # Tokens tauschen Information aus
    x = x + MLP(LayerNorm(x))         # jedes Token denkt für sich nach

Die Residualverbindung (das "x +") heißt: Jeder Block *ergänzt* den Vektor nur ein wenig,
statt ihn zu ersetzen. Dadurch fließen Gradienten auch durch viele Blöcke gut zurück.
"""

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Konfig:
    vokabular: int = 4096
    kontext: int = 512  # so viele Tokens sieht das Modell höchstens auf einmal
    schichten: int = 8  # Anzahl der Blöcke
    koepfe: int = 8  # Attention-Köpfe pro Block
    breite: int = 512  # C: Länge der Vektoren


class SelbstAufmerksamkeit(nn.Module):
    """Kausale Self-Attention: Jedes Token schaut auf die Tokens vor ihm.

    Jedes Token bildet aus seinem Vektor drei neue:
      - eine Frage (Query, q):      "Was suche ich?"
      - einen Schlüssel (Key, k):   "Was biete ich an?"
      - einen Wert (Value, v):      "Was gebe ich weiter, wenn ich gefragt werde?"

    Das Skalarprodukt q·k misst, wie gut ein früheres Token zur Frage passt. Softmax macht
    daraus Gewichte, die zusammen 1 ergeben, und das Ergebnis ist die gewichtete Summe der
    Werte v. "Kausal" heißt: Token 5 darf nur auf Token 1 bis 5 schauen, nicht in die
    Zukunft – sonst könnte es beim Training einfach abschreiben, was als Nächstes kommt.

    Mehrere Köpfe rechnen das parallel mit kleineren Vektoren. So kann ein Kopf z. B.
    auf das letzte Wort achten, ein anderer auf den Anfang des Satzes.
    """

    def __init__(self, k: Konfig):
        super().__init__()
        assert k.breite % k.koepfe == 0
        self.koepfe = k.koepfe
        self.qkv = nn.Linear(k.breite, 3 * k.breite, bias=False)  # alle drei auf einmal
        self.ausgang = nn.Linear(k.breite, k.breite, bias=False)

    def forward(self, x, manuell=False):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) → (B, Köpfe, T, C/Köpfe): jeder Kopf bekommt ein Stück der Kanäle.
        q, k, v = (t.view(B, T, self.koepfe, C // self.koepfe).transpose(1, 2) for t in (q, k, v))

        if manuell:
            # Zum Verstehen ausgeschrieben (und vom Test mit der schnellen Version verglichen):
            punkte = q @ k.transpose(-2, -1) / math.sqrt(k.size(-1))  # (B, H, T, T)
            maske = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
            punkte = punkte.masked_fill(~maske, float("-inf"))  # Zukunft verbieten
            y = punkte.softmax(dim=-1) @ v
        else:
            # Dasselbe als ein fusionierter Kern (Flash Attention): viel schneller und
            # speichersparender, weil die (T, T)-Matrix nie ganz im Speicher steht.
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # Köpfe wieder zusammenkleben
        return self.ausgang(y)


class MLP(nn.Module):
    """Pro Token ein kleines zweischichtiges Netz wie im Bytes-Erkenner: breit machen, GELU, zurück."""

    def __init__(self, k: Konfig):
        super().__init__()
        self.hoch = nn.Linear(k.breite, 4 * k.breite, bias=False)
        self.runter = nn.Linear(4 * k.breite, k.breite, bias=False)

    def forward(self, x):
        return self.runter(F.gelu(self.hoch(x)))


class Block(nn.Module):
    def __init__(self, k: Konfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(k.breite)
        self.aufmerksamkeit = SelbstAufmerksamkeit(k)
        self.norm2 = nn.LayerNorm(k.breite)
        self.mlp = MLP(k)

    def forward(self, x, manuell=False):
        x = x + self.aufmerksamkeit(self.norm1(x), manuell)
        x = x + self.mlp(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(self, k: Konfig):
        super().__init__()
        self.konfig = k
        # Einbettung: eine lernbare Tabelle, Zeile i ist der Vektor für Token i. Ähnliche
        # Tokens bekommen im Lauf des Trainings ähnliche Vektoren.
        self.token_einbettung = nn.Embedding(k.vokabular, k.breite)
        # Attention allein weiß nicht, in welcher Reihenfolge die Tokens stehen. Darum
        # bekommt jede Position ebenfalls einen gelernten Vektor dazuaddiert.
        self.positions_einbettung = nn.Embedding(k.kontext, k.breite)
        self.bloecke = nn.ModuleList(Block(k) for _ in range(k.schichten))
        self.norm = nn.LayerNorm(k.breite)
        self.kopf = nn.Linear(k.breite, k.vokabular, bias=False)
        # Gewichte teilen: Die Ausgabeschicht benutzt dieselbe Tabelle wie die Einbettung.
        # Ein Token, das gut "passt", hat einen Vektor ähnlich dem Ausgangsvektor. Spart
        # Parameter und hilft kleinen Modellen.
        self.kopf.weight = self.token_einbettung.weight
        self.apply(self._initialisieren)
        # Die Ausgänge jedes Blocks summieren sich im Residualstrom auf. Damit er bei vielen
        # Blöcken nicht anwächst, starten diese Schichten kleiner (wie bei GPT-2).
        for name, p in self.named_parameters():
            if name.endswith("ausgang.weight") or name.endswith("runter.weight"):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * k.schichten))

    @staticmethod
    def _initialisieren(modul):
        if isinstance(modul, (nn.Linear, nn.Embedding)):
            nn.init.normal_(modul.weight, std=0.02)

    def anzahl_parameter(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, ziele=None, manuell=False):
        """idx: (B, T) Token-IDs. Ohne `ziele` → Logits (B, T, V), sonst der mittlere Verlust.

        Der Verlust ist die Kreuzentropie: -log(Wahrscheinlichkeit des richtigen nächsten
        Tokens), gemittelt über alle B·T Stellen. Am Anfang rät das Modell gleichverteilt,
        der Verlust liegt dann bei ln(4096) ≈ 8.3.
        """
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.token_einbettung(idx) + self.positions_einbettung(pos)
        for block in self.bloecke:
            x = block(x, manuell)
        logits = self.kopf(self.norm(x))
        if ziele is None:
            return logits
        return F.cross_entropy(logits.float().flatten(0, 1), ziele.flatten())

    @torch.no_grad()
    def erzeugen(self, ids, neue, temperatur=1.0, top_k=None):
        """Text fortsetzen, ein Token nach dem anderen (autoregressiv).

        Jedes Mal: Logits für das letzte Token ausrechnen, durch die Temperatur teilen
        (klein = vorsichtiger, groß = wilder), optional nur die top_k besten behalten,
        und aus der Softmax-Verteilung ein Token *ziehen*. Das gezogene Token hängen wir
        an und wiederholen. (Echte LLMs merken sich dabei Keys und Values der früheren
        Tokens im "KV-Cache"; hier rechnen wir der Einfachheit halber alles neu.)
        """
        self.eval()
        geraet = next(self.parameters()).device
        idx = torch.tensor([ids], device=geraet)
        for _ in range(neue):
            fenster = idx[:, -self.konfig.kontext :]
            logits = self(fenster)[:, -1, :] / max(temperatur, 1e-6)
            if top_k:
                grenze = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1:]
                logits = logits.masked_fill(logits < grenze, float("-inf"))
            naechstes = torch.multinomial(logits.softmax(dim=-1), 1)
            idx = torch.cat([idx, naechstes], dim=1)
        return idx[0].tolist()

    def speichern(self, pfad):
        torch.save({"konfig": asdict(self.konfig), "gewichte": self.state_dict()}, pfad)

    @classmethod
    def laden(cls, pfad):
        if not pfad.exists():
            raise SystemExit(f"{pfad} fehlt. Erst: python -m sprachmodell trainieren")
        stand = torch.load(pfad, map_location="cpu")
        modell = cls(Konfig(**stand["konfig"]))
        modell.load_state_dict(stand["gewichte"])
        return modell
