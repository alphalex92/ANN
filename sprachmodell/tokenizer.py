"""Byte-Pair-Encoding (BPE): Wie ein Sprachmodell Text in Zahlen zerlegt.

Ein neuronales Netz rechnet nur mit Zahlen. Wir brauchen also eine Abbildung
Text → Liste von Ganzzahlen ("Tokens") und zurück.

Die einfachste Wahl wären die rohen UTF-8-Bytes: 256 mögliche Tokens, jeder Text
geht. Aber dann ist "Wikipedia" schon 9 Tokens lang, und das Modell muss sich für
jeden Buchstaben einzeln anstrengen. BPE löst das, indem es häufige Byte-Folgen zu
neuen Tokens verschmilzt:

    1. Start: jedes Byte ist ein Token (IDs 0..255).
    2. Zähle, welches Paar benachbarter Tokens im Trainingstext am häufigsten vorkommt,
       z. B. ("e", "n").
    3. Gib diesem Paar eine neue ID (256) und ersetze es überall.
    4. Wiederhole, bis das Vokabular groß genug ist.

Aus "e"+"n" wird "en", später " d"+"er" → " der", irgendwann ist " Wikipedia" ein
einziges Token. Häufige Wörter werden kurz, seltene zerfallen in Stücke, und da die
256 Bytes immer übrig bleiben, lässt sich *jeder* Text kodieren.

Wie bei GPT-2 zerschneiden wir den Text vorher grob in Wortstücke (MUSTER), und
verschmelzen nur innerhalb eines Stücks. So entsteht kein Token wie "r d", das über
eine Wortgrenze reicht.
"""

import json
import re
from collections import Counter, defaultdict

# Ein optionales Leerzeichen + Buchstaben | bis zu 3 Ziffern | Satzzeichen | Leerraum.
# Das Leerzeichen hängt also *vor* dem Wort: " Haus", nicht "Haus ".
MUSTER = re.compile(r" ?[^\W\d_]+| ?\d{1,3}| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+")

DOKUMENT = "<|dok|>"  # Sondertoken: markiert den Beginn eines neuen Artikels


def _paare(teile):
    """Alle benachbarten Paare einer Token-Folge: (a, b, c) → (a, b), (b, c)."""
    return zip(teile, teile[1:])


def _verschmelzen(teile, paar, neu):
    """Ersetzt jedes Vorkommen von `paar` in `teile` durch das Token `neu`."""
    ergebnis = []
    i = 0
    while i < len(teile):
        if i + 1 < len(teile) and (teile[i], teile[i + 1]) == paar:
            ergebnis.append(neu)
            i += 2
        else:
            ergebnis.append(teile[i])
            i += 1
    return tuple(ergebnis)


class Tokenizer:
    def __init__(self, verschmelzungen=()):
        # verschmelzungen[k] = (a, b) heißt: Token 256 + k entsteht aus a gefolgt von b.
        self.verschmelzungen = [tuple(p) for p in verschmelzungen]
        self.rang = {paar: 256 + k for k, paar in enumerate(self.verschmelzungen)}
        # Für jedes Token die Bytes, für die es steht.
        self.vokabular = [bytes([b]) for b in range(256)]
        for a, b in self.verschmelzungen:
            self.vokabular.append(self.vokabular[a] + self.vokabular[b])
        self.dok_id = len(self.vokabular)  # Sondertoken bekommt die letzte ID
        self._cache = {}

    @property
    def groesse(self):
        return len(self.vokabular) + 1  # + Sondertoken

    @classmethod
    def trainieren(cls, text, groesse, ausgabe=print):
        """Lernt groesse - 257 Verschmelzungen aus `text`.

        Statt den ganzen Text immer wieder zu durchsuchen, zählen wir jedes Wortstück nur
        einmal (mit seiner Häufigkeit) und merken uns, in welchen Stücken ein Paar
        vorkommt. Bei einer Verschmelzung müssen dann nur diese Stücke angepasst werden.
        """
        haeufigkeit = Counter(MUSTER.findall(text))
        woerter = [tuple(stueck.encode("utf-8")) for stueck in haeufigkeit]
        anzahl = list(haeufigkeit.values())

        paar_zahl = Counter()  # wie oft jedes Paar insgesamt vorkommt
        wo = defaultdict(set)  # Paar → Indizes der Wörter, die es enthalten
        for i, wort in enumerate(woerter):
            for paar in _paare(wort):
                paar_zahl[paar] += anzahl[i]
                wo[paar].add(i)

        verschmelzungen = []
        for k in range(groesse - 257):
            if not paar_zahl:
                break
            paar = max(paar_zahl, key=paar_zahl.get)  # das häufigste Paar
            neu = 256 + k
            verschmelzungen.append(paar)
            for i in wo.pop(paar):
                alt = woerter[i]
                # Alte Paare dieses Worts abziehen, neue Paare dazuzählen.
                for p in _paare(alt):
                    paar_zahl[p] -= anzahl[i]
                    if paar_zahl[p] <= 0:
                        del paar_zahl[p]
                woerter[i] = _verschmelzen(alt, paar, neu)
                for p in _paare(woerter[i]):
                    paar_zahl[p] += anzahl[i]
                    wo[p].add(i)
            if (k + 1) % 500 == 0:
                tok = cls(verschmelzungen)
                ausgabe(f"  {k + 1:5d} Verschmelzungen, zuletzt {tok.vokabular[neu]!r}")
        return cls(verschmelzungen)

    def _stueck_kodieren(self, stueck):
        """Ein Wortstück: von Bytes ausgehend die gelernten Verschmelzungen anwenden.

        Immer das Paar mit dem kleinsten Rang zuerst, also in der Reihenfolge, in der sie
        beim Training gelernt wurden. Nur so entstehen dieselben Tokens wie im Training.
        """
        if stueck in self._cache:
            return self._cache[stueck]
        teile = tuple(stueck.encode("utf-8"))
        while len(teile) > 1:
            paar = min(_paare(teile), key=lambda p: self.rang.get(p, float("inf")))
            if paar not in self.rang:
                break  # nichts mehr zu verschmelzen
            teile = _verschmelzen(teile, paar, self.rang[paar])
        if len(self._cache) < 1_000_000:
            self._cache[stueck] = teile
        return teile

    def kodieren(self, text):
        """Text → Liste von Token-IDs (ohne Sondertoken)."""
        ids = []
        for stueck in MUSTER.findall(text):
            ids.extend(self._stueck_kodieren(stueck))
        return ids

    def dekodieren(self, ids):
        """Token-IDs → Text. Das Sondertoken wird als Absatz dargestellt."""
        teile = [self.vokabular[i] if i != self.dok_id else b"\n\n" for i in ids]
        # Ein Token kann mitten in einem mehrbytigen UTF-8-Zeichen enden; "replace"
        # macht daraus ein �, statt abzustürzen.
        return b"".join(teile).decode("utf-8", errors="replace")

    def token_bytes(self):
        """Wie viele Bytes Text jedes Token darstellt (Sondertoken: 0). Für val_bpb."""
        return [len(v) for v in self.vokabular] + [0]

    def speichern(self, pfad):
        with open(pfad, "w") as f:
            json.dump({"verschmelzungen": self.verschmelzungen}, f)

    @classmethod
    def laden(cls, pfad):
        with open(pfad) as f:
            return cls(json.load(f)["verschmelzungen"])
