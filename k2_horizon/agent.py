"""Das Modell beantwortet Fragen und darf dabei die lokale Wikipedia-Suche benutzen.

Ein Durchlauf ist ein Gespräch im Chat-Format von Transformers:

    user       die Frage
    assistant  Gedanken, dann entweder die Antwort oder ein Aufruf von `suche`
    tool       die Suchtreffer (nur nach einem Aufruf)
    assistant  …

Viele Fragen laufen gleichzeitig. In jeder Runde erzeugt das Modell für alle offenen Gespräche
den nächsten Schritt in einem Stapel. Gespräche mit Antwort sind fertig, die anderen bekommen
ihre Suchtreffer und gehen in die nächste Runde.
"""
import re

import torch

from .suche import als_text

WERKZEUG = {"type": "function", "function": {
    "name": "suche",
    "description": "Searches a local copy of English Wikipedia and returns the 5 best matching passages.",
    "parameters": {"type": "object", "properties": {
        "anfrage": {"type": "string", "description": "Search query, e.g. a name plus what you want to know"},
    }, "required": ["anfrage"]},
}}

# Nur beim Erzeugen. Im Trainingsset fehlt er, damit das Modell das Verhalten verinnerlicht.
HINWEIS = ("Your memory of rare facts is unreliable. If you are not certain of the answer, "
           "use the search tool instead of guessing. If you are certain, answer directly.")

FRAGE = "{frage} Answer briefly."
DENK_FELD = {"low": "think_faster", "medium": "think_fast", "high": "think"}
PAD = 1
TOKEN_BUDGET = 24_000  # Tokens je Teilstapel, passt neben 9,4 GB Gewichten in 16 GB

_AUFRUF = re.compile(r"<ifm\|tool_call>(.*?)</ifm\|tool_call>", re.S)
_ARGUMENT = re.compile(r"<ifm\|arg_key>(.*?)</ifm\|arg_key>\s*<ifm\|arg_value>(.*?)</ifm\|arg_value>", re.S)
_ENDE = re.compile(r"<\|ifm\|(im_end|endoftext)\|>.*", re.S)
_DENK_ENDE = re.compile(r"</ifm\|think(_fast|_faster)?>")


def zerlegen(roh, aufwand):
    """Rohausgabe nach dem Generierungs-Prompt → Nachricht für das Gespräch.

    Die Ausgabe beginnt direkt mit den Gedanken, denn das öffnende <ifm|think…> steht schon im
    Prompt. Geschlossen wird oft mit einem anderen Tag als geöffnet (nach <ifm|think_fast> meist
    mit </ifm|think>), deshalb zählt jedes der drei. Ist der Denkteil nicht abgeschlossen
    (Tokenlimit erreicht), gibt es None zurück.
    """
    feld = DENK_FELD[aufwand]
    treffer = _DENK_ENDE.search(roh)
    if treffer is None:
        return None
    gedanken, rest = roh[:treffer.start()], roh[treffer.end():]
    rest = _ENDE.sub("", rest)
    nachricht = {"role": "assistant", feld: gedanken.rstrip("\n") if gedanken.strip() else ""}
    if "<ifm|tool_calls>" in rest:
        inhalt, block = rest.split("<ifm|tool_calls>", 1)
        aufrufe = []
        for a in _AUFRUF.findall(block):
            name = a.strip().split("\n", 1)[0].strip()
            argumente = {k.strip(): v.strip() for k, v in _ARGUMENT.findall(a)}
            aufrufe.append({"type": "function", "function": {"name": name, "arguments": argumente}})
        nachricht["content"] = inhalt.strip()
        nachricht["tool_calls"] = aufrufe
    else:
        nachricht["content"] = rest.strip()
    return nachricht


def gedanken_von(nachricht):
    for feld in DENK_FELD.values():
        if feld in nachricht:
            return nachricht[feld]
    return ""


class Agent:
    def __init__(self, tok, model, suche, aufwand="medium", max_suchen=3, max_neu=1024, ohne_suche=False):
        """ohne_suche: Das Werkzeug steht im Prompt, aber die Tokens für einen Aufruf sind gesperrt.
        So entstehen die Antworten, die das Modell gibt, wenn es rät, mit demselben Prompt wie sonst."""
        self.tok, self.model, self.suche = tok, model, suche
        self.aufwand, self.max_suchen, self.max_neu = aufwand, max_suchen, max_neu
        self.gesperrt = [tok.convert_tokens_to_ids(t) for t in ("<ifm|tool_calls>", "<ifm|tool_call>")] if ohne_suche else None
        tok.padding_side = "left"

    def _prompt(self, gespraech, hinweis):
        nachrichten = ([{"role": "system", "content": HINWEIS}] if hinweis else []) + gespraech
        return self.tok.apply_chat_template(
            nachrichten, tools=[WERKZEUG], add_generation_prompt=True, tokenize=False,
            reasoning_effort=self.aufwand,
        )

    def _erzeugen(self, prompts):
        """Erzeugt für alle Prompts, in Teilstapeln nach Länge sortiert.

        Der KV-Cache braucht etwa 150 KB je Token (36 Schichten × 8 KV-Köpfe × 128 × 2 × bf16).
        Nach drei Suchen ist ein Gespräch schnell 8000 Tokens lang. Deshalb darf ein Teilstapel
        zusammen höchstens TOKEN_BUDGET Tokens haben, mit Auffüllung und neuen Tokens gerechnet.
        """
        laengen = [len(self.tok(p, add_special_tokens=False)["input_ids"]) for p in prompts]
        reihenfolge = sorted(range(len(prompts)), key=lambda i: laengen[i])
        ergebnis = [None] * len(prompts)
        teil = []
        for i in reihenfolge + [None]:
            if i is not None:
                breite = laengen[i] + self.max_neu  # sortiert: das neue ist das längste
                if not teil or (len(teil) + 1) * breite <= TOKEN_BUDGET:
                    teil.append(i)
                    continue
            enc = self.tok([prompts[j] for j in teil], return_tensors="pt", padding=True,
                           add_special_tokens=False).to(self.model.device)
            with torch.no_grad():
                aus = self.model.generate(**enc, max_new_tokens=self.max_neu, do_sample=True,
                                          temperature=1.0, top_p=0.95, pad_token_id=PAD,
                                          suppress_tokens=self.gesperrt)
            for j, x in zip(teil, aus[:, enc["input_ids"].shape[1]:]):
                ergebnis[j] = self.tok.decode(x, skip_special_tokens=False)
            teil = [i] if i is not None else []
        return ergebnis

    def beantworten(self, fragen, hinweis=False):
        """Beantwortet eine Liste von Fragen gemeinsam.

        Gibt je Frage ein dict zurück: gespraech (ohne Hinweis), antwort, suchen (Anzahl),
        status ('fertig', 'abgeschnitten' oder 'zu_viele_suchen').
        """
        return self.fortsetzen([[{"role": "user", "content": FRAGE.format(frage=f)}] for f in fragen],
                               [0] * len(fragen), hinweis)

    def fortsetzen(self, gespraeche, suchen, hinweis=False):
        """Wie `beantworten`, aber ab einem vorhandenen Gesprächsanfang, z. B. direkt nach Suchtreffern.
        `suchen` zählt die Suchen, die im Anfang schon stecken."""
        laeufe = [{"gespraech": list(g), "antwort": None, "suchen": n, "status": None}
                  for g, n in zip(gespraeche, suchen)]
        offen = list(range(len(laeufe)))
        while offen:
            rohe = self._erzeugen([self._prompt(laeufe[i]["gespraech"], hinweis) for i in offen])
            weiter = []
            for i, roh in zip(offen, rohe):
                lauf = laeufe[i]
                nachricht = zerlegen(roh, self.aufwand)
                if nachricht is None:
                    lauf["status"] = "abgeschnitten"
                    continue
                lauf["gespraech"].append(nachricht)
                aufrufe = nachricht.get("tool_calls")
                if not aufrufe:
                    lauf["antwort"], lauf["status"] = nachricht["content"], "fertig"
                    continue
                if lauf["suchen"] >= self.max_suchen:
                    lauf["status"] = "zu_viele_suchen"
                    continue
                for aufruf in aufrufe:
                    anfrage = aufruf["function"]["arguments"].get("anfrage", "")
                    ergebnis = als_text(self.suche.suchen(anfrage)) if anfrage else "Error: missing 'anfrage'."
                    lauf["gespraech"].append({"role": "tool", "name": "suche", "content": ergebnis})
                    lauf["suchen"] += 1
                weiter.append(i)
            offen = weiter
        return laeufe


def laeufe_erzeugen(agent, fragen, ziel, hinweis=True, stapel=16):
    """Beantwortet `fragen` (Zeilen aus der Wissensprüfung) stapelweise und hängt jedes Ergebnis
    als JSON-Zeile an `ziel` an. Bereits erledigte Fragen werden übersprungen."""
    import json
    import os
    import time

    from .bewertung import richtig

    fertig = set()
    if os.path.exists(ziel):
        with open(ziel, encoding="utf-8") as f:
            fertig = {json.loads(z)["id"] for z in f if z.strip()}
    offen = [f for f in fragen if f["id"] not in fertig]
    print(f"{len(fertig)} schon erledigt, {len(offen)} offen", flush=True)
    start = time.perf_counter()
    with open(ziel, "a", encoding="utf-8") as datei:
        for i in range(0, len(offen), stapel):
            teil = offen[i:i + stapel]
            for f, lauf in zip(teil, agent.beantworten([f["frage"] for f in teil], hinweis=hinweis)):
                zeile = {k: f[k] for k in ("id", "frage", "typ", "loesungen", "klasse", "teil")}
                zeile.update(lauf, aufwand=agent.aufwand, hinweis=hinweis,
                             richtig=lauf["antwort"] is not None and richtig(lauf["antwort"], f["loesungen"]))
                datei.write(json.dumps(zeile, ensure_ascii=False) + "\n")
            datei.flush()
            n = i + len(teil)
            rate = n / (time.perf_counter() - start)
            print(f"{n}/{len(offen)} Fragen, {rate:.2f}/s, noch etwa {(len(offen) - n) / rate / 60:.0f} min",
                  flush=True)


def fortsetzungen_erzeugen(agent, stellen, ziel, versuche=8, stapel=16):
    """Setzt jede Stelle `versuche`-mal fort und hängt die Läufe an `ziel` an.

    Eine Stelle ist ein Gesprächsanfang, der mit Suchtreffern endet (siehe datensatz.stellen).
    Jeder Lauf bekommt die ID "<frage>#f<k>" und das Feld `start`: ab dort ist das Gespräch neu.
    Bereits erledigte IDs werden übersprungen.
    """
    import json
    import os
    import time

    from .bewertung import richtig

    fertig = set()
    if os.path.exists(ziel):
        with open(ziel, encoding="utf-8") as f:
            fertig = {json.loads(z)["id"] for z in f if z.strip()}
    auftraege = [(s, f"{s['id']}#f{k}") for s in stellen for k in range(versuche)]
    offen = [(s, i) for s, i in auftraege if i not in fertig]
    print(f"{len(stellen)} Stellen × {versuche}: {len(fertig)} schon erledigt, {len(offen)} offen", flush=True)
    start = time.perf_counter()
    with open(ziel, "a", encoding="utf-8") as datei:
        for i in range(0, len(offen), stapel):
            teil = offen[i:i + stapel]
            laeufe = agent.fortsetzen([s["gespraech"] for s, _ in teil], [s["suchen"] for s, _ in teil])
            for (s, lid), lauf in zip(teil, laeufe):
                zeile = {k: s[k] for k in ("frage", "typ", "loesungen", "klasse", "teil", "quelle")}
                zeile.update(lauf, id=lid, start=len(s["gespraech"]), aufwand=agent.aufwand, hinweis=False,
                             richtig=lauf["antwort"] is not None and richtig(lauf["antwort"], s["loesungen"]))
                datei.write(json.dumps(zeile, ensure_ascii=False) + "\n")
            datei.flush()
            n = i + len(teil)
            rate = n / (time.perf_counter() - start)
            print(f"{n}/{len(offen)} Fortsetzungen, {rate:.2f}/s, noch etwa {(len(offen) - n) / rate / 60:.0f} min",
                  flush=True)
