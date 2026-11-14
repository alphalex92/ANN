"""Ein größeres Modell (Swift über die UkisAI-API) schreibt an einer Stelle, an der K2 hängen
bleibt, den nächsten Zug: die Antwort oder eine weitere Suche.

Der Lehrer sieht das Gespräch bis zur Stelle als Text (Frage, K2s Suchanfragen und Gedanken,
Suchtreffer) und antwortet in einem festen JSON-Format. Daraus wird eine Nachricht in K2s
Format, die direkt ins Gespräch passt. Sucht der Lehrer, läuft die lokale Suche und er ist
wieder dran, bis er antwortet oder die Suchen aufgebraucht sind.

Der Lehrer bekommt die richtige Lösung nicht zu sehen. Ob sein Zug stimmt, wird hinterher wie
bei K2 gegen die Lösungen geprüft.
"""
import json
import re
import time
import urllib.error
import urllib.request

ADRESSEN = {
    "flash-next": "https://ukisai.com/api/flash-next/v1/chat/completions",
    "swift": "https://ukisai.com/api/swift/v1/chat/completions",
}

ANLEITUNG = """You are continuing the work of a small assistant that answers factual questions with a \
Wikipedia search tool. Below is its conversation so far: the question, its own notes and \
searches, and the search results. Write the assistant's next step.

Rules:
- If the results contain the answer, answer. Keep the answer brief, one sentence at most.
- If the question is ambiguous (several works or people share the name) and the results do \
not settle it, name the most likely candidates briefly instead of searching on and on.
- Only search again if the results do not contain the answer and a different query is \
likely to find it. Searches left: {rest}.
- "note" is a short first-person thought (1 to 4 sentences) explaining the step, written as \
the assistant thinking to itself, e.g. "The second result says the film was directed by X, \
so that is the answer."

Reply with JSON only, one of:
{{"note": "...", "answer": "..."}}
{{"note": "...", "search": "query"}}"""


def als_text(gespraech):
    """Das Gespräch so, wie der Lehrer es lesen soll."""
    teile = []
    for m in gespraech:
        if m["role"] == "user":
            teile.append(f"QUESTION: {m['content']}")
        elif m["role"] == "tool":
            teile.append(f"SEARCH RESULTS:\n{m['content']}")
        else:
            gedanken = next((m[f] for f in ("think_faster", "think_fast", "think") if f in m), "")
            if gedanken.strip():
                teile.append(f"ASSISTANT NOTES: {gedanken.strip()}")
            for a in m.get("tool_calls") or []:
                teile.append(f"ASSISTANT SEARCHED: {a['function']['arguments'].get('anfrage', '')}")
            if m.get("content") and not m.get("tool_calls"):
                teile.append(f"ASSISTANT ANSWERED: {m['content']}")
    return "\n\n".join(teile)


def _json(text):
    """Holt das letzte JSON-Objekt aus der Antwort, auch wenn Text oder ```json drumherum steht."""
    for kandidat in reversed(re.findall(r"\{.*?\}", text, re.S)):
        try:
            d = json.loads(kandidat)
        except json.JSONDecodeError:
            continue
        if isinstance(d, dict) and "note" in d and ("answer" in d or "search" in d):
            return d
    return None


def zug(d, feld="think_fast"):
    """JSON des Lehrers → Nachricht in K2s Format."""
    nachricht = {"role": "assistant", feld: d["note"].strip()}
    if d.get("search"):
        nachricht["content"] = ""
        nachricht["tool_calls"] = [{"type": "function", "function": {
            "name": "suche", "arguments": {"anfrage": d["search"].strip()}}}]
    else:
        nachricht["content"] = str(d["answer"]).strip()
    return nachricht


class Lehrer:
    def __init__(self, modell="swift", versuche=4, pause=20, max_tokens=8000):
        self.modell, self.adresse = modell, ADRESSEN[modell]
        self.versuche, self.pause, self.max_tokens = versuche, pause, max_tokens

    def fragen(self, text):
        """Schickt eine Anfrage, wiederholt bei Serverfehlern. Gibt (Inhalt, Sekunden) zurück."""
        daten = json.dumps({"model": self.modell, "max_tokens": self.max_tokens, "temperature": 1.0,
                            "top_p": 0.95, "messages": [{"role": "user", "content": text}]}).encode()
        for versuch in range(self.versuche):
            start = time.perf_counter()
            try:
                anfrage = urllib.request.Request(self.adresse, data=daten,
                                                 headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(anfrage, timeout=600) as r:
                    antwort = json.load(r)
                return antwort["choices"][0]["message"]["content"] or "", time.perf_counter() - start
            except (urllib.error.URLError, TimeoutError, KeyError) as e:
                if versuch == self.versuche - 1:
                    raise
                print(f"  Fehler ({e}), neuer Versuch in {self.pause} s", flush=True)
                time.sleep(self.pause * (versuch + 1))

    def weiter(self, gespraech, suchen, suche, max_suchen=3):
        """Setzt das Gespräch fort, bis der Lehrer antwortet. Gibt (Gespräch, Status, Sekunden) zurück."""
        gespraech, dauer = list(gespraech), 0.0
        while True:
            rest = max(0, max_suchen - suchen)
            text, s = self.fragen(ANLEITUNG.format(rest=rest) + "\n\n---\n\n" + als_text(gespraech))
            dauer += s
            d = _json(text)
            if d is None:
                return gespraech, "unlesbar", dauer
            if d.get("search") and rest == 0:
                return gespraech, "zu_viele_suchen", dauer
            nachricht = zug(d)
            gespraech.append(nachricht)
            if not d.get("search"):
                return gespraech, "fertig", dauer
            from .suche import als_text as treffer_text
            gespraech.append({"role": "tool", "name": "suche",
                              "content": treffer_text(suche.suchen(d["search"].strip()))})
            suchen += 1


def laeufe_erzeugen(lehrer, stellen, suche, ziel, parallel=2):
    """Lässt den Lehrer jede Stelle einmal fortsetzen und hängt die Läufe an `ziel` an.

    Gleiches Zeilenformat wie bei K2s Fortsetzungen, mit der ID "<frage>#l" und `start`.
    Bereits erledigte IDs werden übersprungen, fehlgeschlagene Anfragen beim nächsten Mal wiederholt.
    """
    import os
    from concurrent.futures import ThreadPoolExecutor
    from threading import Lock

    from .bewertung import richtig

    fertig = set()
    if os.path.exists(ziel):
        with open(ziel, encoding="utf-8") as f:
            fertig = {json.loads(z)["id"] for z in f if z.strip()}
    offen = [s for s in stellen if f"{s['id']}#l" not in fertig]
    print(f"{len(stellen)} Stellen: {len(fertig)} schon erledigt, {len(offen)} offen", flush=True)
    sperre, zaehler, start = Lock(), [0, 0], time.perf_counter()

    def eins(s):
        try:
            g, status, dauer = lehrer.weiter(s["gespraech"], s["suchen"], suche)
        except Exception as e:  # Server dauerhaft nicht erreichbar: später erneut versuchen
            print(f"  {s['id']} übersprungen: {e}", flush=True)
            return
        antwort = g[-1].get("content") if status == "fertig" else None
        zeile = {k: s[k] for k in ("frage", "typ", "loesungen", "klasse", "teil", "quelle")}
        zeile.update(id=f"{s['id']}#l", gespraech=g, start=len(s["gespraech"]), status=status, antwort=antwort,
                     suchen=sum(m["role"] == "tool" for m in g), lehrer=lehrer.modell, sekunden=round(dauer, 1),
                     richtig=bool(antwort) and richtig(antwort, s["loesungen"]))
        with sperre:
            datei.write(json.dumps(zeile, ensure_ascii=False) + "\n")
            datei.flush()
            zaehler[0] += 1
            zaehler[1] += zeile["richtig"]
            if zaehler[0] % 10 == 0:
                rate = zaehler[0] / (time.perf_counter() - start)
                print(f"{zaehler[0]}/{len(offen)} Stellen, {zaehler[1]} richtig, "
                      f"noch etwa {(len(offen) - zaehler[0]) / rate / 60:.0f} min", flush=True)

    with open(ziel, "a", encoding="utf-8") as datei, ThreadPoolExecutor(parallel) as pool:
        list(pool.map(eins, offen))
