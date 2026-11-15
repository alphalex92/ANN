"""Die Spielwelt: Wissen gibt es nur draußen, nie in den Gewichten.

Jede Episode würfelt eine neue Welt aus. Es gibt 64 Dinge (e0 … e63) und 8 Beziehungen
(r0 … r7). Jede Beziehung ist eine Zufallsfunktion: r3(e17) = e42 heißt „e17 steht in
Beziehung r3 zu e42“. Weil die Welt jedes Mal neu ist, kann das Modell keine Fakten
auswendig lernen. Es muss sie nachschlagen oder aus dem Kontext lesen.

Eine Frage ist eine Kette: „Start e5, dann r2, dann r7“ meint r7(r2(e5)). Die Anzahl der
Glieder heißt Hops.

Zwei Modi:

werkzeug  Das Modell sieht nur die Frage und muss selbst suchen:
              <s> ? e5 r2 r7   S e5 r2 = e9   S e9 r7 = e30   A e30 </s>
          „S e r“ schreibt das Modell, „= y“ antwortet die Umgebung.

restkette Wie werkzeug, aber die Umgebung sagt nach jedem Treffer, was noch offen ist:
              <s> ? e5 r2 r7   S e5 r2 = e9 | r7   S e9 r7 = e30 |   A e30 </s>
          Das Modell muss nicht mehr mitzählen. Es antwortet, sobald hinter „|“ nichts
          mehr steht. Die Frage ist: Generalisiert es dann auf beliebig lange Ketten?

kompakt   Wie restkette, aber ohne Verlauf: Nach jedem Treffer sieht das Modell nur einen
          frischen Notizzettel mit Frage, aktuellem Stand und Restkette:
              <s> ? e5 r2 r7 = e9 | r7   S e9 r7
          Jeder Schritt sieht gleich aus, egal wie lang die Kette ist.

kontext   Die nötigen Fakten stehen vermischt mit Ablenkern vorne, das Modell antwortet
          sofort, ohne Zwischenschritte im Text:
              <s> F e9 r7 e30  F e1 r0 e8  F e5 r2 e9 …   ? e5 r2 r7   A e30 </s>
          Hier muss die ganze Kette *im Kopf* laufen, also in den Schichten. Genau dafür
          sind die Schleifen gedacht: mehr Durchläufe, mehr Hops.
"""

import random
from dataclasses import dataclass, field

ENTITAETEN = 64
RELATIONEN = 8
FAKTEN_IM_KONTEXT = 16  # so viele Fakten stehen im Modus kontext vorne, Kette plus Ablenker

SONDER = ["·", "<s>", "F", "?", "S", "=", "A", "</s>"]
PAD, BOS, FAKT, FRAGE, SUCHE, TREFFER, ANTWORT, ENDE = range(len(SONDER))
REL0 = len(SONDER)
ENT0 = REL0 + RELATIONEN
REST = ENT0 + ENTITAETEN  # „|“: danach folgen die offenen Glieder der Kette (nur restkette)
VOKABULAR = REST + 1

MODI = ("werkzeug", "restkette", "kompakt", "kontext")
MIT_WERKZEUG = ("werkzeug", "restkette", "kompakt")


def rel(i):
    return REL0 + i


def ent(i):
    return ENT0 + i


def ist_rel(t):
    return REL0 <= t < ENT0


def ist_ent(t):
    return ENT0 <= t < REST


def lesbar(tokens):
    """Token-IDs als Text, zum Anschauen und für Tests."""
    teile = []
    for t in tokens:
        if t < REL0:
            teile.append(SONDER[t])
        elif ist_rel(t):
            teile.append(f"r{t - REL0}")
        elif t == REST:
            teile.append("|")
        else:
            teile.append(f"e{t - ENT0}")
    return " ".join(teile)


@dataclass
class Episode:
    start: int  # Token des Startdings
    kette: list[int]  # Relations-Tokens in der Reihenfolge, in der sie angewendet werden
    rng: random.Random = field(repr=False)
    _welt: dict = field(default_factory=dict, repr=False)

    @property
    def hops(self):
        return len(self.kette)

    def nachschlagen(self, e, r):
        """r(e) in dieser Welt. Die Welt entsteht erst beim Nachschlagen, Stück für Stück.

        Das ist dasselbe wie eine vorab ausgewürfelte Tabelle, nur billiger: Eine Episode
        braucht meist nur eine Handvoll der 512 Einträge.
        """
        schluessel = (e, r)
        if schluessel not in self._welt:
            self._welt[schluessel] = ent(self.rng.randrange(ENTITAETEN))
        return self._welt[schluessel]

    def weg(self):
        """Alle Dinge entlang der Kette: [Start, nach Hop 1, …, Antwort]."""
        dinge = [self.start]
        for r in self.kette:
            dinge.append(self.nachschlagen(dinge[-1], r))
        return dinge

    @property
    def antwort(self):
        return self.weg()[-1]

    def fakten(self):
        """Für den Modus kontext: die Fakten der Kette plus Ablenker, gemischt.

        Die Ablenker benutzen dieselben Dinge und Beziehungen, damit das Modell wirklich
        der Kette folgen muss und nicht einfach „das Ding, das am seltensten vorkommt“
        nehmen kann. Einmal ausgewürfelt, bleiben die Fakten der Episode gleich.
        """
        if "fakten" in self._welt:
            return self._welt["fakten"]
        weg = self.weg()
        noetig = [(weg[i], r) for i, r in enumerate(self.kette)]
        gesehen = set(noetig)
        ablenker = []
        while len(noetig) + len(ablenker) < FAKTEN_IM_KONTEXT:
            # Ablenker hängen oft an Dingen vom Weg, sonst wären sie leicht zu übersehen.
            e = self.rng.choice(weg) if self.rng.random() < 0.5 else ent(self.rng.randrange(ENTITAETEN))
            paar = (e, rel(self.rng.randrange(RELATIONEN)))
            if paar not in gesehen:
                gesehen.add(paar)
                ablenker.append(paar)
        alle = noetig + ablenker
        self.rng.shuffle(alle)
        self._welt["fakten"] = [(e, r, self.nachschlagen(e, r)) for e, r in alle]
        return self._welt["fakten"]


def neue_episode(rng, hops):
    if not 1 <= hops <= FAKTEN_IM_KONTEXT:
        raise ValueError(f"hops muss zwischen 1 und {FAKTEN_IM_KONTEXT} liegen")
    eigener = random.Random(rng.getrandbits(64))
    start = ent(eigener.randrange(ENTITAETEN))
    kette = [rel(eigener.randrange(RELATIONEN)) for _ in range(hops)]
    return Episode(start, kette, eigener)


def anfang(ep, modus):
    """Was das Modell vor seinem ersten eigenen Token sieht."""
    frage = [FRAGE, ep.start, *ep.kette]
    if modus == "kompakt":
        return notiz(ep, [TREFFER, ep.start, REST, *ep.kette])
    if modus in MIT_WERKZEUG:
        return [BOS, *frage]
    if modus == "kontext":
        vorne = [BOS]
        for e, r, y in ep.fakten():
            vorne += [FAKT, e, r, y]
        return vorne + frage
    raise ValueError(f"unbekannter Modus {modus!r}")


def kodieren(ep, modus):
    """Die vorbildliche Lösung als Tokens, dazu je Token: Schreibt es das Modell selbst?

    Gelernt wird nur, was das Modell selbst schreibt. Die Frage, die Fakten und die
    Treffer der Umgebung sind bloß Kontext. Sie vorherzusagen wäre ohnehin unmöglich, sie
    sind ja zufällig.
    """
    if modus == "kompakt":
        raise ValueError("im Modus kompakt gibt es keine durchgehende Folge, siehe folgen()")
    tokens = anfang(ep, modus)
    eigen = [False] * len(tokens)

    def modell(*ts):
        tokens.extend(ts)
        eigen.extend([True] * len(ts))

    def umgebung(*ts):
        tokens.extend(ts)
        eigen.extend([False] * len(ts))

    if modus in MIT_WERKZEUG:
        weg = ep.weg()
        for i, r in enumerate(ep.kette):
            modell(SUCHE, weg[i], r)
            umgebung(*treffer(modus, weg[i + 1], ep.kette[i + 1 :]))
    modell(ANTWORT, ep.antwort, ENDE)
    return tokens, eigen


def notiz(ep, stand):
    """Der Notizzettel im Modus kompakt: Frage plus „= aktueller Stand | offene Glieder“."""
    return [BOS, FRAGE, ep.start, *ep.kette, *stand]


def folgen(ep, modus):
    """Alle Trainingsfolgen einer Episode als (Tokens, eigen).

    Im Modus kompakt ist jeder Schritt eine eigene kurze Folge (ein Notizzettel und die
    Aktion darauf), sonst ist es die eine lange Folge aus `kodieren`.
    """
    if modus != "kompakt":
        return [kodieren(ep, modus)]
    weg = ep.weg()
    ergebnis = []
    for i in range(ep.hops + 1):
        zettel = notiz(ep, [TREFFER, weg[i], REST, *ep.kette[i:]])
        aktion = [SUCHE, weg[i], ep.kette[i]] if i < ep.hops else [ANTWORT, ep.antwort, ENDE]
        ergebnis.append((zettel + aktion, [False] * len(zettel) + [True] * len(aktion)))
    return ergebnis


def treffer(modus, y, offen):
    """Was die Umgebung auf eine Suche antwortet."""
    if modus in ("restkette", "kompakt"):
        return [TREFFER, y, REST, *offen]
    return [TREFFER, y]


# --- Die Umgebung beim Ausrollen ------------------------------------------------------

AKTION_LAENGE = 3  # „S e r“ oder „A e </s>“


@dataclass
class Verlauf:
    """Was in einer Episode beim freien Ausrollen passiert ist."""

    suchen: int = 0
    erledigt: int = 0  # so viele Glieder der Kette hat das Modell der Reihe nach gesucht
    weg_exakt: bool = True  # jede Suche war genau die nächste nötige
    antwort: int | None = None
    ungueltig: bool = False

    @property
    def fertig(self):
        return self.antwort is not None or self.ungueltig


def schritt(ep, modus, verlauf, aktion):
    """Eine Aktion des Modells verarbeiten. Gibt zurück, was die Umgebung anhängt.

    Bei restkette streicht die Umgebung ein Glied erst, wenn das Modell genau die nächste
    nötige Suche gemacht hat. Eine falsche Suche lässt die Restkette also stehen.
    """
    a, b, c = aktion
    if a == ANTWORT and ist_ent(b):
        verlauf.antwort = b
        return []
    if a == SUCHE and modus in MIT_WERKZEUG and ist_ent(b) and ist_rel(c):
        weg = ep.weg()
        i = verlauf.erledigt
        if i < ep.hops and (b, c) == (weg[i], ep.kette[i]):
            verlauf.erledigt += 1
        else:
            verlauf.weg_exakt = False
        verlauf.suchen += 1
        return treffer(modus, ep.nachschlagen(b, c), ep.kette[verlauf.erledigt :])
    verlauf.ungueltig = True
    return []


def max_schritte(ep, modus):
    """Wie viele Aktionen das Modell höchstens bekommt, bevor es als abgebrochen gilt."""
    return ep.hops + 3 if modus in MIT_WERKZEUG else 1


KATEGORIEN = {
    "werkzeug": ["richtig", "gesucht, aber falsch", "geraten", "abgebrochen"],
    "restkette": ["richtig", "gesucht, aber falsch", "geraten", "abgebrochen"],
    "kompakt": ["richtig", "gesucht, aber falsch", "geraten", "abgebrochen"],
    "kontext": ["richtig", "falsch", "abgebrochen"],
}


def einordnen(ep, modus, verlauf):
    if verlauf.antwort is None:
        return "abgebrochen"
    if verlauf.antwort == ep.antwort:
        return "richtig"
    if modus == "kontext":
        return "falsch"
    # Wer weniger sucht, als die Kette lang ist, kann die Antwort nicht kennen.
    return "geraten" if verlauf.suchen < ep.hops else "gesucht, aber falsch"
