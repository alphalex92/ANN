"""Lokale Wikipedia-Suche in zwei Stufen, mit BM25 von Hand in numpy/scipy.

Stufe 1 findet Artikel. Der Index enthält pro Artikel nur den Titel (doppelt gewichtet) und
die ersten 600 Zeichen. Über alle 6,4 Mio. Artikel im Volltext würde der Index die 32 GB RAM
sprengen. Der Anfang eines Artikels sagt aber meist, worum es geht.

Stufe 2 findet die Stelle. Die besten Artikel aus Stufe 1 werden in Abschnitte von etwa
120 Wörtern zerlegt. Diese Abschnitte werden noch einmal mit BM25 bewertet, und die besten
kommen zurück.

BM25 bewertet einen Text für eine Anfrage als Summe über die Anfragewörter:

    idf(w) · tf · (k1 + 1) / (tf + k1 · (1 − b + b · länge / mittlere_länge))

`tf` ist, wie oft das Wort im Text vorkommt. `idf` ist groß für seltene Wörter: "Coppola"
zählt mehr als "film". k1 begrenzt, wie viel Wiederholung bringt, und b bestraft lange Texte.
Weil der Anteil hinter idf nur vom Text abhängt, rechnen wir ihn für Stufe 1 einmal vorab aus
und speichern ihn spaltenweise (Wort → Artikel). Eine Anfrage liest dann nur die Spalten
ihrer paar Wörter.
"""
import json
import os
import re
import time
from collections import Counter

import numpy as np

K1, B = 1.2, 0.75
ANFANG = 600  # Zeichen vom Artikelanfang für Stufe 1
ABSCHNITT_WOERTER = 120

STOPPWOERTER = set("""
a about above after again against all am an and any are as at be because been before being
below between both but by can could did do does doing down during each few for from further
had has have having he her here hers herself him himself his how i if in into is it its itself
just me more most my myself no nor not now of off on once only or other our ours ourselves out
over own same she should so some such than that the their theirs them themselves then there
these they this those through to too under until up very was we were what when where which
while who whom why will with would you your yours yourself yourselves s t
""".split())

_WORT = re.compile(r"\w+")
_stemmer = None


def woerter(text):
    """Text → Liste von Wortstämmen ohne Stoppwörter ('Directed films' → ['direct', 'film'])."""
    global _stemmer
    if _stemmer is None:
        import Stemmer
        _stemmer = Stemmer.Stemmer("english")
    roh = [w for w in _WORT.findall(text.lower()) if w not in STOPPWOERTER]
    return _stemmer.stemWords(roh)


TITEL_BONUS = 10.0     # je Wort eines wörtlich gefundenen Titels, etwa ein seltenes Wort in BM25
ANFANG_FAKTOR = 1.5    # Gewicht des ersten Abschnitts eines Artikels in Stufe 2
VORRANG_GEWICHT = 2.0  # Vorsprung für Abschnitte aus vorne gelisteten Artikeln

# Fragen benutzen oft das Substantiv, Wikipedia das Verb: "director" gegen "directed by".
# Der Stemmer macht daraus verschiedene Stämme, also ergänzen wir die üblichen Formen.
ERWEITERUNG = {
    "director": "directed", "directed": "director",
    "screenwriter": "screenplay written writer", "screenplay": "screenwriter written",
    "author": "written writer novel", "wrote": "written author",
    "producer": "produced", "composer": "composed music", "born": "birth", "birthplace": "born",
}

_KLAMMER = re.compile(r"\s*\([^)]*\)\s*$")

# Wörter, nach denen gefragt wird, nicht worüber. Es gibt einen Artikel "Screenwriter", aber
# bei "Who was the screenwriter for Out 1?" ist er nicht gemeint.
FRAGEWOERTER = set(ERWEITERUNG) | {"capital", "father", "mother", "city", "who", "what", "where"}


def grundtitel(titel):
    """'Victory (1938 film)' → 'victory', 'Up in the Air' → 'up in the air'."""
    return " ".join(_WORT.findall(_KLAMMER.sub("", titel).lower()))


def wortgruppen(anfrage, laengste=8):
    """Alle zusammenhängenden Wortfolgen der Anfrage, die ein Titel sein könnten.

    Gruppen nur aus Stopp- und Fragewörtern ('the', 'of the', 'screenwriter') und einzelne
    Wörter mit weniger als 3 Buchstaben fallen weg.
    """
    w = _WORT.findall(anfrage.lower())
    gruppen = set()
    for n in range(1, laengste + 1):
        for i in range(len(w) - n + 1):
            teil = w[i:i + n]
            if (all(x in STOPPWOERTER or x in FRAGEWOERTER for x in teil)
                    or (n == 1 and len(teil[0]) < 3)):
                continue
            gruppen.add(" ".join(teil))
    return gruppen


def erweitern(anfrage):
    zusatz = [ERWEITERUNG[w] for w in _WORT.findall(anfrage.lower()) if w in ERWEITERUNG]
    return anfrage + (" " + " ".join(zusatz) if zusatz else "")


def abschnitte(titel, text, woerter_pro=ABSCHNITT_WOERTER):
    """Zerlegt einen Artikel an Absatzgrenzen in Stücke von etwa `woerter_pro` Wörtern."""
    teile, aktuell, n = [], [], 0
    for absatz in text.split("\n"):
        absatz = absatz.strip()
        if not absatz:
            continue
        aktuell.append(absatz)
        n += len(absatz.split())
        if n >= woerter_pro:
            teile.append(" ".join(aktuell))
            aktuell, n = [], 0
    if aktuell:
        teile.append(" ".join(aktuell))
    return [(titel, t) for t in teile]


def bm25_gewichte(tf, laengen, mittel):
    return tf * (K1 + 1) / (tf + K1 * (1 - B + B * laengen / mittel))


# ---------------------------------------------------------------- Index bauen

def bauen(parquet_ordner, ziel, max_dateien=None):
    """Liest die Wikipedia-Parquet-Dateien und schreibt Texte und Index nach `ziel`.

    Dateien in `ziel`:
      texte.bin, texte_offsets.npy  alle Artikeltexte hintereinander (UTF-8) und ihre Startpositionen
      titel.json                    Titel je Artikel
      vokabular.json                Wortstamm → Spalte
      idf.npy                       idf je Wort
      spalten_*.npy                 BM25-Gewichte spaltenweise (indptr, Artikel, Gewicht)
    """
    import pyarrow.parquet as pq

    os.makedirs(ziel, exist_ok=True)
    dateien = sorted(f for f in os.listdir(parquet_ordner) if f.endswith(".parquet"))[:max_dateien]
    vokabular, titel_alle, offsets = {}, [], [0]
    docs_teile, wort_teile, tf_teile, laengen = [], [], [], []
    artikel = 0
    t0 = time.perf_counter()

    with open(os.path.join(ziel, "texte.bin"), "wb") as texte:
        for nr, datei in enumerate(dateien):
            tabelle = pq.read_table(os.path.join(parquet_ordner, datei), columns=["title", "text"])
            for titel, text in zip(tabelle.column("title").to_pylist(), tabelle.column("text").to_pylist()):
                roh = text.encode("utf-8")
                texte.write(roh)
                offsets.append(offsets[-1] + len(roh))
                titel_alle.append(titel)
                ids = [vokabular.setdefault(w, len(vokabular))
                       for w in woerter(f"{titel} {titel} {text[:ANFANG]}")]
                zaehler = Counter(ids)
                laengen.append(len(ids))
                docs_teile.append(np.full(len(zaehler), artikel, dtype=np.int32))
                wort_teile.append(np.fromiter(zaehler.keys(), dtype=np.int32, count=len(zaehler)))
                tf_teile.append(np.fromiter(zaehler.values(), dtype=np.float32, count=len(zaehler)))
                artikel += 1
            # Teillisten je Datei zusammenfassen, damit nicht Millionen kleiner Arrays herumliegen
            docs_teile = [np.concatenate(docs_teile)]
            wort_teile = [np.concatenate(wort_teile)]
            tf_teile = [np.concatenate(tf_teile)]
            print(f"Datei {nr + 1}/{len(dateien)}: {artikel:,} Artikel, {len(vokabular):,} Wörter, "
                  f"{time.perf_counter() - t0:.0f} s", flush=True)

    docs, wort, tf = docs_teile[0], wort_teile[0], tf_teile[0]
    del docs_teile, wort_teile, tf_teile
    laengen = np.asarray(laengen, dtype=np.float32)
    tf = bm25_gewichte(tf, laengen[docs], laengen.mean())
    # Spaltenweise anordnen: alle Einträge eines Wortes hintereinander. Ein stabiles Sortieren
    # nach Wort braucht weniger Speicher als scipy.sparse und lässt die Artikel je Wort geordnet.
    reihenfolge = np.argsort(wort, kind="stable")
    df = np.bincount(wort, minlength=len(vokabular))
    del wort
    indptr = np.zeros(len(vokabular) + 1, dtype=np.int64)
    np.cumsum(df, out=indptr[1:])
    np.save(os.path.join(ziel, "spalten_artikel.npy"), docs[reihenfolge])
    del docs
    np.save(os.path.join(ziel, "spalten_gewicht.npy"), tf[reihenfolge])
    anzahl = len(tf)
    del tf, reihenfolge
    idf = np.log(1 + (artikel - df + 0.5) / (df + 0.5)).astype(np.float32)

    np.save(os.path.join(ziel, "texte_offsets.npy"), np.asarray(offsets, dtype=np.int64))
    np.save(os.path.join(ziel, "idf.npy"), idf)
    np.save(os.path.join(ziel, "spalten_indptr.npy"), indptr)
    with open(os.path.join(ziel, "titel.json"), "w", encoding="utf-8") as f:
        json.dump(titel_alle, f, ensure_ascii=False)
    with open(os.path.join(ziel, "vokabular.json"), "w", encoding="utf-8") as f:
        json.dump(vokabular, f, ensure_ascii=False)
    print(f"Fertig: {artikel:,} Artikel, {len(vokabular):,} Wörter, {anzahl:,} Einträge, "
          f"{time.perf_counter() - t0:.0f} s")


# ---------------------------------------------------------------- Suchen

class Suche:
    """Lädt den Index speichersparend (die großen Arrays bleiben per mmap auf der Platte)."""

    def __init__(self, ordner):
        lade = lambda n, **kw: np.load(os.path.join(ordner, n), **kw)
        self.indptr = lade("spalten_indptr.npy")
        self.artikel = lade("spalten_artikel.npy", mmap_mode="r")
        self.gewicht = lade("spalten_gewicht.npy", mmap_mode="r")
        self.idf = lade("idf.npy")
        self.offsets = lade("texte_offsets.npy")
        self.texte = np.memmap(os.path.join(ordner, "texte.bin"), dtype=np.uint8, mode="r")
        with open(os.path.join(ordner, "titel.json"), encoding="utf-8") as f:
            self.titel = json.load(f)
        with open(os.path.join(ordner, "vokabular.json"), encoding="utf-8") as f:
            self.vokabular = json.load(f)

    def text(self, i):
        return bytes(self.texte[self.offsets[i]:self.offsets[i + 1]]).decode("utf-8")

    def _titel_index(self):
        """Grundtitel ohne Klammerzusatz → Artikel. 'Victory (1938 film)' steht unter 'victory'."""
        if not hasattr(self, "_titel_zu_artikel"):
            self._titel_zu_artikel = {}
            for i, t in enumerate(self.titel):
                self._titel_zu_artikel.setdefault(grundtitel(t), []).append(i)
        return self._titel_zu_artikel

    def artikel_finden(self, anfrage, k=20):
        """Stufe 1: Indizes der k besten Artikel.

        BM25 über Titel und Artikelanfang, dazu ein Bonus für Artikel, deren Titel wörtlich in der
        Anfrage steht. Ohne ihn geht 'Victory (1938 film)' unter Tausenden Artikeln mit 'victory' unter.
        """
        punkte = {}
        titel_index = self._titel_index()
        for gruppe in wortgruppen(anfrage):
            for d in titel_index.get(gruppe, ())[:200]:
                punkte[d] = punkte.get(d, 0.0) + TITEL_BONUS * len(gruppe.split())
        for w in set(woerter(erweitern(anfrage))):
            spalte = self.vokabular.get(w)
            if spalte is None:
                continue
            a, b = self.indptr[spalte], self.indptr[spalte + 1]
            docs = np.asarray(self.artikel[a:b])
            werte = np.asarray(self.gewicht[a:b]) * self.idf[spalte]
            for d, v in zip(docs.tolist(), werte.tolist()):
                punkte[d] = punkte.get(d, 0.0) + v
        return [d for d, _ in sorted(punkte.items(), key=lambda x: -x[1])[:k]]

    def suchen(self, anfrage, k_artikel=20, k=5):
        """Stufe 2: die k besten Abschnitte als Liste von (Titel, Text)."""
        kandidaten, vorrang, ist_anfang, titel_treffer = [], [], [], []
        gruppen = wortgruppen(anfrage)
        for rang, d in enumerate(self.artikel_finden(anfrage, k_artikel)):
            teile = abschnitte(self.titel[d], self.text(d))
            kandidaten += teile
            vorrang += [(k_artikel - rang) / k_artikel] * len(teile)
            ist_anfang += [True] + [False] * (len(teile) - 1)
            g = grundtitel(self.titel[d])
            titel_treffer += [len(g.split()) if g in gruppen else 0] * len(teile)
        if not kandidaten:
            return []
        stuecke = [woerter(f"{t} {x}") for t, x in kandidaten]
        laengen = np.array([len(s) for s in stuecke], dtype=np.float32)
        mittel = max(laengen.mean(), 1.0)
        punkte = np.zeros(len(kandidaten), dtype=np.float32)
        for w in set(woerter(erweitern(anfrage))):
            spalte = self.vokabular.get(w)
            idf = self.idf[spalte] if spalte is not None else 0.0
            tf = np.array([s.count(w) for s in stuecke], dtype=np.float32)
            punkte += idf * bm25_gewichte(tf, laengen, mittel)
        # Der Artikelanfang fasst meist das Wichtigste zusammen ("… directed by …"), und Abschnitte
        # aus den in Stufe 1 besser bewerteten Artikeln bekommen einen kleinen Vorsprung.
        # Steht der Titel des Artikels wörtlich in der Frage, geht es fast sicher um ihn.
        punkte += TITEL_BONUS * np.array(titel_treffer, dtype=np.float32)
        punkte *= np.where(ist_anfang, ANFANG_FAKTOR, 1.0)
        punkte += VORRANG_GEWICHT * np.array(vorrang, dtype=np.float32)
        beste = np.argsort(-punkte)[:k]
        return [kandidaten[i] for i in beste if punkte[i] > 0]


def kuerzen(text, max_zeichen):
    """Kürzt an einer Wortgrenze und zeigt mit '…', dass etwas fehlt."""
    if len(text) <= max_zeichen:
        return text
    return text[:max_zeichen].rsplit(" ", 1)[0] + " …"


def als_text(treffer, max_zeichen=1500):
    """Formatiert Suchtreffer so, wie das Modell sie als Werkzeug-Ergebnis sieht.

    Die meisten Abschnitte sind kürzer als 1500 Zeichen und kommen vollständig an. Bei 700
    Zeichen fehlte oft genau der Satz mit der Antwort, und das Modell suchte erneut.
    """
    if not treffer:
        return "No results."
    return "\n\n".join(f"[{i + 1}] {t}\n{kuerzen(x, max_zeichen)}" for i, (t, x) in enumerate(treffer))
