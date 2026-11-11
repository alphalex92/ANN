# ANN

Experimente mit kleinen neuronalen Netzen, zum Lernen und Herumprobieren.

## Projekt 1: Bytes-Erkenner

Ein Netz bekommt **8 rohe Bytes aus dem Speicher** und schätzt, was darin steht:

| Klasse | Beispiel | typisch für |
|---|---|---|
| `i64` | `5f 01 00 00 00 00 00 00` (= 351) | Zähler, Geld, Punkte |
| `f64` | `00 00 00 00 00 40 8f 40` (= 1000.0) | große Werte in Idle-Spielen |
| `2×i32` | zwei Ganzzahlen nebeneinander | Level + Erfahrung |
| `2×f32` | zwei Floats nebeneinander | eine Position (x, y) |
| `zeiger` | `18 9c 1e 9f 61 55 00 00` | Adresse auf ein anderes Objekt |
| `text` | `48 61 6c 6c 6f 00 00 00` (= "Hallo") | Namen, Kennungen |
| `zufall` | Rauschen | komprimierte oder verschlüsselte Daten |

Das Netz ist **von Hand in numpy** geschrieben, ohne PyTorch oder Ähnliches. So steht jede
Rechnung offen da.

### Schnellstart

```bash
python -m venv .venv
```

```bash
source .venv/bin/activate.fish
```

```bash
pip install -r requirements.txt
```

```bash
python -m bytes_erkenner trainieren
```

```bash
python -m bytes_erkenner raten 00 00 00 00 00 40 8f 40
```

`beispiele` zeigt ein paar erzeugte Trainingsdaten, `pytest` startet die Tests.

### Wie es funktioniert

**Daten** ([daten.py](bytes_erkenner/daten.py)): Wir erzeugen die Beispiele selbst, zum Beispiel
einen zufälligen Double und seine 8 Bytes. Damit kennen wir die richtige Antwort und können
beliebig viele Beispiele bauen. Jedes Byte wird in seine 8 Bits zerlegt, das Netz hat also
**64 Eingänge**, jeder ist 0 oder 1.

**Netz** ([netz.py](bytes_erkenner/netz.py)):

```
64 Bits ──► 64 versteckte Neuronen (ReLU) ──► 7 Rohwerte ──► Softmax ──► 7 Wahrscheinlichkeiten
```

- Jedes Neuron bildet eine **gewichtete Summe** seiner Eingänge plus einen Versatz (Bias).
- **ReLU** setzt negative Summen auf 0. Erst dadurch kann das Netz mehr als nur gerade Linien
  lernen.
- **Softmax** macht aus den 7 Rohwerten Wahrscheinlichkeiten, die zusammen 100 % ergeben.

**Lernen** ([training.py](bytes_erkenner/training.py)):

1. Das Netz rät für ein Häppchen von 64 Beispielen (**Mini-Batch**).
2. Der **Verlust** (Kreuzentropie) misst, wie wenig Wahrscheinlichkeit es der richtigen Klasse
   gegeben hat.
3. **Backpropagation** rechnet mit der Kettenregel aus, wie jedes Gewicht den Verlust beeinflusst
   (den **Gradienten**).
4. **Gradientenabstieg**: Jedes Gewicht wandert ein kleines Stück (die **Lernrate**) in die Richtung,
   in der der Verlust sinkt.

Ein Durchlauf durch alle Trainingsdaten heißt **Epoche**. Die **Testdaten** sieht das Netz beim
Lernen nie. Nur an ihnen sieht man, ob es wirklich verallgemeinert oder bloß auswendig lernt.

Der Test `test_gradienten_stimmen_mit_numerischer_ableitung` prüft die Backpropagation. Er wackelt
an einzelnen Gewichten und vergleicht die gemessene Änderung mit dem berechneten Gradienten.
Wer an `netz.py` etwas umbaut, merkt so sofort, wenn die Ableitung nicht mehr stimmt.

### Was man sieht

Nach 20 Epochen liegt das Netz bei etwa **92 %** auf den Testdaten (in ~3 Sekunden). Die
**Verwechslungsmatrix** am Ende zeigt, wo es danebenliegt:

- **`2×f32` ↔ `f64`** werden am häufigsten verwechselt. Beide sind Gleitkommazahlen, nur anders
  aufgeteilt.
- **`zufall`** ist manchmal nicht von echten Werten zu unterscheiden. Zufällige Bytes *können*
  ein gültiger Double sein. 100 % sind hier also gar nicht möglich.
- **`i64` ↔ `2×i32`**: Eine kleine 64-Bit-Zahl und zwei 32-Bit-Zahlen, deren zweite klein ist,
  sehen fast gleich aus.

### Ideen zum Weiterprobieren

- `--lernrate 0.05` oder `--lernrate 1.0`: Was passiert mit der Kurve? Bei 0.2 schwankt die
  Test-Genauigkeit schon sichtbar von Epoche zu Epoche.
- `--versteckt 8` gegen `--versteckt 256`: Ab wann bringt mehr Größe nichts mehr?
- `--epochen 100`: Laufen Training und Test irgendwann auseinander (Überanpassung)?
- Eine zweite versteckte Schicht in `netz.py` einbauen. Der Gradienten-Check sagt dir, ob die
  Backpropagation stimmt.
- Eine neue Klasse in `daten.py`, z. B. `bool`-Felder oder UTF-16-Text.
- Statt 64 Bits die 8 Bytes als Zahlen 0–1 eingeben: Lernt das Netz dann besser oder schlechter?
- Echte Bytes aus einem Spiel mit dem [Trainer](https://github.com/alphalex92/Trainer) auslesen
  und raten lassen.

## Reasoning-Kern: K2 Horizon 3.7B

Die Idee: ein kleines Modell, das vor allem gut **denkt**, statt alles auswendig zu wissen.
Fakten soll es sich später durch Recherche holen und mit der Zeit dazulernen. Als Ausgangspunkt
dient [K2 Horizon 3.7B](https://huggingface.co/IFM/K2-Horizon-3.7B) von MBZUAI (Apache 2.0).
Dort sind auch Trainingsdaten, Code und Zwischenstände veröffentlicht. So lässt sich prüfen, ob
das Modell einen Fakt recherchiert oder schon gekannt hat.

### Einrichten

Das Modell (9,5 GB) liegt außerhalb des Repos. Den Pfad kann man mit `K2_PFAD` ändern,
Standard ist `/mnt/nvme2/modelle/K2-Horizon-3.7B`.

```bash
hf download IFM/K2-Horizon-3.7B --local-dir /mnt/nvme2/modelle/K2-Horizon-3.7B
```

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
```

```bash
uv pip install -r requirements-k2.txt
```

### Benutzen

```bash
python -m k2_horizon fragen "Wie viele Primzahlen liegen zwischen 1 und 50?" --gedanken
```

```bash
python -m k2_horizon tempo
```

```bash
python -m k2_horizon lora-speicher --rang 16
```

`--aufwand low|medium|high` stellt ein, wie lange das Modell nachdenkt. Bei `low` lässt es das
Denken fast weg und verrechnet sich bei einfachen Textaufgaben. Bei `high` denkt es auf
Englisch nach, prüft sein Ergebnis noch einmal und antwortet auf Deutsch.

### Gemessen auf einer RTX 5080 (16 GB)

- **Erzeugen:** etwa 70 Tokens/s, 9,4 GB VRAM in bf16. Der erste Aufruf nach dem Laden ist
  deutlich langsamer.
- **LoRA-Training** (bf16, Rang 16, 34 Mio. trainierbare Parameter, Batch 1):

| Sequenzlänge | Spitze VRAM | Tokens/s |
|---|---|---|
| 1024 | 11,4 GB | 2 184 |
| 2048 | 11,6 GB | 1 974 |
| 4096 | 12,0 GB | 1 785 |
| 8192 | 13,4 GB | 1 578 |
| 16384 | reicht nicht | |

### Zwei Stolperfallen

1. **`model.train()` nicht vergessen.** Nach dem Laden steht das Modell im Auswertungsmodus, und
   dann wirkt Gradient Checkpointing nicht. Schon bei 512 Tokens ist der Speicher voll.
2. **Der Wortschatz ist riesig.** Er umfasst 250 624 Tokens. Für jede Position entsteht ein
   Logit-Vektor dieser Länge, und der normale Verlust von Transformers legt davon Kopien in
   float32 an. Damit war bei 1024 Tokens Schluss. [verlust.py](k2_horizon/verlust.py) rechnet
   den Verlust deshalb blockweise mit Checkpointing. Der Test
   [test_k2_verlust.py](tests/test_k2_verlust.py) prüft, dass Wert und Gradienten exakt dem
   normalen Verlust entsprechen.

Das Modell bringt eigenen Code mit (`trust_remote_code=True`). Er importiert nur torch,
transformers und math. Beim Laden erscheinen Warnungen zu `cache_position`. Sie betreffen den
Code von IFM und sind harmlos.

### Stufe 1: Wissen oder nachschlagen?

Das erste Training soll dem Modell ein **Verhalten** beibringen, keine Fakten: Was es sicher
weiß, beantwortet es direkt. Was es nicht weiß, sucht es nach, statt zu raten.

**Schritt 1, Wissensprüfung** ([wissen.py](k2_horizon/wissen.py)): Fragen aus
[PopQA](https://huggingface.co/datasets/akariasai/PopQA) mit bekannten Lösungen. Jede Frage
wird 5-mal ohne Suche gestellt. Bei mindestens 4 Treffern gilt sie als `gewusst`, bei
höchstens 1 als `unbekannt`, sonst als `unklar`.

```bash
curl -L -o /mnt/nvme2/daten/popqa/test.tsv https://huggingface.co/datasets/akariasai/PopQA/resolve/main/test.tsv
```

```bash
python -m k2_horizon wissen --anzahl 20000
```

```bash
python -m k2_horizon wissen --nur-zusammenfassung
```

Weggelassen sind Fragetypen, bei denen Raten oft trifft. Bei `sport` lautet z. B. 60 % der
Antworten „association football“, bei `religion` 40 % „Catholic Church“. Geprüft wird mit
`--aufwand low`: Das ist 35-mal schneller als `medium` (etwa 3 Fragen/s mit 5 Versuchen), und
beim reinen Faktenabruf hilft Nachdenken kaum.

Was der Probelauf mit 96 Fragen zeigt: Das Modell weiß nur 7 % der Antworten sicher. Bei
unbekannten Fakten **erfindet es meist bei jedem Versuch etwas anderes**. Bei einer
unbekannten Person kamen nacheinander Belém, Bujumbura, Masaka, Mweru und Yaoundé als
Geburtsort.

**Schritt 2, lokale Wikipedia-Suche** ([suche.py](k2_horizon/suche.py)): Die Grundlage ist
der englische Wikipedia-Abzug [wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia)
`20231101.en` (11,6 GB, 6,4 Mio. Artikel, CC-BY-SA). Einzeln abrufen lassen sich die Seiten
nicht: Wikipedia bremst schon nach wenigen Abrufen pro Sekunde.

```bash
hf download wikimedia/wikipedia --repo-type dataset --include "20231101.en/*" --local-dir /mnt/nvme2/daten/wikipedia
```

```bash
python -c "from k2_horizon.suche import bauen; bauen('/mnt/nvme2/daten/wikipedia/20231101.en', '/mnt/nvme2/daten/k2/wiki_index')"
```

```bash
python -m k2_horizon suche "Who was the director of Total Recall?"
```

```bash
python -m k2_horizon suche-messen --teil test
```

Der Index ist in 7 Minuten gebaut und belegt 21 GB, davon 19,6 GB für die Artikeltexte. BM25
ist von Hand in numpy geschrieben, die Formel steht oben in `suche.py`. Die Suche arbeitet in
zwei Stufen. **Stufe 1** findet 20 Artikel über Titel und die ersten 600 Zeichen. Ein Index über
alle Volltexte würde die 32 GB RAM sprengen. **Stufe 2** zerlegt diese Artikel in Abschnitte von
etwa 120 Wörtern und gibt die 5 besten zurück. Eine Suche dauert etwa eine halbe Sekunde.

Gemessen wird mit der PopQA-Frage als Anfrage, ob eine erlaubte Lösung in den Treffern steht:

| Stand | Artikel gefunden | Antwort in Treffern |
|---|---|---|
| reines BM25, 3 Treffer | 42 % | 34 % |
| + Titel aus der Frage bevorzugen, Verbformen ergänzen | 82 % | 43 % |
| + Titelbonus auch in Stufe 2, Artikelanfang bevorzugt | 82 % | 61 % |
| + 5 statt 3 Treffer, Fragewörter nicht als Titel | 83 % | **73 %** (Test: 71 %) |
| *Obergrenze: Antwort steht überhaupt im Artikel* | | *87 %* |

Was hat am meisten gebracht? Bei „Who was the director of Victory?“ heißt der Artikel
„Victory (1938 film)“, und „victory“ steht in Tausenden Artikeln. Artikel, deren Titel ohne
Klammerzusatz wörtlich in der Frage steht, bekommen deshalb einen Bonus. Viele der übrigen
Fehler sind echt mehrdeutig. Bei „Who was the screenwriter for Paradise?“ ist nicht
erkennbar, welcher Film gemeint ist. Hier muss das Modell später seine Anfrage verfeinern.

#### Schritt 3: Lösungswege erzeugen und auswählen

```bash
python -m k2_horizon laeufe
```

Das Modell beantwortet alle gewussten und 1.600 zufällige unbekannte Trainingsfragen mit Zugriff
auf die Suche (`agent.py`). Beim Erzeugen steht im Systemprompt ein Hinweis: „Wenn du dir nicht
sicher bist, such nach, sonst antworte direkt.“ In den gespeicherten Gesprächen fehlt er. Das
Modell soll das Verhalten später ohne Aufforderung zeigen. `datensatz.py` behält nur richtiges
Verhalten: gewusste Fragen ohne Suche richtig beantwortet, unbekannte mit mindestens einer Suche
richtig beantwortet. Mittlerer Denkaufwand, höchstens 3 Suchen, etwa 2,5 Stunden.

| | behalten | Hauptgrund für Verwerfen |
|---|---|---|
| gewusst | 285 / 424 (67 %) | 111× unnötig gesucht |
| unbekannt | 614 / 1.609 (38 %) | 645× nach 3 Suchen keine Antwort |

Bei 390 dieser 645 Abbrüche stand die Antwort schon in den Treffern. Das Modell übersieht sie
oder traut ihnen nicht. Unter den gewussten Beispielen sind 220 von 285 Hauptstädte, deshalb
wird nach Fragetyp getrennt ausgewertet.
