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

### Forschungsschleife (nach dem Vorbild von autoresearch)

Hier verbessert nicht das Netz sich selbst. Ein **KI-Agent** (z. B. Claude Code) spielt den
Forscher und arbeitet am Code des Netzes. Er macht eine Änderung, misst, behält oder verwirft
sie und fängt von vorne an. Vorbild ist
[karpathy/autoresearch](https://github.com/karpathy/autoresearch), nur dauert ein Experiment hier
Sekunden statt 5 Minuten.

| Datei | Wer darf ändern |
|---|---|
| [forschung/program.md](forschung/program.md) | **du**: der Forschungsauftrag, dein eigentlicher Hebel |
| [forschung/auswertung.py](forschung/auswertung.py) | niemand: feste Messung |
| `bytes_erkenner/netz.py`, `bytes_erkenner/training.py` | **der Agent** |
| `bytes_erkenner/daten.py`, `tests/` | niemand |
| `ergebnisse.tsv` | der Agent, als Logbuch (nicht im Git) |

**Messung:** `python -m forschung.auswertung` ruft `fuer_auswertung(sekunden=10, seed)` für
3 Seeds auf, jeweils in einem eigenen Prozess. Bewertet wird auf einem festen Testset mit
7 × 2000 Beispielen. Ergebnis ist die mittlere Genauigkeit. Die Baseline liegt bei **0.933**,
die Streuung bei etwa 0.004, und das Training nutzt erst 0.4 von 10 Sekunden.

**Schutz gegen Schummeln:** Wie im
[Weco-Artikel](https://www.weco.ai/blog/first-evidence-of-recursive-self-improvement)
gibt es drei Schichten:
1. *Anweisung:* `program.md` verbietet z. B., auf dem Testset zu trainieren.
2. *Hart:* Die Auswertung gibt `ungueltig` zurück, wenn eine gesperrte Datei vom Tag
   `forschung-start` abweicht (auch neue Dateien) oder das Zeitbudget überschritten ist.
   Abstürze ergeben `crash`.
3. *Statistisch:* `python -m forschung.auswertung --kontrolle` misst auf einer Kontrollmenge
   mit eigenem Seed. Setz dir einen geheimen Seed (`KONTROLL_SEED=… python -m …`). Steigt im
   Logbuch nur das Test-Ergebnis, aber nicht die Kontrolle, hat sich der Agent ans Testset
   angepasst.

**Starten:** Öffne eine neue Claude-Code-Session in diesem Ordner, am besten im Auto-Modus,
damit sie nicht bei jedem Befehl fragt. Dann schreibst du:

> Lies forschung/program.md und leg los.

Der Agent legt einen Branch `forschung/<datum>` an, misst die Baseline und arbeitet
30 Experimente ab. Danach kannst du `ergebnisse.tsv` und `git log` ansehen und die
Kontrollmenge messen.

Der Agent führt selbst geschriebenen Code aus. Er bleibt zwar im Repo, aber ein Sandbox gibt
es nicht. Schau also ab und zu rein.

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

## Projekt 2: Ein eigenes kleines Sprachmodell (GPT)

Ein **Decoder-only Transformer** wie GPT, nur in Klein. Er lernt auf rund 700 MB deutscher
Wikipedia, das jeweils nächste Stück Text vorherzusagen, und kann danach Text fortsetzen.
Alles ist selbst geschrieben, vom Tokenizer bis zur Trainingsschleife. PyTorch übernimmt nur
die Gradienten und die GPU. Vorbild sind [nanoGPT](https://github.com/karpathy/nanoGPT) und
[nanochat](https://github.com/karpathy/nanochat).

### Schnellstart

Du brauchst eine NVIDIA-GPU. Für die RTX-50er-Reihe muss es PyTorch mit CUDA 12.8 oder
neuer sein. PyTorch gibt es noch nicht für Python 3.14, darum nehmen wir 3.12:

```bash
uv venv --python 3.12 .venv
```

```bash
uv pip install -r requirements-sprachmodell.txt
```

```bash
python -m sprachmodell vorbereiten
```

Lädt ~460 MB von Hugging Face nach `~/.cache/ann-sprachmodell`, lernt den Tokenizer und
kodiert alles (~1 Minute).

```bash
python -m sprachmodell trainieren
```

Trainiert 5 Minuten (plus ~30 s, in denen `torch.compile` das Modell übersetzt) und speichert
`sprachmodell.pt`.

```bash
python -m sprachmodell erzeugen "Die Stadt liegt am Ufer"
```

`tokens "irgendein Text"` zeigt, wie der Tokenizer zerlegt, und `messen` gibt den val_bpb des
gespeicherten Modells aus.

### Wie es funktioniert

**1. Tokenizer** ([tokenizer.py](sprachmodell/tokenizer.py)): Das Netz rechnet mit Zahlen,
also wird Text zuerst in **Tokens** zerlegt. Wir nehmen **Byte-Pair-Encoding** (BPE) wie
GPT-2. Am Anfang ist jedes der 256 Bytes ein Token. Dann wird immer wieder das häufigste
benachbarte Paar zu einem neuen Token verschmolzen, bis es 4096 Tokens gibt:

```
$ python -m sprachmodell tokens "Die Wikipedia ist eine freie Enzyklopädie."
Die |  W | ik | ip | ed | ia |  ist |  eine |  f | re | ie |  En | zy | kl | op | äd | ie | .
18 Tokens für 43 Bytes, IDs: [539, 329, 377, 950, 466, 524, 406, 437, 339, ...]
```

Häufige Wörter wie „ ist“ werden zu einem einzigen Token, seltene zerfallen in Stücke. Das
Leerzeichen gehört dabei zum folgenden Wort. Weil die
256 Bytes immer übrig bleiben, lässt sich jeder Text kodieren, auch Emoji. Ein Token steht im
Schnitt für **3,2 Bytes** deutschen Text. Zusätzlich gibt es das Sondertoken `<|dok|>`. Es
markiert den Anfang jedes Artikels.

**2. Daten** ([daten.py](sprachmodell/daten.py)): Die Artikel werden mit festem Seed gemischt.
3000 davon (2,6 Mio. Tokens) sind die **Validierungsmenge**, die das Modell nie zum Lernen
sieht. Die übrigen 230 Mio. Tokens sind Trainingsmaterial. Ein Trainingsbeispiel ist ein
zufälliger Ausschnitt von 512 Tokens. Das Ziel ist derselbe Ausschnitt, um ein Token
verschoben. So enthält jeder Ausschnitt 512 Aufgaben der Art „Was kommt als Nächstes?“.

**3. Modell** ([modell.py](sprachmodell/modell.py)):

```
Token-IDs ──► Einbettung + Position ──► 8 × Block ──► LayerNorm ──► Logits (4096 je Stelle)

Block:  x = x + Attention(LayerNorm(x))     Tokens tauschen Information aus
        x = x + MLP(LayerNorm(x))           jedes Token rechnet für sich weiter
```

- **Einbettung (Embedding):** Eine lernbare Tabelle mit einem Vektor aus 512 Zahlen pro
  Token. Tokens mit ähnlicher Bedeutung bekommen im Lauf des Trainings ähnliche Vektoren. Ein
  zweiter Vektor pro Position sagt dem Modell, *wo* ein Token steht.
- **Self-Attention:** Jedes Token bildet eine Frage (Query), einen Schlüssel (Key) und einen
  Wert (Value). Passt die Frage eines Tokens gut zum Schlüssel eines früheren, holt es sich
  dessen Wert. So kann „sie“ nachsehen, wer gemeint ist. Die Attention ist **kausal**: Jedes
  Token sieht nur seine Vorgänger, sonst könnte es beim Training die Antwort abschreiben.
  8 **Köpfe** machen das parallel, jeder achtet auf etwas anderes. In `SelbstAufmerksamkeit`
  steht die Rechnung einmal ausgeschrieben (`manuell=True`) und einmal als schneller
  Flash-Attention-Aufruf. Ein Test prüft, dass beide dasselbe ergeben.
- **MLP:** Pro Token ein kleines Netz wie im Bytes-Erkenner (512 → 2048 → 512 mit GELU).
- **Residualverbindung** (`x = x + …`): Jeder Block ergänzt den Vektor nur. Dadurch kommen die
  Gradienten auch durch viele Blöcke gut zurück.
- **Logits:** Am Ende gibt es für jede Stelle 4096 Werte, einen pro möglichem nächsten Token.
  Softmax macht daraus Wahrscheinlichkeiten.

Insgesamt sind das **27 Mio. Parameter**, rund 5000-mal weniger als GPT-3.

**4. Verlust und Training** ([training.py](sprachmodell/training.py)): Der Verlust ist wie
beim Bytes-Erkenner die **Kreuzentropie**: −log(Wahrscheinlichkeit des richtigen nächsten
Tokens). Ein frisches Modell rät gleichverteilt und liegt bei ln(4096) ≈ 8,3. Pro Schritt
sieht das Modell 64 Ausschnitte (32.768 Tokens). PyTorch rechnet die Gradienten aus, und
**AdamW** passt die Gewichte an. Damit das schnell geht, rechnet die GPU in **bfloat16**,
und `torch.compile` fasst Rechenschritte zu eigenen GPU-Kernen zusammen.

Trainiert wird nicht eine feste Anzahl Epochen, sondern ein **festes Zeitbudget von
5 Minuten**. Die Lernrate fährt in den ersten 3 % hoch und fällt danach linear auf 0. So
lassen sich verschiedene Ideen fair vergleichen: Ein größeres Modell lernt pro Schritt mehr,
schafft aber weniger Schritte.

**5. Messung: val_bpb** (`daten.val_bpb`): **Bits pro Byte** auf der Validierungsmenge. Die
Formel lautet: Summe der Verluste ÷ ln 2 ÷ Anzahl der Bytes Text. Das ist die Größe, auf die
das Modell den Text komprimieren könnte. 8 hieße, das Modell hat nichts gelernt. gzip
schafft etwa 2,5 bis 3, große LLMs liegen unter 0,7. Anders als der Verlust pro Token hängt bpb
nicht vom Tokenizer ab. Später darf eine Forschungsschleife also auch den Tokenizer
umbauen, und die Ergebnisse bleiben vergleichbar.

**6. Erzeugen:** Das Modell sagt eine Verteilung für das nächste Token voraus. Wir ziehen
daraus ein Token, hängen es an und wiederholen das. Die **Temperatur** (`--temperatur`) macht
die Verteilung spitzer (< 1, vorsichtiger) oder flacher (> 1, wilder). `--top-k` lässt nur
die k wahrscheinlichsten Tokens zu.

### Was man sieht

Auf einer RTX 5080 schafft das Training etwa **380.000 Tokens pro Sekunde**. In 5 Minuten
sind das rund 3500 Schritte mit 115 Mio. Tokens, also die Hälfte der Trainingsdaten:

| Zeit | Verlust | val_bpb | was das Modell kann |
|---|---|---|---|
| 0 s | 8,3 | ~ 3,8 | nichts, rät gleichverteilt |
| 60 s | ~ 3,6 | 2,13 | schon besser als gzip |
| 300 s | ~ 2,65 | **1,24** | Grammatik, Wikipedia-Aufbau, erfundene Fakten |

Mit `python -m sprachmodell erzeugen "Die Stadt liegt am Ufer"` kommt zum Beispiel heraus:

> Die Stadt liegt am Ufer des Feiertales. Die Siedlung besteht aus einem Dorf (etwa 4 km
> nordöstlich) an der westlichen Randhöhe von Wolga in der Kernstadt an der Kastlachstraße.
>
> Lage
> Die Gemeinde liegt südwestlich der Stadtkernstadt Wolga (Kastlach) an der Grenze zu
> Remscheid und hat eine Fläche von 24,15 km².

Ohne Anfangstext erfindet es ganze Artikel:

> Dieter Kaufmann, eigentlich Josef Kaufmann, war ein jüdischer Maler aus St. Petersburg […]
>
> Leben
> Kaufmann war der Sohn des Kunsthistorikers und Malers Kaufmann und bildete das Studium in
> München, das er 1893 aufstieg.

Die Sätze sind meist grammatisch richtig, und das Modell kennt den Aufbau eines
Wikipedia-Artikels mit Abschnitten wie „Lage“ und „Leben“ und mit Flächenangaben in km².
Inhaltlich ist es reine Erfindung: Die Wolga fließt nicht bei Remscheid. Genau das meint man,
wenn man sagt, ein Sprachmodell **halluziniert**. Es hat gelernt, wie Text *aussieht*, nicht,
was stimmt.

Der Trainings-Verlust fällt am Anfang steil und flacht dann ab. Da das Modell jedes Token
höchstens einmal sieht, gibt es hier keine Überanpassung: Trainings- und
Validierungsverlust liegen gleichauf. Ein größeres Modell oder mehr Zeit würde weiter
helfen.

### Ideen zum Weiterprobieren

- `--sekunden 60` gegen `--sekunden 900`: Wie viel bringt mehr Zeit?
- In `training.py` die `KONFIG` ändern, etwa 4 Blöcke mit Breite 768 oder 12 Blöcke mit
  Breite 384. Was ist in 5 Minuten besser, breit oder tief?
- Die Temperatur beim Erzeugen auf 0,3 oder 1,5 setzen.
- Positions-Einbettung durch **RoPE** ersetzen, LayerNorm durch **RMSNorm**: Das machen
  moderne Modelle wie Llama.
- Einen **KV-Cache** für `erzeugen` bauen, damit nicht jedes Mal der ganze Text neu
  gerechnet wird.
- **Der Crawler träumt:** Das Modell auf die Event-Streams aus dem Crawler-Projekt
  nachtrainieren (Fine-Tuning) und neue Replays erfinden lassen.
- Die Forschungsschleife aus Projekt 1 auf das Sprachmodell anwenden.
  `training.fuer_auswertung(sekunden, seed)` ist dafür schon vorbereitet, gemessen wird mit
  `daten.val_bpb`.
