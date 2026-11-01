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
