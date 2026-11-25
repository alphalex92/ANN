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

## Projekt 2: Denkkern

Ein Spielzeug-Prototyp für eine **hybride Denkarchitektur**: Statt eines großen Netzes arbeiten
mehrere kleine, spezialisierte Module zusammen, wie Regionen im Gehirn. Die Aufgabe ist
**6×6-Sudoku**. Das ist klein genug für die CPU, verlangt aber echte Schlussketten: Eine Ziffer
folgt aus einer anderen, die wiederum aus einer dritten folgt.

Hier kommt **PyTorch** zum Einsatz statt reinem numpy. Ein rekursives Netz von Hand abzuleiten wäre
möglich, würde aber die eigentliche Idee unter sehr viel Rechenarbeit begraben.

### Die Module

```
            Rätsel
              │
              ▼
   ┌──────────────────┐        ┌─────────────────────────┐
   │  Wahrnehmung     │        │  Gedächtnis (Hopfield)  │──► Kenne ich das? → Werkzeug prüft → fertig
   └────────┬─────────┘        └─────────────────────────┘
            ▼                                ▲ jede geprüfte Lösung
   ┌──────────────────────────────────┐      │ wird sofort gespeichert
   │  Arbeitsraum:  x · z · y          │──────┘
   │  (Rätsel · Gedanke · Entwurf)     │
   └───┬───────────────▲──────────┬───┘
       │ Denkschritt   │          │
       ▼               │          ▼
   ┌────────────────┐  │   ┌──────────────────┐   sicher genug?   ┌─────────────────┐
   │ Denkkern       │──┘   │ Metakognition    │──────────────────►│ Werkzeug        │
   │ (rekursiv)     │      │ "Wie sicher?"    │◄──── falsch: ─────│ (Regelprüfer)   │
   └────────────────┘      └──────────────────┘    weiterdenken   └─────────────────┘
```

| Modul | Datei | Vorbild im Gehirn | Was es tut |
|---|---|---|---|
| **Wahrnehmung** | [modell.py](denkkern/modell.py) | Sehrinde | Ziffer + Position jedes Feldes → Vektor |
| **Arbeitsraum** | [modell.py](denkkern/modell.py) | "Global Workspace" | Drei Tensoren, über die alle Module reden: `x` (Rätsel), `z` (latenter Gedanke), `y` (Antwort-Entwurf) |
| **Denkkern** | [modell.py](denkkern/modell.py) | Präfrontaler Kortex | Ein winziges Netz (2 Blöcke), das immer wieder auf seinen eigenen Zustand angewendet wird. Idee aus dem *Tiny Recursive Model* (TRM, 2025) |
| **Metakognition** | [modell.py](denkkern/modell.py) | Anteriorer cingulärer Kortex | Schätzt nach jedem Schritt: "Ist mein Entwurf komplett richtig?" |
| **Werkzeug** | [sudoku.py](denkkern/sudoku.py) | (keins) | Exakte Regelprüfung. Kein Lernen, keine Unsicherheit |
| **Gedächtnis** | [gedaechtnis.py](denkkern/gedaechtnis.py) | Hippocampus | Modernes Hopfield-Netz: speichert ein Erlebnis in einem Schritt, ruft über Ähnlichkeit ab |
| **Steuerung** | [agent.py](denkkern/agent.py) | | Verbindet alles zum Ablauf. Jedes Modul lässt sich abschalten |

### Schnellstart

```bash
pip install -r requirements.txt
```

```bash
python -m denkkern trainieren
```

Das dauert auf 4 CPU-Kernen etwa eine halbe Stunde. Beim ersten Mal werden zusätzlich die Rätsel
erzeugt (etwa 1 Minute) und als `daten_sudoku6.npz` abgelegt.

```bash
python -m denkkern auswerten
```

```bash
python -m denkkern loesen --leer 26
```

`loesen` erzeugt ein neues Rätsel und zeigt nach jedem Denkschritt, wie viele Felder noch falsch
sind, wie sicher sich das Modell ist und was der Regelprüfer sagt. `beispiele` zeigt ein paar
Rätsel, `pytest` startet die Tests.

### Wie es funktioniert

**Rekursives Denken.** Ein gewöhnliches Netz rechnet einmal von vorne nach hinten und ist fertig.
Der Denkkern hat nur zwei Schichten, durchläuft sie aber immer wieder:

```
pro Denkschritt:   3 ×  z ← f(x + y + z)     den Gedanken verfeinern
                   1 ×  y ← f(y + z)         den Entwurf anpassen
```

Dasselbe kleine Netz `f` macht beides. 16 Denkschritte entsprechen 64 Durchläufen, also einem Netz
mit 128 Schichten, aber mit den Parametern von 2 Schichten. **Tiefe kostet Rechenzeit, nicht
Speicher.**

**Tiefe Überwachung.** Beim Training soll die Antwort nach *jedem* Denkschritt schon möglichst
stimmen. Nach jedem Schritt wird zurückgerechnet und der Gradient dann gekappt (`detach`). So muss
die Backpropagation nie durch alle 8 Schritte gleichzeitig laufen, und das Netz lernt, einen
Entwurf schrittweise zu verbessern.

**Metakognition.** Nach jedem Schritt prüfen wir beim Training, ob der Entwurf stimmt (1) oder
nicht (0). Ein kleiner Kopf lernt, das vorherzusagen. Später entscheidet er, wann sich weiteres
Nachdenken nicht mehr lohnt.

**Werkzeug.** Das Netz *schätzt* nur. Der Regelprüfer *weiß*. Die Metakognition entscheidet, wann
geprüft wird. Ist der Entwurf falsch, denkt das Modell weiter, statt Unsinn abzugeben.

**Gedächtnis.** Abruf nach der Formel `softmax(β · Schlüssel · Anfrage) · Werte`. Das ist die
Update-Regel moderner Hopfield-Netze und zugleich die Attention-Formel eines Transformers. Speichern
ist ein einziger Schritt ohne Training: Lernen aus einem Beispiel.

**Denktiefe.** [sudoku.py](denkkern/sudoku.py) misst für jedes Rätsel, wie viele Runden einfacher
Logik ein Mensch mindestens braucht (nackte und versteckte Einer, alle gleichzeitig pro Runde).
"Probieren" heißt: Die einfachen Regeln reichen nicht, man muss weiter vorausdenken.

### Was man sieht

(Messwerte folgen nach dem ersten vollständigen Training.)

### Was fehlt (und Ideen zum Weiterprobieren)

Der Prototyp zeigt die Bausteine im Kleinen. Weltmodell, Sprachkern und Wertsystem der großen
Architektur fehlen, weil Sudoku sie nicht braucht.

- **Konsolidierung ("Schlaf")**: Rätsel, die nur mit vielen Schritten gelöst wurden, aus dem
  Gedächtnis nachtrainieren. Wird das Modell dadurch schneller, ohne Altes zu vergessen?
- `--schritte 16` beim Training: Lernt das Modell tiefere Schlussketten?
- `--d 32` gegen `--d 128`: Wie klein darf der Arbeitsraum werden?
- Die Metakognition mit `schwelle=0.9` statt `0.5` in [agent.py](denkkern/agent.py): weniger
  Fehler, mehr Schritte?
- 9×9-Sudoku: In `sudoku.py` reicht es, `N` und die Kastengröße zu ändern. Das Training wird aber
  deutlich länger.
- Den MLP-Mixer in `modell.py` durch Attention ersetzen und vergleichen.
