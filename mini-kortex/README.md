# Mini-Kortex

Ein Spielzeug-Prototyp für eine **hybride Denkarchitektur**: Statt eines großen Netzes arbeiten
mehrere kleine, spezialisierte Module zusammen, wie Regionen im Gehirn. Die Aufgabe ist
**6×6-Sudoku**. Das ist klein genug für die CPU, verlangt aber echte Schlussketten: Eine Ziffer
folgt aus einer anderen, die wiederum aus einer dritten folgt.

Das Projekt ist eigenständig und liegt nur im selben Repository wie der Bytes-Erkenner. Anders als
dieser nutzt es **PyTorch** statt reinem numpy. Ein rekursives Netz von Hand abzuleiten wäre
möglich, würde aber die eigentliche Idee unter sehr viel Rechenarbeit begraben.

## Die Module

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
| **Wahrnehmung** | [modell.py](mini_kortex/modell.py) | Sehrinde | Ziffer + Position jedes Feldes → Vektor |
| **Arbeitsraum** | [modell.py](mini_kortex/modell.py) | "Global Workspace" | Drei Tensoren, über die alle Module reden: `x` (Rätsel), `z` (latenter Gedanke), `y` (Antwort-Entwurf) |
| **Denkkern** | [modell.py](mini_kortex/modell.py) | Präfrontaler Kortex | Ein winziges Netz (2 Blöcke), das immer wieder auf seinen eigenen Zustand angewendet wird. Idee aus dem *Tiny Recursive Model* (TRM, 2025) |
| **Metakognition** | [modell.py](mini_kortex/modell.py) | Anteriorer cingulärer Kortex | Schätzt nach jedem Schritt: "Ist mein Entwurf komplett richtig?" |
| **Werkzeug** | [sudoku.py](mini_kortex/sudoku.py) | (keins) | Exakte Regelprüfung. Kein Lernen, keine Unsicherheit |
| **Gedächtnis** | [gedaechtnis.py](mini_kortex/gedaechtnis.py) | Hippocampus | Modernes Hopfield-Netz: speichert ein Erlebnis in einem Schritt, ruft über Ähnlichkeit ab |
| **Steuerung** | [agent.py](mini_kortex/agent.py) | | Verbindet alles zum Ablauf. Jedes Modul lässt sich abschalten |

## Schnellstart

Alle Befehle laufen im Ordner `mini-kortex`:

```bash
cd mini-kortex
```

```bash
pip install -r requirements.txt
```

```bash
python -m mini_kortex trainieren
```

Das dauert auf 4 CPU-Kernen etwa eine halbe Stunde. Beim ersten Mal werden zusätzlich die Rätsel
erzeugt (etwa 1 Minute) und als `daten_sudoku6.npz` abgelegt, das Modell landet in `mini_kortex.pt`.

```bash
python -m mini_kortex auswerten
```

```bash
python -m mini_kortex loesen --leer 26
```

`loesen` erzeugt ein neues Rätsel und zeigt nach jedem Denkschritt, wie viele Felder noch falsch
sind, wie sicher sich das Modell ist und was der Regelprüfer sagt. `beispiele` zeigt ein paar
Rätsel, `pytest` startet die Tests.

## Wie es funktioniert

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

**Denktiefe.** [sudoku.py](mini_kortex/sudoku.py) misst für jedes Rätsel, wie viele Runden einfacher
Logik ein Mensch mindestens braucht (nackte und versteckte Einer, alle gleichzeitig pro Runde).
"Probieren" heißt: Die einfachen Regeln reichen nicht, man muss weiter vorausdenken.

## Was man sieht

Nach 3000 Trainingsläufen (ca. 30 Minuten auf 4 CPU-Kernen) mit einem Modell von nur
**88.991 Parametern**, gemessen auf 2000 Testrätseln, die es beim Training nie gesehen hat:

**1. Längeres Nachdenken löst schwerere Rätsel.**

| Denktiefe | Anzahl | 1 Schritt | 4 Schritte | 16 Schritte | Ø Schritte mit Metakognition |
|---|---:|---:|---:|---:|---:|
| 1–3 | 1313 | 100 % | 100 % | 100 % | 1,0 |
| 4 | 355 | 98,6 % | 100 % | 100 % | 1,0 |
| 5 | 249 | 88,8 % | 100 % | 100 % | 1,1 |
| 6 und mehr | 74 | 77,0 % | 98,6 % | 100 % | 1,3 |
| Probieren | 9 | 44,4 % | 66,7 % | 100 % | 3,1 |

Leichte Rätsel schafft das Netz in einem Durchlauf. Je länger die Schlusskette, desto mehr
Denkschritte braucht es, und mit 16 Schritten löst es **alle** Testrätsel. Die rechte Spalte
zeigt, dass die Metakognition das selbst merkt: Bei leichten Rätseln hört sie nach einem Schritt
auf, bei den schwersten denkt sie im Schnitt dreimal so lange nach.

So sieht das bei einem schweren Rätsel aus (`python -m mini_kortex loesen --leer 28 --seed 71`):

```
Denkschritt  1:  7 Felder falsch · Zuversicht 13% · Regelprüfer: ungültig
Denkschritt  2:  5 Felder falsch · Zuversicht 2% · Regelprüfer: ungültig
Denkschritt  3:  0 Felder falsch · Zuversicht 48% · Regelprüfer: gültig ✔
```

**2. Jedes Modul bringt etwas anderes.**

| Konfiguration | gelöst | Ø Denkschritte | Ø Prüfungen |
|---|---:|---:|---:|
| Nur Denkkern, 1 Schritt | 97,2 % | 1,00 | 0 |
| Denkkern, immer 16 Schritte | 100 % | 16,00 | 0 |
| + Metakognition | 99,2 % | 1,04 | 0 |
| + Werkzeug (Regelprüfer) | **100 %** | **1,04** | 1,01 |
| Mit Gedächtnis, 2. Durchgang | 100 % | **0** | 1,00 |
| Verwandte Rätsel (+2 Vorgaben) | 100 % | **0** | 1,00 |

- **Metakognition** spart 94 % der Rechenzeit gegenüber "immer 16 Schritte". Allein ist sie
  aber manchmal zu selbstsicher: 0,8 % der Rätsel gibt sie falsch ab.
- **Werkzeug** schließt genau diese Lücke. Hält der Regelprüfer einen Entwurf für falsch, denkt das
  Modell weiter. Ergebnis: alles richtig bei praktisch gleicher Rechenzeit. Eine falsche Antwort
  mit Prüfsiegel kam nie vor (das prüft auch ein Test).
- **Gedächtnis** beantwortet bekannte Rätsel ganz ohne Nachdenken, auch wenn sie leicht
  verändert sind. Das gelang nach einem einzigen Speichervorgang, ohne Training.

Ohne Metakognition, aber mit Werkzeug (Prüfung nach jedem Schritt) kommt ebenfalls 100 % heraus,
mit 1,04 statt 1,01 Prüfungen pro Rätsel. Hier ist der Unterschied klein, weil fast jedes Rätsel
nach Schritt 1 stimmt. Bei teuren Werkzeugen (ein Beweiser, ein Experiment) wäre er groß.

Eine Einschränkung: 6×6-Sudoku ist für dieses Modell nicht mehr schwer. Die schwierigen Fälle sind
selten (83 von 2000), und deshalb liegen alle Varianten nah beieinander. Wer die Unterschiede
deutlicher sehen will, sollte 9×9 probieren.

## Was fehlt (und Ideen zum Weiterprobieren)

Der Prototyp zeigt die Bausteine im Kleinen. Weltmodell, Sprachkern und Wertsystem der großen
Architektur fehlen, weil Sudoku sie nicht braucht.

- **Konsolidierung ("Schlaf")**: Rätsel, die nur mit vielen Schritten gelöst wurden, aus dem
  Gedächtnis nachtrainieren. Wird das Modell dadurch schneller, ohne Altes zu vergessen?
- `--schritte 16` beim Training: Lernt das Modell tiefere Schlussketten?
- `--d 32` gegen `--d 128`: Wie klein darf der Arbeitsraum werden?
- Die Metakognition mit `schwelle=0.9` statt `0.5` in [agent.py](mini_kortex/agent.py): weniger
  Fehler, mehr Schritte?
- 9×9-Sudoku: In `sudoku.py` reicht es, `N` und die Kastengröße zu ändern. Das Training wird aber
  deutlich länger.
- Den MLP-Mixer in `modell.py` durch Attention ersetzen und vergleichen.
