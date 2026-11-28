# Mini-Kortex

Ein Spielzeug-Prototyp für eine **hybride Denkarchitektur**: Statt eines großen Netzes arbeiten
mehrere kleine, spezialisierte Module zusammen, wie Regionen im Gehirn. Die Aufgabe ist
**Sudoku** (9×9, zum schnellen Ausprobieren auch 6×6). Das ist klein genug für die CPU, verlangt
aber echte Schlussketten: Eine Ziffer folgt aus einer anderen, die wiederum aus einer dritten folgt.

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
| **Konsolidierung** | [konsolidierung.py](mini_kortex/konsolidierung.py) | Schlaf, Hippocampus-Replay | Trainiert das Netz auf eigenen, geprüften Erlebnissen nach, vermischt mit alten Rätseln |
| **Probieren** | [weltmodell.py](mini_kortex/weltmodell.py) | Vorstellungskraft | Trifft Annahmen und lässt das Netz die Folgen durchdenken |
| **Auge** | [sehen.py](mini_kortex/sehen.py) | Sehrinde | Kleines CNN, liest handgeschriebene Ziffern (MNIST). Irrt sich manchmal |
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

Das trainiert ein 9×9-Modell, bewusst nur auf Rätseln mit höchstens 46 leeren Feldern. Die
schwereren bleiben für die Konsolidierung übrig. Auf 4 CPU-Kernen dauert das gut eine Stunde.
Beim ersten Mal werden zusätzlich 42.000 Rätsel erzeugt (etwa 1 Minute) und als
`daten_sudoku9.npz` abgelegt, das Modell landet in `mini_kortex_9.pt`. Alle 250 Läufe wird ein
Zwischenstand gespeichert. Bricht das Training ab, setzt derselbe Befehl dort fort.

```bash
python -m mini_kortex auswerten
```

```bash
python -m mini_kortex konsolidieren
```

```bash
python -m mini_kortex loesen
```

`konsolidieren` spielt drei Wach-Schlaf-Zyklen durch (etwa 30 Minuten) und speichert das Ergebnis
als `mini_kortex_9_konsolidiert.pt`. `loesen` erzeugt ein neues Rätsel und zeigt nach jedem
Denkschritt, wie viele Felder noch falsch sind, wie sicher sich das Modell ist und was der
Regelprüfer sagt. Mit `--modell mini_kortex_9_konsolidiert.pt` schaut man dem ausgeschlafenen
Modell zu. `beispiele` zeigt ein paar Rätsel, `pytest` startet die Tests.

Findet PyTorch eine GPU, rechnet alles automatisch dort. `--geraet cpu` vor dem Befehl erzwingt die
CPU, `--geraet cuda` die GPU. Beim Training legt `--batch` fest, wie viele Rätsel ein Lauf enthält
(Standard 128). Auf der GPU lohnen sich meist größere Batches. Gespeicherte Modelle laden auf beiden.

Für das visuelle Sudoku (`python -m mini_kortex sehen`) braucht es die vier MNIST-Dateien
(`*-ubyte.gz`, zusammen 11 MB, zum Beispiel von https://storage.googleapis.com/cvdf-datasets/mnist/)
in einem Ordner `mnist`. Einen anderen Ort nennt die Umgebungsvariable `MINI_KORTEX_MNIST`. Das Auge
wird bei jedem Aufruf neu trainiert. `--epochen 1 4` vergleicht ein mäßiges mit einem guten Auge.

Mit `--groesse 6` vor dem Befehl läuft alles auf 6×6, das Training dauert dann etwa eine halbe
Stunde (zum Beispiel `python -m mini_kortex --groesse 6 trainieren`).

## Wie es funktioniert

**Rekursives Denken.** Ein gewöhnliches Netz rechnet einmal von vorne nach hinten und ist fertig.
Der Denkkern hat nur zwei Schichten, durchläuft sie aber immer wieder:

```
pro Denkschritt:   3 ×  z ← f(x + y + z)     den Gedanken verfeinern
                   1 ×  y ← f(y + z)         den Entwurf anpassen
```

Dasselbe kleine Netz `f` macht beides. 32 Denkschritte entsprechen 128 Durchläufen, also einem Netz
mit 256 Schichten, aber mit den Parametern von 2 Schichten. **Tiefe kostet Rechenzeit, nicht
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

**Konsolidierung.** Vorbild ist die Theorie der *komplementären Lernsysteme* (McClelland,
McNaughton & O'Reilly 1995): Der Hippocampus speichert Erlebnisse sofort, der Kortex lernt langsam,
und im Schlaf werden Erlebnisse erneut abgespielt. Ein Zyklus besteht aus zwei Phasen:

1. **Wach:** Das Modell bekommt neue, *schwerere* Rätsel als im Training. Mit bis zu 32
   Denkschritten, Metakognition und Regelprüfer löst es einen Teil davon. Jede bestätigte Lösung
   kommt ins Gedächtnis, zusammen mit der Zahl der Denkschritte.
2. **Schlaf:** Das Netz wird auf den Episoden nachtrainiert, die mindestens 2 Denkschritte gekostet
   haben. Die "richtigen Antworten" stammen nur aus eigener, vom Regelprüfer bestätigter Erfahrung;
   die echten Lösungen werden nur zum Messen benutzt. Jede Episode wird in Symmetrie-Varianten
   abgespielt, und jeder Batch ist zur Hälfte mit alten Trainingsrätseln gemischt (Replay).

Gemessen wird danach auf *anderen*, nie gesehenen Rätseln. Die Idee ähnelt *Expert Iteration*
(Anthony u. a. 2017) und *STaR* (Zelikman u. a. 2022): Langsames, gründliches Nachdenken wird zu
schneller Intuition.

**Symmetrien.** Ziffern umbenennen, Zeilen innerhalb eines Bandes tauschen, Bänder tauschen
(dasselbe für Spalten) und an der Diagonale spiegeln: Das Rätsel bleibt gleich schwer, sieht aber
anders aus. Beim Training wird jeder Batch so umgebaut, das Netz sieht dasselbe Rätsel praktisch nie
zweimal.

**Denktiefe.** [sudoku.py](mini_kortex/sudoku.py) misst für jedes Rätsel, wie viele Runden einfacher
Logik ein Mensch mindestens braucht (nackte und versteckte Einer, alle gleichzeitig pro Runde).
"Probieren" heißt: Die einfachen Regeln reichen nicht, man muss weiter vorausdenken.

## Ergebnisse 9×9

Ein Modell mit **134.648 Parametern**, trainiert in 3000 Läufen (62 Minuten auf 4 CPU-Kernen),
nur auf Rätseln mit 30–46 leeren Feldern. Gemessen auf 2000 nie gesehenen Rätseln mit 30–58
leeren Feldern, also auch auf deutlich schwereren als im Training.

**1. Längeres Nachdenken hilft, bis die Schlusskette zu lang wird.**

| Denktiefe | Anzahl | 1 Schritt | 8 Schritte | 32 Schritte | Ø Schritte mit Metakognition |
|---|---:|---:|---:|---:|---:|
| 1–3 | 805 | 100 % | 100 % | 100 % | 1,0 |
| 4–6 | 618 | 56,5 % | 97,6 % | 97,9 % | 1,9 |
| 7–9 | 240 | 1,7 % | 46,2 % | 47,5 % | 15,6 |
| 10 und mehr | 141 | 0 % | 9,2 % | 9,9 % | 27,2 |
| Probieren | 196 | 0 % | 7,1 % | 14,8 % | 27,4 |

Bei mittlerer Tiefe macht Nachdenken aus 57 % fast 98 %. Bei langen Schlussketten stößt das kleine
Netz an seine Grenze: Über 8 Schritte hinaus bringt Weiterdenken kaum noch etwas. Das liegt
vermutlich auch daran, dass beim Training nur 8 Schritte geübt wurden. Die Metakognition merkt, was
schwer ist: Bei leichten Rätseln hört sie nach einem Schritt auf, bei den schwersten denkt sie bis
zum Limit.

**2. Jetzt zeigt sich der Nutzen jedes Moduls.**

| Konfiguration | gelöst | Ø Denkschritte | Ø Prüfungen |
|---|---:|---:|---:|
| Nur Denkkern, 1 Schritt | 57,9 % | 1,00 | 0 |
| Denkkern, immer 32 Schritte | 78,3 % | 32,00 | 0 |
| + Metakognition | 78,0 % | **7,46** | 0 |
| + Werkzeug (Regelprüfer) | **81,4 %** | 7,63 | **0,86** |
| Werkzeug ohne Metakognition | 82,3 % | 7,30 | 7,30 |
| Mit Gedächtnis, 2. Durchgang | 81,4 % | 5,95 | 0,82 |
| Verwandte Rätsel (+2 Vorgaben) | 92,0 % | 3,15 | 0,95 |

- **Metakognition** erreicht dasselbe wie "immer 32 Schritte" mit einem Viertel der Rechenzeit.
- **Werkzeug** hebt die Quote um 3,4 Prozentpunkte: Ist ein Entwurf falsch, denkt das Modell weiter,
  statt abzugeben. Eine falsche Antwort mit Prüfsiegel kam nie vor.
- **Metakognition + Werkzeug**: Prüft man nach jedem Schritt, ist man 0,9 Prozentpunkte besser, braucht
  aber **8,5-mal so viele Prüfungen**. Das ist der Unterschied, der bei 6×6 noch unsichtbar war. Bei
  einem teuren Werkzeug (ein Beweiser, ein echtes Experiment) zählt genau das.
- **Gedächtnis** beantwortet im zweiten Durchgang alle schon gelösten Rätsel (81 %) ohne
  Nachdenken. Die übrigen 19 % kosten weiter das volle Limit, deshalb bleiben im Schnitt 5,95
  Schritte. Verwandte Rätsel erkennt es ebenso wieder, und durch die zwei zusätzlichen Vorgaben
  löst das Modell noch einige der vorher ungelösten.

## Ergebnisse Konsolidierung

Ausgangspunkt ist das Modell von oben. An jedem "Tag" erlebt es 1000 neue Rätsel mit 47–58 leeren
Feldern, also schwerer als alles im Training. Jede "Nacht" hat 300 Trainingsläufe. Gemessen wird
auf je 1000 anderen schweren und leichten Rätseln, die nie erlebt wurden.

| | schwer, 8 Schritte | schwer, Agent | leicht, 1 Schritt | leicht, 8 Schritte |
|---|---:|---:|---:|---:|
| Vor dem ersten Schlaf | 47,1 % | 55,8 % | 90,9 % | 100 % |
| Nacht 1, mit Replay | 74,1 % | 83,3 % | 90,3 % | 99,9 % |
| Nacht 2, mit Replay | 81,7 % | 89,7 % | 90,3 % | 100 % |
| Nacht 3, mit Replay | **83,4 %** | **91,3 %** | 90,4 % | 99,9 % |
| *Zum Vergleich: Nacht 1, nur neue Erlebnisse* | 75,5 % | 86,1 % | 91,2 % | 100 % |

"Agent" heißt: bis zu 32 Schritte mit Metakognition und Regelprüfer.

| Wachphase | selbst gelöst und geprüft | neu zum Üben (≥ 2 Schritte) | falsch, aber geprüft |
|---|---:|---:|---:|
| Tag 1 | 53,3 % | 444 | 0 |
| Tag 2 | 83,7 % | 738 | 0 |
| Tag 3 | 90,3 % | 796 | 0 |

- **Das Modell lernt aus eigener Erfahrung.** Auf ungesehenen schweren Rätseln steigt die Quote in
  drei Nächten von 56 % auf 91 %, ganz ohne Lösungen von außen. Jeden Tag löst es mehr der neuen
  Rätsel selbst und hat damit in der nächsten Nacht mehr zu üben.
- **Aus langsamem Denken wird schnelle Intuition.** Mit festen 8 Schritten steigt die Quote von 47 %
  auf 83 %: Was vorher nur mit langem Nachdenken und Prüfen klappte, schafft das Netz jetzt in
  wenigen Schritten.
- **Vergessen trat nicht auf**, auch nicht ohne Replay. Die leichten Rätsel bleiben bei 90–91 % in
  einem Schritt und 100 % in acht. Replay war hier also nicht nötig. Ohne Replay war das Modell auf
  den schweren Rätseln nach Nacht 1 sogar etwas besser, weil jeder Batch doppelt so viele neue
  Erlebnisse enthielt. Vermutlich liegt das daran, dass schwere und leichte Rätsel nach denselben
  Regeln funktionieren und sich das Üben nicht widerspricht. Vergessen erwartet man, wenn das Neue
  dem Alten widerspricht.

Eine Einschränkung: Das Modell kann nur aus Rätseln lernen, die es mit viel Bedenkzeit schon
selbst lösen kann. Wirklich Neues (andere Regeln) kann es so nicht lernen. Wie nah das Selbstlernen
an Lernen mit Lehrer herankommt, zeigt der Lehrer-Vergleich weiter unten.

## Ergebnisse auf der GPU: breiter und tiefer

Auf einer RTX 5080 ist das Training bei Batch 128 etwa 16-mal schneller als auf 4 CPU-Kernen:
3000 Läufe dauern 2 statt 37 Minuten, mit `--d 128 --schritte 16` sind es 7 Minuten. Damit ließen
sich vier Varianten mit je drei Seeds vergleichen (`--seed 0/1/2`, gleiche Rätsel, andere Startwerte
und Reihenfolge der Batches). Alles andere wie oben: 3000 Läufe, Batch 128, dieselben 2000 Testrätsel.

| Variante | Parameter | Agent (alle Module) | Ø Schritte | Tiefe 7–9, 8 Schritte | Tiefe 10+, 8 Schritte | immer 32 Schritte |
|---|---:|---:|---:|---:|---:|---:|
| d=64, 8 Schritte (wie oben) | 134.648 | 83,0 ± 2,8 % | 7,1 | 55 ± 10 % | 20 ± 10 % | 74 ± 15 % |
| d=128, 8 Schritte | 363.768 | 85,0 ± 0,6 % | 6,6 | 70 ± 6 % | 29 ± 3 % | 74 ± 13 % |
| d=64, 16 Schritte | 134.648 | 87,0 ± 4,6 % | 6,2 | 72 ± 24 % | 33 ± 20 % | 86 ± 5 % |
| **d=128, 16 Schritte** | 363.768 | **88,2 ± 0,9 %** | **5,7** | **81 ± 3 %** | **44 ± 7 %** | 85 ± 2 % |

Mittelwert ± Standardabweichung über drei Seeds. "Agent" heißt bis zu 32 Schritte mit Metakognition und
Regelprüfer.

- **16 Denkschritte im Training bringen am meisten**, bei gleicher Breite 3 bis 4 Punkte. Die langen
  Schlussketten (Tiefe 10 und mehr) löst d=128 mit 16 Schritten mehr als doppelt so oft wie das
  ursprüngliche Modell.
- **Die Breite macht das Training vor allem verlässlich.** Mit d=128 streut das Ergebnis zwischen den
  Seeds um etwa einen Punkt, mit d=64 um 3 bis 5. Ob d=128 im Mittel auch besser ist als d=64 (bei
  16 Schritten), lässt sich mit drei Seeds nicht entscheiden: Der beste Einzellauf war d=64 mit
  16 Schritten (90,8 %), der schlechteste derselben Variante 81,8 %.
- **Zu langes Nachdenken kann schaden.** Über die geübte Schrittzahl hinaus driftet das Netz manchmal
  ab. Bei d=64 mit 8 Schritten löst es nach 32 Schritten je nach Seed 57 bis 84 % der Rätsel, in einem
  Lauf fällt Tiefe 7–9 auf 7 %. Metakognition und Regelprüfer verhindern das, weil sie beim ersten
  geprüft richtigen Entwurf aufhören. Mit 16 geübten Schritten tritt das Abdriften kaum noch auf.
- Seed 0 von d=64/8 erreichte hier 86,1 %, der Lauf in der Cloud mit derselben Einstellung 81,4 %.
  Einzelne Läufe sind also nur auf etwa ±3 Punkte genau.

**Konsolidierung mit dem tieferen Modell.** Dasselbe Experiment wie oben (drei Tage mit je 1000 schweren
Rätseln, 300 Läufe pro Nacht), auf der GPU in drei Minuten. d=128/16 übt im Schlaf mit 16 Denkschritten,
d=64/8 mit 8. Jeweils Seed 0, gemessen auf denselben 1000 schweren, nie erlebten Rätseln.

| | d=128/16, 8 Schritte | d=128/16, Agent | d=64/8, 8 Schritte | d=64/8, Agent |
|---|---:|---:|---:|---:|
| Vor dem ersten Schlaf | 71,1 % | 71,5 % | 53,7 % | 65,7 % |
| Nacht 1, mit Replay | 84,1 % | 91,1 % | 77,8 % | 87,8 % |
| Nacht 2, mit Replay | 86,4 % | 91,9 % | 81,4 % | 89,8 % |
| Nacht 3, mit Replay | **88,0 %** | **94,1 %** | 83,7 % | 91,0 % |
| *Nacht 1, nur neue Erlebnisse* | 85,1 % | 91,0 % | 80,7 % | 89,4 % |

- Das tiefere Modell startet 6 Punkte höher und endet 3 Punkte höher. Den größten Teil holen beide
  in der ersten Nacht.
- **Der Schlaf heilt das Abdriften.** Vor dem Schlaf war d=128/16 nach 32 Schritten schlechter als
  nach 8 (64,8 % gegen 71,1 %), danach besser (94,1 % gegen 88,0 %). Im Schlaf übt es auf eigenen,
  längeren Lösungswegen.
- Leichte Rätsel bleiben bei 100 % in 8 Schritten. In einem einzigen Schritt löst d=128/16 davon
  nur 82–84 % (d=64/8: 90–92 %): Das breitere Modell verteilt auch leichte Rätsel auf mehrere Schritte.
- Wieder keine einzige falsche Lösung mit Prüfsiegel, und Replay bringt nach Nacht 1 keinen Vorteil.

**Lehrer-Vergleich.** `konsolidieren --lehrer` übt nachts auf *allen* bisher erlebten schweren
Rätseln mit den echten Lösungen, auch auf denen, an denen das Modell gescheitert ist. Alles andere
bleibt gleich: dieselben Rätsel, Replay, Schrittzahl, 300 Läufe pro Nacht und Testmengen.

| Agent, schwere Rätsel | d=128/16 selbst | d=128/16 Lehrer | d=64/8 selbst | d=64/8 Lehrer |
|---|---:|---:|---:|---:|
| Vor dem ersten Schlaf | 71,5 % | 71,5 % | 65,7 % | 65,7 % |
| Nacht 1, mit Replay | 91,1 % | 92,1 % | 87,8 % | 88,9 % |
| Nacht 2, mit Replay | 91,9 % | 92,9 % | 89,8 % | 90,5 % |
| Nacht 3, mit Replay | **94,1 %** | **94,2 %** | **91,0 %** | **92,2 %** |
| geübte Rätsel nach Tag 3 | 2.495 | 3.000 | 2.116 | 3.000 |

- **Das Selbstlernen kommt bis auf 0,1 bis 1,2 Punkte an den Lehrer heran**, also innerhalb der
  Schwankung zwischen Seeds. Die Lösungen der 500 bis 900 Rätsel, die das Modell selbst nicht
  schafft, bringen kaum etwas: Das sind die schwersten, und aus ihnen lernt es in 300 Läufen pro
  Nacht offenbar wenig.
- Der Lehrer bekommt mehr Beispiele, aber nicht mehr Rechenzeit. Mit längeren Nächten könnte er
  weiter davonziehen. Und die Rätsel werden nur schwerer, nicht anders: Bei neuen Regeln wäre ein
  Lehrer vermutlich unverzichtbar.
- Jeweils nur Seed 0.

## Ergebnisse: eine neue Regel (Diagonal-Sudoku)

Bisher wurden die Rätsel nur schwerer, die Regeln blieben gleich. Jetzt ändert sich die Welt: Beim
**Diagonal-Sudoku** müssen zusätzlich beide Hauptdiagonalen jede Ziffer genau einmal enthalten. Der
Regelprüfer kennt die neue Regel, das Netz nicht, und im Rätsel steht kein Hinweis, welche Regel gilt.
Ein klassisch trainiertes Modell erlebt drei Tage lang je 1000 Diagonal-Rätsel (30–46 leere Felder,
so schwer wie im Training) und schläft dazwischen. Gemessen wird auf je 1000 nie erlebten Diagonal-
und klassischen Rätseln.

```bash
python -m mini_kortex umlernen --seed 0
```

Etwa 70 % der Diagonal-Rätsel sind schon nach den klassischen Regeln eindeutig, die löst das alte
Modell sofort. Bei den übrigen 30 % gibt es eine klassisch gültige Lösung ab, die die Diagonalen
verletzt. Umgekehrt verletzt jede Lösung eines klassischen Rätsels die Diagonalen: Das Neue
widerspricht dem Alten.

Nach drei Nächten, Agent (bis 32 Schritte mit Metakognition und Diagonal- bzw. klassischem
Regelprüfer), Mittelwert ± Standardabweichung über drei Seeds (je eigenes Modell und eigene Rätsel):

| Bedingung | d=128/16 diagonal | d=128/16 klassisch | d=64/8 diagonal | d=64/8 klassisch |
|---|---:|---:|---:|---:|
| vorher | 69,7 ± 0,8 % | 100,0 % | 71,0 ± 1,0 % | 99,9 % |
| **selbst, mit Replay** | **98,1 ± 0,5 %** | **99,5 ± 0,2 %** | **98,3 ± 1,6 %** | **99,0 ± 0,4 %** |
| selbst, ohne Replay | 99,8 ± 0,1 % | 33,3 ± 13,0 % | 99,9 ± 0,2 % | 18,9 ± 28,1 % |
| Lehrer, mit Replay | 96,6 ± 0,7 % | 99,8 ± 0,1 % | 97,5 ± 1,4 % | 99,7 ± 0,2 % |
| Lehrer, ohne Replay | 99,7 ± 0,2 % | 49,7 ± 14,1 % | 99,9 ± 0,1 % | 20,8 ± 27,0 % |

- **Die neue Regel lernt das Modell selbst.** Die 70 %, die es mit der alten Regel richtig löst,
  bestätigt der Regelprüfer, und alle erfüllen die Diagonalen. Daraus lernt das Netz die Diagonalen
  nebenbei mit und löst jeden Tag mehr der neuen Rätsel selbst (bei d=128/16, Seed 0: 74, 90, 96 %).
  Einen Lehrer braucht es nicht.
- **Ohne Replay wird kräftig vergessen**, in jedem Lauf. Das Modell wendet die Diagonalen jetzt auch
  auf klassische Rätsel an. Wie stark es vergisst, schwankt sehr: Beim kleinen Modell bleiben 3 bis
  51 %, beim breiten 19 bis 65 %.
- **Replay schützt verlässlich.** In allen sechs Läufen bleibt die alte Regel bei 98,7–99,9 %, und die
  neue kostet es nur 1,5 bis 2 Punkte. Bei schwereren Rätseln derselben Regel brachte Replay nichts.
  Hier ist es entscheidend. Das passt zur Theorie der komplementären Lernsysteme: Replay schützt
  dann, wenn das Neue dem Alten widerspricht.
- **Mit Replay ist der Lehrer etwas schlechter als das Selbstlernen**, bei beiden Modellen in allen
  drei Seeds. Vermutlich, weil das Selbstlernen nur Episoden übt, die mindestens 2 Denkschritte
  gekostet haben, während der Lehrer sein Training auch auf viele schon gekonnte Rätsel verteilt.
- In allen 72 Wachphasen hat der Regelprüfer keine einzige falsche Lösung bestätigt.

**Grenze des Selbstlernens.** Ist diese Regel nur deshalb lernbar, weil gut 70 % der Rätsel schon
mit der alten lösbar sind? Mit `umlernen --mehrdeutig` kommen nur Diagonal-Rätsel vor, die nach den
klassischen Regeln *mehrere* Lösungen haben. Die alte Regel allein reicht dort nie.

| Agent nach Nacht 3, drei Seeds | d=128/16 diagonal | d=128/16 klassisch | d=64/8 diagonal | d=64/8 klassisch |
|---|---:|---:|---:|---:|
| vorher | 34,8 ± 0,8 % | 100,0 % | 36,7 ± 0,5 % | 99,9 % |
| **selbst, mit Replay** | **96,8 ± 0,7 %** | **99,7 ± 0,2 %** | **97,8 ± 1,7 %** | **99,0 ± 0,3 %** |
| selbst, ohne Replay | 99,8 ± 0,2 % | 19,6 ± 6,8 % | 99,9 ± 0,2 % | 12,0 ± 17,4 % |
| Lehrer, mit Replay | 96,8 ± 0,2 % | 99,6 ± 0,3 % | 97,5 ± 1,5 % | 99,1 ± 0,4 % |
| Lehrer, ohne Replay | 99,8 ± 0,2 % | 20,0 ± 8,2 % | 99,9 ± 0,1 % | 10,5 ± 14,5 % |

- **Auch hier trägt das Selbstlernen.** Vorher löst das Modell gut ein Drittel dieser Rätsel: Meist
  gibt es nur wenige klassische Lösungen, und weil der Regelprüfer falsche Entwürfe zurückweist,
  denkt der Agent weiter, bis er die richtige trifft. Dieser Halt genügt. Am ersten Tag löst das
  Modell 35 % selbst, am zweiten 87–89 %, am dritten 95–97 %.
- **Der Lehrer hilft nur am Anfang**: nach Nacht 1 mit Replay 92,6 statt 89,5 % (d=128/16) bzw.
  91,9 statt 86,7 % (d=64/8). Nach Nacht 3 ist der Vorsprung verschwunden.
- Vergessen und Replay wie oben, und wieder keine falsche Lösung mit Prüfsiegel.

Die Grenze ist damit noch nicht gefunden: Ein Drittel richtige Antworten zum Start reicht. Wo das
Selbstlernen scheitert, müsste eine Regel zeigen, bei der das alte Modell anfangs fast nichts löst.

**Regel-Hinweis.** Liegt das Vergessen nur daran, dass das Netz nicht weiß, welche Regel gilt? Mit
`umlernen --hinweis` bekommt die Wahrnehmung einen gelernten Vektor pro Regel (0 klassisch,
1 diagonal), der auf jedes Feld addiert wird, wie ein Kontextsignal. Er startet bei null; das
umgebaute Modell rechnet zunächst genau wie vorher.

| Agent nach Nacht 3, drei Seeds | d=128/16 diagonal | d=128/16 klassisch | d=64/8 diagonal | d=64/8 klassisch |
|---|---:|---:|---:|---:|
| selbst, mit Replay | 98,1 → 98,6 % | 99,5 → 99,5 % | 98,3 → 98,9 % | 99,0 → 99,3 % |
| selbst, ohne Replay | 99,8 → 99,9 % | **33,3 → 34,0 %** | 99,9 → 99,9 % | **18,9 → 21,2 %** |
| Lehrer, mit Replay | 96,6 → 97,6 % | 99,8 → 99,6 % | 97,5 → 98,1 % | 99,7 → 99,4 % |
| Lehrer, ohne Replay | 99,7 → 99,6 % | **49,7 → 48,6 %** | 99,9 → 99,8 % | **20,8 → 21,2 %** |

Links ohne, rechts mit Hinweis.

- **Der Hinweis allein schützt nicht vor dem Vergessen.** Ohne Replay sieht das Netz im Schlaf nie
  ein Rätsel mit Hinweis 0. Es lernt also nur "Hinweis 1 heißt Diagonalen", nicht "Hinweis 0 heißt
  keine Diagonalen". Die gemeinsamen Gewichte verschieben sich trotzdem, und bei Hinweis 0 kommt
  ebenfalls die neue Regel heraus. Ein Kontextsignal hilft nur, wenn beide Kontexte geübt werden.
- Mit Replay bringt der Hinweis bei der neuen Regel etwa einen halben bis einen Punkt, kaum mehr als
  die Streuung. Auch ohne Hinweis kann das Netz die Regeln dann offenbar gut genug auseinanderhalten.
- Wieder keine einzige falsche Lösung mit Prüfsiegel.

## Ergebnisse: Probieren in Gedanken (das Netz als Weltmodell)

Bei manchen Rätseln reichen einfache Schlüsse nicht, man muss eine Annahme treffen und schauen, wohin
sie führt. Genau dort war das Modell am schwächsten. Ein **Weltmodell** sagt voraus, was aus einer
Handlung folgt. In [weltmodell.py](mini_kortex/weltmodell.py) übernimmt das vorhandene Netz diese
Rolle, ohne neues Training:

1. **Wählen:** Das Werkzeug kennt die erlaubten Ziffern. Gewählt wird das leere Feld mit den wenigsten
   (mindestens zwei), wie es auch Menschen tun. In der Praxis sind es immer genau zwei.
2. **Vorstellen:** Für jede Ziffer entsteht ein Gedanken-Rätsel mit dieser Ziffer als Vorgabe. Das Netz
   denkt es 16 Schritte durch.
3. **Prüfen:** Ist ein Entwurf regelkonform, ist das Rätsel gelöst. Annahmen, die sofort ein Feld ohne
   erlaubte Ziffer hinterlassen, fallen weg.
4. **Bewerten:** Sonst geht es mit der Annahme weiter, der die Metakognition am meisten zutraut, bis zu
   sechs Annahmen hintereinander.

```bash
python -m mini_kortex probieren
```

2000 Testrätsel, Anteil gelöst (immer mit Regelprüfer nach jedem Schritt), Modell d=128/16, Seed 0:

| | gesamt | Tiefe 7–9 | Tiefe 10+ | Probieren | Ø Schritte |
|---|---:|---:|---:|---:|---:|
| nur nachdenken, 32 Schritte | 90,0 % | 89,2 % | 58,2 % | 41,8 % | 5,0 |
| nur nachdenken, 224 Schritte | 90,1 % | 89,2 % | 58,9 % | 41,8 % | 24,1 |
| **probieren, Metakognition bewertet** | **97,7 %** | **98,3 %** | **95,0 %** | **82,1 %** | 13,1 |
| probieren, Zufall bewertet | 96,2 % | 95,8 % | 88,7 % | 74,0 % | 14,8 |

Mit d=64/8 ebenso, nur auf niedrigerem Niveau: 86,9 % → 93,7 % gesamt, Probieren 37 % → 65 %.

- **Probieren verdoppelt die Quote bei den Rätseln, die Probieren verlangen** (42 % → 82 %), und hebt
  Tiefe 10+ von 58 auf 95 %. Und das mit der Hälfte der Rechenzeit, die langes Nachdenken kostet: Mehr
  Schritte allein bringen nichts mehr.
- **Die Arbeit macht das Netz als Weltmodell.** Nach einer einzigen richtigen Annahme denkt es die
  Folgen meist bis zur Lösung durch: 82 der 2000 Rätsel brauchten eine Annahme, 43 zwei, 14 drei.
  Ein klassischer Löser bräuchte für dieselben Rätsel viele Verzweigungen.
- **Die Metakognition als Bewertung** bringt 6 bis 8 Punkte bei den schwersten Rätseln gegenüber
  zufälligem Weitermachen. Sie sagt also brauchbar voraus, welche Annahme trägt.
- **Die Intuition bei der Zifferwahl spielt kaum eine Rolle**, weil das gewählte Feld immer genau zwei
  erlaubte Ziffern hat und beide ausprobiert werden. Darf nur eine Ziffer geraten werden, trifft die
  Intuition etwas besser als der Zufall (Probieren 63 % statt 58 %).
- Keine falsche Lösung mit Prüfsiegel.

Ein *gelerntes* Weltmodell ist das noch nicht: Es gibt keinen eigenen Teil, der Folgen vorhersagt,
das Netz löst die Gedanken-Rätsel einfach. Die nächste Stufe wäre, die Suche in den Schlaf zu
übernehmen: Was nur mit Probieren klappte, wird nachts geübt, bis es ohne geht.

Das Weltmodell aus dem Gesamtentwurf ([ARCHITEKTUR.md](ARCHITEKTUR.md)) ist damit nicht ersetzt.
Gemeint ist dort ein eigenes Modul, das die Folgen von Handlungen in einer Welt vorhersagt und aus
dem Vorhersagefehler lernt. Dafür ist die Gitterwelt vorgesehen. Hier zeigt sich nur, dass Planen
durch Probehandeln im Kopf schon mit dem vorhandenen Netz viel bringt.

## Ergebnisse: Visuelles Sudoku (das Auge irrt sich)

Bisher bekam der Mini-Kortex das Rätsel fehlerfrei als Zahlen. Jetzt ist jede Vorgabe ein
handgeschriebenes Bild aus dem MNIST-Testteil, und ein **Auge** (kleines CNN, 225.000 Parameter,
trainiert in wenigen Sekunden) muss die Ziffer erst erkennen. Ein Rätsel hat im Schnitt 38
Vorgaben. Liest das Auge 1 % der Ziffern falsch, ist schon jedes dritte Rätsel falsch gelesen.

Verglichen werden drei Abläufe ([sehen.py](mini_kortex/sehen.py), Befehl `sehen`). Gelöst wird
jeweils mit Probieren in Gedanken:

- **Kette:** erst erkennen, dann lösen. Das Auge gibt nur seine beste Lesart weiter.
- **Arbeitsraum:** Erkennen und Lösen tauschen sich aus.
  1. Das Werkzeug meldet Vorgaben, die sich widersprechen. Drei Viertel aller Lesefehler fallen
     so sofort auf.
  2. Findet der Denkkern keine regelkonforme Lösung, ist vermutlich etwas falsch gelesen.
  3. Dann schaut das System bei den verdächtigen Feldern nochmal hin. Verdächtig sind Felder im
     Widerspruch und Felder, bei denen das Auge unter 99 % sicher war. Es liest sie so, wie das
     Auge sie am zweitliebsten gelesen hätte, einzeln und paarweise.
  4. Von allen Lesarten, die aufgehen, gewinnt die, die am besten zu den Bildern passt.
- **Orakel:** das Rätsel als fehlerfreie Zahlen. Das ist die Obergrenze.

Modell d128/16, 2000 Testrätsel, Mittel über drei Seeds des Auges (je andere Startwerte und
andere Bilder):

| Auge | Ablauf | richtig | falsch, aber geprüft | aufgegeben | richtig bei 1 Lesefehler | bei 2+ | Ø Schritte |
|---|---|---:|---:|---:|---:|---:|---:|
| | Orakel | 97,7 % | 0 % | 2,3 % | | | 13 |
| 1 Epoche, 2,1 % Fehler | Kette | 45,0 % | 2,5 % | 52,5 % | 0 % | 0 % | 102 |
| | **Arbeitsraum** | **84,0 %** | 7,2 % | 8,9 % | 88,3 % | 43,4 % | 219 |
| 4 Epochen, 1,1 % Fehler | Kette | 65,9 % | 2,0 % | 32,1 % | 0,3 % | 0 % | 67 |
| | **Arbeitsraum** | **92,8 %** | 4,5 % | 2,7 % | 88,2 % | 61,8 % | 124 |

Mit d64/8 (Seed 0) ist es dasselbe Bild: Bei 1,1 % Fehler im Auge löst die Kette 62,7 % und der
Arbeitsraum 88,8 %, bei einer Obergrenze von 93,7 %.

Was man daraus lernt:

- **Die starre Kette bricht an jedem Lesefehler.** Schon ein falsch gelesenes Feld macht das
  Rätsel unlösbar oder zu einem anderen Rätsel. Mit einem guten Auge (99 % pro Ziffer) bleiben
  zwei Drittel, mit einem mäßigen (98 %) weniger als die Hälfte.
- **Der Austausch holt fast alles zurück.** Bei einem Lesefehler löst der Arbeitsraum 88 %, bei
  einer Obergrenze von 98 %. Insgesamt kommt er mit dem guten Auge auf 92,8 % statt 65,9 %. Es
  braucht keine neuen Gewichte, nur die Rückmeldung von Werkzeug und Denkkern an die Wahrnehmung.
- **Die beiden Signale ergänzen sich.** Das Werkzeug findet Widersprüche zwischen Vorgaben. Der
  Denkkern merkt, wenn ein Rätsel ohne Widerspruch trotzdem nicht aufgeht. Die Unsicherheit des
  Auges sagt, wo man nachsehen sollte: Die richtige Ziffer ist in 84 % der Fälle seine zweite Wahl.
- **Das Prüfsiegel gilt nur noch für das Gelesene.** Bisher hat der Regelprüfer nie eine falsche
  Lösung bestätigt. Jetzt prüft er gegen das, was das Auge gesehen hat. Zwei Arten von Fehlern
  bleiben:
  - Ein falsch gelesenes Rätsel, das trotzdem aufgeht. Das sind 2 % in beiden Abläufen.
  - Der Arbeitsraum erzeugt zusätzlich eigene Fehler. Scheitert der Denkkern an einem richtig
    gelesenen Rätsel, deutet er eine Vorgabe um und löst dann ein leichteres, falsches Rätsel.
  - Deshalb steigt „falsch, aber geprüft“ von 2,0 % auf 4,5 %.
- **Der Preis ist Rechenzeit.** Der Arbeitsraum braucht im Schnitt 124 statt 67 Denkschritte, weil
  er bei Problemen mehrere Lesarten durchdenkt. Bei richtig gelesenen Rätseln kostet er nichts.

## Ergebnisse 6×6

Ein Modell mit 88.991 Parametern, 3000 Läufe (ca. 30 Minuten), 2000 Testrätsel mit 12–28 leeren
Feldern. Mit 16 Denkschritten löst es **alle**. Die Metakognition braucht im Schnitt 1,04 Schritte
(bei den schwersten 3,1), mit Regelprüfer stimmen alle Antworten. 6×6 ist für dieses Modell zu
leicht, um die Module voneinander zu unterscheiden (nur 83 von 2000 Rätseln sind schwer).

So sieht ein schweres 6×6-Rätsel aus (`python -m mini_kortex --groesse 6 loesen --leer 28 --seed 71`):

```
Denkschritt  1:  7 Felder falsch · Zuversicht 13% · Regelprüfer: ungültig
Denkschritt  2:  5 Felder falsch · Zuversicht 2% · Regelprüfer: ungültig
Denkschritt  3:  0 Felder falsch · Zuversicht 48% · Regelprüfer: gültig ✔
```

## Was fehlt (und Ideen zum Weiterprobieren)

Der Prototyp zeigt die Bausteine im Kleinen. Ein eigenes Weltmodell, Sprachkern und Wertsystem der
großen Architektur fehlen, weil Sudoku sie nicht braucht. Gesamtentwurf, vorgeschlagene Reihenfolge
und offene Fragen stehen in [ARCHITEKTUR.md](ARCHITEKTUR.md).

- **Vergessen ohne alte Daten verhindern**: Replay aus dem eigenen Gedächtnis statt aus dem
  Trainingsset, weniger Replay pro Batch, oder wichtige Gewichte schützen (Elastic Weight Consolidation).
- **Unfreundliche neue Regel**: eine Regel, die die alte kaum überschneidet. Wann scheitert das
  Selbstlernen, und hilft dann der Lehrer?
- **Dem Auge misstrauen lernen**: Der Arbeitsraum bestätigt manchmal ein falsch gelesenes Rätsel.
  Abhilfe könnte sein, aufzugeben, wenn eine Lösung sichere Vorgaben umdeuten muss. Oder das Auge
  schaut verdächtige Felder wirklich nochmal an, etwa leicht verschoben.
- **Probieren im Schlaf**: Lösungen, die nur mit Annahmen gefunden wurden, nachts üben. Wird aus dem
  Probieren Intuition, wie zuvor aus dem langen Nachdenken?
- Noch tiefer üben (`--schritte 32`): Wächst der Gewinn bei den langen Schlussketten weiter?
- Die Metakognition mit `schwelle=0.9` statt `0.5` in [agent.py](mini_kortex/agent.py): weniger
  Fehler, mehr Schritte?
- Den MLP-Mixer in `modell.py` durch Attention ersetzen und vergleichen.
