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

Zwei Einschränkungen: Das Modell kann nur aus Rätseln lernen, die es mit viel Bedenkzeit schon
selbst lösen kann. Wirklich Neues (andere Regeln) kann es so nicht lernen. Und es fehlt der
Vergleich mit einem Modell, das auf denselben schweren Rätseln mit den echten Lösungen trainiert
wurde. Der würde zeigen, wie nah das Selbstlernen an Lernen mit Lehrer herankommt.

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

Der Prototyp zeigt die Bausteine im Kleinen. Weltmodell, Sprachkern und Wertsystem der großen
Architektur fehlen, weil Sudoku sie nicht braucht.

- **Lehrer-Vergleich**: Dieselben schweren Rätsel mit den echten Lösungen nachtrainieren. Wie nah
  kommt das Selbstlernen heran?
- **Vergessen provozieren**: Im Schlaf eine Regelvariante üben (zum Beispiel zusätzlich beide
  Diagonalen). Hilft dann Replay?
- Noch tiefer üben (`--schritte 32`): Wächst der Gewinn bei den langen Schlussketten weiter?
- Die Metakognition mit `schwelle=0.9` statt `0.5` in [agent.py](mini_kortex/agent.py): weniger
  Fehler, mehr Schritte?
- Den MLP-Mixer in `modell.py` durch Attention ersetzen und vergleichen.
