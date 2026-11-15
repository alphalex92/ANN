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

## Projekt: Denkkern

Ein kleines GPT mit **Schleife**, das Wissen **nachschlagen** muss, statt es zu kennen. Die
Frage dahinter: Lässt sich „suchen und Treffer verknüpfen“ als eigene Fähigkeit lernen, wenn
das Modell von Anfang an nichts auswendig wissen *kann*? Bei K2 Horizon hängt es genau
daran: Das Modell sucht, wertet die Treffer aber oft nicht aus.

### Die Spielwelt

Jede Episode würfelt eine neue Welt aus: 64 Dinge (`e0` … `e63`) und 8 Beziehungen
(`r0` … `r7`), jede Beziehung eine Zufallsfunktion. Eine Frage ist eine Kette, `? e12 r5 r4 r7`
heißt „r7 von r4 von r5 von e12“. Weil die Welt immer neu ist, hilft Auswendiglernen nichts.

| Modus | Was das Modell sieht | Was es schreibt |
|---|---|---|
| `werkzeug` | nur die Frage | `S e12 r5` → Umgebung: `= e60`, … , am Ende `A e3 </s>` |
| `restkette` | wie werkzeug, aber jeder Treffer nennt die offenen Glieder | `S e12 r5` → `= e60 \| r4 r7`, … , `= e3 \|` → `A e3 </s>` |
| `kompakt` | nach jedem Treffer nur einen frischen Notizzettel: Frage, Stand, Restkette | `<s> ? e12 r5 r4 = e57 \| r4` → `S e57 r4` |
| `kontext` | 16 Fakten (Kette plus Ablenker), dann die Frage | sofort `A e3 </s>`, die Kette muss *in den Schichten* laufen |

Gelernt wird nur, was das Modell selbst schreibt. Frage, Fakten und Treffer sind Kontext.

### Das Modell

`vorne` → `Kern` × Schleifen → `hinten`. Der Kern hat seine Gewichte nur einmal, läuft aber
mehrfach. Vor jedem Durchlauf wird die Eingabe neu eingespeist. Trainiert wird mit zufällig
vielen Durchläufen, gemessen mit mehr als im Training. Positionen über RoPE oder gar keine
(NoPE), damit längere Ketten nicht schon an den Positions-Einbettungen scheitern.

### Schnellstart

```bash
pip install -r requirements-denkkern.txt
```

```bash
python -m denkkern beispiel --modus werkzeug --hops 3
```

```bash
python -m denkkern trainieren --modus werkzeug --name w1 --hops 1-4 --schleifen 1-4 --minuten 5
```

```bash
python -m denkkern messen --name w1 --hops 1-12 --schleifen 1,2,4,8
```

Die Messung rollt das Modell frei aus, die Umgebung beantwortet seine Suchen. Im Modus
`werkzeug` gibt es dieselben Kategorien wie bei K2: richtig, gesucht aber falsch, geraten
(geantwortet, ohne genug gesucht zu haben) und abgebrochen. Dazu „Suchweg exakt“: jede Suche
war genau die nächste nötige. Läufe landen in `laeufe/<name>/`.

### Erster Lauf (6–10 Minuten je Modell, 3,3 M Parameter, RTX 5080)

Trainiert auf Ketten mit 1–4 Hops, gemessen bis 12 Hops, je 500 Fragen. Die Zahlen gelten
bei 4 Schleifen, bei kontext bei 6.

| Hops | werkzeug, RoPE | werkzeug, NoPE | kontext, RoPE |
|---|---|---|---|
| 1–4 | 100 % | 100 % | 88 / 41 / 19 / 13 % |
| 5 (neu) | 5 % | 77 % (87 % bei 8 Schleifen) | 10 % |
| 6–12 (neu) | 0–3 % | 0–3 % | 0–2 % |

- **Im Trainingsbereich ist das Werkzeug perfekt gelernt.** Der Suchweg ist zu 100 % exakt,
  obwohl die Welt jedes Mal neu ist. Nachschlagen und Treffer verketten klappt also.
- **Darüber hinaus antwortet das Modell zu früh.** Es hat „nach höchstens 4 Suchen
  antworten“ gelernt statt „antworten, wenn die Kette abgearbeitet ist“. Das zählt als
  geraten, bei NoPE und 6–9 Hops zu 96–98 %. Ganz lange Ketten brechen ab.
- **Ohne Positionen (NoPE) verallgemeinert es besser als mit RoPE**, auf 5 Hops 77 % statt 5 %.
  Mehr Schleifen helfen dort etwas, 87 % bei 8 statt 77 % bei 4.
- **Im Modus kontext lernt das Modell in 10 Minuten nur 1 Hop richtig** (88 %). Die
  Schleifen bringen nichts: Mit 1 Durchlauf ist es so gut wie mit 12. Der Verlust fiel
  am Ende noch, das Modell ist also untertrainiert.

### Zweiter Lauf: Werkzeug mit Restkette

Die Umgebung hängt an jeden Treffer, welche Glieder der Kette noch offen sind. Gestrichen
wird ein Glied nur, wenn das Modell genau die nötige Suche gemacht hat. Wieder 6 Minuten,
Hops 1–4 im Training. Richtig in %:

| Hops | werkzeug, NoPE | restkette, NoPE, 1 Schl. | restkette, NoPE, 4 Schl. | restkette, RoPE |
|---|---|---|---|---|
| 1–4 | 100 | 100 | 100 | 100 |
| 5 (neu) | 77 | 100 | **100** | 0 |
| 6 (neu) | 3 | 52 | **99** | 0 |
| 7 (neu) | 3 | 5 | **74** | 0 |
| 8 (neu) | 3 | 0 | 25 | 0 |
| 9–12 (neu) | 0–2 | 0 | 0–2 | 0 |

- **Das Raten ist komplett weg**, bei allen Hops 0 %. Die Umgebung sagt, wann die Kette
  fertig ist, und daran hält sich das Modell.
- **Hier helfen die Schleifen zum ersten Mal deutlich.** Mit 4 statt 1 Durchlauf löst das
  Modell 6er-Ketten zu 99 % statt 52 % und 7er-Ketten zu 74 % statt 5 %.
- **RoPE scheitert sofort** an jeder Kette, die länger ist als im Training.
- **Wo es noch bricht:** Ab etwa 7 Hops nimmt das Modell nicht mehr das erste offene Glied
  hinter „|“, sondern wiederholt meist die Beziehung der letzten Suche. Weil die Restkette
  dann stehen bleibt, sucht es im Kreis, bis die Schritte ausgehen (abgebrochen). Lange
  Verläufe mit vielen ähnlichen „| …“-Stücken verwirren es also, die Länge der Restkette
  selbst weniger.

### Dritter Lauf: kompakter Verlauf

Statt den ganzen Verlauf zu behalten, bekommt das Modell nach jedem Treffer einen frischen
Notizzettel mit der Frage, dem aktuellen Stand und der Restkette. Wieder 6 Minuten, Hops 1–4
im Training, gemessen bis 16 Hops. Richtig in %:

| Hops | restkette, NoPE, 4 Schl. | **kompakt, NoPE**, 1 Schl. | **kompakt, NoPE**, 4 Schl. | kompakt, RoPE |
|---|---|---|---|---|
| 1–4 | 100 | 100 | 100 | 100 |
| 5–6 (neu) | 99–100 | 100 | 100 | 0–1 |
| 7–8 (neu) | 25–74 | 100 | 100 | 0 |
| 9–15 (neu) | 0–2 | 100 | 100 | 0 |
| 16 (neu) | – | 98 | **100** | 0 |

- **Mit Notizzettel und ohne Positionsangaben löst das Modell Ketten viermal so lang wie im
  Training fehlerfrei.** Der Suchweg ist jedes Mal exakt, geraten und abgebrochen sind 0 %.
- **Die Schleifen braucht es dafür kaum noch.** Mit 1 Durchlauf sind es bei 16 Hops 98 %,
  mit 4 Durchläufen 100 %. Jeder Schritt ist so einfach geworden, dass ein Durchlauf reicht.
- **RoPE scheitert auch hier** ab der ersten neuen Länge. Die Frage auf dem Zettel ist länger
  als im Training, und schon das wirft RoPE aus der Bahn.

**Was das heißt:** Das Modell hat in allen drei Werkzeug-Varianten dieselbe Fähigkeit
gelernt: suchen und den Treffer weiterverwenden. Gescheitert ist es jedes Mal an der
*Buchführung*, also daran, mitzuzählen, wann es fertig ist, und in einem langen Verlauf den
Faden zu behalten. Legt man die Buchführung in die Umgebung, verallgemeinert schon ein
3-M-Modell nach 6 Minuten auf beliebig lange Ketten. Für K2 hieße das: die Recherche über
einen kurzen Notizzettel führen, statt über den wachsenden Verlauf.

Einschränkung: In dieser Spielwelt ist jeder einzelne Schritt leicht. Wie gut sich das auf
echte Fragen überträgt, bei denen schon das Lesen eines Treffers schwer ist, zeigt erst ein
Versuch mit K2.

### Vierter Lauf: wachsendes Gedächtnis mit Schlaf

`denkkern/gedaechtnis.py`. Die Welt hat jetzt 16 Beziehungen. r0–r3 sind seit der Kindheit
fest, und die Hälfte dieser Fakten lernt das Modell auswendig. r8–r11 werden erst im Leben
fest. Alle anderen Beziehungen bleiben flüchtig und werden in jeder Episode neu gewürfelt,
wie Tagesnachrichten. Auf dem Notizzettel gibt es eine dritte Aktion: `W y` heißt „das
weiß ich“. Die Umgebung prüft das nicht. Ein falsches `W` ist eine Halluzination.

Das Modell ist der Schleifenkern mit NoPE und einer Gedächtnisschicht mit Produkt-Schlüsseln
(4.096 Plätze, Lample et al. 2019) nach der Schleife. Es hat 4,4 M Parameter, davon 1 M für
die Werte des Gedächtnisses.

Ein Lebenstag läuft so ab:
1. Morgens wird auf festen Testfragen gemessen.
2. Das Modell beantwortet 200 Fragen auf r8–r15 mit Suche.
3. Sichten: Ein Fakt gilt als gesichert, wenn die letzten zwei Suchen dasselbe ergaben.
4. Nachts träumt es: Aus den gesicherten Fakten werden Übungsketten erfunden, gemischt mit
   Kindheitsketten, 300 Schritte.

```bash
python -m denkkern kindheit --name g4 --minuten 15 --welten 150 --kern-faktor 0.1 --stapel 64
python -m denkkern leben --name g4 --festigen ganz --sichten pruefen
```

**Die Kindheit entscheidet, ob das Gedächtnis überhaupt benutzt wird.** Lernt alles gemeinsam
in einer Welt, legt das Modell die Fakten in den normalen Schichten ab. Die Gedächtnisschicht
bleibt dann ungenutzt: Setzt man ihre Werte auf 0, ändert sich nichts. Sie liest für alle
1.024 Paare dieselben 6 Plätze.

Genutzt wird sie erst unter drei Bedingungen zusammen:
- 150 kurze Kindheitswelten,
- nach jeder Welt werden die Werte gelöscht,
- das Gedächtnis lernt 100-mal schneller als das Denken.

Löscht man das Gedächtnis dann, fällt die Kindheit von 100 % auf 68 % richtig.

Die Ergebnisse nach 8 Tagen vom selben Kindheitsmodell g4. Gemessen wurde am Morgen von
Tag 9, jeweils auf 500 Testfragen:

| Festigen | Sichten | neu richtig | Suchen/Frage | gewusst (r8–r11) | Halluzinationen/Frage | alt richtig |
|---|---|---|---|---|---|---|
| kein Schlaf | – | 100 % | 2,48 | 0 % | 0,00 | 100 % |
| nur Gedächtnisschicht | prüfen | 76 % | 1,21 | 78 % | 0,26 | 97 % |
| nur Gedächtnisschicht | alles | 18 % | 0,07 | 65 % | 1,54 | 95 % |
| **ganzes Modell** | **prüfen** | **89 %** | **1,11** | **98 %** | **0,11** | **100 %** |
| nur Gedächtnisschicht, Wandel an Tag 5 | prüfen | 58 % | 1,16 | 61 % | 0,53 | 96 % |

Was man sieht:
- **Wissen spart Arbeit, kostet aber Genauigkeit.** Das ganze Modell weiß nach 8 Tagen 98 %
  der neuen Fakten und sucht 55 % weniger. Dafür halluziniert es etwa jede neunte Frage. In
  dieser Welt ist die Suche immer richtig, deshalb kann Wissen hier nur schaden oder Zeit
  sparen. Bei echter, fehlbarer Suche könnte das anders aussehen.
- **Ungeprüft festigen ist verheerend.** Mit „alles“ werden auch flüchtige Fakten gefestigt.
  Das Modell hört praktisch auf zu suchen und behauptet 1,5 falsche Fakten pro Frage.
- **Nur das Gedächtnis zu festigen ist schlechter als das ganze Modell.** Bei 78 % Wissen
  steht es bei 0,26 Halluzinationen pro Frage. Die Fakten in die Schicht zu schreiben,
  gelingt. Die Entscheidung „weiß ich das?“ steckt aber im eingefrorenen Denken. Dieses hat
  nur Gedächtnisinhalte aus der Kindheit gesehen und verlässt sich auf die neuen Inhalte zu
  sehr oder zu wenig.
- **Veraltetes Wissen korrigiert sich nicht.** Nach dem Wandel an Tag 5 bleiben 46 falsche
  Fakten bis zum Ende gesichert. Was das Modell zu wissen glaubt, schlägt es nie mehr nach.
  Darum sieht es die Änderung auch nie. Es bräuchte Stichproben: Gewusstes ab und zu
  trotzdem nachschlagen.
- Die Kindheit mit vielen Welten hilft auch beim Festigen des ganzen Modells. g1 mit einer
  Welt kommt dabei nur auf 83 % richtig und 0,18 Halluzinationen pro Frage.
