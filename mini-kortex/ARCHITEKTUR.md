# Hybride Denkarchitektur: Gesamtentwurf

Diese Notiz hält fest, was in der Cloud-Session "AI-Architekturen jenseits von LLMs" über die
große Architektur besprochen wurde, aus der der Mini-Kortex hervorgegangen ist. Sie trennt dabei,
was **Vorschlag** war und was **Alex entschieden** hat. Ergebnisse, die erst danach auf dem
Branch `mini-kortex` entstanden sind (GPU, breitere Modelle, Lehrer-Vergleich, Diagonal-Sudoku,
Regel-Hinweis), stehen hier nicht drin.

Das interaktive Diagramm dazu: https://claude.ai/artifact/9FWxhEogr8B1wd5v8NPUKm (privat, nur für
Alex sichtbar, solange er es nicht teilt).

## Grundidee

Statt eines einzigen großen Sprachmodells arbeiten spezialisierte Module zusammen, wie Regionen im
Gehirn. Sie tauschen sich über einen kleinen gemeinsamen **Arbeitsraum** aus (Global Workspace
Theory nach Baars und Dehaene, in der KI von Bengio aufgegriffen). Der Arbeitsraum hat nur wenige
"Slots". Was dort landet, sehen alle Module. Der Engpass ist gewollt: Er zwingt zu abstrakten,
kombinierbaren Darstellungen, ähnlich dem begrenzten Arbeitsgedächtnis des Menschen.

Die Leitlinie aus der Diskussion: Eine überlegene Architektur kopiert nicht das Gehirn, sondern
verbindet seine Stärken (Weltmodell, Vorhersage, schnelles und langsames Gedächtnis, lebenslanges
Lernen, Spärlichkeit, Neugier) mit denen des Computers (Geschwindigkeit, exaktes Gedächtnis,
Kopierbarkeit, formale Prüfung, Selbstinspektion). Vergleich: Flugzeuge schlagen nicht mit den
Flügeln, nutzen aber dieselbe Physik.

## Die Module

```
            ┌───────────────────────────────────────────────────────────┐
 Sinne ───► │  WAHRNEHMUNG (Encoder)                                    │
            └───────────────────────────┬───────────────────────────────┘
                                        ▼
  ┌──────────────┐   ┌────────────────────────────────────┐   ┌──────────────────┐
  │ WELTMODELL   │◄─►│        GLOBAL WORKSPACE            │◄─►│ SPRACH-/WISSENS- │
  │              │   │   wenige Slots, gemeinsamer Raum   │   │ KERN + Retrieval │
  └──────────────┘   └───┬──────────┬──────────┬──────────┘   └────────┬─────────┘
                         ▼          ▼          ▼                       ▼
            ┌──────────────┐ ┌─────────────┐ ┌──────────────┐ ┌─────────────────┐
            │ REASONING-   │ │ METAKOGNI-  │ │ WERT- &      │ │ GEDÄCHTNIS      │
            │ KERN         │ │ TION        │ │ MOTIVATION   │ │ episodisch,     │
            └──────┬───────┘ └─────────────┘ └──────────────┘ │ semantisch,     │
                   ▼                                          │ prozedural      │
            ┌──────────────────────────┐                      └─────────────────┘
            │ SYMBOLISCHE WERKZEUGE    │
            └──────────────────────────┘
                                        ▼
                         AUSGABE: Sprache · Handlung · Code

   ⟲ OFFLINE-KONSOLIDIERUNG ("Schlaf"): Replay → langsames Einlernen in die Gewichte
```

| Modul | Aufgabe | Vorbild im Gehirn | Technische Basis | Im Mini-Kortex |
|---|---|---|---|---|
| Wahrnehmung | Rohdaten → abstrakte Vektoren | Sensorische Kortizes | ViT-/JEPA-Encoder, Audio-Encoder | ja (Ziffer + Position → Vektor) |
| Global Workspace | Bündeln, was gerade wichtig ist; Engpass | Präfrontal-parietales Netzwerk | Slot-Attention, kleiner Transformer | vereinfacht: Tensoren x, z, y |
| **Weltmodell** | "Was passiert, wenn …?", Planen durch Probehandeln im Kopf | Prädiktiver Kortex, Kleinhirn | V-JEPA 2, DreamerV3, latente Dynamikmodelle | nein |
| **Sprach- & Wissenskern** | Sprache, Allgemeinwissen, Kommunikation | Sprachareale, Assoziationskortex | kompaktes LM (Transformer/Mamba-Hybrid) mit Datenbank-Zugriff | nein |
| Reasoning-Kern | Tiefes Nachdenken mit variabler Rechenzeit | Präfrontaler Kortex | rekursiv und latent (HRM, TRM) | ja |
| Metakognition | Rechenbudget, Zuversicht, welches Modul hilft | Anteriorer cingulärer Kortex, PFC | kleiner Steuerkopf, per Selbstüberwachung oder RL | ja (Zuversicht, Stopp, wann geprüft wird) |
| **Wert & Motivation** | Ziele, Prioritäten, Neugier, Lernsignal | Basalganglien, Dopaminsystem | Value-Netz, Belohnungsvorhersage, Curiosity-Signal | nein |
| Gedächtnis | Lernen aus einem Erlebnis; Wissen außerhalb der Gewichte | Hippocampus (schnell), Neokortex (langsam) | modernes Hopfield-Netz, Vektor-DB, Titans-Memory | ja (Hopfield, episodisch) |
| Symbolische Werkzeuge | Exakte, beweisbare Ergebnisse | kein Vorbild | Lean, Python, SAT/SMT-Solver | ja (Sudoku-Regelprüfer) |
| Ausgabe | Inhalt des Arbeitsraums → Antwort oder Handlung | Motorkortex, Broca-Areal | Decoder-Köpfe | ja (linearer Kopf) |
| Konsolidierung | Erlebtes langsam in die Gewichte übertragen, ohne Altes zu überschreiben | Schlaf, Hippocampus-Replay | Replay-Training, Continual Learning | ja (Wach-Schlaf-Zyklen) |

### Die drei fehlenden Module im Detail

**Weltmodell.** Sagt abstrakte Folgen von Handlungen voraus, nicht Pixel oder Tokens (JEPA-Idee von
LeCun). Es erlaubt Planen durch Simulation: mehrere Varianten im Kopf durchspielen, bevor man
handelt. Im Gehirn entspricht das der Theorie, dass der Kortex ständig vorhersagt und aus dem
Vorhersagefehler lernt (Predictive Coding). Es liest aus dem Arbeitsraum den aktuellen Zustand und
eine geplante Handlung und schreibt die vorhergesagte Folge zurück. Der Reasoning-Kern kann damit
Alternativen bewerten, ohne sie auszuführen.

**Sprach- und Wissenskern.** Größter Posten der Architektur, weil Sprache und Allgemeinwissen
Parameter brauchen. Sprachmodelle speichern grob 2 Bit Wissen pro Parameter (Allen-Zhu & Li,
"Physics of Language Models"). Ein großer Teil eines Frontier-LLMs ist daher komprimiertes Lexikon,
nicht Denkapparat. Lagert man Wissen in eine Datenbank aus (Retrieval), schrumpft der Kern stark:
DeepMinds RETRO erreichte mit 7,5 Mrd. Parametern plus Datenbank die Leistung von rund 25-mal
größeren Modellen. Der Sprachkern formuliert Antworten, liefert Fakten und versteht Aufgaben in
natürlicher Sprache. Er arbeitet über den Arbeitsraum mit den anderen Modulen, statt wie heute bei
LLM-Agenten alles selbst zu sein (Workspace, Reasoner, Metakognition und Wissensspeicher zugleich).

**Wert und Motivation.** Bewertet Zustände ("lohnt sich das?"), setzt Prioritäten und liefert das
Lernsignal, analog zur Belohnungsvorhersage des Dopaminsystems. Dazu gehört ein Neugier-Signal:
Überraschung (großer Vorhersagefehler des Weltmodells) ist selbst eine Belohnung und treibt
Erkundung an. Es arbeitet eng mit Metakognition und Weltmodell zusammen: Das Weltmodell sagt
Folgen voraus, das Wertsystem bewertet sie, die Metakognition entscheidet, wie viel Aufwand sich
lohnt.

### Zusammenspiel: ein Denkvorgang

Beispiel aus der Diskussion: "Passt das Sofa durch meine Tür?"

1. **Wahrnehmung** codiert das Foto von Sofa und Türrahmen; ins Workspace kommt nur das Wesentliche.
2. **Metakognition** erkennt eine räumliche Aufgabe mit unsicherem Ausgang und gibt mehr Rechenbudget frei.
3. **Gedächtnis** meldet ein ähnliches Erlebnis: Beim letzten Umzug mussten die Füße ab.
4. **Weltmodell** simuliert Drehungen und Kippen.
5. **Reasoning-Kern** iteriert über Lösungen; die Zuversicht steigt.
6. **Werkzeug** rechnet bei knappen Maßen exakt nach.
7. **Sprachkern** formuliert: "Ja, gekippt, aber nur ohne Füße."
8. **Konsolidierung** überträgt das Erlebnis nachts langsam in die Gewichte.

## Größenabschätzung

Spekulativ, aber begründet. Kernaussage: **Reasoning braucht wenig Parameter, Wissen braucht viele.**

| Modul | Parameter | Begründung |
|---|---|---|
| Wahrnehmung | 0,5–2 Mrd. | heutige starke Encoder |
| Global Workspace | 0,1–0,5 Mrd. | bewusst klein (Engpass) |
| Weltmodell | 1–5 Mrd. | V-JEPA 2 hat rund 1,2 Mrd. |
| Sprach- & Wissenskern | 3–30 Mrd. | größter Posten; Wissen liegt großteils in der Datenbank |
| Reasoning-Kern | 0,01–1 Mrd. | HRM 27 Mio., TRM 7 Mio.: klein, aber viele Iterationen |
| Metakognition + Wertsystem | 0,1–1 Mrd. | Steuerung, kein Wissensspeicher |
| **Summe** | **ca. 5–40 Mrd.** | plus externes Gedächtnis (GB bis TB) |

Zum Vergleich: Frontier-LLMs werden auf mehrere hundert Mrd. bis einige Billionen Parameter
geschätzt. Die Rechenlast verschiebt sich aber vom Training in die Inferenz: Der Reasoning-Kern
iteriert bei schweren Problemen hunderte Male, das Weltmodell simuliert viele Varianten.

## Umsetzungsideen für Weltmodell, Sprachkern und Wertsystem

Grundsatz, den ich vorgeschlagen habe: **Kleine Module selbst bauen, große fertig übernehmen.**

- **Große Module fertig übernehmen und einfrieren** (Wahrnehmung, Sprache, später Weltmodell). Von
  Grund auf wären sie auf dieser Hardware nicht in brauchbarer Qualität machbar, und man lernt
  dabei wenig über die Architektur.
- **Kleine Adapter dazwischen trainieren**, die fremde Vektorräume in den gemeinsamen Arbeitsraum
  projizieren. So arbeiten heutige Bild-Sprach-Modelle wie LLaVA: fertiger Bild-Encoder, kleine
  gelernte Projektion, fertiges Sprachmodell.
- **Arbeitsraum, Reasoning-Kern, Metakognition, Gedächtnis und Steuerung selbst bauen.** Dort liegt
  die eigentliche Forschungsfrage, und diese Teile laufen auf der CPU.
- **Jedes neue Modul braucht eine Aufgabe, die es wirklich braucht.** Sudoku braucht weder
  Weltmodell noch Sprache noch Wertsystem.

Daraus folgte diese vorgeschlagene Reihenfolge:

1. **9×9 + Konsolidierung.** Erledigt (siehe README).
2. **Visuelles Sudoku.** Ziffern kommen als handgeschriebene Bilder (MNIST). Die Wahrnehmung wird
   ein echtes, fehlbares Modul (kleines CNN, trainiert in Minuten, oder fertig übernommen). Die
   Module müssen wirklich zusammenarbeiten: Hält die Wahrnehmung eine 7 für eine 1, entsteht ein
   Widerspruch, den Werkzeug und Denkkern finden müssen, und die Metakognition muss merken, welches
   Feld sie nochmal "ansehen" sollte. Messbar: Ist ein System mit gemeinsamem Arbeitsraum robuster
   als eine starre Kette "erst erkennen, dann lösen"?
3. **Gitterwelt mit Weltmodell und Wertsystem.** Eine kleine, selbst entworfene Gitterwelt mit
   Handlungen, in der ein Agent plant (Weltmodell im latenten Raum, Dreamer- oder JEPA-artig) und
   neugierig erkundet (Wertsystem mit Neugier-Signal aus dem Vorhersagefehler).
4. **Optional: Sprachmodul.** Ein kleines fertiges Sprachmodell (etwa 0,5 Mrd. Parameter, läuft auch
   auf der CPU), eingefroren und über einen Adapter angebunden, etwa damit das System seine
   Lösungswege erklärt.

### Was Alex entschieden oder gewünscht hat

- Prototyp im Repo und ein Diagramm der Architektur: gewünscht und gebaut.
- Eigenes Projekt im Ordner `mini-kortex`, nicht unter dem Namen `denkkern`, weil Alex lokal schon
  an einem eigenen `denkkern` arbeitete. Name `mini-kortex` von Alex gewählt. Ein eigenes Repo war
  eine Option; Alex wollte vorerst einen eigenen Ordner in ANN.
- **Mit 9×9 und Konsolidierung anfangen**: Alex' Entscheidung auf die vorgeschlagene Reihenfolge.
  Zu Schritt 2 bis 4 gibt es **noch keine Entscheidung**. Auch der Grundsatz "kleine Module selbst
  bauen, große übernehmen" war mein Vorschlag; Alex hat ihm nicht widersprochen, ihn aber auch
  nicht ausdrücklich bestätigt.
- GPU stellt Alex bereit, sobald er am PC ist. Der Versuch, diese Cloud-Session per Remote Control
  mit dem PC zu verbinden, scheiterte: Desktop-Alex listet die Ordner ANN, Crawler und Trainer, bietet
  sie Cloud-Sessions aber nicht an (Ursache unklar, vermutlich App-Version unter Windows). Daher die
  lokale Session.

## Offene Fragen

- **Gemeinsame Sprache der Module:** Separat trainierte Module verstehen sich zunächst nicht. Reichen
  kleine Adapter, oder muss der Arbeitsraum gemeinsam mit allen Modulen feinjustiert werden?
- **Echter Workspace mit Slots:** Der Mini-Kortex nutzt nur drei Tensoren (x, z, y) als Arbeitsraum.
  Ab wann braucht man einen echten Engpass mit wenigen Slots und Konkurrenz um den Zugang?
- **Training des Gesamtsystems:** End-to-End geht nicht, weil Werkzeuge und Datenbanken nicht
  differenzierbar sind. Vorgeschlagen war ein gestuftes Vorgehen (Encoder und Weltmodell
  selbstüberwacht, Sprachkern klassisch, Reasoning-Kern auf Logik, dann über den Workspace verbinden
  und per Reinforcement Learning gemeinsam feintunen). Erprobt ist davon nichts.
- **Credit Assignment:** Wenn die Antwort falsch ist, welches Modul war schuld? Im Gehirn wie in der
  KI ungelöst.
- **Metakognition als Router:** Im Mini-Kortex entscheidet sie nur, wann aufgehört und wann geprüft
  wird. Soll sie später auch entscheiden, welches Modul gefragt wird (etwa per RL)?
- **Die "Bitter Lesson"** (Sutton): Handentworfene Modularität hat historisch oft gegen einfache
  Architekturen plus Skalierung verloren. Gegenvorschlag aus der Diskussion: Modularität lernen
  lassen statt festlegen (Mixture-of-Experts als erster Schritt).
- **Welche Aufgabe für das Weltmodell?** Die Gitterwelt ist nur grob umrissen: Regeln, Größe und
  Ziel sind offen.
- **Grenzen des Selbstlernens:** Die Konsolidierung lernt nur aus Rätseln, die das Modell mit viel
  Bedenkzeit schon selbst schafft. Wie kommt wirklich Neues hinein?

## Verworfene Ideen

- **Große Module von Grund auf bauen** (Sprache, Wahrnehmung, Weltmodell): verworfen wegen Hardware
  und geringem Erkenntnisgewinn; stattdessen fertig übernehmen und per Adapter anbinden.
- **Weltmodell oder Sprachmodul einfach an Sudoku anhängen:** verworfen, weil Sudoku sie nicht
  braucht. Erst eine passende Aufgabe, dann das Modul.
- **End-to-End-Training des ganzen Systems:** nicht machbar (Werkzeuge nicht differenzierbar),
  daher gestuftes Training plus RL.
- **6×6 als Hauptmaßstab:** aufgegeben, weil das Modell alle Rätsel löst und sich die Module nicht
  mehr unterscheiden lassen.
- **9×9 auf allen Schwierigkeiten trainieren:** bewusst nicht gemacht. Die schweren Rätsel (mehr als
  46 leere Felder) blieben für die Konsolidierung übrig, damit das Modell sie aus eigener Erfahrung
  lernen muss.
- **Attention im Denkkern:** nicht verworfen, aber zurückgestellt. Gewählt wurde ein MLP-Mixer, weil
  TRM damit auf Sudoku am besten abschnitt; der Vergleich steht noch aus.
- **Ein einzelnes biologisch nachgebautes Netz** (Spiking, Predictive Coding als Hauptarchitektur):
  als langfristig interessant eingeordnet, für den Prototyp aber zu weit von brauchbarer Leistung
  entfernt.
