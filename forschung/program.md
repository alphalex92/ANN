# Forschungsauftrag: Bytes-Erkenner verbessern

Du bist ein autonomer ML-Forscher. Dein Ziel: Der Bytes-Erkenner soll in **10 Sekunden
Training** eine möglichst hohe Genauigkeit erreichen. Gemessen wird das auf einem festen Testset
und gemittelt über 3 Seeds.

## Spielregeln

**Du darfst ändern:** `bytes_erkenner/netz.py` und `bytes_erkenner/training.py`, also Architektur,
Eingabe-Kodierung, Optimizer, Lernrate, Batchgröße, Datenmenge, Epochen und Regularisierung.

**Gesperrt** (die Auswertung verweigert sonst das Ergebnis):
- `bytes_erkenner/daten.py`: der Datengenerator und damit die Aufgabe selbst
- `forschung/`: diese Anleitung und die Auswertung
- `tests/`

**Außerdem verboten**, auch wenn es technisch ginge:
- das Testset nachbauen oder darauf trainieren (also nie `TEST_SEED` oder `auswertung.py` verwenden)
- Zeitmessung austricksen, z. B. Arbeit vor `fuer_auswertung` verlagern, Ergebnisse
  zwischenspeichern oder Dateien mit vortrainierten Gewichten anlegen
- neue Abhängigkeiten: Es gibt nur numpy und die Standardbibliothek
- Netzwerkzugriffe und alles außerhalb dieses Repos

Der Vertrag bleibt gleich: `fuer_auswertung(sekunden, seed)` gibt eine Funktion zurück, die
uint8-Bytes der Form (N, 8) auf Klassen-Indizes der Form (N,) abbildet. Außerdem muss `pytest -q`
grün bleiben. Die Tests nutzen `Netz(...)`, `verlust_und_gradienten`, `trainieren(...)`,
`speichern` und `laden`. Diese Schnittstellen müssen also erhalten bleiben.

## Start (einmalig)

1. `git switch -c forschung/<datum>` (z. B. `forschung/2026-11-02`)
2. `git tag -f forschung-start`
3. Baseline messen: `.venv/bin/python -m forschung.auswertung`
4. `ergebnisse.tsv` anlegen (ist in `.gitignore`, wird nicht committet) mit der Kopfzeile
   `commit	genauigkeit	streuung	sekunden	status	beschreibung`
   und der Baseline als erster Zeile mit Status `keep`.

## Die Schleife

Wiederhole, bis 30 Experimente gelaufen sind oder ich dich stoppe:

1. **Idee wählen.** Lies dafür `ergebnisse.tsv`. Pro Experiment ändest du genau eine Sache, damit
   klar bleibt, was gewirkt hat.
2. **Ändern** und `git commit -am "<kurze Beschreibung>"`.
3. **Prüfen:** `.venv/bin/python -m pytest -q`. Sind die Tests rot, reparierst du kurz oder
   verwirfst die Änderung.
4. **Messen:** `.venv/bin/python -m forschung.auswertung` gibt eine JSON-Zeile aus.
5. **Entscheiden:**
   - `status` ist `ok` **und** `genauigkeit` liegt um mindestens **0.002** über dem bisher besten
     `keep`: Die Änderung bleibt, Status `keep`.
   - sonst: `git reset --hard HEAD~1`, Status `discard`. Bei `crash` oder `ungueltig` setzt du
     diesen Status.
6. **Protokollieren:** Hänge eine Zeile an `ergebnisse.tsv` an, mit dem Kurz-Hash des Commits,
   den Werten aus der JSON-Zeile und einer Beschreibung in einem Satz.

Frag zwischendurch nicht nach, ob du weitermachen sollst. Arbeite selbstständig weiter.
Am Ende schreibst du eine kurze Zusammenfassung: was gewirkt hat, was nicht und was du als
Nächstes probieren würdest.

## Ideen, falls dir keine einfallen

Lernrate oder Lernraten-Plan, Momentum, Adam, Batchgröße, Breite der versteckten Schicht,
eine zweite Schicht, mehr oder weniger Trainingsdaten im Zeitbudget, Gewichts-Regularisierung,
andere Eingabe-Kodierung (Bits, Bytes als Zahlen, beides), Initialisierung, float32 statt float64
für mehr Tempo.

Denk daran: Das Ergebnis streut. Ein Gewinn knapp über 0.002 kann Glück sein. Wenn etwas
überraschend gut aussieht, miss es ruhig noch einmal.
