"""Konsolidierung ("Schlaf"): aus eigener Erfahrung lernen, ohne Altes zu vergessen.

Vorbild ist die Theorie der *komplementären Lernsysteme* (McClelland, McNaughton & O'Reilly 1995):
Der Hippocampus speichert Erlebnisse sofort, der Neokortex lernt langsam. Im Schlaf spielt der
Hippocampus Erlebnisse erneut ab, und zwar vermischt mit älterem Wissen. Nur so kann der Kortex
Neues aufnehmen, ohne Altes zu überschreiben.

Hier heißt das:

    WACH     Das Modell bekommt neue, schwerere Rätsel als im Training. Mit viel Bedenkzeit,
             Metakognition und Regelprüfer löst es einen Teil davon. Jede vom Prüfer bestätigte
             Lösung landet im Gedächtnis, zusammen mit der Zahl der Denkschritte.

    SCHLAF   Das Modell wird auf diesen Episoden nachtrainiert. Die "richtigen Antworten" stammen
             nur aus eigener, geprüfter Erfahrung; es gibt keinen Lehrer. Jede Episode wird dabei
             in Symmetrie-Varianten abgespielt (wie Träume, die Erlebtes abwandeln), und auf Wunsch
             mit alten Trainingsrätseln vermischt (Replay).

    MESSEN   Löst das Modell danach *andere*, nie gesehene schwere Rätsel schneller und öfter?
             Und kann es die leichten noch?
"""

import copy

import numpy as np

from .agent import Agent
from .sudoku import datensatz, varianten
from .training import genauigkeit_pro_schritt, trainingslauf


def wach(modell, raetsel, loesungen, gedaechtnis=None, max_schritte=32):
    """Eine Wachphase. Gibt (Gedächtnis, Ergebnis) zurück.

    `loesungen` dient nur zur Auswertung. Ins Gedächtnis kommt ausschließlich, was der
    Regelprüfer bestätigt hat.
    """
    agent = Agent(modell, metakognition=True, werkzeug=True, gedaechtnis=True, max_schritte=max_schritte)
    if gedaechtnis is not None:
        agent.gedaechtnis = gedaechtnis
    ergebnis = agent.loesen(raetsel, loesungen)
    return agent.gedaechtnis, ergebnis


def schlafen(modell, gedaechtnis, alte=None, anteil_alt=0.5, mindestens_schritte=2, laeufe=300,
             batch=128, schritte=8, lernrate=3e-4, seed=0, protokoll=print):
    """Eine Schlafphase. Trainiert eine *Kopie* des Modells und gibt sie zurück.

    alte:                (Rätsel, Lösungen) aus dem ursprünglichen Training, fürs Replay
    anteil_alt:          Anteil alter Rätsel pro Batch (0 = nur neue Erlebnisse)
    mindestens_schritte: nur Episoden abspielen, die so viel Nachdenken gekostet haben.
                         Was sofort klappte, muss man nicht üben.
    """
    neu_r, neu_l, _ = gedaechtnis.episoden(mindestens_schritte)
    if len(neu_r) == 0:
        protokoll("Keine passenden Episoden im Gedächtnis, Schlaf fällt aus.")
        return copy.deepcopy(modell)
    anzahl_alt = int(round(batch * anteil_alt)) if alte is not None else 0
    anzahl_neu = batch - anzahl_alt
    protokoll(f"Schlaf: {len(neu_r)} Episoden, pro Batch {anzahl_neu} neue + {anzahl_alt} alte Rätsel, "
              f"{laeufe} Läufe")

    def ziehen(rng):
        i = rng.integers(0, len(neu_r), anzahl_neu)
        r, l = neu_r[i], neu_l[i]
        if anzahl_alt:
            j = rng.integers(0, len(alte[0]), anzahl_alt)
            r, l = np.concatenate([r, alte[0][j]]), np.concatenate([l, alte[1][j]])
        return varianten(r, l, rng)

    kopie = copy.deepcopy(modell)
    return trainingslauf(kopie, ziehen, laeufe, schritte, lernrate, aufwaermen=20, seed=seed, protokoll=protokoll)


def messen(modell, schwer, leicht, max_schritte=32):
    """Kennzahlen auf zwei festen Testmengen (beide nie gesehen)."""
    acc_s = genauigkeit_pro_schritt(modell, schwer[0], schwer[1], max_schritte)
    acc_l = genauigkeit_pro_schritt(modell, leicht[0], leicht[1], 8)
    agent = Agent(modell, metakognition=True, werkzeug=True, gedaechtnis=False, max_schritte=max_schritte)
    e = agent.loesen(schwer[0], schwer[1])
    return {
        "schwer_8": acc_s[7],
        "schwer_max": acc_s[-1],
        "schwer_agent": e.richtig.mean(),
        "schwer_schritte": e.schritte[e.richtig].mean() if e.richtig.any() else float("nan"),
        "leicht_1": acc_l[0],
        "leicht_8": acc_l[-1],
    }


SPALTEN = [
    ("schwer_8", "schwer, 8 Schritte", "{:.1%}"),
    ("schwer_max", "schwer, 32 Schritte", "{:.1%}"),
    ("schwer_agent", "schwer, Agent", "{:.1%}"),
    ("schwer_schritte", "Ø Schritte (gelöst)", "{:.1f}"),
    ("leicht_1", "leicht, 1 Schritt", "{:.1%}"),
    ("leicht_8", "leicht, 8 Schritte", "{:.1%}"),
]


def tabelle(zeilen, protokoll=print):
    kopf = f"  {'':<34}" + "".join(f"{titel:>21}" for _, titel, _ in SPALTEN)
    protokoll(kopf)
    for name, werte in zeilen:
        protokoll(f"  {name:<34}" + "".join(f"{fmt.format(werte[k]):>21}" for k, _, fmt in SPALTEN))


def experiment_daten(n, leer_basis, anzahl_wach, anzahl_test, seed=0, prozesse=4):
    """Drei frische Rätselmengen: zum Erleben (schwer), zum Testen (schwer) und zum Testen (leicht)."""
    leer_max = {6: 28, 9: 58}[n]
    wach_daten = datensatz(anzahl_wach, np.random.default_rng(seed + 2_000_000), n=n,
                           leer_min=leer_basis + 1, leer_max=leer_max, prozesse=prozesse)
    schwer = datensatz(anzahl_test, np.random.default_rng(seed + 3_000_000), n=n,
                       leer_min=leer_basis + 1, leer_max=leer_max, prozesse=prozesse)
    leicht = datensatz(anzahl_test, np.random.default_rng(seed + 4_000_000), n=n,
                       leer_max=leer_basis, prozesse=prozesse)
    return wach_daten, schwer, leicht


def experiment(modell, alte, wach_daten, schwer, leicht, zyklen=3, laeufe=300, max_schritte=32,
               schritte=8, protokoll=print):
    """Mehrere Wach-Schlaf-Zyklen; nach dem ersten zusätzlich der Vergleich "Schlaf ohne Replay".

    Die Erlebnis-Rätsel werden auf die Zyklen aufgeteilt: Jede Nacht kommt neue Erfahrung dazu.
    `schritte`: Denkschritte pro Rätsel beim nächtlichen Üben, am besten so viele wie im Training.
    Gibt die Tabellenzeilen und eine Liste mit Kennzahlen pro Wachphase zurück.
    """
    zeilen = [("Vor dem ersten Schlaf", messen(modell, schwer, leicht, max_schritte))]
    tabelle(zeilen[-1:], protokoll)
    wachphasen = []
    gedaechtnis = None
    portionen = np.array_split(np.arange(len(wach_daten[0])), zyklen)
    for zyklus, idx in enumerate(portionen, start=1):
        gedaechtnis, e = wach(modell, wach_daten[0][idx], wach_daten[1][idx], gedaechtnis, max_schritte)
        falsch_geprueft = int((e.geprueft & ~e.richtig).sum())
        wachphasen.append({"zyklus": zyklus, "raetsel": len(idx), "geloest": float(e.geprueft.mean()),
                           "schritte": float(e.schritte[e.geprueft].mean()) if e.geprueft.any() else float("nan"),
                           "episoden": len(gedaechtnis), "falsch_geprueft": falsch_geprueft})
        protokoll(f"\nTag {zyklus}: {len(idx)} neue schwere Rätsel, {e.geprueft.mean():.1%} selbst gelöst und geprüft "
                  f"(Ø {wachphasen[-1]['schritte']:.1f} Schritte). Gedächtnis: {len(gedaechtnis)} Episoden. "
                  f"Falsch, aber geprüft: {falsch_geprueft}")
        if zyklus == 1:
            ohne = schlafen(modell, gedaechtnis, alte=None, laeufe=laeufe, schritte=schritte, protokoll=protokoll)
            zeilen.append(("Nacht 1, nur neue Erlebnisse", messen(ohne, schwer, leicht, max_schritte)))
            tabelle(zeilen[-1:], protokoll)
        modell = schlafen(modell, gedaechtnis, alte=alte, laeufe=laeufe, schritte=schritte, seed=zyklus,
                          protokoll=protokoll)
        zeilen.append((f"Nacht {zyklus}, mit Replay", messen(modell, schwer, leicht, max_schritte)))
        tabelle(zeilen[-1:], protokoll)
    return modell, zeilen, wachphasen
