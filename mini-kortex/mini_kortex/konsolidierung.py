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

PROBIEREN IM SCHLAF: Statt nur nachzudenken, darf das Modell tagsüber auch Annahmen treffen und
durchdenken (siehe `weltmodell.py`). So löst es mehr schwere Rätsel, auch solche, die echtes
Probieren verlangen. Geübt wird nachts trotzdem nur die geprüfte Lösung, nicht der Weg dorthin.
Die Frage: Wird aus dem Probieren Intuition, die beim nächsten Mal ohne Annahmen auskommt?

Zum Vergleich gibt es einen LEHRER-Modus: Nachts wird auf *allen* bisher erlebten Rätseln mit den
echten Lösungen geübt, auch auf denen, die das Modell selbst nicht geschafft hat. Der Abstand
zum Selbstlernen zeigt, wie viel es kostet, nur aus eigener, geprüfter Erfahrung zu lernen.
"""

import copy

import numpy as np

from .agent import Agent
from .sudoku import datensatz, varianten
from .training import genauigkeit_pro_schritt, trainingslauf


def wach(modell, raetsel, loesungen, gedaechtnis=None, max_schritte=32, diagonal=False, regel=None):
    """Eine Wachphase. Gibt (Gedächtnis, Ergebnis) zurück.

    `loesungen` dient nur zur Auswertung. Ins Gedächtnis kommt ausschließlich, was der
    Regelprüfer bestätigt hat.
    """
    agent = Agent(modell, metakognition=True, werkzeug=True, gedaechtnis=True, max_schritte=max_schritte,
                  diagonal=diagonal, regel=regel)
    if gedaechtnis is not None:
        agent.gedaechtnis = gedaechtnis
    ergebnis = agent.loesen(raetsel, loesungen)
    return agent.gedaechtnis, ergebnis


def wach_probieren(modell, raetsel, loesungen, max_schritte=32, tiefe=6, breite=2, schritte=16):
    """Wachphase mit Probieren in Gedanken. Rückgabe: (Episoden (Rätsel, Lösungen), Kennzahlen).

    Episoden sind alle geprüften Lösungen, die Nachdenken gekostet haben (mindestens 2 Schritte),
    also auch alle, die nur mit Annahmen gefunden wurden. `loesungen` dient nur zur Auswertung.
    """
    from .weltmodell import probieren
    e = probieren(modell, raetsel, max_schritte, tiefe, breite, schritte)
    nehmen = e["geloest"] & (e["schritte"] >= 2)
    kennzahlen = {"geloest": float(e["geloest"].mean()),
                  "mit_annahmen": int((e["annahmen"] > 0).sum()),
                  "falsch_geprueft": int((e["geloest"] & ~(e["antwort"] == loesungen).all(1)).sum()),
                  "schritte": float(e["schritte"][e["geloest"]].mean()) if e["geloest"].any() else float("nan")}
    return (raetsel[nehmen], e["antwort"][nehmen]), kennzahlen


def schlafen(modell, gedaechtnis, alte=None, anteil_alt=0.5, mindestens_schritte=2, laeufe=300,
             batch=128, schritte=8, lernrate=3e-4, seed=0, protokoll=print, episoden=None, diagonal=False,
             regel_neu=None, strafe=None):
    """Eine Schlafphase. Trainiert eine *Kopie* des Modells und gibt sie zurück.

    alte:                (Rätsel, Lösungen) aus dem ursprünglichen Training, fürs Replay
    anteil_alt:          Anteil alter Rätsel pro Batch (0 = nur neue Erlebnisse)
    mindestens_schritte: nur Episoden abspielen, die so viel Nachdenken gekostet haben.
                         Was sofort klappte, muss man nicht üben.
    episoden:            (Rätsel, Lösungen), die statt des Gedächtnisses geübt werden (Lehrer-Modus)
    diagonal:            Regelvariante der neuen Episoden (siehe `sudoku.py`); abgewandelt wird nur mit
                         Symmetrien, die diese Regel erhalten
    regel_neu:           Regel-Hinweis für die neuen Episoden (die alten bekommen 0 = klassisch).
                         None = Modell ohne Hinweis-Eingang.
    strafe:              zusätzlicher Verlust beim Üben, etwa EWC (siehe `training.ewc_strafe`)
    """
    if episoden is not None:
        neu_r, neu_l = episoden
    else:
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
        if diagonal == "versetzt":
            # Diese Symmetrien erhalten die klassischen Kästen nicht: alte Rätsel klassisch abwandeln.
            r, l = varianten(r, l, rng, diagonal)
            if anzahl_alt:
                j = rng.integers(0, len(alte[0]), anzahl_alt)
                ra, la = varianten(alte[0][j], alte[1][j], rng)
                r, l = np.concatenate([r, ra]), np.concatenate([l, la])
        else:
            if anzahl_alt:
                j = rng.integers(0, len(alte[0]), anzahl_alt)
                r, l = np.concatenate([r, alte[0][j]]), np.concatenate([l, alte[1][j]])
            r, l = varianten(r, l, rng, diagonal)
        if regel_neu is None:
            return r, l
        return r, l, np.array([regel_neu] * anzahl_neu + [0] * anzahl_alt)

    kopie = copy.deepcopy(modell)
    return trainingslauf(kopie, ziehen, laeufe, schritte, lernrate, aufwaermen=20, seed=seed, protokoll=protokoll,
                         strafe=strafe)


def messen(modell, schwer, leicht, max_schritte=32, ausfuehrlich=False):
    """Kennzahlen auf zwei festen Testmengen (beide nie gesehen).

    ausfuehrlich: zusätzlich die Rätsel, die Probieren verlangen (Denktiefe 0, braucht `schwer[2]`),
    und wie viele schwere Rätsel Probieren in Gedanken löst.
    """
    acc_s = genauigkeit_pro_schritt(modell, schwer[0], schwer[1], max_schritte)
    acc_l = genauigkeit_pro_schritt(modell, leicht[0], leicht[1], 8)
    agent = Agent(modell, metakognition=True, werkzeug=True, gedaechtnis=False, max_schritte=max_schritte)
    e = agent.loesen(schwer[0], schwer[1])
    extra = {}
    if ausfuehrlich:
        from .weltmodell import probieren
        probier = schwer[2] == 0
        extra["probier_agent"] = e.richtig[probier].mean()
        p = probieren(modell, schwer[0], max_schritte, 6, 2, 16)
        extra["schwer_probieren"] = (p["geloest"] & (p["antwort"] == schwer[1]).all(1)).mean()
        extra["probier_probieren"] = p["geloest"][probier].mean()
    return extra | {
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


SPALTEN_PROBIEREN = [
    ("probier_agent", "Probier-Rätsel, Agent", "{:.1%}"),
    ("schwer_probieren", "schwer, probieren", "{:.1%}"),
    ("probier_probieren", "Probier-R., probieren", "{:.1%}"),
]


def spalten(werte):
    return SPALTEN + [s for s in SPALTEN_PROBIEREN if s[0] in werte]


def tabelle(zeilen, protokoll=print):
    kopf = f"  {'':<34}" + "".join(f"{titel:>23}" for _, titel, _ in spalten(zeilen[0][1]))
    protokoll(kopf)
    for name, werte in zeilen:
        protokoll(f"  {name:<34}" + "".join(f"{fmt.format(werte[k]):>23}" for k, _, fmt in spalten(werte)))


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
               schritte=8, lehrer=False, protokoll=print, probieren_wach=False, ausfuehrlich=False):
    """Mehrere Wach-Schlaf-Zyklen; nach dem ersten zusätzlich der Vergleich "Schlaf ohne Replay".

    Die Erlebnis-Rätsel werden auf die Zyklen aufgeteilt: Jede Nacht kommt neue Erfahrung dazu.
    `schritte`: Denkschritte pro Rätsel beim nächtlichen Üben, am besten so viele wie im Training.
    `lehrer`: nachts auf allen bisher erlebten Rätseln mit den echten Lösungen üben statt auf dem Gedächtnis.
    `probieren_wach`: tagsüber mit Probieren in Gedanken lösen; geübt werden alle so gefundenen Lösungen.
    `ausfuehrlich`: zusätzliche Kennzahlen zu Rätseln, die Probieren verlangen (siehe `messen`).
    Gibt die Tabellenzeilen und eine Liste mit Kennzahlen pro Wachphase zurück.
    """
    def m(mod):
        return messen(mod, schwer, leicht, max_schritte, ausfuehrlich)

    zeilen = [("Vor dem ersten Schlaf", m(modell))]
    tabelle(zeilen[-1:], protokoll)
    wachphasen = []
    gedaechtnis = None
    gesammelt = (wach_daten[0][:0], wach_daten[1][:0])
    portionen = np.array_split(np.arange(len(wach_daten[0])), zyklen)
    for zyklus, idx in enumerate(portionen, start=1):
        if probieren_wach:
            neu, k = wach_probieren(modell, wach_daten[0][idx], wach_daten[1][idx], max_schritte)
            gesammelt = (np.concatenate([gesammelt[0], neu[0]]), np.concatenate([gesammelt[1], neu[1]]))
            episoden = gesammelt
            wachphasen.append({"zyklus": zyklus, "raetsel": len(idx), "geloest": k["geloest"], "schritte": k["schritte"],
                               "episoden": len(gesammelt[0]), "mit_annahmen": k["mit_annahmen"],
                               "falsch_geprueft": k["falsch_geprueft"]})
            protokoll(f"\nTag {zyklus}: {len(idx)} neue schwere Rätsel, {k['geloest']:.1%} mit Probieren gelöst und "
                      f"geprüft, davon {k['mit_annahmen']} nur mit Annahmen. Episoden: {len(gesammelt[0])}. "
                      f"Falsch, aber geprüft: {k['falsch_geprueft']}")
        else:
            gedaechtnis, e = wach(modell, wach_daten[0][idx], wach_daten[1][idx], gedaechtnis, max_schritte)
            bisher = np.concatenate(portionen[:zyklus])
            episoden = (wach_daten[0][bisher], wach_daten[1][bisher]) if lehrer else None
            falsch_geprueft = int((e.geprueft & ~e.richtig).sum())
            wachphasen.append({"zyklus": zyklus, "raetsel": len(idx), "geloest": float(e.geprueft.mean()),
                               "schritte": float(e.schritte[e.geprueft].mean()) if e.geprueft.any() else float("nan"),
                               "episoden": len(gedaechtnis), "falsch_geprueft": falsch_geprueft})
            protokoll(f"\nTag {zyklus}: {len(idx)} neue schwere Rätsel, {e.geprueft.mean():.1%} selbst gelöst und "
                      f"geprüft (Ø {wachphasen[-1]['schritte']:.1f} Schritte). Gedächtnis: {len(gedaechtnis)} Episoden. "
                      f"Falsch, aber geprüft: {falsch_geprueft}")
        if zyklus == 1:
            ohne = schlafen(modell, gedaechtnis, alte=None, laeufe=laeufe, schritte=schritte, protokoll=protokoll,
                            episoden=episoden)
            zeilen.append(("Nacht 1, nur neue Erlebnisse", m(ohne)))
            tabelle(zeilen[-1:], protokoll)
        modell = schlafen(modell, gedaechtnis, alte=alte, laeufe=laeufe, schritte=schritte, seed=zyklus,
                          protokoll=protokoll, episoden=episoden)
        zeilen.append((f"Nacht {zyklus}, mit Replay", m(modell)))
        tabelle(zeilen[-1:], protokoll)
    return modell, zeilen, wachphasen
