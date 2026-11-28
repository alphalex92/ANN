"""Umlernen: eine neue Regel lernen, ohne die alte zu vergessen.

Bisher wurden die Rätsel in der Konsolidierung nur *schwerer*, die Regeln blieben gleich. Jetzt
ändert sich die Welt: Das Modell kennt klassisches Sudoku und bekommt **Diagonal-Sudoku**, bei dem
zusätzlich beide Hauptdiagonalen jede Ziffer genau einmal enthalten müssen. Der Regelprüfer kennt
die neue Regel, das Netz nicht. Es gibt auch keinen Hinweis im Rätsel, welche Regel gilt.

Drei Fragen:

1. **Selbstlernen**: Kann sich das Modell die neue Regel beibringen, wenn nur geprüft richtige
   eigene Lösungen geübt werden? Oder löst es zu wenig, um überhaupt etwas zu lernen?
2. **Vergessen**: Diesmal widerspricht das Neue teilweise dem Alten. Kann das Modell danach noch
   klassisches Sudoku?
3. **Replay**: Hilft es jetzt, alte Rätsel in jede Nacht zu mischen?

Verglichen werden vier Bedingungen: selbst oder mit Lehrer (echte Lösungen für alle erlebten
Diagonal-Rätsel), jeweils mit und ohne Replay alter klassischer Rätsel.

Mit `neu="versetzt"` ist die neue Regel **unfreundlich**: Statt eine Regel dazuzunehmen, gelten
andere Kästen (versetzt um eine Zeile und Spalte, siehe `sudoku.py`). Kein einziges dieser Rätsel ist
nach den alten Regeln lösbar. Kann das Modell sich dann noch selbst etwas beibringen?

Mit `hinweis=True` bekommt das Modell zusätzlich einen **Regel-Hinweis**: Es erfährt bei jedem Rätsel,
ob es klassisch (0) oder diagonal (1) ist. Liegt das Vergessen ohne Replay nur daran, dass das Netz
die beiden Regeln nicht auseinanderhalten kann?
"""

import copy

import numpy as np

from .agent import Agent
from .konsolidierung import schlafen, wach
from .modell import mit_hinweis, regel_tensor
from .sudoku import anzahl_loesungen, datensatz, gueltig_batch
from .training import genauigkeit_pro_schritt


def nur_mehrdeutige(anzahl, rng, **kw):
    """Diagonal-Rätsel, die nach den *klassischen* Regeln mehrere Lösungen haben.

    Bei diesen hilft die alte Regel allein nicht: Ein Modell, das nur klassisches Sudoku kann, findet
    höchstens zufällig die Lösung, die auch die Diagonalen erfüllt. Etwa 30 % aller Diagonal-Rätsel
    sind so. Es wird in Portionen erzeugt und gefiltert, bis genug beisammen sind.
    """
    teile_r, teile_l, teile_t, gesamt = [], [], [], 0
    while gesamt < anzahl:
        r, l, t = datensatz(max(64, 4 * (anzahl - gesamt)), rng, diagonal=True, **kw)
        mehrdeutig = np.array([anzahl_loesungen(x) > 1 for x in r])
        teile_r.append(r[mehrdeutig]), teile_l.append(l[mehrdeutig]), teile_t.append(t[mehrdeutig])
        gesamt += int(mehrdeutig.sum())
    return (np.concatenate(teile_r)[:anzahl], np.concatenate(teile_l)[:anzahl], np.concatenate(teile_t)[:anzahl])


def umlernen_daten(n, anzahl_wach, anzahl_test, leer_min, leer_max, seed=0, prozesse=4, mehrdeutig=False, neu=True):
    """Rätsel der neuen Regel zum Erleben und zum Testen, dazu klassische Testrätsel gleicher Schwere.

    `mehrdeutig`: nur Diagonal-Rätsel, die die alte Regel nicht eindeutig löst (siehe `nur_mehrdeutige`).
    `neu`: die neue Regelvariante (True = Diagonal-Sudoku, "versetzt" = versetzte Kästen).
    """
    kw = dict(n=n, leer_min=leer_min, leer_max=leer_max, prozesse=prozesse)
    erzeugen = (lambda a, rng: nur_mehrdeutige(a, rng, **kw)) if mehrdeutig else \
        (lambda a, rng: datensatz(a, rng, diagonal=neu, **kw))
    wach_daten = erzeugen(anzahl_wach, np.random.default_rng(seed + 5_000_000))
    diag = erzeugen(anzahl_test, np.random.default_rng(seed + 6_000_000))
    klass = datensatz(anzahl_test, np.random.default_rng(seed + 7_000_000), **kw)
    return wach_daten, diag, klass


def entwuerfe(modell, raetsel, schritte, batch=500, regel=None):
    """Die Antwort des Netzes nach genau `schritte` Denkschritten (ohne Agent)."""
    import torch
    modell.eval()
    teile = []
    with torch.no_grad():
        for i in range(0, len(raetsel), batch):
            r = torch.as_tensor(raetsel[i:i + batch]).to(modell.geraet)
            hinweis = regel_tensor(regel, len(r), modell.geraet)
            for s, (y, *_rest) in enumerate(modell.denken(r, schritte, hinweis), start=1):
                if s == schritte:
                    teile.append(modell.ablesen(r, y).cpu().numpy())
    return np.concatenate(teile)


def messen(modell, diag, klass, max_schritte=32, schritte=8, neu=True):
    """Kennzahlen auf Diagonal- und klassischen Testrätseln (beide nie erlebt).
    Hat das Modell einen Hinweis-Eingang, bekommt es den richtigen Hinweis (1 diagonal, 0 klassisch)."""
    rd, rk = (1, 0) if modell.hat_hinweis else (None, None)
    acc_d = genauigkeit_pro_schritt(modell, diag[0], diag[1], schritte, regel=rd)
    acc_k = genauigkeit_pro_schritt(modell, klass[0], klass[1], schritte, regel=rk)
    agent_d = Agent(modell, gedaechtnis=False, max_schritte=max_schritte, diagonal=neu,
                    regel=rd).loesen(diag[0], diag[1])
    agent_k = Agent(modell, gedaechtnis=False, max_schritte=max_schritte, regel=rk).loesen(klass[0], klass[1])
    # Woran scheitert das Modell bei den Diagonal-Rätseln? Ist der Entwurf eine *klassisch* gültige
    # Lösung, hat es die alte Regel angewandt und nur die Diagonalen ignoriert.
    e = entwuerfe(modell, diag[0], schritte, regel=rd)
    alte_regel = gueltig_batch(e, diag[0]) & ~(e == diag[1]).all(1)
    return {
        "diag_8": acc_d[-1],
        "diag_agent": agent_d.richtig.mean(),
        "diag_alte_regel": alte_regel.mean(),
        "klass_8": acc_k[-1],
        "klass_agent": agent_k.richtig.mean(),
    }


SPALTEN = [
    ("diag_8", "Diagonal, 8 Schritte", "{:.1%}"),
    ("diag_agent", "Diagonal, Agent", "{:.1%}"),
    ("diag_alte_regel", "Diag.: alte Regel", "{:.1%}"),
    ("klass_8", "Klassisch, 8 Schritte", "{:.1%}"),
    ("klass_agent", "Klassisch, Agent", "{:.1%}"),
]


def spalten(neu=True):
    """Die Spalten mit dem Namen der neuen Regel (die Schlüssel bleiben diag_…)."""
    if neu is True:
        return SPALTEN
    return [(k, t.replace("Diagonal", "Versetzt").replace("Diag.", "Vers."), f) for k, t, f in SPALTEN]


def tabelle(zeilen, protokoll=print, neu=True):
    protokoll(f"  {'':<36}" + "".join(f"{titel:>23}" for _, titel, _ in spalten(neu)))
    for name, werte in zeilen:
        protokoll(f"  {name:<36}" + "".join(f"{fmt.format(werte[k]):>23}" for k, _, fmt in spalten(neu)))


def bedingung(modell, alte, wach_daten, diag, klass, lehrer, replay, zyklen=3, laeufe=300, schritte=8,
              max_schritte=32, seed=0, hinweis=False, protokoll=print, anteil_alt=0.5, strafe=None, neu=True):
    """Mehrere Wach-Schlaf-Zyklen auf Diagonal-Rätseln. Gibt (Modell, Zeilen, Wachphasen) zurück.
    `hinweis`: das Modell vorher um einen Regel-Hinweis-Eingang erweitern (startet bei null).
    `anteil_alt`: Anteil der Replay-Rätsel pro Batch. `strafe`: zusätzlicher Verlust im Schlaf (EWC)."""
    modell = mit_hinweis(modell) if hinweis else copy.deepcopy(modell)
    regel = 1 if hinweis else None
    zeilen, wachphasen = [], []
    gedaechtnis = None
    portionen = np.array_split(np.arange(len(wach_daten[0])), zyklen)
    for zyklus, idx in enumerate(portionen, start=1):
        gedaechtnis, e = wach(modell, wach_daten[0][idx], wach_daten[1][idx], gedaechtnis, max_schritte,
                              diagonal=neu, regel=regel)
        falsch_geprueft = int((e.geprueft & ~e.richtig).sum())
        wachphasen.append({"zyklus": zyklus, "geloest": float(e.geprueft.mean()), "episoden": len(gedaechtnis),
                           "falsch_geprueft": falsch_geprueft})
        protokoll(f"Tag {zyklus}: {e.geprueft.mean():.1%} der neuen Rätsel selbst gelöst und geprüft. "
                  f"Gedächtnis: {len(gedaechtnis)} Episoden. Falsch, aber geprüft: {falsch_geprueft}")
        bisher = np.concatenate(portionen[:zyklus])
        episoden = (wach_daten[0][bisher], wach_daten[1][bisher]) if lehrer else None
        modell = schlafen(modell, gedaechtnis, alte=alte if replay else None, laeufe=laeufe, schritte=schritte,
                          seed=1000 * seed + zyklus, protokoll=protokoll, episoden=episoden, diagonal=neu,
                          regel_neu=regel, anteil_alt=anteil_alt, strafe=strafe)
        zeilen.append((f"Nacht {zyklus}", messen(modell, diag, klass, max_schritte, neu=neu)))
        tabelle(zeilen[-1:], protokoll, neu)
    return modell, zeilen, wachphasen


BEDINGUNGEN = [
    ("selbst, mit Replay", False, True),
    ("selbst, ohne Replay", False, False),
    ("Lehrer, mit Replay", True, True),
    ("Lehrer, ohne Replay", True, False),
]


def experiment(modell, alte, wach_daten, diag, klass, zyklen=3, laeufe=300, schritte=8, max_schritte=32,
               seed=0, hinweis=False, protokoll=print, neu=True):
    """Alle vier Bedingungen, jeweils vom selben Ausgangsmodell. Gibt ein Ergebnis-Dict zurück."""
    start = messen(modell, diag, klass, max_schritte, neu=neu)
    protokoll("Ausgangsmodell (kennt nur klassisches Sudoku):")
    tabelle([("vorher", start)], protokoll, neu)
    ergebnis = {"vorher": start, "bedingungen": {}}
    for name, lehrer, replay in BEDINGUNGEN:
        protokoll(f"\n=== {name} ===")
        _, zeilen, wachphasen = bedingung(modell, alte, wach_daten, diag, klass, lehrer, replay, zyklen, laeufe,
                                          schritte, max_schritte, seed, hinweis, protokoll, neu=neu)
        ergebnis["bedingungen"][name] = {"zeilen": zeilen, "wachphasen": wachphasen}
    protokoll("\nZusammenfassung nach der letzten Nacht:")
    tabelle([("vorher", start)] + [(name, b["zeilen"][-1][1]) for name, b in ergebnis["bedingungen"].items()],
            protokoll, neu)
    return ergebnis


def eigene_erinnerungen(modell, raetsel, loesungen, anzahl, max_schritte=32):
    """Ein kleiner Vorrat eigener Erlebnisse mit klassischem Sudoku, statt der Trainingsdaten.

    Das Modell löst `anzahl` klassische Rätsel selbst. Was der Regelprüfer bestätigt, wird mit der
    *eigenen* Antwort aufbewahrt, so wie der Hippocampus Erlebtes speichert, nicht das Lehrbuch.
    `loesungen` dient nur zur Auswertung.
    """
    gedaechtnis, e = wach(modell, raetsel[:anzahl], loesungen[:anzahl], max_schritte=max_schritte)
    r, l, _ = gedaechtnis.episoden(0)
    return r, l


def vergessen_bedingungen(modell, alte, anzahl_fisher=500, staerken=(1e2, 1e3, 1e4), max_schritte=32,
                          schritte=8, protokoll=print):
    """Die Bedingungen für das Vergessen-Experiment: (Name, Replay-Rätsel oder None, Anteil, Strafe)."""
    from .training import ewc_strafe
    gross = [("ohne Replay", None, 0.5, None),
             ("Trainingsset, 50 % pro Batch", alte, 0.5, None),
             ("Trainingsset, 10 % pro Batch", alte, 0.1, None)]
    for anzahl in (500, 100):
        r, l = eigene_erinnerungen(modell, alte[0], alte[1], anzahl, max_schritte)
        protokoll(f"Eigene Erinnerungen: {len(r)} von {anzahl} klassischen Rätseln selbst gelöst und geprüft.")
        gross.append((f"{len(r)} eigene Erinnerungen, 50 %", (r, l), 0.5, None))
    for staerke in staerken:
        strafe = ewc_strafe(modell, alte[0][:anzahl_fisher], alte[1][:anzahl_fisher], staerke, schritte)
        gross.append((f"EWC {staerke:g}, ohne Replay", None, 0.5, strafe))
    return gross


def vergessen_experiment(modell, alte, wach_daten, diag, klass, zyklen=3, laeufe=300, schritte=8, max_schritte=32,
                         seed=0, protokoll=print):
    """Selbstlernen der Diagonalregel unter verschiedenen Gegenmitteln gegen das Vergessen."""
    start = messen(modell, diag, klass, max_schritte)
    protokoll("Ausgangsmodell (kennt nur klassisches Sudoku):")
    tabelle([("vorher", start)], protokoll)
    ergebnis = {"vorher": start, "bedingungen": {}}
    for name, replay, anteil, strafe in vergessen_bedingungen(modell, alte, max_schritte=max_schritte,
                                                              schritte=schritte, protokoll=protokoll):
        protokoll(f"\n=== {name} ===")
        _, zeilen, wachphasen = bedingung(modell, replay, wach_daten, diag, klass, False, replay is not None, zyklen,
                                          laeufe, schritte, max_schritte, seed, False, protokoll, anteil, strafe)
        ergebnis["bedingungen"][name] = {"zeilen": zeilen, "wachphasen": wachphasen}
    protokoll("\nZusammenfassung nach der letzten Nacht:")
    tabelle([("vorher", start)] + [(name, b["zeilen"][-1][1]) for name, b in ergebnis["bedingungen"].items()],
            protokoll)
    return ergebnis
