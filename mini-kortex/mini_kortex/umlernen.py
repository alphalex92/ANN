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
"""

import copy

import numpy as np

from .agent import Agent
from .konsolidierung import schlafen, wach
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


def umlernen_daten(n, anzahl_wach, anzahl_test, leer_min, leer_max, seed=0, prozesse=4, mehrdeutig=False):
    """Diagonal-Rätsel zum Erleben und zum Testen, dazu klassische Testrätsel gleicher Schwere.

    `mehrdeutig`: nur Diagonal-Rätsel, die die alte Regel nicht eindeutig löst (siehe `nur_mehrdeutige`).
    """
    kw = dict(n=n, leer_min=leer_min, leer_max=leer_max, prozesse=prozesse)
    erzeugen = (lambda a, rng: nur_mehrdeutige(a, rng, **kw)) if mehrdeutig else \
        (lambda a, rng: datensatz(a, rng, diagonal=True, **kw))
    wach_daten = erzeugen(anzahl_wach, np.random.default_rng(seed + 5_000_000))
    diag = erzeugen(anzahl_test, np.random.default_rng(seed + 6_000_000))
    klass = datensatz(anzahl_test, np.random.default_rng(seed + 7_000_000), **kw)
    return wach_daten, diag, klass


def entwuerfe(modell, raetsel, schritte, batch=500):
    """Die Antwort des Netzes nach genau `schritte` Denkschritten (ohne Agent)."""
    import torch
    modell.eval()
    teile = []
    with torch.no_grad():
        for i in range(0, len(raetsel), batch):
            r = torch.as_tensor(raetsel[i:i + batch]).to(modell.geraet)
            for s, (y, *_rest) in enumerate(modell.denken(r, schritte), start=1):
                if s == schritte:
                    teile.append(modell.ablesen(r, y).cpu().numpy())
    return np.concatenate(teile)


def messen(modell, diag, klass, max_schritte=32, schritte=8):
    """Kennzahlen auf Diagonal- und klassischen Testrätseln (beide nie erlebt)."""
    acc_d = genauigkeit_pro_schritt(modell, diag[0], diag[1], schritte)
    acc_k = genauigkeit_pro_schritt(modell, klass[0], klass[1], schritte)
    agent_d = Agent(modell, gedaechtnis=False, max_schritte=max_schritte, diagonal=True).loesen(diag[0], diag[1])
    agent_k = Agent(modell, gedaechtnis=False, max_schritte=max_schritte).loesen(klass[0], klass[1])
    # Woran scheitert das Modell bei den Diagonal-Rätseln? Ist der Entwurf eine *klassisch* gültige
    # Lösung, hat es die alte Regel angewandt und nur die Diagonalen ignoriert.
    e = entwuerfe(modell, diag[0], schritte)
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


def tabelle(zeilen, protokoll=print):
    protokoll(f"  {'':<36}" + "".join(f"{titel:>23}" for _, titel, _ in SPALTEN))
    for name, werte in zeilen:
        protokoll(f"  {name:<36}" + "".join(f"{fmt.format(werte[k]):>23}" for k, _, fmt in SPALTEN))


def bedingung(modell, alte, wach_daten, diag, klass, lehrer, replay, zyklen=3, laeufe=300, schritte=8,
              max_schritte=32, seed=0, protokoll=print):
    """Mehrere Wach-Schlaf-Zyklen auf Diagonal-Rätseln. Gibt (Modell, Zeilen, Wachphasen) zurück."""
    modell = copy.deepcopy(modell)
    zeilen, wachphasen = [], []
    gedaechtnis = None
    portionen = np.array_split(np.arange(len(wach_daten[0])), zyklen)
    for zyklus, idx in enumerate(portionen, start=1):
        gedaechtnis, e = wach(modell, wach_daten[0][idx], wach_daten[1][idx], gedaechtnis, max_schritte,
                              diagonal=True)
        falsch_geprueft = int((e.geprueft & ~e.richtig).sum())
        wachphasen.append({"zyklus": zyklus, "geloest": float(e.geprueft.mean()), "episoden": len(gedaechtnis),
                           "falsch_geprueft": falsch_geprueft})
        protokoll(f"Tag {zyklus}: {e.geprueft.mean():.1%} der neuen Diagonal-Rätsel selbst gelöst und geprüft. "
                  f"Gedächtnis: {len(gedaechtnis)} Episoden. Falsch, aber geprüft: {falsch_geprueft}")
        bisher = np.concatenate(portionen[:zyklus])
        episoden = (wach_daten[0][bisher], wach_daten[1][bisher]) if lehrer else None
        modell = schlafen(modell, gedaechtnis, alte=alte if replay else None, laeufe=laeufe, schritte=schritte,
                          seed=1000 * seed + zyklus, protokoll=protokoll, episoden=episoden, diagonal=True)
        zeilen.append((f"Nacht {zyklus}", messen(modell, diag, klass, max_schritte)))
        tabelle(zeilen[-1:], protokoll)
    return modell, zeilen, wachphasen


BEDINGUNGEN = [
    ("selbst, mit Replay", False, True),
    ("selbst, ohne Replay", False, False),
    ("Lehrer, mit Replay", True, True),
    ("Lehrer, ohne Replay", True, False),
]


def experiment(modell, alte, wach_daten, diag, klass, zyklen=3, laeufe=300, schritte=8, max_schritte=32,
               seed=0, protokoll=print):
    """Alle vier Bedingungen, jeweils vom selben Ausgangsmodell. Gibt ein Ergebnis-Dict zurück."""
    start = messen(modell, diag, klass, max_schritte)
    protokoll("Ausgangsmodell (kennt nur klassisches Sudoku):")
    tabelle([("vorher", start)], protokoll)
    ergebnis = {"vorher": start, "bedingungen": {}}
    for name, lehrer, replay in BEDINGUNGEN:
        protokoll(f"\n=== {name} ===")
        _, zeilen, wachphasen = bedingung(modell, alte, wach_daten, diag, klass, lehrer, replay, zyklen, laeufe,
                                          schritte, max_schritte, seed, protokoll)
        ergebnis["bedingungen"][name] = {"zeilen": zeilen, "wachphasen": wachphasen}
    protokoll("\nZusammenfassung nach der letzten Nacht:")
    tabelle([("vorher", start)] + [(name, b["zeilen"][-1][1]) for name, b in ergebnis["bedingungen"].items()],
            protokoll)
    return ergebnis
