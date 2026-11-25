import numpy as np
import torch

from mini_kortex.agent import Agent
from mini_kortex.gedaechtnis import HopfieldGedaechtnis
from mini_kortex.modell import DenkModell, verlust
from mini_kortex.sudoku import (anzahl_loesungen, datensatz, denktiefe, gueltig_batch, ist_gueltige_loesung,
                             regelverstoesse, zufaelliges_gitter)


def test_zufaellige_gitter_sind_gueltig():
    rng = np.random.default_rng(0)
    for _ in range(50):
        assert ist_gueltige_loesung(zufaelliges_gitter(rng))


def test_raetsel_sind_eindeutig_und_passen_zur_loesung():
    raetsel, loesungen, tiefe = datensatz(30, np.random.default_rng(1))
    for r, l, t in zip(raetsel, loesungen, tiefe):
        assert anzahl_loesungen(r) == 1
        assert ist_gueltige_loesung(l, r)
        assert t == denktiefe(r) and t >= 0  # 0 = einfache Regeln reichen nicht


def test_regelpruefer_erkennt_fehler():
    g = zufaelliges_gitter(np.random.default_rng(2))
    kaputt = g.copy()
    kaputt[0], kaputt[1] = kaputt[1], kaputt[0]  # zwei Ziffern in einer Zeile tauschen bricht Spalten
    assert not ist_gueltige_loesung(kaputt)
    assert regelverstoesse(kaputt) > 0
    assert regelverstoesse(g) == 0


def test_batch_pruefer_stimmt_mit_einzelpruefer_ueberein():
    rng = np.random.default_rng(3)
    raetsel, loesungen, _ = datensatz(20, rng)
    kandidaten = loesungen.copy()
    kandidaten[::2, 5] = kandidaten[::2, 5] % 6 + 1  # jede zweite Lösung verfälschen
    erwartet = [ist_gueltige_loesung(k, r) for k, r in zip(kandidaten, raetsel)]
    assert gueltig_batch(kandidaten, raetsel).tolist() == erwartet


def test_gedaechtnis_findet_gleiches_und_verwandtes_raetsel():
    rng = np.random.default_rng(4)
    raetsel, loesungen, _ = datensatz(50, rng)
    g = HopfieldGedaechtnis()
    for r, l in zip(raetsel, loesungen):
        g.speichern(r, l)
    vorschlag, aehnlichkeit = g.abrufen(raetsel[17])
    assert np.array_equal(vorschlag, loesungen[17]) and aehnlichkeit > 0.99
    variante = raetsel[17].copy()
    leer = np.flatnonzero(variante == 0)[:2]
    variante[leer] = loesungen[17, leer]
    vorschlag, _ = g.abrufen(variante)
    assert np.array_equal(vorschlag, loesungen[17])


def test_agent_gibt_mit_werkzeug_nie_bestaetigte_falsche_antworten_ab():
    torch.manual_seed(0)
    raetsel, loesungen, _ = datensatz(40, np.random.default_rng(5))
    agent = Agent(DenkModell(d=16), max_schritte=3)  # untrainiert: fast alles falsch
    e = agent.loesen(raetsel, loesungen)
    assert not np.any(e.geprueft & ~e.richtig)
    assert np.all(e.antwort[raetsel > 0] == raetsel[raetsel > 0])  # Vorgaben bleiben stehen


def test_agent_lernt_sofort_ueber_das_gedaechtnis():
    raetsel, loesungen, _ = datensatz(20, np.random.default_rng(6))
    agent = Agent(DenkModell(d=16), max_schritte=1)
    for r, l in zip(raetsel[:10], loesungen[:10]):  # so tun, als hätte er diese schon gelöst
        agent.gedaechtnis.speichern(r, l)
    e = agent.loesen(raetsel, loesungen)
    assert np.all(e.quelle[:10] == "gedaechtnis") and np.all(e.richtig[:10]) and np.all(e.schritte[:10] == 0)


def test_training_senkt_den_verlust():
    torch.manual_seed(0)
    raetsel, loesungen, _ = datensatz(64, np.random.default_rng(7))
    r, l = torch.as_tensor(raetsel), torch.as_tensor(loesungen)
    modell = DenkModell(d=32)
    opt = torch.optim.AdamW(modell.parameters(), lr=3e-3)

    def gesamt():
        with torch.no_grad():
            return sum(t.item() for t, _ in verlust(modell, r, l, 2))

    vorher = gesamt()
    for _ in range(30):
        for teil, _ in verlust(modell, r, l, 2):
            opt.zero_grad()
            teil.backward()
            opt.step()
    assert gesamt() < 0.8 * vorher
