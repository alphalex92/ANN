import copy

import numpy as np
import pytest
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


def test_9x9_raetsel_sind_eindeutig():
    raetsel, loesungen, tiefe = datensatz(5, np.random.default_rng(8), n=9)
    assert raetsel.shape == (5, 81)
    for r, l in zip(raetsel, loesungen):
        assert anzahl_loesungen(r) == 1 and ist_gueltige_loesung(l, r)


def test_symmetrie_varianten_bleiben_gueltig_und_gleich_schwer():
    from mini_kortex.sudoku import varianten
    for n in (6, 9):
        raetsel, loesungen, tiefe = datensatz(5, np.random.default_rng(9), n=n)
        vr, vl = varianten(raetsel, loesungen, np.random.default_rng(10))
        assert gueltig_batch(vl, vr).all()
        assert ((vr == 0).sum(1) == (raetsel == 0).sum(1)).all()
        assert [denktiefe(r) for r in vr] == tiefe.tolist()
        assert not np.array_equal(vr, raetsel)


def test_9x9_modell_und_agent_laufen():
    torch.manual_seed(0)
    raetsel, loesungen, _ = datensatz(6, np.random.default_rng(11), n=9)
    e = Agent(DenkModell(n=9, d=16), max_schritte=2).loesen(raetsel, loesungen)
    assert e.antwort.shape == (6, 81)
    assert not np.any(e.geprueft & ~e.richtig)


def test_gedaechtnis_merkt_sich_die_denkschritte():
    raetsel, loesungen, _ = datensatz(3, np.random.default_rng(12))
    g = HopfieldGedaechtnis()
    for r, l, s in zip(raetsel, loesungen, [1, 5, 9]):
        g.speichern(r, l, s)
    r, l, s = g.episoden(mindestens_schritte=5)
    assert s.tolist() == [5, 9] and np.array_equal(r, raetsel[1:]) and np.array_equal(l, loesungen[1:])


def test_schlaf_trainiert_eine_kopie_auf_eigenen_episoden():
    from mini_kortex.konsolidierung import schlafen
    torch.manual_seed(0)
    raetsel, loesungen, _ = datensatz(8, np.random.default_rng(13))
    modell = DenkModell(d=16)
    vorher = {k: v.clone() for k, v in modell.state_dict().items()}
    g = HopfieldGedaechtnis()
    for r, l in zip(raetsel, loesungen):
        g.speichern(r, l, 3)
    neu = schlafen(modell, g, alte=(raetsel, loesungen), laeufe=3, batch=8, schritte=2, protokoll=lambda *_: None)
    assert all(torch.equal(v, modell.state_dict()[k]) for k, v in vorher.items())  # Original unverändert
    assert any(not torch.equal(v, neu.state_dict()[k]) for k, v in vorher.items())  # Kopie hat gelernt


def test_training_setzt_beim_zwischenstand_fort(tmp_path):
    from mini_kortex.training import trainingslauf, ziehen_aus
    raetsel, loesungen, _ = datensatz(16, np.random.default_rng(14))
    ziehen = ziehen_aus(raetsel, loesungen, 4)
    still = dict(schritte=1, aufwaermen=1, alle=2, protokoll=lambda *_: None)

    torch.manual_seed(0)
    durchgehend = trainingslauf(DenkModell(d=8), ziehen, 4, **still)

    stand = tmp_path / "stand.pt"
    torch.manual_seed(0)
    trainingslauf(DenkModell(d=8), ziehen, 4, zwischenstand=stand, **still)  # schreibt Stand nach Lauf 2
    # "Absturz" nach Lauf 2 nachstellen: Ein frisches Modell setzt beim Zwischenstand fort.
    torch.manual_seed(1)
    fortgesetzt = trainingslauf(DenkModell(d=8), ziehen, 4, zwischenstand=stand, **still)
    for k, v in durchgehend.state_dict().items():
        assert torch.allclose(v, fortgesetzt.state_dict()[k], atol=1e-6), k


@pytest.mark.skipif(not torch.cuda.is_available(), reason="keine GPU")
def test_auf_der_gpu_wie_auf_der_cpu(tmp_path):
    from mini_kortex.training import genauigkeit_pro_schritt, laden, speichern, trainingslauf, ziehen_aus
    raetsel, loesungen, _ = datensatz(16, np.random.default_rng(15))
    torch.manual_seed(0)
    modell = DenkModell(d=16)
    auf_gpu = copy.deepcopy(modell).cuda()
    assert auf_gpu.geraet.type == "cuda"
    cpu = Agent(modell, max_schritte=3).loesen(raetsel, loesungen)
    gpu = Agent(auf_gpu, max_schritte=3).loesen(raetsel, loesungen)
    assert (cpu.antwort == gpu.antwort).mean() > 0.99  # bis auf winzige Rundungsunterschiede gleich
    trainingslauf(auf_gpu, ziehen_aus(raetsel, loesungen, 4), 2, schritte=1, aufwaermen=1, protokoll=lambda *_: None)
    acc = genauigkeit_pro_schritt(auf_gpu, raetsel, loesungen, 2)
    assert acc.shape == (2,)
    pfad = tmp_path / "m.pt"
    speichern(auf_gpu, pfad)
    assert laden(pfad).geraet.type == "cpu"  # Datei lädt auch ohne GPU


def test_lehrer_schlaf_uebt_auch_ungeloeste_raetsel():
    from mini_kortex.konsolidierung import schlafen
    torch.manual_seed(0)
    raetsel, loesungen, _ = datensatz(8, np.random.default_rng(16))
    modell = DenkModell(d=16)
    leer = HopfieldGedaechtnis()  # nichts selbst gelöst: Ohne Lehrer fällt der Schlaf aus
    still = dict(laeufe=2, batch=4, schritte=1, protokoll=lambda *_: None)
    ohne = schlafen(modell, leer, **still)
    mit = schlafen(modell, leer, episoden=(raetsel, loesungen), **still)
    assert all(torch.equal(v, ohne.state_dict()[k]) for k, v in modell.state_dict().items())
    assert any(not torch.equal(v, mit.state_dict()[k]) for k, v in modell.state_dict().items())


def test_diagonal_raetsel_sind_eindeutig_und_brauchen_die_neue_regel():
    raetsel, loesungen, _ = datensatz(10, np.random.default_rng(17), n=9, leer_min=40, leer_max=46, diagonal=True)
    assert gueltig_batch(loesungen, raetsel, diagonal=True).all()
    assert all(anzahl_loesungen(r, diagonal=True) == 1 for r in raetsel)
    kr, kl, _ = datensatz(10, np.random.default_rng(18), n=9)
    assert not gueltig_batch(kl, kr, diagonal=True).any()  # klassische Lösungen verletzen die Diagonalen


def test_diagonal_symmetrien_erhalten_beide_regeln():
    from mini_kortex.sudoku import varianten
    raetsel, loesungen, tiefe = datensatz(5, np.random.default_rng(19), n=9, diagonal=True)
    vr, vl = varianten(raetsel, loesungen, np.random.default_rng(20), diagonal=True)
    assert gueltig_batch(vl, vr, diagonal=True).all()
    assert [denktiefe(r, diagonal=True) for r in vr] == tiefe.tolist()
    kr, kl, _ = datensatz(5, np.random.default_rng(21), n=9)
    vr, vl = varianten(kr, kl, np.random.default_rng(22), diagonal=True)
    assert gueltig_batch(vl, vr).all()


def test_umlernen_laeuft_und_misst_beide_regeln():
    from mini_kortex import umlernen
    torch.manual_seed(0)
    wach_daten = datensatz(6, np.random.default_rng(23), n=9, diagonal=True)
    diag = datensatz(4, np.random.default_rng(24), n=9, diagonal=True)
    klass = datensatz(4, np.random.default_rng(25), n=9)
    modell = DenkModell(n=9, d=8)
    _, zeilen, wachphasen = umlernen.bedingung(modell, klass[:2], wach_daten, diag, klass, lehrer=True, replay=True,
                                               zyklen=2, laeufe=2, schritte=1, max_schritte=2,
                                               protokoll=lambda *_: None)
    assert len(zeilen) == 2 and set(zeilen[0][1]) == {k for k, _, _ in umlernen.SPALTEN}
    assert all(w["falsch_geprueft"] == 0 for w in wachphasen)


def test_mehrdeutige_diagonal_raetsel_brauchen_die_diagonalen():
    from mini_kortex.umlernen import nur_mehrdeutige
    r, l, _ = nur_mehrdeutige(5, np.random.default_rng(26), n=9, leer_min=36, leer_max=46)
    assert len(r) == 5
    assert all(anzahl_loesungen(x) > 1 and anzahl_loesungen(x, diagonal=True) == 1 for x in r)
    assert gueltig_batch(l, r, diagonal=True).all()


def test_regel_hinweis_startet_wirkungslos_und_wird_gelernt():
    from mini_kortex.konsolidierung import schlafen
    from mini_kortex.modell import mit_hinweis
    torch.manual_seed(0)
    raetsel, loesungen, _ = datensatz(8, np.random.default_rng(27), n=9, diagonal=True)
    alt = DenkModell(n=9, d=8)
    neu = mit_hinweis(alt)
    assert neu.hat_hinweis and not alt.hat_hinweis
    r = torch.as_tensor(raetsel)
    vorher = [y for y, *_ in alt.denken(r, 2)]
    nachher = [y for y, *_ in neu.denken(r, 2, torch.ones(len(r), dtype=torch.long))]
    assert all(torch.allclose(a, b) for a, b in zip(vorher, nachher))  # Hinweis startet bei null
    g = HopfieldGedaechtnis()
    for x, l in zip(raetsel, loesungen):
        g.speichern(x, l, 3)
    kr, kl, _ = datensatz(8, np.random.default_rng(28), n=9)
    gelernt = schlafen(neu, g, alte=(kr, kl), laeufe=3, batch=8, schritte=1, diagonal=True, regel_neu=1,
                       protokoll=lambda *_: None)
    assert gelernt.wahrnehmung.hinweis.weight.abs().sum() > 0
    e = Agent(gelernt, max_schritte=2, diagonal=True, regel=1).loesen(raetsel, loesungen)
    assert not np.any(e.geprueft & ~e.richtig)


def test_modell_mit_hinweis_laesst_sich_speichern_und_laden(tmp_path):
    from mini_kortex.modell import mit_hinweis
    from mini_kortex.training import laden, speichern
    m = mit_hinweis(DenkModell(n=9, d=8))
    speichern(m, tmp_path / "m.pt")
    assert laden(tmp_path / "m.pt").hat_hinweis
