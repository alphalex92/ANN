import random
from collections import defaultdict

import torch

from denkkern import gedaechtnis as g
from denkkern.modell import Konfig, SchleifenGPT


def kleines_modell(plaetze=8):
    torch.manual_seed(0)
    return SchleifenGPT(Konfig(vokabular=g.VOKABULAR, kontext=g.KONTEXT, breite=32, koepfe=2,
                               positionen="keine", gedaechtnis=plaetze, gedaechtnis_k=4))


def test_welt_teilt_fest_und_fluechtig():
    welt = g.neue_welt(0)
    assert len(welt.kindheit) == 64 * 4 and len(welt.leben) == 64 * 4
    assert len(welt.gelernt) == 128 and welt.gelernt <= set(welt.kindheit)
    rng = random.Random(0)
    fest = welt.fest()
    e = g.ent(3)
    # feste Beziehung: in jeder Episode gleich; flüchtige: wechselt
    assert len({g.episode(rng, fest, g.ALLE).nachschlagen(e, g.rel(9)) for _ in range(20)}) == 1
    assert len({g.episode(rng, fest, g.ALLE).nachschlagen(e, g.rel(13)) for _ in range(20)}) > 5


def test_vorbild_weiss_nur_bekanntes():
    welt = g.neue_welt(0)
    fest = welt.fest()
    paar = sorted(welt.leben)[0]
    ep = g.episode(random.Random(0), fest, g.ALLE, start=paar[0], kette=[paar[1], g.rel(13)])
    weg = ep.weg()
    schritte = [t[-3:] for t, _ in g.folgen(ep, {paar})]
    assert schritte == [[g.WEISS, weg[1], g.PAD], [g.SUCHE, weg[1], g.rel(13)], [g.ANTWORT, weg[2], g.ENDE]]
    schritte = [t[-3:] for t, _ in g.folgen(ep, set())]
    assert schritte[0] == [g.SUCHE, paar[0], paar[1]]


def test_umgebung_mit_vorbildlichen_aktionen():
    welt = g.neue_welt(0)
    fest = welt.fest()
    rng = random.Random(1)
    for _ in range(50):
        ep = g.episode(rng, fest, g.NEUE)
        bekannt = {p for p in fest if p[1] in g.LEBEN_FEST}
        zustand, v = [ep.start, 0], g.Verlauf()
        for tokens, _ in g.folgen(ep, bekannt):
            g.schritt(ep, zustand, v, tokens[-3:])
        assert v.antwort == ep.antwort and v.falsch_gewusst == 0
        assert v.gewusst == v.neu_gewusst == sum(r in g.LEBEN_FEST for r in ep.kette)
        assert v.suchen == ep.hops - v.gewusst
        assert all(ep.nachschlagen(e, r) == y for e, r, y in v.beobachtet)


def test_falsches_wissen_ist_halluzination():
    welt = g.neue_welt(0)
    ep = g.episode(random.Random(0), welt.fest(), g.ALLE, kette=[g.rel(9)])
    falsch = g.ent(0) if ep.antwort != g.ent(0) else g.ent(1)
    zustand, v = [ep.start, 0], g.Verlauf()
    g.schritt(ep, zustand, v, [g.WEISS, falsch, g.PAD])
    g.schritt(ep, zustand, v, [g.ANTWORT, zustand[0], g.ENDE])
    assert v.falsch_gewusst == 1 and v.antwort == falsch != ep.antwort


def test_sichten():
    beobachtet = defaultdict(list, {"a": [1, 1], "b": [1, 2], "c": [3], "d": [1, 2, 2]})
    gesichert, verworfen = g.sichten(beobachtet, "pruefen")
    assert gesichert == {"a": 1, "d": 2} and verworfen == {"b"}
    gesichert, _ = g.sichten(beobachtet, "alles")
    assert gesichert == {"a": 1, "b": 2, "c": 3, "d": 2}


def test_traum_weiss_genau_das_gesicherte():
    welt = g.neue_welt(0)
    gesichert = dict(sorted(welt.leben.items())[:10])
    verworfen = {sorted(welt.leben)[20]}
    ziehen = g.traumziehen(welt, gesichert, verworfen)
    rng = random.Random(0)
    gesehen = set()
    for _ in range(300):
        ep, bekannt = ziehen(rng)
        weg = ep.weg()
        for i, r in enumerate(ep.kette):
            paar = (weg[i], r)
            gesehen.add(paar)
            if paar in gesichert:
                assert ep.nachschlagen(*paar) == gesichert[paar]
            assert (paar in bekannt) == (paar in gesichert or paar in welt.gelernt)
    assert set(gesichert) <= gesehen and verworfen <= gesehen


def test_wandel_aendert_lebensfakten():
    welt = g.neue_welt(0)
    vorher = dict(welt.leben)
    paare = g.wandeln(welt, 0.25, 5)
    assert len(paare) == 64
    assert all(welt.leben[p] != vorher[p] for p in paare)
    assert sum(welt.leben[p] != vorher[p] for p in vorher) == 64


def test_gedaechtnisschicht_und_alte_modelle():
    modell = kleines_modell()
    x = torch.randint(0, g.VOKABULAR, (3, 10))
    assert modell(x, 2).shape == (3, 10, g.VOKABULAR)
    ohne = SchleifenGPT(Konfig(vokabular=10, breite=32, koepfe=2))
    assert ohne.gedaechtnis is None


def test_schlaf_mit_gedaechtnis_aendert_nur_das_gedaechtnis():
    welt = g.neue_welt(0)
    modell = kleines_modell()
    vorher = {n: p.detach().clone() for n, p in modell.named_parameters()}
    gesichert = dict(sorted(welt.leben.items())[:10])
    g.schlafen(modell, welt, gesichert, set(), "gedaechtnis", schritte=3, seed=0)
    for n, p in modell.named_parameters():
        geaendert = not torch.equal(p, vorher[n])
        assert geaendert == (n == "gedaechtnis.werte.weight"), n
        assert p.requires_grad


def test_ein_kurzes_leben_laeuft():
    welt = g.neue_welt(0)
    modell = kleines_modell()
    tagebuch = g.leben(modell, welt, tage=1, fragen=8, schlaf_schritte=2, melden=lambda *_: None)
    assert [e["tag"] for e in tagebuch] == [1, 2]
    assert set(tagebuch[0]["neu"]) >= {"richtig", "suchen", "neu_gewusst", "falsch_gewusst"}


def test_kindheit_mit_mehreren_welten_laeuft():
    modell, info = g.kindheit(g.neue_welt(0), minuten=0.02, breite=32, plaetze=8, stapel=4, welten=2,
                              geraet="cpu", melden=lambda *_: None)
    assert info["welten"] == 2 and modell.gedaechtnis is not None
