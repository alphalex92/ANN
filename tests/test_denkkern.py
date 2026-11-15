import random

import pytest
import torch

from denkkern import messung, welt
from denkkern.modell import Konfig, SchleifenGPT
from denkkern.training import bereich, kontext_fuer, stapeln, trainieren


def episode(hops=3, seed=0):
    return welt.neue_episode(random.Random(seed), hops)


def test_welt_ist_eine_funktion():
    ep = episode()
    e, r = welt.ent(5), welt.rel(2)
    assert ep.nachschlagen(e, r) == ep.nachschlagen(e, r)
    assert welt.ist_ent(ep.nachschlagen(e, r))


def test_welten_unterscheiden_sich():
    rng = random.Random(1)
    a, b = welt.neue_episode(rng, 2), welt.neue_episode(rng, 2)
    e, r = welt.ent(0), welt.rel(0)
    antworten = {welt.neue_episode(rng, 1).nachschlagen(e, r) for _ in range(50)}
    assert len(antworten) > 10
    assert (a.start, a.kette) != (b.start, b.kette) or a.antwort != b.antwort


def test_werkzeug_kodierung_folgt_dem_weg():
    ep = episode(hops=3)
    tokens, eigen = welt.kodieren(ep, "werkzeug")
    weg = ep.weg()
    assert tokens[:4] == [welt.BOS, welt.FRAGE, ep.start, ep.kette[0]]
    rest = tokens[3 + ep.hops :]
    for i in range(ep.hops):
        assert rest[5 * i : 5 * i + 5] == [welt.SUCHE, weg[i], ep.kette[i], welt.TREFFER, weg[i + 1]]
    assert tokens[-3:] == [welt.ANTWORT, ep.antwort, welt.ENDE]
    # Gelernt werden die Suchen und die Antwort, nicht die Treffer und nicht die Frage.
    assert sum(eigen) == 3 * ep.hops + 3
    assert not any(eigen[: 3 + ep.hops])
    for i, t in enumerate(tokens):
        if t == welt.TREFFER:
            assert not eigen[i] and not eigen[i + 1]


def test_kontext_enthaelt_die_kette_und_ablenker():
    ep = episode(hops=4)
    fakten = ep.fakten()
    assert len(fakten) == welt.FAKTEN_IM_KONTEXT
    assert len({(e, r) for e, r, _ in fakten}) == len(fakten)
    weg = ep.weg()
    for i, r in enumerate(ep.kette):
        assert (weg[i], r, weg[i + 1]) in fakten
    assert ep.fakten() is fakten  # bleibt gleich
    tokens, eigen = welt.kodieren(ep, "kontext")
    assert tokens[-3:] == [welt.ANTWORT, ep.antwort, welt.ENDE]
    assert sum(eigen) == 3


def test_schritt_und_einordnen():
    ep = episode(hops=2)
    weg = ep.weg()
    v = welt.Verlauf()
    assert welt.schritt(ep, "werkzeug", v, [welt.SUCHE, weg[0], ep.kette[0]]) == [welt.TREFFER, weg[1]]
    assert welt.schritt(ep, "werkzeug", v, [welt.SUCHE, weg[1], ep.kette[1]]) == [welt.TREFFER, weg[2]]
    welt.schritt(ep, "werkzeug", v, [welt.ANTWORT, ep.antwort, welt.ENDE])
    assert v.fertig and v.weg_exakt and welt.einordnen(ep, "werkzeug", v) == "richtig"

    falsch = ep.antwort + 1 if ep.antwort + 1 < welt.VOKABULAR else ep.antwort - 1
    geraten = welt.Verlauf()
    welt.schritt(ep, "werkzeug", geraten, [welt.ANTWORT, falsch, welt.ENDE])
    assert welt.einordnen(ep, "werkzeug", geraten) == "geraten"

    kaputt = welt.Verlauf()
    welt.schritt(ep, "werkzeug", kaputt, [welt.ENDE, welt.PAD, welt.PAD])
    assert kaputt.ungueltig and welt.einordnen(ep, "werkzeug", kaputt) == "abgebrochen"

    # Im Modus kontext gibt es kein Werkzeug.
    ohne = welt.Verlauf()
    welt.schritt(ep, "kontext", ohne, [welt.SUCHE, weg[0], ep.kette[0]])
    assert ohne.ungueltig


def test_stapeln_maskiert_fremde_tokens():
    eps = [episode(h, s) for s, h in enumerate([1, 3, 2])]
    x, y = stapeln(eps, "werkzeug", "cpu")
    assert x.shape == y.shape
    for i, ep in enumerate(eps):
        tokens, eigen = welt.kodieren(ep, "werkzeug")
        assert (y[i] != -100).sum().item() == sum(eigen[1:])


def test_bereich():
    assert bereich("1-4") == (1, 4)
    assert bereich("3") == (3, 3)
    with pytest.raises(ValueError):
        bereich("4-1")


@pytest.mark.parametrize("positionen", ["rope", "keine"])
def test_modell_laeuft_mit_beliebig_vielen_schleifen(positionen):
    k = Konfig(vokabular=welt.VOKABULAR, kontext=64, breite=32, koepfe=2, positionen=positionen)
    m = SchleifenGPT(k)
    idx = torch.randint(0, welt.VOKABULAR, (2, 20))
    for s in (1, 3, 9):
        assert m(idx, s).shape == (2, 20, welt.VOKABULAR)
    # Kausal: Spätere Tokens ändern frühere Logits nicht.
    anders = idx.clone()
    anders[:, 10:] = 0
    assert torch.allclose(m(idx, 2)[:, :10], m(anders, 2)[:, :10], atol=1e-5)


def test_speichern_und_laden(tmp_path):
    m = SchleifenGPT(Konfig(vokabular=welt.VOKABULAR, kontext=32, breite=32, koepfe=2))
    m.speichern(tmp_path / "m.pt", modus="werkzeug")
    neu, info = SchleifenGPT.laden(tmp_path / "m.pt")
    idx = torch.randint(0, welt.VOKABULAR, (1, 8))
    assert info["modus"] == "werkzeug"
    assert torch.allclose(m(idx, 2), neu(idx, 2))


@pytest.mark.parametrize("modus", welt.MODI)
def test_ausrollen_und_messen_ohne_absturz(modus):
    max_hops = 3
    k = Konfig(vokabular=welt.VOKABULAR, kontext=kontext_fuer(modus, max_hops), breite=32, koepfe=2)
    m = SchleifenGPT(k)
    ergebnis = messung.messen(m, modus, [1, max_hops], [1, 2], anzahl=6, stapel=4)
    for r in ergebnis.values():
        assert sum(r["kategorien"].values()) == 6
        assert set(r["kategorien"]) <= set(welt.KATEGORIEN[modus])
    assert "Hops" in messung.tabelle_richtig(ergebnis, [1, max_hops], [1, 2], (1, 2))
    assert "Hops" in messung.tabelle_kategorien(ergebnis, modus, [1, max_hops], 2)


def test_training_lernt_ein_hop_mit_werkzeug():
    """Ein Hop mit Werkzeug ist reines Abschreiben, das schafft ein winziges Modell schnell."""
    torch.manual_seed(0)
    modell, info = trainieren("werkzeug", hops="1", schleifen="1", minuten=0.4, stapel=64, breite=64,
                              kern=1, geraet="cpu", mess_hops=1, melden=lambda *_: None)
    assert info["verlust"] < 1.0
    ergebnis = messung.messen(modell, "werkzeug", [1], [1], anzahl=50)
    assert ergebnis[(1, 1)]["kategorien"]["richtig"] >= 30


def test_restkette_kodierung_zeigt_was_offen_ist():
    ep = episode(hops=3)
    tokens, eigen = welt.kodieren(ep, "restkette")
    weg = ep.weg()
    text = welt.lesbar(tokens)
    for i in range(ep.hops):
        offen = " ".join(welt.lesbar([r]) for r in ep.kette[i + 1 :])
        treffer = f"= {welt.lesbar([weg[i + 1]])} |" + (f" {offen}" if offen else "")
        assert treffer in text
    assert sum(eigen) == 3 * ep.hops + 3
    for i, t in enumerate(tokens):
        if t == welt.REST:
            assert not eigen[i]


def test_restkette_streicht_nur_bei_richtiger_suche():
    ep = episode(hops=2)
    weg = ep.weg()
    v = welt.Verlauf()
    falsche_rel = welt.rel((ep.kette[0] - welt.REL0 + 1) % welt.RELATIONEN)
    antwort = welt.schritt(ep, "restkette", v, [welt.SUCHE, weg[0], falsche_rel])
    assert antwort[2:] == [welt.REST, *ep.kette] and not v.weg_exakt
    antwort = welt.schritt(ep, "restkette", v, [welt.SUCHE, weg[0], ep.kette[0]])
    assert antwort == [welt.TREFFER, weg[1], welt.REST, ep.kette[1]]
    antwort = welt.schritt(ep, "restkette", v, [welt.SUCHE, weg[1], ep.kette[1]])
    assert antwort == [welt.TREFFER, weg[2], welt.REST]
    assert v.erledigt == 2 and v.suchen == 3


class Orakel(torch.nn.Module):
    """Ein „Modell“, das die vorbildliche Lösung abspielt: prüft, dass ausrollen mit
    verschieden langen Folgen jede Folge an ihrer eigenen Stelle fortsetzt."""

    def __init__(self, episoden, modus, kontext):
        super().__init__()
        self.loesungen = [[f for f, _ in welt.folgen(ep, modus)] for ep in episoden]
        self.konfig = Konfig(vokabular=welt.VOKABULAR, kontext=kontext)
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, idx, schleifen):
        B, T = idx.shape
        logits = torch.zeros(B, T, welt.VOKABULAR)
        for b, loesungen in enumerate(self.loesungen):
            for loesung in loesungen:
                for t in range(min(T, len(loesung) - 1)):
                    if idx[b, : t + 1].tolist() == loesung[: t + 1]:
                        logits[b, t, loesung[t + 1]] = 1.0
        return logits


@pytest.mark.parametrize("modus", ["werkzeug", "restkette", "kompakt"])
def test_ausrollen_mit_orakel_ist_immer_richtig(modus):
    rng = random.Random(3)
    episoden = [welt.neue_episode(rng, h) for h in (1, 4, 2, 3)]
    orakel = Orakel(episoden, modus, kontext_fuer(modus, 4))
    verlaeufe, folgen = messung.ausrollen(orakel, episoden, modus, schleifen=1)
    for ep, v, f in zip(episoden, verlaeufe, folgen):
        assert welt.einordnen(ep, modus, v) == "richtig"
        assert v.weg_exakt and v.suchen == ep.hops
        assert f == welt.folgen(ep, modus)[-1][0]


def test_kompakt_ist_eine_folge_je_schritt():
    ep = episode(hops=3)
    schritte = welt.folgen(ep, "kompakt")
    weg = ep.weg()
    assert len(schritte) == ep.hops + 1
    frage = [welt.BOS, welt.FRAGE, ep.start, *ep.kette]
    for i, (tokens, eigen) in enumerate(schritte):
        assert tokens[: len(frage)] == frage
        assert tokens[len(frage) : len(frage) + 3] == [welt.TREFFER, weg[i], welt.REST]
        assert sum(eigen) == 3 and all(eigen[-3:])
    assert schritte[-1][0][-3:] == [welt.ANTWORT, ep.antwort, welt.ENDE]
    assert welt.anfang(ep, "kompakt") == schritte[0][0][:-3]
    with pytest.raises(ValueError):
        welt.kodieren(ep, "kompakt")
